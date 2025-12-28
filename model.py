import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

"""
input_channels=111,    # Env Input
num_moves=4672,        # Discrete Output
d_model=384,           # Embedding size
n_head=12, 
d_ff=768, 
num_layers=16, 
dropout=0,
lr=3e-4,
weight_decay=1e-4
"""

class ResBlock(nn.Module):
    """
    Standard ResNet block for the Policy Head.
    Preserves spatial resolution (8x8).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.res_weight = nn.Parameter(torch.tensor([0.0]))
        self.res = nn.Sequential(
            nn.GroupNorm(1, in_channels), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.GroupNorm(1, out_channels), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.res_weight * self.res(x) + self.skip(x)

class ff_linear(nn.Module):
    """
    MLP with skip connections
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.block_weight = nn.Parameter(torch.tensor([0.0]))
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim)
        )
        
        self.skip = nn.Identity()
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.block_weight * self.block(x) + self.skip(x)

class Smolgen(nn.Module):
    def __init__(self, d_model, n_head, d_smol=16):
        super().__init__()
        
        self.n_head = n_head
        self.d_smol = d_smol
        
        # 1. Project to a smaller 'coordinate' space
        self.q_proj = nn.Linear(d_model, n_head * d_smol, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * d_smol, bias=False)
        
        # 2. Learnable scalar to scale the bias
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_smol)))
        
    def forward(self, x):
        # x: (Batch, Seq_Len=64, d_model)
        b, seq, dim = x.shape
        
        # Generate spatial queries and keys
        # Shape: (B, 64, n_head * d_smol) -> (B, 64, n_head, d_smol)
        q = self.q_proj(x).view(b, seq, self.n_head, self.d_smol)
        k = self.k_proj(x).view(b, seq, self.n_head, self.d_smol)
        
        # Permute for dot product: (B, n_head, 64, d_smol)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        
        # Calculate pair-wise relationships
        # (B, n_head, 64, d_smol) @ (B, n_head, d_smol, 64) -> (B, n_head, 64, 64)
        bias = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Flatten batch and head for PyTorch's attn_mask compatibility
        # Output: (B * n_head, 64, 64)
        return bias.reshape(b * self.n_head, seq, seq)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # SwiGLU has 3 linear layers instead of 2.
        # To keep parameter count comparable to standard FFN, 
        # hidden_dim is usually 2/3 of standard d_ff.
        self.gate_value = nn.Linear(dim, hidden_dim * 2, bias=False) # w1 and w2
        self.w3 = nn.Linear(hidden_dim, dim, bias=False) # Output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        gate_value = self.gate_value(x) 
        gate, value = gate_value.chunk(2, dim=-1)
        
        return self.dropout(self.w3(F.silu(gate) * value))

class SwiGLUTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        
        self.smolgen = Smolgen(d_model, n_head)
        self.n_head = n_head
        
        self.dropout = nn.Dropout(dropout)

    # Arguments match nn.TransformerEncoder's expectations
    def forward(self, src):
        batch_size, tokens, _ = src.shape
        norm_x = self.norm1(src)
        
        # Get smolgen
        spatial_tokens = src[:, 1:, :]
        bias_64 = self.smolgen(spatial_tokens)        
        full_bias = F.pad(bias_64, (1, 0, 1, 0), "constant", 0)
        
        # Run attn
        attn_out, _ = self.attn(
            query=norm_x, 
            key=norm_x, 
            value=norm_x, 
            key_padding_mask=None,
            attn_mask=full_bias,
            is_causal=False,
            need_weights=False
        )
        
        # Residual connection
        x = src + self.dropout(attn_out)
        
        # 2. SwiGLU FFN Block
        # Norm -> SwiGLU -> Add
        x = x + self.ffn(self.norm2(x))
        
        return x

class TinyChessTransformer(nn.Module):
    def __init__(
        self, 
        input_channels=111,    # Env Input
        num_moves=4672,        # Discrete Output
        d_model=512,           # Embedding size
        n_head=16, 
        d_ff=512, 
        num_layers=28, 
        dropout=0.0,
        lr=1e-4,
        weight_decay=1e-4,
        logit_cap=5,
    ):
        super().__init__()
        
        # 1. Tokenizer & Embedding
        self.input_projection = nn.Sequential(
            ResBlock(input_channels, d_model),
            ResBlock(d_model, d_model)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # CLS token predicts value
        self.pos_embedding = nn.Parameter(torch.randn(1, 64+1, d_model)) # +1 for cls token

        # 2. Transformer Body (BT4-style Pre-Norm)
        self.attn_layers = nn.ModuleList([
            SwiGLUTransformerEncoderLayer(
                d_model=d_model, 
                n_head=n_head, 
                d_ff=d_ff, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.post_attn_norm = nn.LayerNorm(d_model)

        # P head
        self.policy_conv_block = nn.Sequential(
            ResBlock(d_model, d_model),
            nn.Conv2d(d_model, int(num_moves / 64), kernel_size=1) # 1x1 Conv to map to 73 move planes
        )
        self.logit_cap = logit_cap

        # Value head
        self.value_head = nn.Sequential(
            ff_linear(d_model, 1024),
            ff_linear(1024, 1024),
            nn.Linear(1024, 3)
        )
        
        # Optim
        self.optim = self.configure_optimizers(weight_decay, lr)
    
    def forward(self, x, training: bool):
        """
        x: (Batch, 111, 8, 8)
        Returns: 
          policy_logits: (Batch, 4672)
          value: (Batch, 1)
        """
        b, c, h, w = x.shape
        
        # --- Embedding ---
        # new x: (B, d_model, 8, 8)
        x = self.input_projection(x)
        
        # flatten to (B, 64, d_model)
        x = x.flatten(2).permute(0, 2, 1)
        
        # add CLS token
        cls_token = self.cls_token.expand(b, -1, -1)
        
        # concat cls token
        # x: (B, 65, d_model)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add Pos and Normalize
        x = x + self.pos_embedding
        
        # --- Transformer Body ---
        # x: (B, 64, d_model)
        for layer in self.attn_layers:
            if training:
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = layer(x)
        
        x = self.post_attn_norm(x)
        
        # --- Split CLS token ---
        cls_out = x[:, 0, :]
        spatial_out = x[:, 1:, :] # (B, 64, d_model)

        # --- Policy Head ---
        # Reshape tokens back to spatial grid: (B, d_model, 8, 8)
        spatial_img = spatial_out.permute(0, 2, 1).view(b, -1, h, w)
        
        # Convolutional Policy: Output (B, 73, 8, 8)
        p = self.policy_conv_block(spatial_img)
        
        # Flatten to discrete moves
        # Note reshape to (B, 8, 8, 73) as that is output desired
        raw_policy_logits = p.permute(0, 3, 2, 1).flatten(1)
        
        # Clip logits
        clip_policy_logits = self.logit_cap * torch.tanh(raw_policy_logits / self.logit_cap)

        # --- Value Head ---
        value = self.value_head(cls_out)

        return clip_policy_logits, value, raw_policy_logits
    
    def save_checkpoint(self, path):
        torch.save({
            'model': self.state_dict(),
            'optim': self.optim.state_dict(),
            # 'scaler': self.scaler.state_dict()
        }, path)
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        # self.scaler.load_state_dict(checkpoint['scaler'])
    
    def configure_optimizers(self, weight_decay, learning_rate):
        # 1. Create empty lists
        decay_params = []
        nodecay_params = []
        
        # 2. Iterate over named parameters
        for pn, p in self.named_parameters():
            if p.requires_grad:
                
                # CHECK 1: Dimensions
                # - Weights are usually >= 2D
                # - Biases and Norms are 1D
                is_bias_or_norm = p.dim() < 2
                
                # CHECK 2: Explicit Names to EXCLUDE from decay
                # If 'pos_embedding' is in the name, force it to no_decay
                is_pos_embed = 'pos_embedding' in pn
                is_cls_token = 'cls_token' in pn

                # Logic: If it's big (2D+) AND NOT an embedding, decay it.
                if is_bias_or_norm or is_pos_embed or is_cls_token:
                    nodecay_params.append(p)
                else:
                    decay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(optim_groups, lr=learning_rate)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyChessTransformer().to(device)
    
    # Dummy Input: 111 channels
    dummy_input = torch.randn(1024, 111, 8, 8).to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        policy, val, _ = model(dummy_input, training=True)
        loss = policy.mean()
        
        # This backward pass uses the FlashAttention gradient kernels
        loss.backward() 
    
    import time
    t = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        policy, val, _ = model(dummy_input, training=True)
        loss = policy.mean()
        
        # This backward pass uses the FlashAttention gradient kernels
        loss.backward() 
    print(f"{(time.time() - t)}s")
    
    print(f"Policy Shape: {policy.shape}") # Expect [4, 4672]
    print(f"Value Shape:  {val.shape}")    # Expect [4, 1]
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    
    print(f"MAX: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")