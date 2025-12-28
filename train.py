import numpy as np
import torch, math, chess, random, time
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from stockfish import Stockfish

from model import *
from chess_fn import *
from chess_env import chess_gym

BATCH_SIZE = 768
GROUP_BATCH_SIZE = 768 * 4 # INCREASE THIS AS TRAINING GOES ON
EPSILON = 0.2
PPO_RETRAIN_RATIO = 5
KL_LIMIT = 0.015
LAMBDA = 0.99
GAMMA = 0.999
WARMUP_STEPS = 100
TARGET_ENTROPY_BOUND = 0.8
ENTROPY_MULTIPLIER = 1.02

# Gen data vars
TOTAL_GAMES = 64
SELF_PLAY_GAMES = math.ceil(TOTAL_GAMES * 0.8)
MODEL_V_MODEL_GAMES = 2 * (TOTAL_GAMES - SELF_PLAY_GAMES)
TOTAL_ENVS = SELF_PLAY_GAMES + MODEL_V_MODEL_GAMES

MAX_PLYS = 230 # Trim at 115 moves
TEMPERATURE = 2.0
TEMP_DECAY = 0.98
MIN_TEMP = 0.33

# Chess stats
SF_DEPTH = 14
MAX_NO_MATE_EVAL = 0.8
MAX_HAVE_MATE_EVAL = 0.9
MOVE_PENALTY = 0.001 # Max count is 100 (50 plys)
PUZZLE_FEN_PROB = 0.85
PUZZLE_MAX_PLYS = 16 # 8 moves max

VALUE_LOSS_COEF = 0.25

LOGIT_LOSS_COEF = 5e-4
LOGIT_CAP = 5.0

ENTROPY_COEF = 0.01 # Set to last entropy shown in console
EPOCHS_TRAINED = 2900

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model = TinyChessTransformer(
    logit_cap=LOGIT_CAP
).to(device)
torch.set_float32_matmul_precision('high')

eff_dtype = torch.bfloat16

# Compile model
torch._inductor.config.max_autotune_gemm = False
model.load_checkpoint("models/chess.pt")
model = torch.compile(
    model,
    options={
        "triton.cudagraphs": False,
        "max_autotune": False,
        "epilogue_fusion": True,
        "shape_padding": True,
    }
)

decay_sched = lr_scheduler.CosineAnnealingLR(
    model.optim, T_max=30000, eta_min=5e-7, last_epoch=EPOCHS_TRAINED-1
)
warmup_steps = WARMUP_STEPS - EPOCHS_TRAINED

if warmup_steps > 0:
    warmup_sched = lr_scheduler.LinearLR(
        model.optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps, last_epoch=EPOCHS_TRAINED
    )

    scheduler = lr_scheduler.SequentialLR(
        model.optim, 
        schedulers=[warmup_sched, decay_sched], 
        milestones=[warmup_steps],
    )
else:
    scheduler = decay_sched

# Create old models
model_names = ["smoke", "flame", "anchor"]

old_models = {
    name: TinyChessTransformer().to('cpu') for name in model_names
}

# load model
for name, m in old_models.items():
    m.load_state_dict(torch.load(f"models/{name}.pt", map_location=device, weights_only=True))
    m.eval()
    
# for v in old_models.values():
#     v.load_state_dict(model._orig_mod.state_dict())
#     v.eval()

# Initialise the envs for gen data
env_list = [chess_gym() for _ in range(TOTAL_ENVS)]
all_obs = [e.reset()[0] for e in env_list] # reset returns (obs, info)
split_idx = SELF_PLAY_GAMES

# Main model side will always show the side of our main model even if game ends and we swap sides. For stockfish eval at the end
main_model_side = torch.tensor([random.choice([True, False]) for _ in range(TOTAL_ENVS - split_idx)], dtype=torch.bool, device=device) # True is white false is Black
obs_mask = torch.full((TOTAL_ENVS,), dtype=torch.bool, fill_value=True, device=device)
obs_mask[split_idx:] = main_model_side
puzzle_boards = torch.tensor([False for _ in range(SELF_PLAY_GAMES)], dtype=torch.bool, device='cpu')

# temperature
temp_tensor = torch.full((TOTAL_ENVS,), dtype=torch.float32, fill_value=TEMPERATURE, device=device) / TEMP_DECAY # we decay the first time
min_temp_tensor = torch.tensor(MIN_TEMP, dtype=torch.float32, device=device)

# buffers for current episode being generated per environment
current_obs = [[] for _ in range(TOTAL_ENVS)]
current_actions = [[] for _ in range(TOTAL_ENVS)]
current_log_probs = [[] for _ in range(TOTAL_ENVS)]
current_dones = [[] for _ in range(TOTAL_ENVS)]
current_rewards = [[] for _ in range(TOTAL_ENVS)] # We store 0 initially, backfill later
current_illegal_mask = [[] for _ in range(TOTAL_ENVS)]
current_is_self = [[] for _ in range(TOTAL_ENVS)]
    
rival_epochs = [20, 300]
rival_names = ["smoke", "flame"]

# Load puzzles
with open('puzzles.txt', 'r', encoding='utf-8') as f:
    # Use a list comprehension for speed
    puzzles = [line.strip() for line in f]

try:
    sf = Stockfish(path="stockfish.exe", parameters={
        "Threads": 8,
        "Hash": 128,
    }) # Adjust path if on Linux/Mac
    sf.set_depth(SF_DEPTH) # Set appropriate depth
except Exception as e:
    raise RuntimeError(f"Stockfish failed to load! {e}")

def gen_parallel_data(stockfish_traj: int, model2, print_ratio=50, max_samples=8192):
    global env_list, split_idx, main_model_side, obs_mask, temp_tensor, min_temp_tensor, all_obs
    global current_actions, current_dones, current_illegal_mask, current_is_self, current_log_probs, current_obs, current_rewards

    # Storage for completed episodes (flattened later)
    completed_data = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "dones": [],
        "rewards": [],
        "illegal_mask": [],
        "is_self": []
    }

    # Initialize envs
    total_samples = sum(len(x) for x in current_rewards)
    
    # Random moves
    num_random_envs = SELF_PLAY_GAMES // 2
    for i in range(num_random_envs):
        for _ in range(12):
            # Get random legal action
            action = env_list[i].get_random_legal_move()
            # Step the environment (we ignore rewards/done here, just setting the stage)
            obs, _, term, trunc, _ = env_list[i].step(action)
            
            # If the game somehow ends in 6 moves, reset and stop randomizing for this env
            if term or trunc:
                all_obs[i], _ = env_list[i].reset()
                break
            else:
                all_obs[i] = obs
    
    # Helper to calculate alternating rewards for a finished episode
    def backfill_rewards(rewards_list, final_reward, alternate: bool):
        alt_value = -1
        if not alternate:
            alt_value = 1
        
        num_steps = len(rewards_list)
        filled_rewards = np.zeros(num_steps, dtype=np.float32)
        # Iterate backwards
        for i in range(num_steps):
            # i=0 is last move (creates 1), i=1 is prev move (creates -1), etc.
            filled_rewards[num_steps - 1 - i] = (alt_value**i) * final_reward
        return filled_rewards.tolist()

    # Helper to move an episode from current buffers to completed storage
    def flush_episode(env_idx, final_reward):
        # We check if env idx is more than or equal to split idx
        rewards = backfill_rewards(current_rewards[env_idx], final_reward, alternate=(env_idx < split_idx))
        
        # Extend completed data
        completed_data["observations"].extend(current_obs[env_idx])
        completed_data["actions"].extend(current_actions[env_idx])
        completed_data["log_probs"].extend(current_log_probs[env_idx])
        completed_data["dones"].extend(current_dones[env_idx])
        completed_data["rewards"].extend(rewards)
        completed_data["illegal_mask"].extend(current_illegal_mask[env_idx])
        completed_data["is_self"].extend(current_is_self[env_idx])
        
        # Clear buffers for this env
        current_obs[env_idx] = []
        current_actions[env_idx] = []
        current_log_probs[env_idx] = []
        current_dones[env_idx] = []
        current_rewards[env_idx] = []
        current_illegal_mask[env_idx] = []
        current_is_self[env_idx] = []

    # --- Main Generation Loop ---
    end_indices = []
    while total_samples < max_samples:
        for step in range(stockfish_traj):     
            # Check for max samples
            if total_samples >= max_samples:
                break
            
            # Decay temp
            temp_tensor *= TEMP_DECAY if step != 0 else 1
            
            # Handle game ends
            for idx in end_indices:
                main_model_side[idx - split_idx] = random.choice([True, False]) # shuffle sides
                obs_mask[idx] = main_model_side[idx - split_idx]
            end_indices = []
            
            # 1. Batch Inference
            mask_array = [e.get_illegal_mask() for e in env_list]
            unmasked_obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
            unmasked_illegal_mask = torch.tensor(np.array(mask_array), dtype=torch.bool, device=device)
            
            obs_tensor = unmasked_obs_tensor[obs_mask]
            illegal_mask = unmasked_illegal_mask[obs_mask]
            masked_temp = temp_tensor[obs_mask]
            
            with torch.inference_mode(), torch.amp.autocast(device_type=device_name, dtype=eff_dtype):
                logits, _, _ = model._orig_mod(obs_tensor, training=False)
                
                # Apply temperature
                logits_temp = logits / torch.maximum(masked_temp, min_temp_tensor).unsqueeze(1)
                
                # Mask illegal logits
                logits = logits.masked_fill(illegal_mask, -1e9)
                logits_temp = logits_temp.masked_fill(illegal_mask, -1e9)
                
                # Since we get from logits_temp, we need to get original log probs too
                logits = torch.log_softmax(logits, dim=-1)
                
                # Optimized: Create one batch distribution instead of list comprehension
                dist = torch.distributions.Categorical(logits=logits_temp)
                actions = dist.sample() # (obs_mask trues,)
                
                log_probs = torch.gather(logits, 1, actions.unsqueeze(1)).view(-1)
                
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            # 2. Step Environments
            k_loop = torch.where(obs_mask)[0].tolist()
            total_samples += len(k_loop)
            for j in range(len(k_loop)):
                k = k_loop[j]
                action = actions_np[j]
                log_prob = log_probs_np[j] # j here because log probs isnt size envs
                
                # Store data BEFORE step (Current Observation)
                current_obs[k].append(all_obs[k])
                current_actions[k].append(action)
                current_log_probs[k].append(log_prob)
                current_rewards[k].append(0) # Placeholder
                current_illegal_mask[k].append(mask_array[k])
                
                # Step
                next_o, reward, termination, truncation, info = env_list[k].step(action)
                done = termination or truncation
                
                # Truncate long games
                if k < SELF_PLAY_GAMES and not puzzle_boards[k]:
                    done = done or env_list[k].get_num_moves() > MAX_PLYS
                else:
                    done = done or env_list[k].get_num_moves() > PUZZLE_MAX_PLYS
                
                current_dones[k].append(done)
                current_is_self[k].append(k < split_idx)

                if done:
                    # Episode finished normally
                    flush_episode(k, reward)
                    # Reset environment immediately
                    all_obs[k], _ = env_list[k].reset()
                    
                    # Reset temp
                    temp_tensor[k] = TEMPERATURE # (there will be auto temp decay at the start)
                    
                    # Append to done indices
                    if k >= split_idx:
                        end_indices.append(k)
                    else:
                        puzzle_boards[k] = False
                        if random.random() < PUZZLE_FEN_PROB:
                            # Set a random puzzle
                            try:
                                rand_puzzle_fen = random.choice(puzzles)
                                
                                if random.random() < 0.5:
                                    rand_puzzle_fen = mirror_fen(rand_puzzle_fen)
                                
                                all_obs[k], _ = env_list[k].reset(fen=random.choice(puzzles))
                                puzzle_boards[k] = True
                                    
                            except ValueError:
                                # Invalid board
                                all_obs[k], _ = env_list[k].reset()
                                puzzle_boards[k] = False
                else:
                    # Update observation for next iteration
                    all_obs[k] = next_o
            
            # We do the same for opposite
            if not (~obs_mask).any():
                obs_mask[split_idx:] = ~obs_mask[split_idx:]
                continue
            
            obs_tensor = unmasked_obs_tensor[~obs_mask]
            illegal_mask = unmasked_illegal_mask[~obs_mask]
            masked_temp = temp_tensor[~obs_mask]
            
            with torch.inference_mode(), torch.amp.autocast(device_type=device_name, dtype=eff_dtype):
                logits, _, _ = model2(obs_tensor, training=False)
                
                # Apply temperature
                logits = logits / torch.maximum(masked_temp, min_temp_tensor).unsqueeze(1)
                
                # Mask illegal logits
                logits = logits.masked_fill(illegal_mask, -1e9)
                
                # Optimized: Create one batch distribution instead of list comprehension
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample() # (~obs_mask true size,)
                
            actions_np = actions.cpu().numpy()

            # 2. Step Environments
            k_loop = torch.where(~obs_mask)[0].tolist()
            for j in range(len(k_loop)):
                k = k_loop[j]
                action = actions_np[j]
                
                # There is no need to store data here: this is the other model
                
                # Step
                next_o, reward, termination, truncation, info = env_list[k].step(action)
                done = termination or truncation

                if done:
                    # Episode finished normally
                    # We flush negative reward as this is opponent playing but our data is us playing
                    flush_episode(k, -reward)
                    # Reset environment immediately
                    all_obs[k], _ = env_list[k].reset()
                    
                    # Reset temp
                    temp_tensor[k] = TEMPERATURE
                    
                    # Add to done indices
                    if k >= split_idx:
                        end_indices.append(k)
                else:
                    # Update observation for next iteration
                    all_obs[k] = next_o
            
            # We need to flip the side (so model wont get eval opponents side)
            obs_mask[split_idx:] = ~obs_mask[split_idx:]

        # --- Post-Loop: Handle Ongoing Games with Stockfish ---
        for k in range(TOTAL_ENVS):
            # Dont eval puzzle boards
            if k < SELF_PLAY_GAMES:
                if puzzle_boards[k]:
                    continue
            
            if len(current_obs[k]) > 0: # If there is data in the buffer
                final_reward = 0
                
                if sf:
                    # Get FEN from environment (Assuming env has get_fen() or similar)
                    # If your env doesn't have get_fen(), you might need access to env.board.fen()
                    try:
                        fen = env_list[k].get_fen() 
                        if k >= split_idx: # if we are looking at one of the model v old model
                            turn = 1 if main_model_side[k - split_idx] else -1 # True if white False if black
                        else:
                            turn = -1 * env_list[k].get_turn()  # youre looking at the other perspective, last player was diff.
                        sf.set_fen_position(fen)
                        eval_data = sf.get_evaluation()
                        
                        score = eval_data['value']
                        if eval_data['type'] == 'mate':
                            # If mate in X, it's a win/loss (mate > 0 is win for white)
                            final_reward = 1 if score > 0 else -1
                            final_reward *= max(GAMMA ** MAX_HAVE_MATE_EVAL, MAX_NO_MATE_EVAL) # Clipped to max no mate eval
                        else:
                            # Centipawns
                            final_reward = stockfish_win_probability(score, max_eval=MAX_NO_MATE_EVAL)
                        
                        # Apply move loss
                        num_moves = env_list[k].get_50move_clock()
                        if final_reward < 0:
                            final_reward = min(final_reward + MOVE_PENALTY * num_moves, 0)
                        elif final_reward > 0:
                            final_reward = max(final_reward - MOVE_PENALTY * num_moves, 0)
                        
                        # If its blacks turn, a final_reward = 1 (white winning) should be -1
                        final_reward *= turn
                        
                        if k % print_ratio == 0:
                            print(f"{env_list[k].get_fen():<90}{score}")
                        
                    except Exception as e:
                        print(f"Error evaluating env {k}: {e}")
                        final_reward = 0
                
                # Flush the unfinished episode with the estimated reward
                # We treat the truncation as a 'done' here
                current_dones[k][-1] = True 
                flush_episode(k, final_reward)

    # Convert lists to numpy arrays
    return {
        "observations": np.array(completed_data["observations"], dtype=np.float32),
        "actions": np.array(completed_data["actions"], dtype=np.int64),
        "log_probs": np.array(completed_data["log_probs"], dtype=np.float32),
        "dones": np.array(completed_data["dones"], dtype=np.int32),
        "rewards": np.array(completed_data["rewards"], dtype=np.float32),
        "illegal_mask": np.array(completed_data["illegal_mask"], dtype=np.int32),
        "is_self": np.array(completed_data["is_self"], dtype=np.int32),
    }

def transform_rewards(full_rewards):
    # Ensure input is a float tensor
    x = full_rewards.float()
    
    # Weight for index 0 (Value -1):
    # If x is -0.5, w0 is 0.5. If x is positive, w0 is 0.0.
    w0 = torch.clamp(-x, min=0.0, max=1.0)
    
    # Weight for index 2 (Value 1):
    # If x is 0.7, w2 is 0.7. If x is negative, w2 is 0.0.
    w2 = torch.clamp(x, min=0.0, max=1.0)
    
    # Weight for index 1 (Value 0):
    # If x is 0, w1 is 1.0. If x is 0.7, w1 is 0.3.
    w1 = 1.0 - (w0 + w2)
    
    # Stack together: (Batch, 3)
    return torch.stack([w0, w1, w2], dim=-1)

def train(epochs: int, rivals: list, offset=0):
    global ENTROPY_COEF
    
    rival_odds = [0.75, 0.2, 0.05]
    for i in range(epochs):
        t_start = time.time()
        sel_model = random.choices(rivals, weights=rival_odds, k=1)[0]
        model.eval()
        sel_model.to(device)
        torch.cuda.empty_cache()
        
        # We should randomise trajectory lengths, but still ensure total samples is 8192
        # Equation for total sampels: for g(a, b, model2, c), total_s = (a + 0.5c) * b
        traj_len = random.randint(16, 30)        
        data = gen_parallel_data(stockfish_traj=traj_len, model2=sel_model, max_samples=GROUP_BATCH_SIZE) # model2 count is halved
        model.train() 
        sel_model.to('cpu')
        print("Training on gen data...")
        
        # Get full data
        full_obs = torch.tensor(data["observations"], dtype=torch.float32, device='cpu')
        full_actions = torch.tensor(data["actions"], dtype=torch.long, device='cpu')
        full_log_probs = torch.tensor(data["log_probs"], dtype=torch.float32, device='cpu')
        full_rewards = torch.tensor(data["rewards"], dtype=torch.float32, device='cpu')
        full_masks = torch.tensor(data["illegal_mask"], dtype=torch.bool, device='cpu')
                
        # Get batch size
        b_size = len(full_rewards)
        
        v_now = torch.zeros((b_size, 3), dtype=torch.float32, device=device)
        # Pre compute advantages
        torch.cuda.empty_cache()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device_name, dtype=eff_dtype):
                for k in range(0, b_size, BATCH_SIZE):
                    obs_chunk = full_obs[k:k+BATCH_SIZE].to(device)
                    
                    _, v_chunk, _ = model._orig_mod(obs_chunk, training=False)
                    
                    v_now[k:k+BATCH_SIZE] = v_chunk
                
            v_now = torch.softmax(v_now, dim=-1)
            ev_now = v_now[:, 2] - v_now[:, 0]
            
            # Move all to numpy
            ev_now = ev_now.float().cpu().numpy()
            full_rewards_np = data["rewards"]
            is_selfplay_np = data["is_self"]
            dones_np = data["dones"]
                
            full_adv_np = np.zeros_like(ev_now)
            gae_val = 0
            for n in reversed(range(len(ev_now))):
                if dones_np[n]:
                    delta = full_rewards_np[n] - ev_now[n]
                    gae_val = delta
                else:
                    if is_selfplay_np[n]:
                        # model vs self
                        delta = (GAMMA * (-ev_now[n+1])) - ev_now[n]
                        gae_val = delta - (GAMMA * LAMBDA * gae_val) # how much better the previous move was (for the opponent)
                    else:
                        # model vs old model: every state is our models viewing
                        delta = (GAMMA * GAMMA * ev_now[n+1]) - ev_now[n]
                        gae_val = delta + (GAMMA * GAMMA * LAMBDA * gae_val) # GAMMA twice because here every turn is 2 moves
                
                full_adv_np[n] = gae_val
            
            full_target_np = full_adv_np + ev_now # This is the target value for the model
        
        # Convert numpy advantages to torch
        full_adv = torch.tensor(full_adv_np, dtype=torch.float32, device=device)
        full_returns = torch.tensor(full_target_np, dtype=torch.float32, device=device)
        
        # Update mean and std used for training
        full_adv_mean = full_adv.mean()
        full_adv_std = torch.clamp(full_adv.std(), min=1e-5)
        # From here on out, full_dones and full_is_selfplay should NOT be used
        
        # track loss
        total_val_loss = 0
        total_policy_loss = 0
        total_entropy_loss = 0
        total_logit_loss = 0
        
        stop_training = False # check for KL divergence
        ppo_steps = 0
        
        # Zero grad & clear cache before loading data
        torch.cuda.empty_cache()
        model.optim.zero_grad(set_to_none=True)
        for p in range(PPO_RETRAIN_RATIO):           
            ppo_steps += 1
             
            # Shuffle data
            rand_indices = torch.randperm(b_size, device='cpu')
            
            # Shuffle data
            full_obs = full_obs[rand_indices]
            full_actions = full_actions[rand_indices]
            full_log_probs = full_log_probs[rand_indices]
            full_rewards = full_rewards[rand_indices]
            full_masks = full_masks[rand_indices]
            full_adv = full_adv[rand_indices] # no need to update std or mean as they are 1 dimensional
            full_returns = full_returns[rand_indices]
            
            sub_batches = b_size // BATCH_SIZE
            for s in range(sub_batches):
                start = s * BATCH_SIZE
                end = start + BATCH_SIZE
                obs = full_obs[start:end].to(device)
                actions = full_actions[start:end].to(device)
                log_probs = full_log_probs[start:end].to(device)
                rewards = full_rewards[start:end].to(device)
                masks = full_masks[start:end].to(device)
                adv = full_adv[start:end].to(device)
                ret = full_returns[start:end].to(device)
                
                # Conv rewards to (batch, 3)
                target_wdl = transform_rewards(ret)
                
                with torch.amp.autocast(device_type=device_name, dtype=eff_dtype):
                    p, v, raw_p = model(obs, training=True) # p: (batch, 4672), v: (batch, 3)
                    p = p.masked_fill(masks, -1e9)
                    
                    # Get logit loss
                    logit_loss = ((raw_p * (~masks).float()).pow(2).sum()) / (~masks).float().sum().clamp(min=1.0)
                    
                    # Get value net loss
                    val_loss = F.cross_entropy(v, target_wdl)
                    
                    adv = (adv - full_adv_mean) / full_adv_std
                    
                    log_p = F.log_softmax(p, dim=-1) # (batch, 4672)
                    log_ratio = torch.gather(log_p, 1, actions.unsqueeze(1)).view(-1) - log_probs
                    r_val = torch.exp(log_ratio) # (batch,)
                    
                    with torch.no_grad():
                        approx_kl = (r_val - 1 - log_ratio).mean()
                        
                        if approx_kl > KL_LIMIT:
                            stop_training = True
                            break
                    
                    clipped_ratio = torch.clamp(r_val, min=1-EPSILON, max=1+EPSILON) # (batch,)
                    policy_loss = -torch.min(r_val * adv, clipped_ratio * adv).mean()
                    
                    entropy = -torch.sum(log_p * torch.exp(log_p), dim=-1).mean()
                    
                    loss = (VALUE_LOSS_COEF * val_loss + 
                            logit_loss * LOGIT_LOSS_COEF + 
                            policy_loss - 
                            ENTROPY_COEF * entropy) / sub_batches  # max policy obj and entropy but scale by subbatches
                
                    # Update total losses
                    total_val_loss += val_loss.detach().item() / sub_batches
                    total_policy_loss += abs(policy_loss.detach().item()) / sub_batches
                    total_entropy_loss += entropy.detach().item() / sub_batches
                    total_logit_loss += logit_loss.detach().item() / sub_batches
                
                # Backprop the loss first to clear vram
                loss.backward()
                        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optim.step()
            model.optim.zero_grad(set_to_none=True)
            
            if stop_training:
                break
        
        if not stop_training:
            scheduler.step()
        
        # Scale everything else to num steps
        total_val_loss /= ppo_steps
        total_policy_loss /= ppo_steps
        total_entropy_loss /= ppo_steps
        total_logit_loss /= ppo_steps
        
        # Scale entropy
        if total_entropy_loss < TARGET_ENTROPY_BOUND:
            ENTROPY_COEF = min(ENTROPY_COEF * ENTROPY_MULTIPLIER, 0.2)
        else:
            ENTROPY_COEF = max(ENTROPY_COEF / ENTROPY_MULTIPLIER, 0.01)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print(f"\n\nEpoch {offset+i+1}\nVal loss: {total_val_loss:.4f}\nPolicy loss: {total_policy_loss:.4f}\n\nLogit Loss: {total_logit_loss:.4f}\nEntropy: {total_entropy_loss:.4f}\nEntropy coefficient: {ENTROPY_COEF:.4f}\n")
        print(f"PPO Retrain: {ppo_steps}\nTime: {time.time() - t_start:.2f}s\n\n")
        
        
# --- Example Usage ---
if __name__ == "__main__":
    gym = chess_gym()
    
    try:
        for i in range(EPOCHS_TRAINED // 10, 1_000_000, 1):
            train(10, [old_models["smoke"], old_models["flame"], old_models["anchor"]], i*10)
            model.save_checkpoint("chess.pt")
            
            # Check
            for x in range(len(rival_epochs)):
                if (i * 10 + 10) % rival_epochs[x] == 0:
                    old_models[rival_names[x]].load_state_dict(model._orig_mod.state_dict())
            if is_power_of_two(i+1):
                old_models["anchor"].load_state_dict(model._orig_mod.state_dict())
                
    except KeyboardInterrupt:
        pass
    
    model.save_checkpoint("models/chess.pt")
    torch.save(old_models["anchor"].state_dict(), "models/anchor.pt")
    torch.save(old_models["smoke"].state_dict(), "models/smoke.pt")
    torch.save(old_models["flame"].state_dict(), "models/flame.pt")