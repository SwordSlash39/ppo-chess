import numpy as np
import torch, pygame, time
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from stockfish import Stockfish
import chess.pgn
"""
/usr/games/stockfish
"""
from model import *
from chess_env import chess_gym

torch.set_printoptions(threshold=float('inf'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyChessTransformer().to(device).to(torch.float32)
model.load_checkpoint("chess.pt")
model.eval()

# Compile model
torch._inductor.config.max_autotune_gemm = False
# model = torch.compile(
#     model,
#     options={
#         "triton.cudagraphs": False,
#         "max_autotune": False,
#         "epilogue_fusion": True,
#         "shape_padding": True,
#     }
# )

eff_dtype = torch.bfloat16
temp = 0.7
temp_decay = 0.98
temp_min = 0.05

R_LIST = ["human", "rgb_array"]
RENDER = "rgb_array"
with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=eff_dtype):
    env = chess_gym(render_mode=RENDER)
    
    observation, info = env.reset()

    for i in range(1000000):
        # board = env.env.unwrapped.board
        # uci_history = [move.uci() for move in board.move_stack]
        # print("Move History (UCI Strings):", uci_history)
        # this is where you would insert your policy
        illegalMask = torch.tensor(env.get_illegal_mask(), dtype=torch.bool, device=device)
        obv_tensor = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
        logits, val, _ = model(obv_tensor, training=False)
        logits = logits.view(-1)
        
        logits[illegalMask] = -1e9  
        prob = torch.softmax(logits, dim=-1)
        color = "White" if i % 2 == 0 else "Black"
        WDL = torch.softmax(val.view(-1), dim=-1).tolist()
        WDL = [round(x, 2) for x in WDL]
        
        # Apply temp
        logits = logits / max(temp, temp_min)
        temp *= temp_decay
        
        logits[illegalMask] = -1e9
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        # action = torch.argmax(logits.squeeze()).item()

        print(f"Move: {i//2 + 1}   Color:   {color}   WDL: {WDL}")
        print(prob[action])
        
        observation, reward, termination, truncation, info = env.step(action)

        if RENDER == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    print("Window closed by user.")
                    exit() # Stop the script

        if termination:
            if RENDER == "human":
                time.sleep(4)
            break
        
        if RENDER == "human":
            env.env.render()
            time.sleep(5)
        
        # observation, reward, termination, truncation, info = env.step(env.get_random_legal_move())
        # if termination:
        #     break

# --- PGN EXPORT LOGIC ---
print("Generating PGN...")

# 1. Access the internal python-chess board
final_board = env.env.unwrapped.board

# 2. Create a PGN Game from the board's move stack
game = chess.pgn.Game.from_board(final_board)

# 3. Add some headers (optional)
game.headers["Event"] = "PettingZoo AI Match"
game.headers["Site"] = "Local Machine"
game.headers["White"] = "TinyChessTransformer"
game.headers["Black"] = "TinyChessTransformer"
game.headers["Result"] = final_board.result()

# 4. Write to pgn.txt
with open("pgn.txt", "w") as f:
    f.write(str(game))

print("PGN saved to pgn.txt")
print("Final FEN:", final_board.fen())