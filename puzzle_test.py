import numpy as np
import torch, pygame, time
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from stockfish import Stockfish
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


with open('puzzles.txt', 'r', encoding='utf-8') as f:
    # Use a list comprehension for speed
    puzzles = [line.strip() for line in f]
    
eff_dtype = torch.bfloat16
temp = 0.7
temp_decay = 0.98
temp_min = 0.05

R_LIST = ["human", "rgb_array"]
RENDER = "rgb_array"

env = chess_gym(render_mode=RENDER)
total = 0
checkmates = 0
for p in puzzles:
    observation, info = env.reset(fen=p)
    
    hasCheckmate = False
    for i in range(16):
        illegalMask = torch.tensor(env.get_illegal_mask(), dtype=torch.bool, device=device)
        obv_tensor = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
        
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=eff_dtype):
            logits, _, _ = model(obv_tensor, training=False)
            logits = logits.view(-1)
            
            logits[illegalMask] = -1e9
        
        action = torch.argmax(logits.squeeze()).item()
        
        observation, reward, termination, truncation, info = env.step(action)
        if reward != 0:
            hasCheckmate = True
        
        if termination or truncation:
            break
    
    total += 1
    if hasCheckmate:
        checkmates += 1
    
    if total % 5 == 0:
        print(f"Puzzles: {total}\nAccuracy: {(checkmates * 100 / total):.2f}%")

