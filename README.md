PPO Implementation for chess!!

You can read on PPO here: https://huggingface.co/blog/deep-rl-ppo

This is a trial implementation for chess, but slightly supervised (stockfish evals are given every few moves) in order to speed up training
Model is 75M parameters, attention model with SwiGLU and smolgen (from lc0)
Some games also start from puzzles (you need to upload your own puzzles.txt), so the model quickly learns how to mate

Format for puzzles.txt:
<fen>
<fen>
...
