"""
List of chess functions used in train
"""
import math, chess

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def stockfish_win_probability(cp_eval, max_eval):
    """
    Estimates Win Probability based on Stockfish CP Eval
    """
    return max_eval * math.tanh(cp_eval / 360)

def mirror_fen(fen):
    """
    Converts a 'White to move' puzzle into a 'Black to move' puzzle.
    The board is flipped, colors are swapped, and the turn is changed.
    """
    board = chess.Board(fen)
    mirrored_board = board.mirror() # Flips board vertically and swaps colors
    return mirrored_board.fen()