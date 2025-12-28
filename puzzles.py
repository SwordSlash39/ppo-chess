from datasets import load_dataset
import chess
ds = load_dataset("Lichess/chess-puzzles", split="train")

with open("puzzles.txt", "w") as f:
    count = 0
    limit = 2000000  # Number of puzzles to save
    
    print(f"Generating puzzles (Depth 1-8)...")

    for puzzle in ds:
        # Filter for mate puzzles
        if any("mate" in t.lower() for t in puzzle["Themes"]):
            
            moves = puzzle["Moves"].split()
            
            # Filter: Only puzzles up to Mate in 8
            # (Original moves include 1 opponent move + up to 15 solution moves)
            if len(moves) <= 16 and int(puzzle["Rating"]) < 1200:
                
                # --- START CONVERSION LOGIC ---
                # 1. Load the "before" FEN
                board = chess.Board(puzzle["FEN"])
                
                # 2. Play the first move (the opponent's setup move)
                first_move = chess.Move.from_uci(moves[0])
                board.push(first_move)
                
                # 3. This is the FEN where the puzzle actually starts
                actual_start_fen = board.fen()
                
                # 4. These are the moves the player must make to solve it
                solution_moves = " ".join(moves[1:])
                # --- END CONVERSION LOGIC ---

                # Save to file
                line = f"{actual_start_fen}\n"
                f.write(line)
                
                count += 1
                if count % 50000 == 0:
                    print(f"Processed {count} puzzles...")

        if count >= limit:
            break

print(f"Done! Saved {count} puzzles to puzzles.txt")