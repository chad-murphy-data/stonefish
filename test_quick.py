"""Quick test to make sure everything works."""

import chess
import chess.engine
import time

from engine import STOCKFISH_PATH

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"Threads": 2, "Hash": 256})

board = chess.Board()

# Test 1: Get top moves from starting position
print("Test 1: Top 5 moves from starting position")
start = time.time()
result = engine.analyse(board, chess.engine.Limit(depth=12), multipv=5)
for info in result:
    move = info["pv"][0]
    score = info["score"].white()
    print(f"  {board.san(move)}: {score}")
print(f"  Time: {time.time() - start:.2f}s")

# Test 2: Score a candidate move for nettlesomeness
print("\nTest 2: Scoring e4 for opponent difficulty")
board.push(chess.Move.from_uci("e2e4"))
start = time.time()
result = engine.analyse(board, chess.engine.Limit(depth=12), multipv=3)
print("  Opponent's top 3 responses to e4:")
evals = []
for info in result:
    move = info["pv"][0]
    score = info["score"].white()
    cp = score.score() if not score.is_mate() else 10000
    evals.append(cp / 100.0)
    print(f"    {board.san(move)}: {score} ({cp/100:.2f} pawns)")

if len(evals) >= 2:
    gap = evals[1] - evals[0]
    print(f"  Gap between best and 2nd best response: {gap:.3f} pawns")

board.pop()
print(f"  Time: {time.time() - start:.2f}s")

# Test 3: Full nettlesome evaluation of starting position
print("\nTest 3: Nettlesome scoring of top 5 opening moves")
candidates = engine.analyse(board, chess.engine.Limit(depth=12), multipv=5)

for info in candidates:
    cand_move = info["pv"][0]
    cand_eval = info["score"].white()
    
    # Push candidate and check opponent responses
    board.push(cand_move)
    responses = engine.analyse(board, chess.engine.Limit(depth=12), multipv=3)
    
    resp_evals = []
    resp_moves = []
    for r in responses:
        rm = r["pv"][0]
        rs = r["score"].white()
        rcp = rs.score() if not rs.is_mate() else 10000
        resp_evals.append(rcp / 100.0)
        resp_moves.append(board.san(rm))
    
    board.pop()
    
    gap_12 = resp_evals[1] - resp_evals[0] if len(resp_evals) >= 2 else 0
    gap_13 = resp_evals[2] - resp_evals[0] if len(resp_evals) >= 3 else 0
    difficulty = 0.6 * gap_12 + 0.4 * gap_13
    
    print(f"  {board.san(cand_move)}: eval={cand_eval}, "
          f"responses={resp_moves}, "
          f"gap_1_2={gap_12:.3f}, gap_1_3={gap_13:.3f}, "
          f"difficulty={difficulty:.3f}")

print("\nAll tests passed!")
engine.quit()
