"""
Interactive Stonefish vs Maia 1900
===================================
You play Stonefish's moves manually. Maia 1900 plays the opponent.
At each of your turns, you see SF top-25 candidates with post-move
evals and pick the one you want. Designed to investigate whether a
human maintainer can hold a target eval against real Maia drift --
i.e., whether the trap mechanic CAN drive outcomes if the baseline
moves are picked well.

Commands at each of your turns:
  <number>      play candidate at that index from the SF top-25 list
  <UCI or SAN>  play a specific move (e.g. e2e4 or Nf3)
  target X.XX   change the target eval (in pawns, from your POV)
  refresh       re-show the candidate list at the current target
  show          show the maintainer's pick (closest to target in top-25)
  hint          one-line summary of best target-hitter
  fen           print current FEN
  quit          exit

Usage:
    python3 interactive_game.py [w|b]   # your color (default white)
"""

import sys
import chess
import chess.engine

from engine import STOCKFISH_PATH, MAIA_WEIGHTS_PATH, MaiaBot, get_top_moves


def eval_position(sf, board, depth=10):
    """Eval in pawns from white's perspective."""
    if board.is_game_over():
        res = board.result()
        if res == "1-0": return 99.99
        if res == "0-1": return -99.99
        return 0.0
    info = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        return 99.99 if score.mate() > 0 else -99.99
    return score.score() / 100.0


def build_candidates(sf, board, depth, num_candidates=25):
    """SF top-N candidates with eval-for-current-side, SAN, drift-from-target."""
    raw = get_top_moves(sf, board, num_moves=num_candidates, depth=depth)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    out = []
    for move, eval_cp in raw:
        eval_for_us = max(-10.0, min(10.0, eval_cp * sign))
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        out.append((move, eval_for_us, san))
    return out


def show_candidates(candidates, target):
    """Print top-N candidates, marking the closest-to-target."""
    if not candidates:
        print("  (no legal candidates)")
        return
    closest_idx = min(range(len(candidates)),
                      key=lambda i: abs(candidates[i][1] - target))
    print(f"\n  {'#':>3}  {'move':<10}  {'eval(us)':>10}  {'drift':>8}  {'  '}")
    print(f"  {'-'*46}")
    for i, (mv, ev, san) in enumerate(candidates, 1):
        drift = ev - target
        marker = "  *" if (i - 1) == closest_idx else "   "
        print(f"  {i:>3}  {san:<10}  {ev:>+9.2f}p  {drift:>+7.2f}p {marker}")
    print(f"  (* = closest to target {target:+.2f}p; what the maintainer would pick)")


def parse_move(s, board, candidates):
    """Parse user input as index, UCI, or SAN. Returns chess.Move."""
    s = s.strip()
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx][0]
        raise ValueError(f"index {idx + 1} out of range (1-{len(candidates)})")
    try:
        m = chess.Move.from_uci(s)
        if m in board.legal_moves:
            return m
    except Exception:
        pass
    try:
        return board.parse_san(s)
    except Exception:
        pass
    raise ValueError(f"can't parse move: {s!r}")


def main():
    user_color_str = sys.argv[1] if len(sys.argv) > 1 else "w"
    user_color = chess.WHITE if user_color_str.lower() in ("w", "white") else chess.BLACK
    depth = 10
    num_candidates = 25

    color_label = "WHITE" if user_color == chess.WHITE else "BLACK"
    print(f"\n=== Interactive Stonefish vs Maia 1900 ===")
    print(f"You play: {color_label}")
    print(f"SF depth: {depth}, top-{num_candidates} candidates shown each turn")
    print(f"Type 'help' at any prompt for commands.\n")

    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    maia = MaiaBot(MAIA_WEIGHTS_PATH,
                   rating=1900, temperature=1.0, seed=None)

    board = chess.Board()
    target = 0.0
    history = []  # list of (ply, san, eval_for_user_after)

    sign = 1.0 if user_color == chess.WHITE else -1.0

    try:
        while not board.is_game_over():
            cur_eval = eval_position(sf, board, depth) * sign
            print(f"\n{'='*52}")
            print(f"Ply {board.ply()+1}: "
                  f"{'WHITE' if board.turn == chess.WHITE else 'BLACK'} to move")
            print(f"  Eval (yours): {cur_eval:+.2f}p     "
                  f"Target: {target:+.2f}p     "
                  f"Drift: {cur_eval - target:+.2f}p")
            if history:
                tail = " ".join(s for _, s, _ in history[-8:])
                print(f"  Recent: {tail}")

            if board.turn != user_color:
                print("\n  Maia 1900 is thinking...", flush=True)
                move = maia.choose_move(board)
                try:
                    san = board.san(move)
                except Exception:
                    san = move.uci()
                board.push(move)
                post_eval = eval_position(sf, board, depth) * sign
                history.append((board.ply(), san, post_eval))
                print(f"  Maia plays: {san}   (eval now {post_eval:+.2f}p, "
                      f"drift {post_eval - target:+.2f}p)")
                continue

            # User's turn
            candidates = build_candidates(sf, board, depth, num_candidates)
            show_candidates(candidates, target)

            while True:
                try:
                    s = input("\n  Your move (or command): ").strip()
                except EOFError:
                    print("\n  EOF, exiting.")
                    return
                if not s:
                    continue
                cmd = s.lower()
                if cmd in ("quit", "q", "exit"):
                    print("  Goodbye.")
                    return
                if cmd == "help":
                    print(__doc__)
                    continue
                if cmd == "fen":
                    print(f"  {board.fen()}")
                    continue
                if cmd == "refresh":
                    candidates = build_candidates(sf, board, depth, num_candidates)
                    show_candidates(candidates, target)
                    continue
                if cmd in ("show", "hint"):
                    if not candidates:
                        print("  no candidates")
                        continue
                    best_i = min(range(len(candidates)),
                                 key=lambda i: abs(candidates[i][1] - target))
                    bm, be, bs = candidates[best_i]
                    print(f"  Maintainer pick: #{best_i+1} {bs}  "
                          f"eval {be:+.2f}p  drift {be - target:+.2f}p")
                    continue
                if cmd.startswith("target "):
                    try:
                        target = float(s.split(maxsplit=1)[1])
                        print(f"  Target = {target:+.2f}p")
                        show_candidates(candidates, target)
                    except Exception as e:
                        print(f"  bad target: {e}")
                    continue
                try:
                    move = parse_move(s, board, candidates)
                except Exception as e:
                    print(f"  {e}")
                    continue
                try:
                    san = board.san(move)
                except Exception:
                    san = move.uci()
                board.push(move)
                post_eval = eval_position(sf, board, depth) * sign
                history.append((board.ply(), san, post_eval))
                print(f"  You played: {san}   (eval now {post_eval:+.2f}p, "
                      f"drift {post_eval - target:+.2f}p)")
                break

        print(f"\n{'='*52}")
        print(f"Game over: {board.result()}")
        print(f"Final eval (yours): {eval_position(sf, board, depth) * sign:+.2f}p")
        print(f"\nFull PGN moves:")
        print(f"  {' '.join(s for _, s, _ in history)}")
    finally:
        try: sf.quit()
        except: pass
        try: maia.quit()
        except: pass


if __name__ == "__main__":
    main()
