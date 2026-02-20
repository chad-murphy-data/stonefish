"""
Puzzle Detection Test Across ELO Brackets
==========================================
Plays self-play games at each ELO preset and reports whether the
puzzle detector is finding puzzles -- how many, what types, eval gaps,
difficulty, and solve rates.

Uses the simple bot.choose_move() interface which handles opponent
tracking internally via _infer_opponent_move.

Run:  python test_puzzle_detection.py [num_games_per_preset]
"""

import chess
import chess.engine
import sys
import time
from collections import Counter

from engine import STOCKFISH_PATH, PureStockfishBot
from stonefish.config import StonefishConfig
from stonefish.bot import StonefishBot
from stonefish.presets import ELO_PRESETS, apply_preset


def test_elo(engine, elo, preset, num_games):
    """Play self-play games at a given elo and collect puzzle stats."""
    config = StonefishConfig(deep_depth=10, base_depth=6)
    apply_preset(config, elo)
    # Cap lookahead to 1 for speed (skip the expensive two-ahead screening)
    config.max_lookahead = 1

    stockfish = PureStockfishBot(engine, depth=10)

    total_puzzles = 0
    total_solved = 0
    total_our_moves = 0
    total_half_moves = 0
    puzzle_types = Counter()
    puzzle_scores = []
    puzzle_eval_gaps = []
    puzzle_floor_probs = []
    puzzle_disagreements = Counter()
    game_results = []

    for game_num in range(num_games):
        sf_white = game_num % 2 == 0
        stonefish = StonefishBot(engine, config)
        stonefish.start_game(chess.WHITE if sf_white else chess.BLACK)

        board = chess.Board()
        move_count = 0
        game_start = time.time()

        for _ in range(200):
            if board.is_game_over():
                break

            if (board.turn == chess.WHITE) == sf_white:
                # Stonefish's turn — uses choose_move which calls choose_move_full
                # and infers opponent moves internally
                move = stonefish.choose_move(board)
            else:
                # Opponent (PureStockfish) plays
                move = stockfish.choose_move(board)

            board.push(move)
            move_count += 1

        game_elapsed = time.time() - game_start
        state = stonefish.end_game(board)

        # Collect puzzle data from our moves
        game_puzzles_created = 0
        if state:
            for mr in state.move_results:
                total_our_moves += 1
                if mr.puzzle_found:
                    game_puzzles_created += 1
                    puzzle_types[mr.puzzle_type or "unknown"] += 1
                    if mr.puzzle_score is not None and mr.puzzle_score != 0:
                        puzzle_scores.append(mr.puzzle_score)
                    if mr.puzzle_eval_gap is not None and mr.puzzle_eval_gap != 0:
                        puzzle_eval_gaps.append(mr.puzzle_eval_gap)
                    if mr.puzzle_floor_prob is not None:
                        puzzle_floor_probs.append(mr.puzzle_floor_prob)
                    if mr.puzzle_disagreement:
                        puzzle_disagreements[mr.puzzle_disagreement] += 1

            # Opponent solve rate
            for opp in state.opponent_results:
                if opp.was_puzzle:
                    total_puzzles += 1
                    if opp.found_puzzle:
                        total_solved += 1

        total_half_moves += move_count

        # Game result
        board_result = board.result()
        if board_result == "1-0":
            outcome = "W" if sf_white else "L"
        elif board_result == "0-1":
            outcome = "L" if sf_white else "W"
        else:
            outcome = "D"
        game_results.append(outcome)

        color = "W" if sf_white else "B"
        print(f"    Game {game_num+1} ({color}): {board_result} "
              f"in {move_count} moves, puzzles: {game_puzzles_created}, "
              f"time: {game_elapsed:.0f}s", flush=True)

    return {
        "elo": elo,
        "games": num_games,
        "total_half_moves": total_half_moves,
        "total_our_moves": total_our_moves,
        "puzzles_presented": total_puzzles,
        "puzzles_solved": total_solved,
        "puzzle_types": dict(puzzle_types),
        "puzzle_scores": puzzle_scores,
        "puzzle_eval_gaps": puzzle_eval_gaps,
        "puzzle_floor_probs": puzzle_floor_probs,
        "puzzle_disagreements": dict(puzzle_disagreements),
        "results": game_results,
    }


def print_results(all_results):
    """Print summary table."""
    print("\n" + "=" * 90)
    print("PUZZLE DETECTION RESULTS BY ELO")
    print("=" * 90)

    header = (f"  {'ELO':>5}  {'Games':>5}  {'Our Moves':>9}  {'Puzzles':>8}  "
              f"{'Per Game':>8}  {'Solved':>7}  {'Types'}")
    print(header)
    print("  " + "-" * 85)

    for r in all_results:
        n_games = r["games"]
        per_game = r["puzzles_presented"] / n_games if n_games > 0 else 0
        solve_str = (f"{r['puzzles_solved']}/{r['puzzles_presented']}"
                     if r["puzzles_presented"] > 0 else "n/a")
        types_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["puzzle_types"].items()))
        if not types_str:
            types_str = "none"

        print(f"  {r['elo']:>5}  {n_games:>5}  {r['total_our_moves']:>9}  "
              f"{r['puzzles_presented']:>8}  {per_game:>7.1f}  "
              f"{solve_str:>7}  {types_str}")

    # Detailed stats for presets that found puzzles
    print("\n" + "=" * 90)
    print("PUZZLE QUALITY DETAILS")
    print("=" * 90)

    for r in all_results:
        p = ELO_PRESETS[r["elo"]]
        print(f"\n  ELO {r['elo']} (Floor={p['floor_rating']}, "
              f"Stretch={p['stretch_rating'] or 'SF'}, "
              f"Reach={p['reach_rating'] or 'SF'})")
        print(f"  {'-' * 60}")

        if not r["puzzle_eval_gaps"] and not r["puzzle_types"]:
            print(f"    No puzzles detected!")
            print(f"    Game results: {Counter(r['results'])}")
            continue

        if r["puzzle_types"]:
            print(f"    Types: {r['puzzle_types']}")

        if r["puzzle_disagreements"]:
            print(f"    Disagreements: {r['puzzle_disagreements']}")

        if r["puzzle_eval_gaps"]:
            gaps = r["puzzle_eval_gaps"]
            print(f"    Eval gaps:  avg={sum(gaps)/len(gaps):.2f}  "
                  f"min={min(gaps):.2f}  max={max(gaps):.2f}")

        if r["puzzle_scores"]:
            scores = r["puzzle_scores"]
            print(f"    Scores:     avg={sum(scores)/len(scores):.2f}  "
                  f"min={min(scores):.2f}  max={max(scores):.2f}")

        if r["puzzle_floor_probs"]:
            probs = r["puzzle_floor_probs"]
            avg_p = sum(probs) / len(probs)
            very_hard = sum(1 for p in probs if p < 0.05)
            hard = sum(1 for p in probs if 0.05 <= p < 0.15)
            moderate = sum(1 for p in probs if 0.15 <= p < 0.30)
            findable = sum(1 for p in probs if p >= 0.30)
            print(f"    Floor prob: avg={avg_p:.1%}")
            print(f"      Very hard (<5%): {very_hard}  |  Hard (5-15%): {hard}  "
                  f"|  Moderate (15-30%): {moderate}  |  Findable (30%+): {findable}")

        print(f"    Game results: {Counter(r['results'])}")


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print("=" * 90)
    print("Puzzle Detection Test Across ELO Brackets")
    print("=" * 90)
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Games per preset: {num_games}")
    print(f"Settings: deep_depth=10, base_depth=6, max_lookahead=1 (fast mode)")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    all_results = []
    total_start = time.time()

    for elo in sorted(ELO_PRESETS.keys()):
        preset = ELO_PRESETS[elo]
        print(f"\n  --- ELO {elo} (Floor={preset['floor_rating']}, "
              f"Stretch={preset['stretch_rating'] or 'SF'}, "
              f"Reach={preset['reach_rating'] or 'SF'}) ---")

        start = time.time()
        result = test_elo(engine, elo, preset, num_games)
        elapsed = time.time() - start

        print(f"    Preset done in {elapsed:.0f}s")
        all_results.append(result)

    engine.quit()

    print_results(all_results)

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed:.0f}s")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
