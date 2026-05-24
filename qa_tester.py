"""
QA against a synthetic Coin-Flip tester
========================================
Runs Stonefish v1 against CoinFlipTesterBot at a sweep of find probabilities.
This isolates the trap mechanic from opponent variance: the tester plays
Stockfish baseline (consistent) and only randomizes its response at the
trap moments themselves.

For each find_probability we should see:
  - Trap volume roughly equal across runs (Stonefish-determined)
  - Opp win/draw rate that scales monotonically with find_probability
  - Cleanly testable QA rules

Usage:
    python3 qa_tester.py [games_per_prob] [depth] [probs...]
    e.g. python3 qa_tester.py 20 10 0.3 0.5 0.7 0.9
"""

import sys
import time
import json
import chess
import chess.engine

from engine import (
    NettlesomeBot, CoinFlipTesterBot, EquilibriumBaselineBot, STOCKFISH_PATH,
    play_game,
)
from maia_policy import MaiaPolicyEngine


def make_stonefish(sf_engine, maia_oracle, baseline, give_back, depth):
    """Build the canonical Stonefish v1."""
    return NettlesomeBot(
        sf_engine, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.5,
        maia_oracle=maia_oracle, baseline_bot=baseline,
        puzzle_mode=True,
        min_eval_cost=0.20,
        min_gap=0.50,
        min_gap_ratio=1.5,
        p_maia_min=0.20, p_maia_max=0.80,
        post_trap_baseline=give_back,
        post_trap_duration=5,
        endgame_mode=True, endgame_threshold=1.0,
        endgame_min_move=30, endgame_min_pieces=14,
        conversion_mode=True, conversion_probability=0.80,
        conversion_gap_threshold=0.7,
    )


def run_for_probability(find_prob, num_games, depth, sf, oracle):
    baseline = EquilibriumBaselineBot(sf, oracle, target_delta=0.3, depth=depth, rating=1900)
    give_back = EquilibriumBaselineBot(sf, oracle, target_delta=0.7, depth=depth, rating=1900)
    stonefish = make_stonefish(sf, oracle, baseline, give_back, depth)
    # Seed the tester deterministically (varies across prob-buckets via the prob)
    # baseline_target_delta=0 -> tester plays SF top in non-trap positions
    # (matches the original behavior the p=0.30 data came from).
    tester = CoinFlipTesterBot(sf, depth=depth, find_probability=find_prob,
                                trap_gap_threshold=0.5,
                                baseline_target_delta=0.0,
                                seed=int(find_prob * 1000))

    games = []
    print(f"\n=== find_probability = {find_prob:.2f} ({num_games} games) ===", flush=True)
    for g in range(num_games):
        if g % 2 == 0:
            white, black = stonefish, tester
            stone_white = True
        else:
            white, black = tester, stonefish
            stone_white = False

        start = time.time()
        white_score, num_moves, _ = play_game(white, black, verbose=False)
        elapsed = time.time() - start
        bot_score = white_score if stone_white else (1.0 - white_score)

        stats = stonefish.stats
        moments = stats.n_moments
        found = stats.n_opp_found_top
        find_rate = (found / moments) if moments > 0 else None

        games.append({
            "g": g + 1, "moves": num_moves, "moments": moments, "found": found,
            "find_rate": find_rate, "bot_score": bot_score, "elapsed": elapsed,
        })
        outcome = "W" if bot_score == 1.0 else ("L" if bot_score == 0.0 else "D")
        find_str = f"{find_rate*100:>4.0f}%" if find_rate is not None else "  n/a"
        print(f"  G{g + 1:>2}: {outcome}  {num_moves:>3}mv  "
              f"moments={moments:>2}  found={found:>2}  find={find_str}  "
              f"{elapsed:.1f}s", flush=True)

    return games


def main(num_games, depth, probs):
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    oracle = MaiaPolicyEngine("/home/user/.maia/maia-1900.pb.gz")

    print(f"QA-tester: {num_games} games per find-prob, depth={depth}")
    print(f"Find probabilities: {probs}")
    print(f"Opponent: CoinFlipTesterBot (SF baseline + weighted coin at trap moments)")
    print("=" * 70, flush=True)

    results = {}
    for p in probs:
        games = run_for_probability(p, num_games, depth, sf, oracle)
        wins = sum(1 for g in games if g["bot_score"] == 1.0)
        losses = sum(1 for g in games if g["bot_score"] == 0.0)
        draws = len(games) - wins - losses
        score = sum(g["bot_score"] for g in games) / len(games)
        opp_score = 1.0 - score
        results[p] = {
            "games": games,
            "stone_score": score,
            "opp_score": opp_score,
            "W": wins, "L": losses, "D": draws,
        }

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'find_prob':>9} {'games':>6} {'W-D-L':>10} {'Stonefish':>11} {'Opp':>8}  "
          f"{'avg moments':>11} {'actual find':>12}")
    print("-" * 80)
    for p, r in results.items():
        games = r["games"]
        n = len(games)
        n_with_moments = sum(1 for g in games if g["moments"] > 0)
        total_moments = sum(g["moments"] for g in games)
        avg_m = total_moments / n
        actual_finds = sum(g["found"] for g in games)
        actual_find_rate = actual_finds / total_moments if total_moments > 0 else 0
        wdl = f"{r['W']}-{r['D']}-{r['L']}"
        print(f"{p:>9.2f} {n:>6} {wdl:>10} "
              f"{r['stone_score']:>10.1%}  {r['opp_score']:>7.1%}  "
              f"{avg_m:>10.1f}  {actual_find_rate*100:>10.0f}%")

    # Save raw data
    out_path = f"results/qa_tester_n{num_games}_d{depth}.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "num_games_per_prob": num_games,
            "depth": depth,
            "probs": probs,
            "results": {str(p): r for p, r in results.items()},
        }, f, indent=2)
    print(f"\nRaw data: {out_path}")

    try: oracle.quit()
    except: pass
    try: sf.quit()
    except: pass


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    depth     = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if len(sys.argv) > 3:
        probs = [float(x) for x in sys.argv[3:]]
    else:
        probs = [0.3, 0.5, 0.7, 0.9]
    main(num_games, depth, probs)
