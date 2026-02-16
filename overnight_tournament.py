"""
Overnight Tournament
====================
Runs a large tournament between Nettlesome, PureStockfish, and imperfect
opponents to measure whether the nettlesome strategy wins faster or more
often than pure best-move play.

Usage:
    python overnight_tournament.py

    # Custom settings
    python overnight_tournament.py --games 100 --depth 16

Results are saved to tournament_results.json and printed as a summary.
"""

import sys
import json
import time
import argparse
from datetime import datetime, timedelta

import chess
import chess.engine

from engine import (
    NettlesomeBot, RandomTopNBot, PureStockfishBot,
    play_game, STOCKFISH_PATH,
)


def run_matchup(engine, bot_a, bot_b, num_games, label):
    """Run a matchup and return detailed results."""
    results = {
        "label": label,
        "bot_a": bot_a.name,
        "bot_b": bot_b.name,
        "games": [],
        "bot_a_wins": 0,
        "bot_b_wins": 0,
        "draws": 0,
        "total_moves": 0,
        "bot_a_win_moves": [],  # Move counts for bot_a wins
        "bot_b_win_moves": [],  # Move counts for bot_b wins
    }

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            white, black = bot_a, bot_b
            a_is_white = True
        else:
            white, black = bot_b, bot_a
            a_is_white = False

        start = time.time()
        score, num_moves, _ = play_game(white, black, max_moves=300, verbose=False)
        elapsed = time.time() - start

        # Convert to bot_a's perspective
        a_score = score if a_is_white else 1.0 - score

        game_record = {
            "game_num": i + 1,
            "a_is_white": a_is_white,
            "score": a_score,
            "moves": num_moves,
            "time": round(elapsed, 1),
        }
        results["games"].append(game_record)
        results["total_moves"] += num_moves

        if a_score == 1.0:
            results["bot_a_wins"] += 1
            results["bot_a_win_moves"].append(num_moves)
            outcome = f"{bot_a.name} wins"
        elif a_score == 0.0:
            results["bot_b_wins"] += 1
            results["bot_b_win_moves"].append(num_moves)
            outcome = f"{bot_b.name} wins"
        else:
            results["draws"] += 1
            outcome = "Draw"

        total = i + 1
        a_pct = (results["bot_a_wins"] + 0.5 * results["draws"]) / total
        avg_moves = results["total_moves"] / total
        eta_games = num_games - total
        eta_secs = eta_games * elapsed
        eta_str = str(timedelta(seconds=int(eta_secs)))

        print(f"  [{label}] Game {total}/{num_games}: {outcome} "
              f"in {num_moves} moves ({elapsed:.0f}s) | "
              f"Running: {a_pct:.0%} | ETA: {eta_str}")

    # Compute summary stats
    total = num_games
    results["a_score_pct"] = (results["bot_a_wins"] + 0.5 * results["draws"]) / total
    results["avg_moves"] = results["total_moves"] / total
    results["avg_a_win_moves"] = (
        sum(results["bot_a_win_moves"]) / len(results["bot_a_win_moves"])
        if results["bot_a_win_moves"] else 0
    )
    results["avg_b_win_moves"] = (
        sum(results["bot_b_win_moves"]) / len(results["bot_b_win_moves"])
        if results["bot_b_win_moves"] else 0
    )

    return results


def print_summary(all_results):
    """Print a nice summary table."""
    print()
    print("=" * 80)
    print("OVERNIGHT TOURNAMENT RESULTS")
    print("=" * 80)
    print(f"{'Matchup':<45} {'Score':>6} {'W-D-L':>10} {'AvgLen':>7} {'AvgWinLen':>9}")
    print("-" * 80)

    for r in all_results:
        wdl = f"{r['bot_a_wins']}-{r['draws']}-{r['bot_b_wins']}"
        avg_win = r["avg_a_win_moves"]
        print(f"{r['label']:<45} {r['a_score_pct']:>5.1%} {wdl:>10} "
              f"{r['avg_moves']:>6.0f} {avg_win:>8.0f}")

    # Key comparison: Nettlesome vs PureStockfish against the same opponents
    print()
    print("=" * 80)
    print("KEY COMPARISON: Nettlesome vs PureStockfish against imperfect opponents")
    print("=" * 80)

    nett_results = {r["label"]: r for r in all_results if "Nettlesome" in r["bot_a"]}
    sf_results = {r["label"]: r for r in all_results if "PureStockfish" in r["bot_a"]}

    comparisons = [
        ("RandomTop3_70/20/10", "Nett vs RT3-Weighted", "SF vs RT3-Weighted"),
        ("RandomTop5", "Nett vs RT5", "SF vs RT5"),
    ]

    for opp_tag, nett_label, sf_label in comparisons:
        n = nett_results.get(nett_label)
        s = sf_results.get(sf_label)
        if not n or not s:
            continue

        print(f"\n  vs {opp_tag}:")
        print(f"    Nettlesome:    {n['a_score_pct']:.1%} "
              f"(W{n['bot_a_wins']}/D{n['draws']}/L{n['bot_b_wins']}, "
              f"avg win in {n['avg_a_win_moves']:.0f} moves)")
        print(f"    PureStockfish: {s['a_score_pct']:.1%} "
              f"(W{s['bot_a_wins']}/D{s['draws']}/L{s['bot_b_wins']}, "
              f"avg win in {s['avg_a_win_moves']:.0f} moves)")

        if n["a_score_pct"] > s["a_score_pct"]:
            print(f"    >>> Nettlesome wins MORE ({n['a_score_pct'] - s['a_score_pct']:+.1%})")
        elif n["a_score_pct"] == s["a_score_pct"]:
            if n["avg_a_win_moves"] and s["avg_a_win_moves"]:
                diff = s["avg_a_win_moves"] - n["avg_a_win_moves"]
                if diff > 0:
                    print(f"    >>> Same win rate, but Nettlesome wins {diff:.0f} moves FASTER")
                else:
                    print(f"    >>> Same win rate and similar speed")
            else:
                print(f"    >>> Tied")
        else:
            print(f"    >>> PureStockfish wins more ({s['a_score_pct'] - n['a_score_pct']:+.1%})")
            if n["avg_a_win_moves"] and s["avg_a_win_moves"]:
                diff = s["avg_a_win_moves"] - n["avg_a_win_moves"]
                if diff > 0:
                    print(f"    >>> But Nettlesome wins {diff:.0f} moves faster when it does win")


def main():
    parser = argparse.ArgumentParser(description="Overnight Nettlesome Tournament")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per matchup (default: 50)")
    parser.add_argument("--depth", type=int, default=14,
                        help="Engine depth (default: 14)")
    args = parser.parse_args()

    num_games = args.games
    depth = args.depth

    print(f"NETTLESOME OVERNIGHT TOURNAMENT")
    print(f"Games per matchup: {num_games}")
    print(f"Depth: {depth}")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    # Create bots
    nettlesome = NettlesomeBot(engine, num_candidates=7, num_responses=3,
                               depth=depth, max_eval_cost=1.0)
    stockfish = PureStockfishBot(engine, depth=depth)
    random_top5 = RandomTopNBot(engine, top_n=5, depth=depth)
    random_top3_weighted = RandomTopNBot(engine, top_n=3, depth=depth,
                                         weights=[0.7, 0.2, 0.1])

    matchups = [
        ("Nett vs RT3-Weighted", nettlesome, random_top3_weighted),
        ("SF vs RT3-Weighted", stockfish, random_top3_weighted),
        ("Nett vs RT5", nettlesome, random_top5),
        ("SF vs RT5", stockfish, random_top5),
        ("Nett vs SF", nettlesome, stockfish),
    ]

    all_results = []
    total_start = time.time()

    for label, bot_a, bot_b in matchups:
        print(f"\n{'='*60}")
        print(f"MATCHUP: {label} ({bot_a.name} vs {bot_b.name})")
        print(f"{'='*60}")
        matchup_start = time.time()

        result = run_matchup(engine, bot_a, bot_b, num_games, label)
        all_results.append(result)

        matchup_elapsed = time.time() - matchup_start
        print(f"\n  Matchup complete in {timedelta(seconds=int(matchup_elapsed))}")

        # Save intermediate results after each matchup
        output = {
            "config": {"games_per_matchup": num_games, "depth": depth},
            "started": datetime.now().isoformat(),
            "matchups": all_results,
        }
        with open("tournament_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  (Results saved to tournament_results.json)")

    total_elapsed = time.time() - total_start

    print_summary(all_results)

    print(f"\nTotal time: {timedelta(seconds=int(total_elapsed))}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final save
    output = {
        "config": {"games_per_matchup": num_games, "depth": depth},
        "started": datetime.now().isoformat(),
        "total_time_seconds": int(total_elapsed),
        "matchups": all_results,
    }
    with open("tournament_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    engine.quit()


if __name__ == "__main__":
    main()
