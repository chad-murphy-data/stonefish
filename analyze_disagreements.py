"""
Maia Tier Disagreement Analysis
================================
Plays self-play games at each elo preset and measures how often
the three Maia tiers (Floor/Stretch/Reach) disagree on the best move.

This is the key signal for puzzle detection -- disagreement between
tiers means a position where a weaker player would get it wrong
but a stronger one would get it right.

Run: python analyze_disagreements.py [num_games_per_preset]
"""

import chess
import chess.engine
import sys
import time
from collections import Counter, defaultdict

from engine import STOCKFISH_PATH, get_top_moves
from stonefish.maia import MaiaEngine
from stonefish.presets import ELO_PRESETS
from stonefish.scoring import involves_material_difference


def analyze_preset(engine, maia, elo, preset, num_games=2):
    """Play self-play games at a given elo preset and measure disagreements."""

    floor_r = preset["floor_rating"]
    stretch_r = preset["stretch_rating"]
    reach_r = preset["reach_rating"]

    stats = {
        "positions": 0,
        "any_disagreement": 0,
        "floor_v_stretch": 0,
        "floor_v_reach": 0,
        "stretch_v_reach": 0,
        "all_three_differ": 0,
        "material_relevant": 0,
        "floor_prob_when_disagree": [],
        "eval_gaps": [],
    }

    for game_num in range(num_games):
        board = chess.Board()
        move_count = 0

        for _ in range(200):
            if board.is_game_over():
                break

            # Play Stockfish move (depth 8 for speed)
            result = engine.analyse(board, chess.engine.Limit(depth=8))
            if "pv" not in result:
                break
            move = result["pv"][0]

            # Before playing, analyze this position from the mover's perspective
            if move_count >= 6:  # skip opening
                tiers = maia.predict_three_tier(board, floor_r, stretch_r, reach_r)

                floor_move = tiers.floor.top_move
                stretch_move = tiers.stretch.top_move
                reach_move = tiers.reach.top_move

                stats["positions"] += 1

                fvs = floor_move != stretch_move
                fvr = floor_move != reach_move
                svr = stretch_move != reach_move

                if fvs:
                    stats["floor_v_stretch"] += 1
                if fvr:
                    stats["floor_v_reach"] += 1
                if svr:
                    stats["stretch_v_reach"] += 1

                if fvs or fvr or svr:
                    stats["any_disagreement"] += 1

                    # Check if all three are different
                    if len({floor_move, stretch_move, reach_move}) == 3:
                        stats["all_three_differ"] += 1

                    # Check material relevance for the most important disagreement
                    if fvs and involves_material_difference(board, floor_move, stretch_move):
                        stats["material_relevant"] += 1
                    elif fvr and involves_material_difference(board, floor_move, reach_move):
                        stats["material_relevant"] += 1

                    # Floor's probability on the better move
                    better = stretch_move if fvs else reach_move
                    floor_prob = tiers.floor.prob_for(better)
                    stats["floor_prob_when_disagree"].append(floor_prob)

                    # Quick eval gap check (only when disagreeing)
                    if fvs or fvr:
                        worse = floor_move
                        better = stretch_move if fvs else reach_move
                        sign = 1.0 if board.turn == chess.WHITE else -1.0

                        board.push(worse)
                        w_top = get_top_moves(engine, board, num_moves=1, depth=10)
                        board.pop()

                        board.push(better)
                        b_top = get_top_moves(engine, board, num_moves=1, depth=10)
                        board.pop()

                        if w_top and b_top:
                            w_eval = w_top[0][1] * sign
                            b_eval = b_top[0][1] * sign
                            gap = w_eval - b_eval  # positive = worse move costs eval
                            stats["eval_gaps"].append(gap)

            board.push(move)
            move_count += 1

    return stats


def print_stats(elo, preset, stats):
    """Print formatted stats for one preset."""
    n = stats["positions"]
    if n == 0:
        print(f"  {elo}: no positions analyzed")
        return

    floor_r = preset["floor_rating"]
    stretch_r = preset["stretch_rating"] or "SF"
    reach_r = preset["reach_rating"] or "SF"

    any_pct = stats["any_disagreement"] / n * 100
    fvs_pct = stats["floor_v_stretch"] / n * 100
    fvr_pct = stats["floor_v_reach"] / n * 100
    svr_pct = stats["stretch_v_reach"] / n * 100
    tri_pct = stats["all_three_differ"] / n * 100

    mat_pct = (stats["material_relevant"] / stats["any_disagreement"] * 100
               if stats["any_disagreement"] > 0 else 0)

    print(f"\n  ELO {elo} (Floor={floor_r}, Stretch={stretch_r}, Reach={reach_r})")
    print(f"  {'—' * 55}")
    print(f"  Positions analyzed:     {n}")
    print(f"  Any disagreement:       {stats['any_disagreement']:>4} ({any_pct:>5.1f}%)")
    print(f"    Floor vs Stretch:     {stats['floor_v_stretch']:>4} ({fvs_pct:>5.1f}%)")
    print(f"    Floor vs Reach:       {stats['floor_v_reach']:>4} ({fvr_pct:>5.1f}%)")
    print(f"    Stretch vs Reach:     {stats['stretch_v_reach']:>4} ({svr_pct:>5.1f}%)")
    print(f"    All 3 differ:         {stats['all_three_differ']:>4} ({tri_pct:>5.1f}%)")
    print(f"    Material-relevant:    {stats['material_relevant']:>4} ({mat_pct:>5.1f}% of disagreements)")

    if stats["floor_prob_when_disagree"]:
        probs = stats["floor_prob_when_disagree"]
        avg_prob = sum(probs) / len(probs)
        under_5 = sum(1 for p in probs if p < 0.05) / len(probs) * 100
        under_20 = sum(1 for p in probs if p < 0.20) / len(probs) * 100
        print(f"  Floor prob on correct (when disagree):")
        print(f"    Avg: {avg_prob:.1%}  |  <5%: {under_5:.0f}%  |  <20%: {under_20:.0f}%")

    if stats["eval_gaps"]:
        gaps = stats["eval_gaps"]
        avg_gap = sum(gaps) / len(gaps)
        over_05 = sum(1 for g in gaps if g > 0.5) / len(gaps) * 100
        over_10 = sum(1 for g in gaps if g > 1.0) / len(gaps) * 100
        over_20 = sum(1 for g in gaps if g > 2.0) / len(gaps) * 100
        print(f"  Eval gap (floor move vs better move):")
        print(f"    Avg: {avg_gap:+.2f}  |  >0.5: {over_05:.0f}%  |  >1.0: {over_10:.0f}%  |  >2.0: {over_20:.0f}%")


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print("=" * 60)
    print("Maia Tier Disagreement Analysis")
    print("=" * 60)
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Games per preset: {num_games}")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 128})

    maia = MaiaEngine(stockfish_engine=engine)

    all_stats = {}
    total_start = time.time()

    for elo in sorted(ELO_PRESETS.keys()):
        preset = ELO_PRESETS[elo]
        start = time.time()
        print(f"\n  Running elo={elo}...", end=" ", flush=True)

        stats = analyze_preset(engine, maia, elo, preset, num_games=num_games)
        elapsed = time.time() - start
        print(f"done ({elapsed:.0f}s)")

        all_stats[elo] = stats

    engine.quit()

    total_elapsed = time.time() - total_start

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for elo in sorted(all_stats.keys()):
        print_stats(elo, ELO_PRESETS[elo], all_stats[elo])

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Disagreement Rate by Elo")
    print("=" * 60)
    print(f"  {'Elo':>6}  {'Tiers':>22}  {'Positions':>9}  {'Disagree%':>9}  {'FvS%':>6}  {'FvR%':>6}  {'SvR%':>6}")
    print(f"  {'—' * 70}")

    for elo in sorted(all_stats.keys()):
        s = all_stats[elo]
        p = ELO_PRESETS[elo]
        n = s["positions"]
        if n == 0:
            continue

        tiers = f"{p['floor_rating']}/{p['stretch_rating'] or 'SF'}/{p['reach_rating'] or 'SF'}"
        any_pct = s["any_disagreement"] / n * 100
        fvs = s["floor_v_stretch"] / n * 100
        fvr = s["floor_v_reach"] / n * 100
        svr = s["stretch_v_reach"] / n * 100

        print(f"  {elo:>6}  {tiers:>22}  {n:>9}  {any_pct:>8.1f}%  {fvs:>5.1f}%  {fvr:>5.1f}%  {svr:>5.1f}%")

    print(f"\n  Total time: {total_elapsed:.0f}s")
    print("\n  Analysis complete.")


if __name__ == "__main__":
    main()
