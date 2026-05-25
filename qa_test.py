"""
Stonefish QA test
=================
Runs N games of Stonefish v1 (equilibrium baseline + symmetric puzzle
filter) against stochastic Maia 1900 and checks four rules:

  1. On average, ~4-6 traps per game (mean in [4, 6] AND
     >= 80% of games have at least 3 moments).
  2. When opponent finds the top reply >60% of trap moments,
     they should win (or draw): success rate of opp >= 70%.
  3. When opponent fails to find the top reply >40% of moments,
     they should lose (or draw): bot success rate >= 70%.
  4. Trap-finding should be the dominant signal: correlation
     between find_rate and (1 - bot_score) >= 0.5.

Usage:
    python3 qa_test.py [num_games] [depth]
"""

import sys
import time
import json
import statistics
import chess
import chess.engine

from engine import (
    NettlesomeBot, MaiaBot, EquilibriumBaselineBot,
    EquilibriumMaintainerBot, STOCKFISH_PATH,
    play_game,
)
from maia_policy import MaiaPolicyEngine


def make_stonefish(sf_engine, maia_oracle, baseline, give_back, depth):
    """Build the canonical Stonefish v1 the QA targets."""
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
        endgame_mode=True,
        endgame_min_move=30, endgame_min_pieces=14,
    )


def correlation(xs, ys):
    """Pearson correlation between two equal-length sequences."""
    n = len(xs)
    if n < 2: return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0 or dy == 0: return 0.0
    return num / ((dx * dy) ** 0.5)


def run_qa(num_games: int, depth: int):
    maia_weights = "/home/user/.maia/maia-1900.pb.gz"

    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    oracle = MaiaPolicyEngine(maia_weights)
    # Baseline = EquilibriumMaintainer: holds eval at a target set by trap
    # events. Between traps, trap resolution is the only thing that moves
    # the score. Replaced WeakenedStockfishBot(1900) baseline whose bursty
    # randomized weakening was drowning out the trap signal.
    baseline = EquilibriumMaintainerBot(sf, depth=depth, num_candidates=8,
                                         initial_target=0.0)
    # Give-back = SF at UCI_Elo=1500 for 5 moves after find
    from engine import WeakenedStockfishBot
    give_back = WeakenedStockfishBot(STOCKFISH_PATH, target_elo=1500, move_time=0.3)
    stonefish = make_stonefish(sf, oracle, baseline, give_back, depth)
    maia_opponent = MaiaBot(maia_weights, temperature=1.0, seed=99)

    per_game = []  # one row per game

    print(f"Stonefish QA: N={num_games}, depth={depth}", flush=True)
    print(f"vs Maia 1900 (stochastic, T=1)")
    print("=" * 70, flush=True)

    for g in range(num_games):
        # Alternate colors
        if g % 2 == 0:
            white, black = stonefish, maia_opponent
            sf_white = True
        else:
            white, black = maia_opponent, stonefish
            sf_white = False

        start = time.time()
        result_white, num_moves, _ = play_game(white, black, verbose=False)
        elapsed = time.time() - start

        # Our score from Stonefish's POV: 1.0 win, 0.0 loss, 0.5 draw
        bot_score = result_white if sf_white else (1.0 - result_white)

        stats = stonefish.stats
        moments = stats.n_moments
        found = stats.n_opp_found_top
        find_rate = (found / moments) if moments > 0 else None

        per_game.append({
            "game": g + 1,
            "color": "W" if sf_white else "B",
            "moves": num_moves,
            "moments": moments,
            "found": found,
            "find_rate": find_rate,
            "bot_score": bot_score,
            "elapsed_s": round(elapsed, 1),
        })
        outcome = "W" if bot_score == 1.0 else ("L" if bot_score == 0.0 else "D")
        find_str = f"{find_rate*100:>4.0f}%" if find_rate is not None else "  n/a"
        print(f"  G{g + 1:>2}/{num_games} ({stats.color and 'W' or 'B'}): "
              f"{outcome}  {num_moves:>3}mv  "
              f"moments={moments:>2}  found={found:>2}/{moments:<2}  "
              f"find_rate={find_str}  {elapsed:.1f}s", flush=True)

    # ----- analysis -----
    print()
    print("=" * 70)
    print(f"RESULTS  ({num_games} games)")
    print("=" * 70)

    bot_wins = sum(1 for r in per_game if r["bot_score"] == 1.0)
    bot_losses = sum(1 for r in per_game if r["bot_score"] == 0.0)
    draws = num_games - bot_wins - bot_losses
    overall = sum(r["bot_score"] for r in per_game) / num_games
    print(f"  Stonefish: {bot_wins}W / {draws}D / {bot_losses}L  (score {overall:.1%})")

    # ---- Rule 1: trap volume ----
    moments_per_game = [r["moments"] for r in per_game]
    mean_moments = statistics.mean(moments_per_game)
    games_with_3plus = sum(1 for m in moments_per_game if m >= 3)
    pct_3plus = games_with_3plus / num_games
    rule_1 = (4 <= mean_moments <= 6) and (pct_3plus >= 0.8)
    print()
    print(f"  RULE 1 -- ~4-6 traps/game")
    print(f"    mean moments/game:        {mean_moments:.1f}  (target [4, 6])")
    print(f"    games with >=3 moments:   {games_with_3plus}/{num_games} ({pct_3plus:.0%})  (target >=80%)")
    print(f"    PASS" if rule_1 else "    FAIL")

    # ---- Rules 2 & 3: find-rate determining outcome ----
    games_with_finds = [r for r in per_game if r["find_rate"] is not None]
    high_find = [r for r in games_with_finds if r["find_rate"] > 0.60]
    low_find  = [r for r in games_with_finds if r["find_rate"] < 0.40]

    print()
    if not high_find:
        rule_2 = False
        print(f"  RULE 2 -- when opp finds >60%, opp wins or draws")
        print(f"    no games with find_rate > 60%; can't evaluate")
        print(f"    FAIL (insufficient data)")
    else:
        opp_success = sum(1.0 - r["bot_score"] for r in high_find) / len(high_find)
        rule_2 = opp_success >= 0.70
        print(f"  RULE 2 -- when opp finds >60%, opp wins or draws")
        print(f"    games matching:           {len(high_find)}/{num_games}")
        print(f"    opp avg score:            {opp_success:.1%}  (target >=70%)")
        print(f"    PASS" if rule_2 else "    FAIL")

    print()
    if not low_find:
        rule_3 = False
        print(f"  RULE 3 -- when opp finds <40%, opp loses or draws")
        print(f"    no games with find_rate < 40%; can't evaluate")
        print(f"    FAIL (insufficient data)")
    else:
        bot_success = sum(r["bot_score"] for r in low_find) / len(low_find)
        rule_3 = bot_success >= 0.70
        print(f"  RULE 3 -- when opp finds <40%, opp loses or draws")
        print(f"    games matching:           {len(low_find)}/{num_games}")
        print(f"    Stonefish avg score:      {bot_success:.1%}  (target >=70%)")
        print(f"    PASS" if rule_3 else "    FAIL")

    # ---- Rule 4: correlation ----
    xs = [r["find_rate"] for r in games_with_finds]
    ys = [1.0 - r["bot_score"] for r in games_with_finds]  # opp score
    corr = correlation(xs, ys)
    rule_4 = corr >= 0.5
    print()
    print(f"  RULE 4 -- find_rate predicts outcome")
    print(f"    Pearson r:                {corr:+.2f}  (target >=+0.50)")
    print(f"    n:                        {len(xs)} games with at least one moment")
    print(f"    PASS" if rule_4 else "    FAIL")

    # ---- find-rate bin scoreboard ----
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print()
    print(f"  Score by find-rate bin (opp's POV):")
    print(f"    {'find rate':<14} {'n':>4} {'opp score':>11}")
    print(f"    {'-' * 32}")
    for lo, hi in bins:
        games = [r for r in games_with_finds if lo <= r["find_rate"] < hi]
        if not games:
            print(f"    [{lo*100:>3.0f}%, {hi*100:>3.0f}%)   {0:>4}   --")
        else:
            opp_score = sum(1.0 - r["bot_score"] for r in games) / len(games)
            print(f"    [{lo*100:>3.0f}%, {hi*100:>3.0f}%)   {len(games):>4}   {opp_score:>10.1%}")

    games_no_moments = sum(1 for r in per_game if r["find_rate"] is None)
    if games_no_moments:
        bot_score_no_moments = (
            sum(r["bot_score"] for r in per_game if r["find_rate"] is None)
            / games_no_moments
        )
        print(f"    no moments       {games_no_moments:>4}   "
              f"{1 - bot_score_no_moments:>10.1%}")

    # ---- overall ----
    print()
    rules = [rule_1, rule_2, rule_3, rule_4]
    n_pass = sum(rules)
    print(f"  Rules passed: {n_pass}/4")
    if n_pass == 4:
        print(f"  STONEFISH v1 PASSES CALIBRATION")
    else:
        print(f"  Calibration is OFF on {4 - n_pass} rule(s)")

    # Dump raw data for further analysis
    out_path = f"results/qa_v1_n{num_games}_d{depth}.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "num_games": num_games,
            "depth": depth,
            "per_game": per_game,
            "rules_passed": rules,
            "moments_mean": mean_moments,
            "correlation": corr,
        }, f, indent=2)
    print(f"\n  Data: {out_path}")

    # cleanup
    try: maia_opponent.quit()
    except: pass
    try: oracle.quit()
    except: pass
    try: sf.quit()
    except: pass


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    depth     = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_qa(num_games, depth)
