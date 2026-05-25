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
from dataclasses import asdict

from engine import (
    NettlesomeBot, CoinFlipTesterBot, EquilibriumBaselineBot,
    EquilibriumMaintainerBot, STOCKFISH_PATH,
    play_game, get_top_moves,
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
        endgame_mode=True,
        endgame_min_move=30, endgame_min_pieces=14,
    )


def play_game_logged(stonefish, opponent, stone_white, sf, depth, max_moves=200):
    """Play Stonefish vs CoinFlipTester and log per-Stonefish-move data for
    noise analysis. Mirrors engine.play_game but additionally records, for
    each Stonefish move:
      - mode: "trap" | "give_back" | "endgame_precise" | "baseline"
      - eval_cost: full-SF top-eval minus eval-after-played-move (in pawns,
        from Stonefish's perspective)
      - top_eval, played_eval (both in side-to-move-friendly units)
      - trap_idx: index into stonefish.stats.moments for "trap" rows
    """
    board = chess.Board()
    stone_color = chess.WHITE if stone_white else chess.BLACK
    stonefish.reset_stats(stone_color)

    white = stonefish if stone_white else opponent
    black = opponent if stone_white else stonefish
    move_log = []

    ply = 0
    while ply < max_moves and not board.is_game_over():
        ply += 1
        is_stone = (board.turn == stone_color)
        active = white if board.turn == chess.WHITE else black
        passive = black if board.turn == chess.WHITE else white
        sign = 1.0 if board.turn == chess.WHITE else -1.0

        if is_stone:
            pre_moments = len(stonefish.stats.moments)
            pre_post_trap = stonefish._post_trap_remaining
            # Snapshot baseline's equilibrium target for the analyzer
            target_eval = getattr(stonefish.baseline_bot, 'target_eval', None)
            # Full-strength SF top moves (depth = bot's analysis depth) so
            # we can score the played move's eval_cost vs SF top.
            top_pre = get_top_moves(sf, board, num_moves=8, depth=depth)
            top_eval = (top_pre[0][1] * sign) if top_pre else 0.0
            # Cap mate-y spikes
            top_eval = max(-10.0, min(10.0, top_eval))
            top_by_uci = {m.uci(): max(-10.0, min(10.0, e * sign))
                          for m, e in top_pre}

        move = active.choose_move(board)

        if is_stone:
            post_moments = len(stonefish.stats.moments)
            if post_moments > pre_moments:
                mode = "trap"
                trap_idx = post_moments - 1
            elif pre_post_trap > 0:
                mode = "give_back"
                trap_idx = None
            else:
                # Endgame-precise vs baseline check (mirrors NettlesomeBot
                # logic; _move_counter has already been incremented by
                # choose_move so _in_endgame sees the up-to-date count)
                solve_rate = (stonefish.stats.n_opp_found_top /
                              stonefish.stats.n_moments
                              if stonefish.stats.n_moments > 0 else 0.0)
                if (stonefish.endgame_mode and stonefish._in_endgame(board)
                        and stonefish.stats.n_moments > 0
                        and solve_rate < 0.5):
                    mode = "endgame_precise"
                else:
                    mode = "baseline"
                trap_idx = None

            played_uci = move.uci()
            if played_uci in top_by_uci:
                played_eval = top_by_uci[played_uci]
            else:
                # Played move outside top-8: evaluate fresh after the push
                tmp = board.copy(); tmp.push(move)
                if tmp.is_game_over():
                    res = tmp.result()
                    raw_white = (99.99 if res == "1-0" else
                                 -99.99 if res == "0-1" else 0.0)
                else:
                    info = sf.analyse(tmp, chess.engine.Limit(depth=depth))
                    s = info["score"].white()
                    raw_white = ((99.99 if s.mate() > 0 else -99.99)
                                 if s.is_mate() else s.score() / 100.0)
                # Convert to side-to-move (us, who just played)
                played_eval = max(-10.0, min(10.0, raw_white * sign))

            move_log.append({
                "ply": ply,
                "mode": mode,
                "eval_cost": round(top_eval - played_eval, 3),
                "top_eval": round(top_eval, 3),
                "played_eval": round(played_eval, 3),
                "target_eval": (round(target_eval, 3)
                                if target_eval is not None else None),
                "in_top8": played_uci in top_by_uci,
                "trap_idx": trap_idx,
            })

        if passive is stonefish:
            stonefish.note_opponent_reply(move, board=board)

        board.push(move)

    result_str = board.result()
    result = 1.0 if result_str == "1-0" else (0.0 if result_str == "0-1" else 0.5)
    bot_score = result if stone_white else (1.0 - result)
    stonefish.stats.result = bot_score
    stonefish.stats.total_moves = ply
    return bot_score, ply, move_log


def run_for_probability(find_prob, num_games, depth, sf, oracle):
    from engine import WeakenedStockfishBot
    # EquilibriumMaintainer: holds eval at target_eval (initial 0.0,
    # updated by NettlesomeBot after each trap resolves / after give-back).
    # Between traps, eval is locked. The only thing that moves the score
    # is trap resolution.
    # SF top-15 + Maia top-10 expansion -- Maia surfaces 1900-plausible
    # moves SF didn't include in its top-N, letting the maintainer reach
    # further from SF #1 when needed.
    baseline = EquilibriumMaintainerBot(sf, maia_oracle=oracle, depth=depth,
                                         num_sf_candidates=15,
                                         num_maia_candidates=10,
                                         initial_target=0.0)
    give_back = WeakenedStockfishBot(STOCKFISH_PATH, target_elo=1500, move_time=0.3)
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
        stone_white = (g % 2 == 0)

        start = time.time()
        bot_score, num_moves, move_log = play_game_logged(
            stonefish, tester, stone_white, sf, depth,
        )
        elapsed = time.time() - start

        stats = stonefish.stats
        moments = stats.n_moments
        found = stats.n_opp_found_top
        find_rate = (found / moments) if moments > 0 else None

        games.append({
            "g": g + 1, "moves": num_moves, "moments": moments, "found": found,
            "find_rate": find_rate, "bot_score": bot_score, "elapsed": elapsed,
            "stone_white": stone_white,
            "moments_data": [asdict(m) for m in stats.moments],
            "moves_log": move_log,
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
