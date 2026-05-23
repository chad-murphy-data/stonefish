"""
Maia moment diagnostic
======================
Runs N games of NettlesomeBot vs Maia 1900 and captures, for each
nettlesome moment played, Maia's predicted policy probability of Stockfish's
#1 reply. This tells us, at threshold X, how many moments per game we'd
keep if we filtered to "Maia finds the right reply < X% of the time".

Usage:
    python3 maia_moment_diagnostic.py [num_games] [depth]
"""

import sys
import json
import time
import chess
import chess.engine

from engine import NettlesomeBot, MaiaBot, STOCKFISH_PATH
from maia_policy import MaiaPolicyEngine


def run_diagnostic(num_games: int = 5, depth: int = 10,
                   maia_weights: str = "/home/user/.maia/maia-1900.pb.gz",
                   max_eval_cost: float = 1.0,
                   out_path: str = "moment_diagnostic.json"):
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})

    maia_opponent = MaiaBot(maia_weights, rating=1900)
    maia_oracle = MaiaPolicyEngine(maia_weights, rating=1900)

    nettlesome = NettlesomeBot(sf, num_candidates=7, num_responses=3,
                                depth=depth, max_eval_cost=max_eval_cost)

    all_moments = []  # flat list across all games

    for game_idx in range(num_games):
        # Alternate colors
        we_white = (game_idx % 2 == 0)
        nettlesome.reset_stats(chess.WHITE if we_white else chess.BLACK)
        board = chess.Board()
        moves = []
        game_start = time.time()
        print(f"\n[Game {game_idx + 1}/{num_games}] {'White' if we_white else 'Black'} = Nettlesome",
              flush=True)

        for ply in range(300):
            if board.is_game_over():
                break

            our_turn = (board.turn == chess.WHITE) == we_white

            if our_turn:
                pre_moments = len(nettlesome.stats.moments)
                move = nettlesome.choose_move(board)
                # If this generated a new moment, immediately query Maia's
                # policy on board-after-our-move to enrich the moment.
                if len(nettlesome.stats.moments) > pre_moments:
                    moment = nettlesome.stats.moments[-1]
                    next_board = board.copy()
                    next_board.push(move)
                    maia_policy = maia_oracle.predict_policy(next_board)
                    # Probability Maia plays SF #1
                    if moment.top_response_ucis:
                        sf_top_uci = moment.top_response_ucis[0]
                        sf_top_move = chess.Move.from_uci(sf_top_uci)
                        p_sf_top = maia_policy.get(sf_top_move, 0.0)
                    else:
                        p_sf_top = 0.0
                    # Maia's own top choice + its probability
                    if maia_policy:
                        maia_top_move = max(maia_policy, key=maia_policy.get)
                        p_maia_top = maia_policy[maia_top_move]
                        try:
                            maia_top_san = next_board.san(maia_top_move)
                        except Exception:
                            maia_top_san = maia_top_move.uci()
                    else:
                        maia_top_san = "?"
                        p_maia_top = 0.0
                    moment_data = {
                        "game": game_idx + 1,
                        "ply": ply + 1,
                        "our_move_san": moment.our_move_san,
                        "sf_top_san": moment.sf_top_san,
                        "eval_cost": moment.eval_cost,
                        "gap_1_2": moment.gap_1_2,
                        "p_maia_sf_top": round(p_sf_top, 4),
                        "p_maia_top": round(p_maia_top, 4),
                        "maia_top_san": maia_top_san,
                        "maia_agrees_with_sf": maia_top_move == sf_top_move if maia_policy else None,
                    }
                    all_moments.append(moment_data)
                    print(f"  ply {ply+1:3d}: {moment.our_move_san:>6} "
                          f"[N! cost={moment.eval_cost:.2f} gap={moment.gap_1_2:.2f}] -> "
                          f"P(Maia plays {moment.top_response_ucis[0]}) = {p_sf_top*100:5.1f}%  "
                          f"P(Maia top {maia_top_san}) = {p_maia_top*100:5.1f}%",
                          flush=True)
            else:
                move = maia_opponent.choose_move(board)
                # Let nettlesome bot classify opp reply
                nettlesome.note_opponent_reply(move)

            moves.append(move)
            board.push(move)

        result_str = board.result()
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = 0.0
        else:
            result = 0.5
        our_result = result if we_white else 1.0 - result
        elapsed = time.time() - game_start
        n_game_moments = len(nettlesome.stats.moments)
        outcome_str = "WIN" if our_result == 1.0 else ("LOSS" if our_result == 0.0 else "DRAW")
        print(f"  [Game {game_idx + 1}] {outcome_str} in {len(moves)} moves "
              f"({elapsed:.1f}s) -- {n_game_moments} moments", flush=True)

    # Write everything to disk
    with open(out_path, "w") as f:
        json.dump({
            "num_games": num_games,
            "depth": depth,
            "max_eval_cost": max_eval_cost,
            "weights": maia_weights,
            "moments": all_moments,
        }, f, indent=2)

    # Analysis
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC SUMMARY  (data written to {out_path})")
    print(f"{'=' * 70}")
    n = len(all_moments)
    print(f"Total moments across {num_games} games: {n} ({n/num_games:.1f}/game)")

    if n == 0:
        return

    probs = [m["p_maia_sf_top"] for m in all_moments]
    # Histogram
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    print(f"\nDistribution of P(Maia plays SF #1) across all moments:")
    print(f"{'range':<14} {'count':>6} {'pct':>7} {'cum/game':>10}")
    cum = 0
    for lo, hi in zip(bins, bins[1:]):
        cnt = sum(1 for p in probs if lo <= p < hi)
        cum += cnt
        cum_per_game = cum / num_games
        pct = cnt / n * 100
        print(f"[{lo*100:>3.0f}%, {hi*100:>3.0f}%)   {cnt:>5}  {pct:>5.1f}%  "
              f"{cum_per_game:>7.1f}/game")

    # Threshold table: "if we keep moments where P < threshold..."
    print(f"\nIf we filtered to moments with P(Maia plays SF #1) < threshold:")
    print(f"{'threshold':>10} {'kept':>6} {'/game':>7} {'agree-rate':>11}")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        kept = sum(1 for p in probs if p < t)
        kpg = kept / num_games
        # How often did the "hard" subset have Maia agreeing with SF anyway?
        # (We expect 0% by definition once we filter, but check)
        print(f"  P < {t*100:>3.0f}%   {kept:>5}  {kpg:>5.1f}  --")

    # Specifically highlight a target of ~5/game
    target = 5.0
    print(f"\nTarget: {target:.1f} moments/game")
    best_t = None
    for t in [round(x * 0.01, 2) for x in range(1, 100)]:
        kept = sum(1 for p in probs if p < t)
        kpg = kept / num_games
        if kpg >= target:
            best_t = t
            break
    if best_t is not None:
        kept = sum(1 for p in probs if p < best_t)
        print(f"  Threshold {best_t*100:.0f}% -> {kept} kept ({kept/num_games:.1f}/game)")
    else:
        print(f"  Even no threshold yields only {n/num_games:.1f}/game (less than target)")

    sf.quit()
    maia_opponent.quit()
    maia_oracle.quit()


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_diagnostic(num_games=num_games, depth=depth)
