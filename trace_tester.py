"""
Trace a single Stonefish vs CoinFlipTester game move-by-move
=============================================================
Same trace format as trace_game.py but uses CoinFlipTesterBot as
opponent so we can see what happens when the opponent's baseline is
consistent (Stockfish) and only the trap responses are random.

Usage:
    python3 trace_tester.py [our_color w|b] [find_prob] [depth] [tester_seed]
"""

import sys
import chess
import chess.engine

from engine import (
    NettlesomeBot, CoinFlipTesterBot, EquilibriumBaselineBot, STOCKFISH_PATH,
)
from maia_policy import MaiaPolicyEngine


def eval_for_white(sf, board, depth=10):
    if board.is_game_over():
        result = board.result()
        if result == "1-0": return 99.99
        if result == "0-1": return -99.99
        return 0.0
    info = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    if score.is_mate():
        return 99.99 if score.mate() > 0 else -99.99
    return score.score() / 100.0


def main(our_color_str: str, find_prob: float, depth: int, tester_seed: int):
    weights = "/home/user/.maia/maia-1900.pb.gz"
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    oracle = MaiaPolicyEngine(weights)

    baseline = EquilibriumBaselineBot(sf, oracle, target_delta=0.3, depth=depth, rating=1900)
    give_back = EquilibriumBaselineBot(sf, oracle, target_delta=0.7, depth=depth, rating=1900)

    stonefish = NettlesomeBot(
        sf, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.5,
        maia_oracle=oracle, baseline_bot=baseline,
        puzzle_mode=True,
        min_eval_cost=0.20, min_gap=0.50, min_gap_ratio=1.5,
        p_maia_min=0.20, p_maia_max=0.80,
        post_trap_baseline=give_back, post_trap_duration=5,
        endgame_mode=True,
        endgame_min_move=30, endgame_min_pieces=14,
    )

    opponent = CoinFlipTesterBot(sf, depth=depth, find_probability=find_prob,
                                  trap_gap_threshold=0.5, seed=tester_seed)

    our_color = chess.WHITE if our_color_str.lower() in ("w", "white") else chess.BLACK
    color_label = "W" if our_color == chess.WHITE else "B"
    stonefish.reset_stats(our_color)

    board = chess.Board()
    print(f"=== Stonefish (color={color_label}) vs CoinFlipTester(p={find_prob:.2f}, seed={tester_seed}) ===")
    print(f"depth={depth}")
    print()
    header = f"{'#':>3} {'Side':<4} {'Move':<8} {'Mode':<14} {'Eval-W':>7} {'Stone':>7} {'Δ':>6}  Notes"
    print(header)
    print("-" * len(header))

    prev_eval_w = 0.0
    ply = 0
    pending_trap_str = None

    while not board.is_game_over() and ply < 200:
        ply += 1
        side = "W" if board.turn == chess.WHITE else "B"
        is_us = (board.turn == our_color)

        if is_us:
            pre_moments = len(stonefish.stats.moments)
            in_give_back = stonefish._post_trap_remaining > 0
            move = stonefish.choose_move(board)
            if len(stonefish.stats.moments) > pre_moments:
                mode = "TRAP"
                m = stonefish.stats.moments[-1]
                pending_trap_str = (
                    f"set: SF would play {m.sf_top_san}; "
                    f"cost={m.eval_cost:.2f}p gap={m.gap_1_2:.2f}p "
                    f"P_maia(top)={m.p_maia_top*100:.0f}%"
                ) if m.p_maia_top is not None else "set"
            elif in_give_back:
                mode = "GIVE-BACK"
            else:
                mode = "baseline"
        else:
            move = opponent.choose_move(board)
            mode = "opp"
            stonefish.note_opponent_reply(move)

        try: san = board.san(move)
        except Exception: san = move.uci()
        board.push(move)

        eval_w = eval_for_white(sf, board, depth=depth)
        stone_eval = eval_w if our_color == chess.WHITE else -eval_w
        delta = stone_eval - (prev_eval_w if our_color == chess.WHITE else -prev_eval_w)

        notes = ""
        if mode == "TRAP" and pending_trap_str:
            notes = pending_trap_str
            pending_trap_str = None
        elif mode == "opp":
            if stonefish.stats.moments:
                last = stonefish.stats.moments[-1]
                if last.opponent_reply_uci == move.uci() and last.opponent_rank is not None:
                    rank = last.opponent_rank
                    if rank == 1:
                        notes = "TRAP FOUND -- 5-move give-back armed"
                    elif rank <= 3:
                        notes = f"TRAP missed (#{rank})"
                    else:
                        notes = "TRAP missed (outside top-3)"

        print(f"{ply:>3} {side:<4} {san:<8} {mode:<14} "
              f"{eval_w:+7.2f} {stone_eval:+7.2f} {delta:+6.2f}  {notes}",
              flush=True)
        prev_eval_w = eval_w

    print()
    print(f"Result: {board.result()}")
    s = stonefish.stats
    if s.n_moments:
        print(f"Moments: {s.n_moments}, opp found #1: {s.n_opp_found_top} "
              f"({s.n_opp_found_top/s.n_moments*100:.0f}%)")
    else:
        print("No moments played")
    oracle.quit(); sf.quit()


if __name__ == "__main__":
    color = sys.argv[1] if len(sys.argv) > 1 else "w"
    prob  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed  = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    main(color, prob, depth, seed)
