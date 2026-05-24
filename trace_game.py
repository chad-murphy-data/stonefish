"""
Trace a single Stonefish v1 game move-by-move
==============================================
Runs one game and prints per-move:
  - Move number, who played, move (SAN)
  - Eval before/after from white's POV (and from Stonefish's POV)
  - Stonefish's mode that move: BASELINE / TRAP / GIVE-BACK
  - Trap markers: TRAP set with cost/gap/p_maia,
                  later TRAP FOUND / TRAP MISSED + give-back activation
  - Optional eval delta (gain/loss) per move

Usage:
    python3 trace_game.py [our_color w|b] [depth]
"""

import sys
import chess
import chess.engine

from engine import (
    NettlesomeBot, MaiaBot, EquilibriumBaselineBot, STOCKFISH_PATH,
    get_top_moves,
)
from maia_policy import MaiaPolicyEngine


def eval_for_white(sf, board, depth=10):
    """Return SF eval at the given position from white's POV (pawns)."""
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


def main(our_color_str: str, depth: int):
    weights = "/home/user/.maia/maia-1900.pb.gz"
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    oracle = MaiaPolicyEngine(weights)

    baseline = MaiaBot(weights, temperature=1.0, seed=3)
    give_back = EquilibriumBaselineBot(sf, oracle, target_delta=0.2, depth=depth, rating=1900)

    stonefish = NettlesomeBot(
        sf, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.5,
        maia_oracle=oracle, baseline_bot=baseline,
        puzzle_mode=True,
        min_eval_cost=0.20, min_gap=0.50, min_gap_ratio=1.5,
        p_maia_min=0.20, p_maia_max=0.80,
        post_trap_baseline=give_back, post_trap_duration=5,
    )

    opponent = MaiaBot(weights, temperature=1.0, seed=99)

    our_color = chess.WHITE if our_color_str.lower() in ("w", "white") else chess.BLACK
    color_label = "W" if our_color == chess.WHITE else "B"
    stonefish.reset_stats(our_color)

    board = chess.Board()
    print(f"=== Stonefish (color={color_label}) vs Maia 1900 ===")
    print(f"depth={depth}, baseline=Maia1900(T=1), give-back=Equilibrium(T-d=0.2) for 5 plies after find")
    print()
    header = f"{'#':>3} {'Side':<4} {'Move':<8} {'Mode':<14} {'Eval-W':>7} {'Stone':>7} {'Δ':>6}  Notes"
    print(header)
    print("-" * len(header))

    prev_eval_w = 0.0  # start eval
    ply = 0
    pending_trap_str = None
    give_back_remaining = 0  # local mirror for display

    while not board.is_game_over() and ply < 200:
        ply += 1
        side = "W" if board.turn == chess.WHITE else "B"
        is_us = (board.turn == our_color)

        if is_us:
            pre_moments = len(stonefish.stats.moments)
            in_give_back = stonefish._post_trap_remaining > 0
            move = stonefish.choose_move(board)

            # Identify mode AFTER the call (since _post_trap_remaining may have decremented)
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
            # Did opp just classify a trap?
            # NettlesomeBot.notify happens in play_game normally; here we
            # call note_opponent_reply directly.
            stonefish.note_opponent_reply(move)

        try:
            san = board.san(move)
        except Exception:
            san = move.uci()

        board.push(move)

        eval_w = eval_for_white(sf, board, depth=depth)
        # Convert to Stonefish POV
        stone_eval = eval_w if our_color == chess.WHITE else -eval_w
        delta = stone_eval - (prev_eval_w if our_color == chess.WHITE else -prev_eval_w)

        notes = ""
        if mode == "TRAP" and pending_trap_str:
            notes = pending_trap_str
            pending_trap_str = None
        elif mode == "opp":
            # If a moment just got classified, surface the rank
            if stonefish.stats.moments:
                last = stonefish.stats.moments[-1]
                if last.opponent_reply_uci == move.uci() and last.opponent_rank is not None:
                    rank = last.opponent_rank
                    if rank == 1:
                        notes = f"TRAP FOUND (opp played SF #1) -- 5-move give-back armed"
                    elif rank <= 3:
                        notes = f"TRAP missed (opp played #{rank})"
                    else:
                        notes = "TRAP missed (opp played outside top-3)"

        print(f"{ply:>3} {side:<4} {san:<8} {mode:<14} "
              f"{eval_w:+7.2f} {stone_eval:+7.2f} {delta:+6.2f}  {notes}",
              flush=True)
        prev_eval_w = eval_w

    print()
    result = board.result()
    print(f"Result: {result}")
    stats = stonefish.stats
    if stats.n_moments:
        print(f"Moments: {stats.n_moments}, opp found #1: {stats.n_opp_found_top} "
              f"({stats.n_opp_found_top/stats.n_moments*100:.0f}%)")
    else:
        print("No moments played")

    opponent.quit()
    oracle.quit()
    sf.quit()


if __name__ == "__main__":
    our_color = sys.argv[1] if len(sys.argv) > 1 else "w"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(our_color, depth)
