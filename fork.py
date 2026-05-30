"""
Fork: re-run a user's game with NettlesomeBot taking over from any ply.
=========================================================================
Given a saved history of user-vs-Maia moves (UCI list) + a starting ply,
spawns Stonefish v1 (NettlesomeBot with the v1 config) and Maia 1900
(deterministic via seed), plays out the game from that ply to end,
and returns the resulting move sequence + final outcome.

Used by /api/compare to detect divergences (compare bot.choose_move at
each user-turn position to what the user actually played) and by
/fork/<snap>/<ply> to render a continuation in a new tab.

Bot state at fork point:
  - _move_counter aligned to actual game ply
  - target_eval set to the current position's eval (so maintainer holds
    where the game is, not where it was at trap reset)
  - Empty stats (no trap memory from before the fork)
  - No active give-back

This is not a perfectly-faithful replay of "what if the bot had been
playing from move 1" -- it's "you've been playing, the bot takes over
now". That's the question the comparison answers.
"""

import chess
import chess.engine

from engine import (
    STOCKFISH_PATH, MAIA_WEIGHTS_PATH,
    NettlesomeBot, MaiaBot, EquilibriumMaintainerBot, WeakenedStockfishBot,
    play_game,
)
from maia_policy import MaiaPolicyEngine


def make_stonefish(sf, oracle, depth):
    """Same v1 config as qa_test.py."""
    baseline = EquilibriumMaintainerBot(sf, maia_oracle=oracle, depth=depth,
                                         num_sf_candidates=15,
                                         num_maia_candidates=10,
                                         initial_target=0.0)
    give_back = WeakenedStockfishBot(STOCKFISH_PATH, target_elo=1500,
                                      move_time=0.3)
    return NettlesomeBot(
        sf, num_candidates=7, num_responses=3,
        depth=depth, max_eval_cost=1.5,
        maia_oracle=oracle, baseline_bot=baseline,
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


def reconstruct_board(history_uci, up_to_ply):
    """Replay UCI moves through `up_to_ply` half-moves."""
    board = chess.Board()
    for i, uci in enumerate(history_uci):
        if i >= up_to_ply:
            break
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            raise ValueError(f"bad uci at index {i}: {uci!r}")
        if move not in board.legal_moves and len(uci) == 4:
            move = chess.Move.from_uci(uci + "q")  # default promotion
        if move not in board.legal_moves:
            raise ValueError(f"illegal move at index {i}: {uci!r}")
        board.push(move)
    return board


def eval_for(board, sf, depth, color):
    """Current eval in pawns from `color`'s perspective."""
    if board.is_game_over():
        res = board.result()
        raw = 99.99 if res == "1-0" else (-99.99 if res == "0-1" else 0.0)
    else:
        info = sf.analyse(board, chess.engine.Limit(depth=depth))
        s = info["score"].white()
        if s.is_mate():
            raw = 99.99 if s.mate() > 0 else -99.99
        else:
            raw = s.score() / 100.0
    return raw if color == chess.WHITE else -raw


def bot_pick_at_ply(history_uci, up_to_ply, user_color, sf, oracle, depth,
                     target_eval=0.0):
    """What would NettlesomeBot pick at this user-turn position?

    `target_eval` should be the maintainer target the user was holding
    (typically the live UI's target, e.g. 0.0). The bot picks moves that
    aim at the SAME target, so the comparison is honest -- "given that we
    were both trying to maintain target T, what would the bot pick?".

    Returns dict {uci, san, was_trap (bool), mode (str)}.
    `mode` is one of: "trap", "give-back" (n/a here, give-back is internal
    state we don't have), "endgame-precise", "baseline".
    """
    board = reconstruct_board(history_uci, up_to_ply)
    if board.turn != user_color:
        raise ValueError(f"ply {up_to_ply} is not user's turn")

    stonefish = make_stonefish(sf, oracle, depth)
    stonefish.reset_stats(user_color)
    stonefish._move_counter = up_to_ply // 2  # approximate game move number
    stonefish.baseline_bot.set_equilibrium(target_eval)

    pre_moments = len(stonefish.stats.moments)
    move = stonefish.choose_move(board)
    post_moments = len(stonefish.stats.moments)
    try:
        san = board.san(move)
    except Exception:
        san = move.uci()

    if post_moments > pre_moments:
        mode = "trap"
    elif stonefish.endgame_mode and stonefish._in_endgame(board):
        if (stonefish.stats.n_moments > 0
                and stonefish.stats.n_opp_found_top / stonefish.stats.n_moments < 0.5):
            mode = "endgame-precise"
        else:
            mode = "baseline"
    else:
        mode = "baseline"

    return {"uci": move.uci(), "san": san,
            "was_trap": post_moments > pre_moments, "mode": mode}


def play_fork(history_uci, fork_ply, user_color, maia_seed, depth,
              target_eval=0.0, max_moves=200):
    """Run the bot from fork_ply to the end of the game.

    `target_eval` is the maintainer target the user was holding -- the bot
    inherits it on takeover so it picks moves aimed at the same target.

    Returns dict {original_moves, fork_moves, fork_result, fen_at_fork,
                  starting_eval, trap_count, find_count}.
    """
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"Threads": 2, "Hash": 256})
    oracle = MaiaPolicyEngine(MAIA_WEIGHTS_PATH, rating=1900)
    try:
        board = reconstruct_board(history_uci, fork_ply)
        stonefish = make_stonefish(sf, oracle, depth)
        stonefish.reset_stats(user_color)
        stonefish._move_counter = fork_ply // 2
        cur_eval = eval_for(board, sf, depth, user_color)
        stonefish.baseline_bot.set_equilibrium(target_eval)
        maia = MaiaBot(MAIA_WEIGHTS_PATH, rating=1900,
                       temperature=1.0, seed=maia_seed)

        white_bot = stonefish if user_color == chess.WHITE else maia
        black_bot = maia if user_color == chess.WHITE else stonefish

        # Play from current board to end (custom inline loop -- play_game
        # resets the board to start, which we don't want).
        fork_moves = []
        ply = fork_ply
        while ply < max_moves and not board.is_game_over():
            active = white_bot if board.turn == chess.WHITE else black_bot
            passive = black_bot if board.turn == chess.WHITE else white_bot
            move = active.choose_move(board)
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            if isinstance(passive, NettlesomeBot):
                passive.note_opponent_reply(move, board=board)
            ev = eval_for(board, sf, depth, user_color) if False else None
            board.push(move)
            post_ev = eval_for(board, sf, depth, user_color)
            fork_moves.append({
                "ply": ply + 1, "san": san, "uci": move.uci(),
                "by": "stonefish" if active is stonefish else "maia",
                "eval": round(post_ev, 2),
            })
            ply += 1

        # Result + bot stats
        result_str = board.result()
        if result_str == "1-0": result = "stonefish" if user_color == chess.WHITE else "maia"
        elif result_str == "0-1": result = "maia" if user_color == chess.WHITE else "stonefish"
        else: result = "draw"

        # Build SAN list of the original prefix (so the UI doesn't have to
        # depend on a JS chess library to translate UCI -> SAN).
        original_san = []
        tmp = chess.Board()
        for uci in history_uci[:fork_ply]:
            try:
                mv = chess.Move.from_uci(uci)
                if mv not in tmp.legal_moves and len(uci) == 4:
                    mv = chess.Move.from_uci(uci + "q")
                original_san.append(tmp.san(mv))
                tmp.push(mv)
            except Exception:
                original_san.append(uci)

        return {
            "original_san": original_san,
            "starting_eval": round(cur_eval, 2),
            "starting_fen": reconstruct_board(history_uci, fork_ply).fen(),
            "fork_moves": fork_moves,
            "fork_result": result,
            "result_pgn": result_str,
            "trap_count": stonefish.stats.n_moments,
            "find_count": stonefish.stats.n_opp_found_top,
        }
    finally:
        try: sf.quit()
        except Exception: pass
        try: oracle.quit()
        except Exception: pass


def compare_history(history_uci, user_color, sf, oracle, depth,
                     target_eval=0.0, targets_by_ply=None):
    """For each user-turn ply, compare what bot would pick to what user did.

    If `targets_by_ply` is supplied (list of len(history_uci)), use the
    per-ply target the user was holding at that moment. Otherwise fall
    back to the single `target_eval` for every ply.

    Returns list of divergences:
      [{ply, user_uci, user_san, bot_uci, bot_san, mode, was_trap, target}]
    """
    divergences = []
    for i, uci in enumerate(history_uci):
        # Determine whose turn played this ply by replaying
        board = reconstruct_board(history_uci, i)
        if board.turn != user_color:
            continue
        t = (targets_by_ply[i] if targets_by_ply and i < len(targets_by_ply)
             else target_eval)
        try:
            pick = bot_pick_at_ply(history_uci, i, user_color, sf, oracle,
                                    depth, target_eval=t)
        except Exception as e:
            divergences.append({
                "ply": i, "error": str(e), "target": round(t, 2),
            })
            continue
        # User's actual move
        try:
            user_move = chess.Move.from_uci(uci)
            if user_move not in board.legal_moves and len(uci) == 4:
                user_move = chess.Move.from_uci(uci + "q")
            user_san = board.san(user_move)
        except Exception:
            user_san = uci
        if pick["uci"] != uci:
            divergences.append({
                "ply": i,
                "user_uci": uci, "user_san": user_san,
                "bot_uci": pick["uci"], "bot_san": pick["san"],
                "mode": pick["mode"],
                "was_trap": pick["was_trap"],
                "target": round(t, 2),
            })
    return divergences
