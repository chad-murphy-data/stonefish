"""
Web UI: play Stonefish manually vs Maia 1900
=============================================
Flask app with a drag-and-drop chessboard. You play Stonefish; Maia 1900
plays the opponent. At each of your turns the sidebar shows SF top-25
candidates with their evals + drift from target; the maintainer's pick
is highlighted. Click a candidate to play it, or drag a piece on the
board.

Usage:
    python3 web_game.py
    # then open http://localhost:5000 in your browser

Requires:
    pip install flask python-chess
    + Stockfish on PATH (or STOCKFISH_PATH env)
    + lc0 on PATH (or LC0_PATH env)
    + Maia 1900 weights at ~/.maia/maia-1900.pb.gz (or MAIA_WEIGHTS env)
"""

import threading

import chess
import chess.engine
from flask import Flask, request, jsonify, render_template_string

from engine import (
    STOCKFISH_PATH, MAIA_WEIGHTS_PATH,
    MaiaBot, get_top_moves, find_puzzle_trap, apply_maintainer_rules,
)
from maia_policy import MaiaPolicyEngine

app = Flask(__name__)

# Single-game global state. Single-user app; one lock is enough.
state_lock = threading.Lock()
state = {
    "board": chess.Board(),
    "history": [],   # list of dicts: {ply, san, eval, by}
    "target": 0.0,
    "user_color": chess.WHITE,
    "sf": None,
    "maia": None,
    "depth": 10,     # SF analysis depth -- set in init_engines() from CLI
    "maia_seed": None,  # int or None; when set, Maia plays deterministically
    # Last trap-resolution event, used by the UI for notifications and
    # to auto-update the target. Set after Maia replies to a user move
    # that qualified as a trap. Has shape:
    # {"ply", "found": bool, "trap_san", "expected_reply_san",
    #  "actual_reply_san", "old_target", "new_target",
    #  "eval_cost", "gap", "p_maia"}
    "last_trap_event": None,
    "event_seq": 0,  # monotonic counter so UI can detect new events
    # Cache: FEN -> trap_info from find_puzzle_trap. Populated by
    # serialize_state, consumed by api_move. Necessary because SF with
    # Threads>1 is non-deterministic; running find_puzzle_trap twice on
    # the same position (once for the UI display, once when validating
    # the user's move) was producing inconsistent trap qualifications,
    # which made auto-target-updates silently fail.
    "trap_cache": {},
    # Give-back: after Maia FINDS a trap, the maintainer enters give-back
    # for 5 user moves, where the effective target is lowered by
    # give_back_drop. The bot's actual NettlesomeBot uses WeakenedSF(1500)
    # for this; in the manual UI we approximate by holding the lower
    # target so the rec naturally picks sub-optimal moves. After the 5
    # moves complete, state["target"] is updated to whatever eval landed
    # at -- the "post-give-back equilibrium". Mirrors NettlesomeBot's
    # _refresh_equilibrium_next deferred-refresh on found traps.
    "give_back_remaining": 0,
    "give_back_drop": 0.5,
}

# Snapshots of completed/in-progress games keyed by snapshot_id, for forks.
# {snap_id: {"history": [...], "user_color": "w"|"b", "maia_seed": int|None,
#            "depth": int, "starting_target": float}}
snapshots = {}
import uuid


def init_engines(depth=10, maia_seed=None):
    state["sf"] = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    state["sf"].configure({"Threads": 2, "Hash": 256})
    state["maia_seed"] = maia_seed
    state["maia"] = MaiaBot(MAIA_WEIGHTS_PATH,
                            rating=1900, temperature=1.0, seed=maia_seed)
    # Separate MaiaPolicyEngine for the trap-finder (uses predict_policy,
    # different API than MaiaBot.choose_move).
    state["maia_oracle"] = MaiaPolicyEngine(MAIA_WEIGHTS_PATH, rating=1900)
    state["depth"] = depth


def eval_white(board):
    """Eval in pawns from white's perspective."""
    if board.is_game_over():
        res = board.result()
        if res == "1-0": return 99.99
        if res == "0-1": return -99.99
        return 0.0
    info = state["sf"].analyse(board, chess.engine.Limit(depth=state["depth"]))
    s = info["score"].white()
    if s.is_mate():
        return 99.99 if s.mate() > 0 else -99.99
    return s.score() / 100.0


def maia_plays():
    """Play Maia's move (assumes it's Maia's turn)."""
    target_before = state["target"]
    move = state["maia"].choose_move(state["board"])
    try:
        san = state["board"].san(move)
    except Exception:
        san = move.uci()
    state["board"].push(move)
    sign = 1.0 if state["user_color"] == chess.WHITE else -1.0
    post = eval_white(state["board"]) * sign
    state["history"].append({
        "ply": state["board"].ply(), "san": san,
        "eval": round(post, 2), "by": "maia",
        "uci": move.uci(),
        "target_before": round(target_before, 2),
    })


def serialize_state():
    board = state["board"]
    user_color = state["user_color"]
    sign = 1.0 if user_color == chess.WHITE else -1.0
    cur_eval = eval_white(board) * sign

    candidates = []
    trap_info = None
    maintainer_pick_uci = None
    maintainer_reason = None
    if (board.turn == user_color
            and not board.is_game_over()):
        # Effective target accounts for give-back: while it's active, the
        # maintainer aims at a lower target so the rec naturally picks
        # sub-optimal moves and eval bleeds over the give-back window.
        effective_target = state["target"]
        if state["give_back_remaining"] > 0:
            effective_target -= state["give_back_drop"]
        raw = get_top_moves(state["sf"], board, num_moves=25,
                            depth=state["depth"])
        cand_tuples = []
        for move, eval_cp in raw:
            eval_for_us = max(-10.0, min(10.0, eval_cp * sign))
            try:
                san = board.san(move)
            except Exception:
                san = move.uci()
            candidates.append({
                "uci": move.uci(),
                "san": san,
                "eval": round(eval_for_us, 2),
                "drift": round(eval_for_us - effective_target, 2),
            })
            cand_tuples.append((move, eval_for_us))
        # Detect forced mate from our POV (rule 2)
        mate_dist = None
        try:
            info = state["sf"].analyse(board,
                                        chess.engine.Limit(depth=state["depth"]))
            sc = info["score"].pov(board.turn)
            if sc.is_mate():
                mate_dist = sc.mate()
        except Exception:
            pass
        # Apply the maintainer's decision rules to pick the recommended move
        chosen, maintainer_reason = apply_maintainer_rules(
            cand_tuples, effective_target, mate_distance=mate_dist,
        )
        if chosen is not None:
            maintainer_pick_uci = chosen.uci()
        # Look for an available puzzle trap (Stonefish's actual trap logic).
        # Cache the result keyed on FEN -- api_move re-uses this when checking
        # whether the user's move was a trap, avoiding SF threading-related
        # non-determinism that would otherwise make the second find_puzzle_trap
        # call disagree with this one.
        fen = board.fen()
        if fen in state["trap_cache"]:
            trap_info = state["trap_cache"][fen]
        else:
            try:
                trap_info = find_puzzle_trap(
                    state["sf"], state["maia_oracle"], board,
                    depth=state["depth"],
                )
            except Exception:
                trap_info = None
            state["trap_cache"][fen] = trap_info

    return {
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "user_color": "w" if user_color == chess.WHITE else "b",
        "eval": round(cur_eval, 2),
        "target": round(state["target"], 2),
        "drift": round(cur_eval - state["target"], 2),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "history": state["history"],
        "candidates": candidates,
        "trap": trap_info,
        "maintainer_pick_uci": maintainer_pick_uci,
        "maintainer_reason": maintainer_reason,
        "give_back_remaining": state["give_back_remaining"],
        "give_back_drop": state["give_back_drop"],
        "effective_target": round(effective_target, 2)
            if board.turn == user_color and not board.is_game_over()
            else round(state["target"], 2),
        "last_trap_event": state["last_trap_event"],
        "ply": board.ply(),
        "move_number": board.fullmove_number,
    }


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(serialize_state())


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json() or {}
    uci = data.get("uci", "")
    with state_lock:
        board = state["board"]
        if board.is_game_over():
            return jsonify({"error": "game over"}), 400
        if board.turn != state["user_color"]:
            return jsonify({"error": "not your turn"}), 400
        try:
            move = chess.Move.from_uci(uci)
        except Exception as e:
            return jsonify({"error": f"bad uci: {e}"}), 400
        # Handle pawn-promotion default to queen if not specified
        if move not in board.legal_moves and len(uci) == 4:
            promo = chess.Move.from_uci(uci + "q")
            if promo in board.legal_moves:
                move = promo
        if move not in board.legal_moves:
            return jsonify({"error": "illegal move"}), 400
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        # Snapshot the live target the user was holding at decision time so
        # /api/compare and /api/fork can reconstruct what the user was aiming
        # at on each ply (target updates mid-game after trap resolutions).
        target_before = state["target"]
        # Check whether THIS move qualifies as a trap, BEFORE we push it.
        # Prefer the cached trap_info that serialize_state populated when
        # rendering this position -- SF with Threads>1 can give different
        # trap qualifications on a second call, which would silently break
        # the auto-target-update on a trap that the UI told the user about.
        fen_before = board.fen()
        pending_trap = None
        if fen_before in state["trap_cache"]:
            tinfo = state["trap_cache"][fen_before]
        else:
            try:
                tinfo = find_puzzle_trap(state["sf"], state["maia_oracle"],
                                          board, depth=state["depth"])
            except Exception:
                tinfo = None
            state["trap_cache"][fen_before] = tinfo
        if tinfo:
            for q in tinfo["all_qualifying"]:
                if q["uci"] == move.uci():
                    pending_trap = q
                    break
        board.push(move)
        sign = 1.0 if state["user_color"] == chess.WHITE else -1.0
        post = eval_white(board) * sign
        state["history"].append({
            "ply": board.ply(), "san": san,
            "eval": round(post, 2), "by": "user",
            "uci": move.uci(),
            "target_before": round(target_before, 2),
        })
        # Maia replies if game still on
        if not board.is_game_over():
            maia_plays()
            # If the user move was a trap, resolve it now (Maia just replied).
            # On MISS: target updates immediately to post-resolution eval.
            # On FOUND: enter give-back (defer target update by 5 user moves;
            # during give-back the effective target is lowered to bleed eval).
            if pending_trap is not None:
                reply = state["history"][-1]
                actual_uci = reply.get("uci")
                found = (actual_uci == pending_trap.get("sf_top_reply_uci"))
                post_eval = eval_white(board) * sign
                post_eval = max(-10.0, min(10.0, post_eval))
                state["event_seq"] += 1
                if found:
                    # Give-back: keep target where it was, drop the effective
                    # target by give_back_drop for the next 5 user moves.
                    state["give_back_remaining"] = 5
                    new_target_for_event = round(state["target"], 2)
                else:
                    state["target"] = post_eval
                    new_target_for_event = round(post_eval, 2)
                state["last_trap_event"] = {
                    "id": state["event_seq"],
                    "ply": board.ply(),
                    "found": found,
                    "trap_san": pending_trap["san"],
                    "expected_reply_san": pending_trap["sf_top_reply_san"],
                    "actual_reply_san": reply.get("san"),
                    "old_target": round(target_before, 2),
                    "new_target": new_target_for_event,
                    "eval_cost": pending_trap["eval_cost"],
                    "gap": pending_trap["gap"],
                    "p_maia": pending_trap["p_maia_top"],
                    "give_back_started": found,
                    "give_back_drop": state["give_back_drop"],
                }
            elif state["give_back_remaining"] > 0:
                # Mid give-back, no new trap fired -- decrement the counter.
                # When it reaches 0, lock in the new target = current eval.
                state["give_back_remaining"] -= 1
                if state["give_back_remaining"] == 0:
                    post_eval = eval_white(board) * sign
                    post_eval = max(-10.0, min(10.0, post_eval))
                    state["target"] = post_eval
                    state["event_seq"] += 1
                    state["last_trap_event"] = {
                        "id": state["event_seq"],
                        "ply": board.ply(),
                        "give_back_ended": True,
                        "new_target": round(post_eval, 2),
                    }
        return jsonify(serialize_state())


@app.route("/api/target", methods=["POST"])
def api_target():
    data = request.get_json() or {}
    try:
        t = float(data.get("target", 0.0))
    except Exception:
        return jsonify({"error": "bad target"}), 400
    with state_lock:
        state["target"] = max(-10.0, min(10.0, t))
        return jsonify(serialize_state())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json() or {}
    color = data.get("color", "w")
    with state_lock:
        state["board"] = chess.Board()
        state["history"] = []
        state["target"] = 0.0
        state["last_trap_event"] = None
        state["trap_cache"] = {}
        state["give_back_remaining"] = 0
        state["user_color"] = (chess.WHITE if color == "w" else chess.BLACK)
        if state["user_color"] == chess.BLACK:
            maia_plays()
        return jsonify(serialize_state())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    """Undo back to the user's previous turn (pop user move + maia reply)."""
    with state_lock:
        board = state["board"]
        # If it's user's turn now, the last two moves were maia-then-user-now.
        # Pop until it's user's turn AGAIN (so user can re-pick their move).
        popped = 0
        while board.move_stack and popped < 4:
            board.pop()
            if state["history"]:
                state["history"].pop()
            popped += 1
            if board.turn == state["user_color"] and popped >= 2:
                break
        return jsonify(serialize_state())


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Scan the current game's history; return every position where the bot
    would have picked differently from the user. Returns a snapshot_id that
    can be used to open /fork/<snap>/<ply> URLs in a new tab."""
    from fork import compare_history
    with state_lock:
        entries = [h for h in state["history"] if h.get("uci")]
        history_uci = [h["uci"] for h in entries]
        if not history_uci:
            return jsonify({"error": "no game to compare"}), 400
        # Per-ply target the user was holding at the time of each move.
        # entries[i]["target_before"] is the target the user was aiming at
        # when they decided their move at that ply. Falls back to 0.0 for
        # legacy entries written before per-ply tracking.
        targets_by_ply = [h.get("target_before", 0.0) for h in entries]
        snap_id = uuid.uuid4().hex[:12]
        snapshots[snap_id] = {
            "history": history_uci,
            "targets_by_ply": targets_by_ply,
            "user_color": "w" if state["user_color"] == chess.WHITE else "b",
            "maia_seed": state["maia_seed"],
            "depth": state["depth"],
            "target": state["target"],
        }
        try:
            divergences = compare_history(
                history_uci, state["user_color"],
                state["sf"], state["maia_oracle"], state["depth"],
                targets_by_ply=targets_by_ply,
            )
        except Exception as e:
            return jsonify({"error": f"compare failed: {e}"}), 500
        return jsonify({
            "snapshot_id": snap_id,
            "divergences": divergences,
            "total_user_turns": sum(
                1 for i in range(len(history_uci))
                if (i % 2 == 0) == (state["user_color"] == chess.WHITE)
            ),
        })


@app.route("/api/fork/<snap_id>/<int:ply>")
def api_fork(snap_id, ply):
    """Return the bot's continuation from ply N as JSON.
    Used by the /fork page to render."""
    from fork import play_fork
    snap = snapshots.get(snap_id)
    if snap is None:
        return jsonify({"error": "snapshot not found"}), 404
    user_color = chess.WHITE if snap["user_color"] == "w" else chess.BLACK
    # The bot takes over at fork_ply -- inherit the target the user was
    # actually holding at that moment in the live game.
    targets = snap.get("targets_by_ply") or []
    target_at_fork = (targets[ply] if 0 <= ply < len(targets)
                      else snap.get("target", 0.0))
    try:
        result = play_fork(
            snap["history"], ply, user_color,
            snap["maia_seed"], snap["depth"],
            target_eval=target_at_fork,
        )
        result["snapshot_id"] = snap_id
        result["fork_ply"] = ply
        result["user_color"] = snap["user_color"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"fork failed: {e}"}), 500


@app.route("/fork/<snap_id>/<int:ply>")
def fork_page(snap_id, ply):
    """Render a fork as a static read-only game in a new tab."""
    return render_template_string(FORK_HTML_PAGE,
                                   snap_id=snap_id, ply=ply)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stonefish vs Maia 1900</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 16px;
    background: #1e1e1e; color: #ddd;
    display: flex; gap: 24px;
  }
  #board-area { width: 480px; }
  #board { width: 480px; }
  #board-area h2 { margin: 8px 0; font-size: 16px; font-weight: 500; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
  .row label { color: #888; font-size: 13px; min-width: 60px; }
  .row input, .row select, .row button {
    background: #2a2a2a; color: #ddd; border: 1px solid #444;
    padding: 6px 10px; border-radius: 4px; font-size: 13px;
  }
  .row button { cursor: pointer; }
  .row button:hover { background: #333; }
  .stats { background: #2a2a2a; padding: 12px; border-radius: 6px; margin: 8px 0; }
  .stats .big { font-size: 20px; font-weight: bold; }
  .stats .row { margin: 4px 0; gap: 8px; }
  .stats .row label { min-width: 80px; }
  .positive { color: #6bd968; }
  .negative { color: #e36a6a; }
  .neutral { color: #ddd; }
  #side { flex: 1; min-width: 380px; max-width: 500px; }
  #candidates { max-height: 500px; overflow-y: auto; }
  .cand {
    display: flex; padding: 6px 8px; gap: 12px; align-items: center;
    border-bottom: 1px solid #2a2a2a; cursor: pointer; font-size: 13px;
  }
  .cand:hover { background: #2a2a2a; }
  .cand .idx { color: #666; width: 22px; }
  .cand .san { font-weight: 500; min-width: 60px; }
  .cand .eval { min-width: 60px; text-align: right; font-family: monospace; }
  .cand .drift { min-width: 60px; text-align: right; font-family: monospace; }
  .cand.maintainer-pick { background: #2c4a2c; }
  .cand.maintainer-pick:hover { background: #355c35; }
  .cand.trap-pick { background: #5a3a1a; border-left: 4px solid #ffaa00; }
  .cand.trap-pick:hover { background: #6b4523; }
  .cand.trap-pick.maintainer-pick { background: #4a4520; }
  .trap-panel {
    background: #3a2c14; border: 1px solid #6b4523; padding: 12px;
    border-radius: 6px; margin: 12px 0; font-size: 13px;
  }
  .trap-panel .label { color: #ffaa00; font-weight: bold; }
  .trap-panel .row { margin: 4px 0; }
  .trap-panel .small { font-size: 11px; color: #aaa; }
  .notification {
    position: fixed; top: 16px; right: 16px;
    background: #2a2a2a; border-left: 4px solid #ffaa00;
    padding: 12px 16px; border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    max-width: 360px; font-size: 13px;
    transition: opacity 0.3s ease;
    z-index: 100;
  }
  .notification.found { border-left-color: #e36a6a; }
  .notification.missed { border-left-color: #6bd968; }
  .notification .title { font-weight: bold; margin-bottom: 4px; }
  .notification .body { color: #ccc; line-height: 1.4; }
  .notification.hiding { opacity: 0; }
  #history {
    max-height: 200px; overflow-y: auto; padding: 8px;
    background: #2a2a2a; border-radius: 6px; font-family: monospace;
    font-size: 12px; line-height: 1.5;
  }
  .hist-user { color: #6bd968; }
  .hist-maia { color: #6bb5e3; }
  .hist-eval { color: #888; margin-left: 8px; }
  #status { color: #888; font-size: 12px; margin-top: 8px; }
  .gameover { color: #ffaa00; font-weight: bold; margin: 12px 0; }
</style>
</head>
<body>

<div id="board-area">
  <h2>Stonefish vs Maia 1900</h2>
  <div id="board"></div>
  <div class="row">
    <label>Color:</label>
    <select id="color-select">
      <option value="w">White</option>
      <option value="b">Black</option>
    </select>
    <button id="reset-btn">New game</button>
    <button id="undo-btn">Undo</button>
    <button id="compare-btn">Compare to bot</button>
  </div>
  <div id="status">Ready.</div>
</div>

<div id="side">
  <div class="stats">
    <div class="row">
      <label>Eval (you):</label>
      <span id="eval" class="big neutral">+0.00p</span>
    </div>
    <div class="row">
      <label>Target:</label>
      <input id="target-input" type="number" step="0.1" value="0.0" style="width: 70px;">
      <span style="color:#666; font-size:12px;">(your POV, pawns)</span>
    </div>
    <div class="row">
      <label>Drift:</label>
      <span id="drift" class="neutral">+0.00p</span>
    </div>
    <div class="row" id="gameover-row" style="display:none;">
      <span id="gameover" class="gameover"></span>
    </div>
  </div>

  <div id="give-back-panel" class="trap-panel" style="display:none;
       background: #3a2c14; border-color: #6b4523;">
    <div class="row"><span class="label" style="color:#e36a6a;">
      GIVE-BACK ACTIVE</span></div>
    <div id="give-back-detail" class="row"></div>
    <div class="small">
      Maia found your last trap. The maintainer is targeting a lower
      eval for the next few moves so you bleed advantage; after
      give-back ends the target locks in at whatever eval you've
      settled at.
    </div>
  </div>

  <div id="trap-panel" class="trap-panel" style="display:none;">
    <div class="row"><span class="label">PUZZLE MOMENT AVAILABLE</span></div>
    <div id="trap-detail" class="row"></div>
    <div class="row"><button id="play-trap-btn">Play the trap</button></div>
    <div class="small">
      The trap qualifies under Stonefish's filter (eval_cost
      &ge; 0.2p, gap &ge; 0.5p, gap/cost &ge; 1.5, P_maia &isin; [0.2, 0.8]).
      Sacrifices a little eval for a position only a precise reply holds.
    </div>
  </div>

  <h3 style="margin: 16px 0 8px; font-size: 14px; color: #aaa;">
    SF top-25 candidates (click to play)
    <span style="font-size:11px; color:#666;">
      — green = maintainer pick (closest from below target)
      — orange = trap move
    </span>
  </h3>
  <div id="candidates"></div>

  <h3 style="margin: 16px 0 8px; font-size: 14px; color: #aaa;">Move history</h3>
  <div id="history"></div>

  <div id="compare-panel" style="display:none;">
    <h3 style="margin: 16px 0 8px; font-size: 14px; color: #aaa;">
      Divergences (your picks vs bot's picks)
    </h3>
    <div id="divergences" style="background: #2a2a2a; padding: 10px;
         border-radius: 6px; font-size: 13px; max-height: 320px;
         overflow-y: auto;"></div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
let board = null;
let cur_state = null;
let lastSeenTrapEventId = 0;

const evalEl = document.getElementById('eval');
const driftEl = document.getElementById('drift');
const targetInput = document.getElementById('target-input');
const candidatesEl = document.getElementById('candidates');
const historyEl = document.getElementById('history');
const statusEl = document.getElementById('status');
const colorSelect = document.getElementById('color-select');
const gameoverRow = document.getElementById('gameover-row');
const gameoverEl = document.getElementById('gameover');

function fmtEval(v) {
  if (v >= 99) return 'M+';
  if (v <= -99) return 'M-';
  return (v >= 0 ? '+' : '') + v.toFixed(2) + 'p';
}

function evalClass(v) {
  if (v > 0.3) return 'positive';
  if (v < -0.3) return 'negative';
  return 'neutral';
}

function showNotification(html, kind) {
  const n = document.createElement('div');
  n.className = 'notification ' + (kind || '');
  n.innerHTML = html;
  document.body.appendChild(n);
  // Auto-dismiss after 8s
  setTimeout(() => {
    n.classList.add('hiding');
    setTimeout(() => n.remove(), 400);
  }, 8000);
  // Click to dismiss
  n.onclick = () => {
    n.classList.add('hiding');
    setTimeout(() => n.remove(), 400);
  };
}

function checkTrapEvent(state) {
  const ev = state.last_trap_event;
  if (!ev) return;
  if (ev.id <= lastSeenTrapEventId) return;
  lastSeenTrapEventId = ev.id;
  // give-back end notification (no trap details, just confirms new target)
  if (ev.give_back_ended) {
    showNotification(
      '<div class="title">Give-back complete</div>' +
      '<div class="body">Target locks in at <b>' + fmtEval(ev.new_target) +
      '</b>. Resume normal maintenance.</div>',
      'missed'
    );
    return;
  }
  const title = ev.found
    ? "Maia found the trap"
    : "Maia missed the trap";
  const detail = ev.found
    ? "She played " + ev.actual_reply_san + " (the precise reply)."
    : "Expected " + (ev.expected_reply_san || "?") +
      "; she played " + ev.actual_reply_san + ".";
  let tgt;
  if (ev.give_back_started) {
    tgt = "Give-back active: aim ~" + ev.give_back_drop.toFixed(1) +
          "p below target for 5 moves.";
  } else {
    tgt = "Target updated: " + fmtEval(ev.old_target) + " &rarr; <b>" +
          fmtEval(ev.new_target) + "</b>";
  }
  showNotification(
    '<div class="title">' + title + '</div>' +
    '<div class="body">' + ev.trap_san + ' &middot; ' + detail + '<br>' +
    tgt + '</div>',
    ev.found ? 'found' : 'missed'
  );
}

function render() {
  if (!cur_state) return;
  // Fire trap notification first (so the target update animates right after)
  checkTrapEvent(cur_state);
  // Board
  board.position(cur_state.fen, false);
  if (cur_state.user_color === 'b') {
    board.orientation('black');
  } else {
    board.orientation('white');
  }

  // Eval and drift
  evalEl.textContent = fmtEval(cur_state.eval);
  evalEl.className = 'big ' + evalClass(cur_state.eval);
  driftEl.textContent = fmtEval(cur_state.drift);
  driftEl.className = evalClass(cur_state.drift);
  targetInput.value = cur_state.target;

  // Game over
  if (cur_state.is_game_over) {
    gameoverRow.style.display = '';
    gameoverEl.textContent = 'Game over: ' + cur_state.result;
  } else {
    gameoverRow.style.display = 'none';
  }

  // Give-back panel
  const gbPanel = document.getElementById('give-back-panel');
  if (cur_state.give_back_remaining > 0) {
    gbPanel.style.display = '';
    document.getElementById('give-back-detail').innerHTML =
      `<b>${cur_state.give_back_remaining}</b> of your moves left. ` +
      `Effective target lowered to <b>${fmtEval(cur_state.effective_target)}</b> ` +
      `(was ${fmtEval(cur_state.target)}).`;
  } else {
    gbPanel.style.display = 'none';
  }

  // Trap panel
  const trapPanel = document.getElementById('trap-panel');
  const trapDetail = document.getElementById('trap-detail');
  let trapUci = null;
  if (cur_state.trap && cur_state.trap.trap) {
    const t = cur_state.trap.trap;
    trapUci = t.uci;
    trapPanel.style.display = '';
    trapDetail.innerHTML =
      '<b>' + t.san + '</b> instead of <b>' + t.sf_top_san + '</b>' +
      ' (SF top). Sacrifice ' + t.eval_cost.toFixed(2) + 'p.' +
      ' If opp finds <b>' + (t.sf_top_reply_san || '?') + '</b>, they hold' +
      ' (gap ' + t.gap.toFixed(2) + 'p, P_maia=' + (t.p_maia_top*100).toFixed(0) + '%).';
    document.getElementById('play-trap-btn').onclick = () => playMove(trapUci);
  } else {
    trapPanel.style.display = 'none';
  }

  // Candidates
  candidatesEl.innerHTML = '';
  if (cur_state.candidates.length === 0) {
    candidatesEl.innerHTML = '<div style="padding:8px;color:#666;">(opponent to move)</div>';
  } else {
    // Maintainer pick comes from the server (apply_maintainer_rules)
    const pickUci = cur_state.maintainer_pick_uci;
    const reason = cur_state.maintainer_reason;
    const reasonLabel = {
      'few-moves': 'forced (<3 legal moves)',
      'mate': 'mate available',
      'below-target': 'below target',
      'above-target-no-below': 'no below-target option',
      'empty': '(no candidates)',
    }[reason] || reason;
    cur_state.candidates.forEach((c, i) => {
      const isTrap = (c.uci === trapUci);
      const isMaintainer = (c.uci === pickUci);
      let cls = 'cand';
      if (isMaintainer) cls += ' maintainer-pick';
      if (isTrap) cls += ' trap-pick';
      const div = document.createElement('div');
      div.className = cls;
      const reasonBadge = isMaintainer && reasonLabel
        ? '<span style="color:#888;font-size:11px;font-style:italic;margin-left:8px;">— ' + reasonLabel + '</span>'
        : '';
      div.innerHTML =
        '<span class="idx">' + (i+1) + '</span>' +
        '<span class="san">' + c.san + '</span>' +
        '<span class="eval ' + evalClass(c.eval) + '">' + fmtEval(c.eval) + '</span>' +
        '<span class="drift ' + evalClass(c.drift) + '">' + fmtEval(c.drift) + '</span>' +
        reasonBadge;
      div.onclick = () => playMove(c.uci);
      candidatesEl.appendChild(div);
    });
  }

  // History
  historyEl.innerHTML = '';
  cur_state.history.forEach(h => {
    const div = document.createElement('div');
    div.className = h.by === 'user' ? 'hist-user' : 'hist-maia';
    const moveNum = Math.floor((h.ply - 1) / 2) + 1;
    const dot = (h.ply % 2 === 1) ? '.' : '...';
    div.innerHTML =
      moveNum + dot + ' <b>' + h.san + '</b>' +
      '<span class="hist-eval">(' + fmtEval(h.eval) + ')</span>';
    historyEl.appendChild(div);
  });
  historyEl.scrollTop = historyEl.scrollHeight;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function fetchState() {
  const r = await fetch('/api/state');
  cur_state = await r.json();
  render();
}

async function playMove(uci) {
  setStatus('Playing ' + uci + ' ... Maia is thinking...');
  const r = await fetch('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({uci: uci}),
  });
  if (r.ok) {
    cur_state = await r.json();
    render();
    setStatus('Your turn.');
  } else {
    const err = await r.json();
    setStatus('Error: ' + err.error);
    fetchState();
  }
}

function onDragStart(source, piece, position, orientation) {
  if (!cur_state || cur_state.is_game_over) return false;
  if (cur_state.turn !== cur_state.user_color) return false;
  if (orientation === 'white' && piece.search(/^b/) !== -1) return false;
  if (orientation === 'black' && piece.search(/^w/) !== -1) return false;
}

function onDrop(source, target) {
  if (source === target) return 'snapback';
  // Build UCI; promotion defaults to queen if not specified
  let uci = source + target;
  // Check if it's a promotion (pawn to rank 8 or 1)
  if (cur_state) {
    const moving = cur_state.fen.split(' ')[0];
    // Server handles promotion default
  }
  playMove(uci);
  return; // let the move happen; will refresh from server state
}

function onSnapEnd() {
  // After drop animation completes, sync position from server in case
  // promotion/castling/en-passant required adjustments
  if (cur_state) board.position(cur_state.fen, false);
}

board = Chessboard('board', {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
});

targetInput.addEventListener('change', async (e) => {
  const v = parseFloat(e.target.value);
  const r = await fetch('/api/target', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({target: v}),
  });
  cur_state = await r.json();
  render();
});

document.getElementById('reset-btn').addEventListener('click', async () => {
  const color = colorSelect.value;
  setStatus('Resetting...');
  const r = await fetch('/api/reset', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({color: color}),
  });
  cur_state = await r.json();
  render();
  setStatus('Ready.');
});

document.getElementById('undo-btn').addEventListener('click', async () => {
  setStatus('Undoing...');
  const r = await fetch('/api/undo', {method: 'POST'});
  cur_state = await r.json();
  render();
  setStatus('Ready.');
});

document.getElementById('compare-btn').addEventListener('click', async () => {
  const btn = document.getElementById('compare-btn');
  btn.disabled = true;
  setStatus('Comparing your picks to the bot (this can take 20-60s)...');
  try {
    const r = await fetch('/api/compare', {method: 'POST'});
    if (!r.ok) {
      const e = await r.json();
      setStatus('Compare failed: ' + (e.error || 'unknown'));
      return;
    }
    const data = await r.json();
    const panel = document.getElementById('compare-panel');
    const list = document.getElementById('divergences');
    panel.style.display = '';
    list.innerHTML = '';
    if (data.divergences.length === 0) {
      list.innerHTML = '<div style="color:#aaa;">No divergences ' +
                       '&mdash; you picked the same move as the bot every turn.</div>';
    } else {
      const header = document.createElement('div');
      header.style.cssText = 'color:#aaa; margin-bottom:8px;';
      header.textContent = `${data.divergences.length} divergence(s) out of ` +
                           `${data.total_user_turns} of your moves`;
      list.appendChild(header);
      data.divergences.forEach(d => {
        if (d.error) {
          const row = document.createElement('div');
          row.style.cssText = 'padding:4px 0; color:#e36a6a;';
          row.textContent = `Ply ${d.ply}: error ${d.error}`;
          list.appendChild(row);
          return;
        }
        const moveNum = Math.floor(d.ply / 2) + 1;
        const tag = (d.ply % 2 === 0) ? `${moveNum}.` : `${moveNum}...`;
        const row = document.createElement('div');
        row.style.cssText = 'padding:6px 0; border-bottom:1px solid #333;';
        const trapTag = d.was_trap
          ? '<span style="color:#ffaa00;font-size:11px;margin-left:6px;">[TRAP]</span>'
          : '';
        const targetTag = (d.target !== undefined)
          ? `<span style="color:#666;font-size:11px;margin-left:6px;">target ${d.target >= 0 ? '+' : ''}${d.target}p</span>`
          : '';
        row.innerHTML =
          `<b style="color:#888;">${tag}</b> ` +
          `You: <b>${d.user_san}</b> &middot; ` +
          `Bot: <b style="color:#6bd968;">${d.bot_san}</b> ` +
          `<span style="color:#888;font-size:11px;">(${d.mode})</span> ${trapTag}${targetTag}` +
          ` <a href="/fork/${data.snapshot_id}/${d.ply}" target="_blank" ` +
          `style="margin-left:8px; color:#6bb5e3;">view fork &rarr;</a>`;
        list.appendChild(row);
      });
    }
    setStatus('Compare done. ' + data.divergences.length + ' divergences.');
  } finally {
    btn.disabled = false;
  }
});

fetchState();
</script>

</body>
</html>
"""


FORK_HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fork: bot takes over at ply {{ ply }}</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 16px;
    background: #1e1e1e; color: #ddd;
    display: flex; gap: 24px;
  }
  #board { width: 480px; }
  #side { flex: 1; min-width: 380px; max-width: 600px; }
  h2 { margin: 0 0 8px; font-size: 18px; font-weight: 500; }
  .meta { color: #888; font-size: 13px; margin-bottom: 12px; }
  .result {
    font-size: 18px; font-weight: bold; padding: 10px 14px;
    border-radius: 6px; margin: 12px 0;
  }
  .result.stonefish { background: #2c4a2c; color: #6bd968; }
  .result.maia { background: #4a2c2c; color: #e36a6a; }
  .result.draw { background: #2a2a2a; color: #ddd; }
  .stats { background: #2a2a2a; padding: 12px; border-radius: 6px; margin: 12px 0;
           font-size: 13px; }
  .stats .row { display: flex; justify-content: space-between; padding: 3px 0; }
  .stats .row label { color: #888; }
  #moves {
    max-height: 540px; overflow-y: auto;
    padding: 8px; background: #2a2a2a; border-radius: 6px;
    font-family: monospace; font-size: 12px; line-height: 1.6;
  }
  .move-row { display: flex; gap: 8px; align-items: baseline; }
  .move-num { color: #666; min-width: 24px; }
  .move-stonefish { color: #6bd968; min-width: 80px; }
  .move-maia { color: #6bb5e3; min-width: 80px; }
  .move-original { color: #444; min-width: 80px; font-style: italic; }
  .move-eval { color: #888; font-size: 11px; margin-left: auto; }
  .divider {
    border-top: 1px dashed #555; margin: 6px 0; padding-top: 6px;
    color: #aaa; font-style: italic;
  }
</style>
</head>
<body>
<div>
  <h2>Bot's continuation from ply {{ ply }}</h2>
  <div id="board"></div>
  <div class="meta" id="meta">Loading...</div>
</div>
<div id="side">
  <div id="result"></div>
  <div id="stats" class="stats" style="display:none;"></div>
  <h3 style="margin: 12px 0 6px; font-size: 14px; color: #aaa;">Move list</h3>
  <div id="moves">Loading...</div>
</div>
<script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
const SNAP_ID = "{{ snap_id }}";
const FORK_PLY = {{ ply }};
let board = null;

async function load() {
  const r = await fetch(`/api/fork/${SNAP_ID}/${FORK_PLY}`);
  if (!r.ok) {
    const e = await r.json();
    document.getElementById('meta').textContent = 'Error: ' + (e.error || 'unknown');
    return;
  }
  const data = await r.json();
  document.getElementById('meta').innerHTML =
    `User color: <b>${data.user_color === 'w' ? 'White' : 'Black'}</b> &middot; ` +
    `starting eval at fork: <b>${data.starting_eval > 0 ? '+' : ''}${data.starting_eval}p</b>`;

  const resultEl = document.getElementById('result');
  const resName = data.fork_result;
  const resLabel = resName === 'stonefish' ? 'Stonefish wins' :
                   resName === 'maia' ? 'Maia wins' : 'Draw';
  resultEl.className = 'result ' + resName;
  resultEl.textContent = resLabel + ' (' + data.result_pgn + ')';

  const statsEl = document.getElementById('stats');
  statsEl.style.display = '';
  statsEl.innerHTML =
    `<div class="row"><label>Bot traps set:</label><b>${data.trap_count}</b></div>` +
    `<div class="row"><label>Maia found:</label><b>${data.find_count}/${data.trap_count}</b></div>` +
    `<div class="row"><label>Fork moves:</label><b>${data.fork_moves.length}</b></div>`;

  // Move list: original up to fork ply, then fork from there
  const movesEl = document.getElementById('moves');
  movesEl.innerHTML = '';
  // original moves (italic gray)
  for (let i = 0; i < data.original_san.length; i++) {
    const moveNum = Math.floor(i / 2) + 1;
    const tag = (i % 2 === 0) ? `${moveNum}.` : `${moveNum}...`;
    const div = document.createElement('div');
    div.className = 'move-row';
    div.innerHTML = `<span class="move-num">${tag}</span>` +
                    `<span class="move-original">${data.original_san[i]}</span>`;
    movesEl.appendChild(div);
  }
  // divider
  const div = document.createElement('div');
  div.className = 'divider';
  div.textContent = '— bot takes over —';
  movesEl.appendChild(div);
  // fork moves
  data.fork_moves.forEach(m => {
    const moveNum = Math.floor((m.ply - 1) / 2) + 1;
    const tag = (m.ply % 2 === 1) ? `${moveNum}.` : `${moveNum}...`;
    const row = document.createElement('div');
    row.className = 'move-row';
    const cls = m.by === 'stonefish' ? 'move-stonefish' : 'move-maia';
    const evalStr = (m.eval >= 0 ? '+' : '') + m.eval + 'p';
    row.innerHTML = `<span class="move-num">${tag}</span>` +
                    `<span class="${cls}">${m.san}</span>` +
                    `<span class="move-eval">${evalStr}</span>`;
    movesEl.appendChild(row);
  });
  // initial board = starting fen
  board = Chessboard('board', {
    position: data.starting_fen,
    draggable: false,
  });
  if (data.user_color === 'b') board.orientation('black');
}

load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    maia_seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    init_engines(depth=depth, maia_seed=maia_seed)
    if maia_seed is not None:
        print(f"  Maia is seeded ({maia_seed}) -- her moves are deterministic.")
    print(f"Starting Stonefish web UI on http://localhost:{port} (depth={depth})")
    if depth < 12:
        print(f"  note: depth={depth} can show inaccurate evals on tactical "
              f"positions (multipv search non-determinism). Bump to 14-16 for "
              f"more reliable evals -- slower per move, especially for 25 candidates.")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
