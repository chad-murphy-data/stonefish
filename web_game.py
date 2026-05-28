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

from engine import STOCKFISH_PATH, MAIA_WEIGHTS_PATH, MaiaBot, get_top_moves

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
}


def init_engines(depth=10):
    state["sf"] = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    state["sf"].configure({"Threads": 2, "Hash": 256})
    state["maia"] = MaiaBot(MAIA_WEIGHTS_PATH,
                            rating=1900, temperature=1.0, seed=None)
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
    })


def serialize_state():
    board = state["board"]
    user_color = state["user_color"]
    sign = 1.0 if user_color == chess.WHITE else -1.0
    cur_eval = eval_white(board) * sign

    candidates = []
    if (board.turn == user_color
            and not board.is_game_over()):
        raw = get_top_moves(state["sf"], board, num_moves=25,
                            depth=state["depth"])
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
                "drift": round(eval_for_us - state["target"], 2),
            })

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
        board.push(move)
        sign = 1.0 if state["user_color"] == chess.WHITE else -1.0
        post = eval_white(board) * sign
        state["history"].append({
            "ply": board.ply(), "san": san,
            "eval": round(post, 2), "by": "user",
            "uci": move.uci(),
        })
        # Maia replies if game still on
        if not board.is_game_over():
            maia_plays()
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

  <h3 style="margin: 16px 0 8px; font-size: 14px; color: #aaa;">
    SF top-25 candidates (click to play)
    <span style="font-size:11px; color:#666;">— green = maintainer pick</span>
  </h3>
  <div id="candidates"></div>

  <h3 style="margin: 16px 0 8px; font-size: 14px; color: #aaa;">Move history</h3>
  <div id="history"></div>
</div>

<script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
let board = null;
let cur_state = null;

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

function render() {
  if (!cur_state) return;
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

  // Candidates
  candidatesEl.innerHTML = '';
  if (cur_state.candidates.length === 0) {
    candidatesEl.innerHTML = '<div style="padding:8px;color:#666;">(opponent to move)</div>';
  } else {
    // Identify maintainer's pick (smallest |drift|)
    let bestIdx = 0;
    let bestDrift = Math.abs(cur_state.candidates[0].drift);
    cur_state.candidates.forEach((c, i) => {
      if (Math.abs(c.drift) < bestDrift) {
        bestDrift = Math.abs(c.drift);
        bestIdx = i;
      }
    });
    cur_state.candidates.forEach((c, i) => {
      const div = document.createElement('div');
      div.className = 'cand' + (i === bestIdx ? ' maintainer-pick' : '');
      div.innerHTML =
        '<span class="idx">' + (i+1) + '</span>' +
        '<span class="san">' + c.san + '</span>' +
        '<span class="eval ' + evalClass(c.eval) + '">' + fmtEval(c.eval) + '</span>' +
        '<span class="drift ' + evalClass(c.drift) + '">' + fmtEval(c.drift) + '</span>';
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

fetchState();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    init_engines(depth=depth)
    print(f"Starting Stonefish web UI on http://localhost:{port} (depth={depth})")
    if depth < 12:
        print(f"  note: depth={depth} can show inaccurate evals on tactical "
              f"positions (multipv search non-determinism). Bump to 14-16 for "
              f"more reliable evals -- slower per move, especially for 25 candidates.")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
