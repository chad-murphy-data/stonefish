"""
Live Maia Tier Disagreement Viewer
===================================
Runs the same analysis as analyze_disagreements.py but with a live
browser-based chessboard that updates move-by-move, showing tier
predictions and highlighting disagreements in real time.

Run:  python watch_analysis.py [num_games_per_preset]
"""

import chess
import chess.engine
import json
import math
import queue
import sys
import threading
import time
import webbrowser
from flask import Flask, Response, render_template_string

from engine import STOCKFISH_PATH, get_top_moves
from stonefish.maia import MaiaEngine
from stonefish.presets import ELO_PRESETS
from stonefish.scoring import net_material_difference

# ---------------------------------------------------------------------------
# Flask app + SSE
# ---------------------------------------------------------------------------

app = Flask(__name__)
event_queue = queue.Queue()

MOVE_DELAY = 0.4  # seconds between moves so you can watch


def sse_stream():
    """Generator that yields SSE events from the queue."""
    while True:
        try:
            data = event_queue.get(timeout=30)
        except queue.Empty:
            # Send keepalive
            yield ":\n\n"
            continue
        if data is None:
            break
        yield f"data: {json.dumps(data)}\n\n"


@app.route("/stream")
def stream():
    return Response(sse_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ---------------------------------------------------------------------------
# Analysis thread (mirrors analyze_disagreements.py logic)
# ---------------------------------------------------------------------------

def emit(event_type, **kwargs):
    """Push an event to the browser."""
    kwargs["type"] = event_type
    event_queue.put(kwargs)


def run_analysis(num_games):
    """Run full disagreement analysis, emitting events for the viewer."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 128})
    maia = MaiaEngine(stockfish_engine=engine)

    all_stats = {}

    for elo in sorted(ELO_PRESETS.keys()):
        preset = ELO_PRESETS[elo]
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
        }

        emit("preset_start", elo=elo,
             floor=floor_r,
             stretch=stretch_r or "SF",
             reach=reach_r or "SF")

        for game_num in range(num_games):
            board = chess.Board()
            move_count = 0

            emit("game_start", elo=elo, game=game_num + 1, total_games=num_games)

            for _ in range(200):
                if board.is_game_over():
                    break

                result = engine.analyse(board, chess.engine.Limit(depth=8))
                if "pv" not in result:
                    break
                move = result["pv"][0]

                # Tier analysis (skip opening)
                tier_data = None
                if move_count >= 6:
                    tiers = maia.predict_three_tier(board, floor_r, stretch_r, reach_r)

                    floor_move = tiers.floor.top_move
                    stretch_move = tiers.stretch.top_move
                    reach_move = tiers.reach.top_move

                    stats["positions"] += 1

                    fvs = floor_move != stretch_move
                    fvr = floor_move != reach_move
                    svr = stretch_move != reach_move

                    has_disagreement = fvs or fvr or svr

                    if fvs:
                        stats["floor_v_stretch"] += 1
                    if fvr:
                        stats["floor_v_reach"] += 1
                    if svr:
                        stats["stretch_v_reach"] += 1

                    if has_disagreement:
                        stats["any_disagreement"] += 1
                        if len({floor_move, stretch_move, reach_move}) == 3:
                            stats["all_three_differ"] += 1
                        if fvs and abs(net_material_difference(board, floor_move, stretch_move, engine)) >= 0.3:
                            stats["material_relevant"] += 1
                        elif fvr and abs(net_material_difference(board, floor_move, reach_move, engine)) >= 0.3:
                            stats["material_relevant"] += 1

                    tier_data = {
                        "floor_move": floor_move.uci(),
                        "stretch_move": stretch_move.uci(),
                        "reach_move": reach_move.uci(),
                        "disagree": has_disagreement,
                        "fvs": fvs,
                        "fvr": fvr,
                        "svr": svr,
                    }

                # Play the move
                san = board.san(move)
                board.push(move)

                emit("move", fen=board.fen(), move=move.uci(), san=san,
                     move_num=move_count + 1, tiers=tier_data,
                     stats={
                         "positions": stats["positions"],
                         "disagreements": stats["any_disagreement"],
                         "rate": (stats["any_disagreement"] / stats["positions"] * 100
                                  if stats["positions"] > 0 else 0),
                     })

                move_count += 1
                time.sleep(MOVE_DELAY)

            # Game over
            outcome = board.outcome()
            result_str = outcome.result() if outcome else "*"
            emit("game_end", elo=elo, game=game_num + 1, result=result_str,
                 moves=move_count)

        all_stats[elo] = stats

    engine.quit()

    # Send final summary
    summary = []
    for elo in sorted(all_stats.keys()):
        s = all_stats[elo]
        p = ELO_PRESETS[elo]
        n = s["positions"]
        if n == 0:
            continue
        summary.append({
            "elo": elo,
            "tiers": f"{p['floor_rating']}/{p['stretch_rating'] or 'SF'}/{p['reach_rating'] or 'SF'}",
            "positions": n,
            "disagree_pct": round(s["any_disagreement"] / n * 100, 1),
            "fvs_pct": round(s["floor_v_stretch"] / n * 100, 1),
            "fvr_pct": round(s["floor_v_reach"] / n * 100, 1),
            "svr_pct": round(s["stretch_v_reach"] / n * 100, 1),
        })

    emit("analysis_complete", summary=summary)


# ---------------------------------------------------------------------------
# HTML template (inline)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stonefish - Live Disagreement Analysis</title>
<link rel="stylesheet"
      href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    justify-content: center;
    padding: 24px;
    min-height: 100vh;
  }
  .container {
    display: flex;
    gap: 28px;
    max-width: 1100px;
    width: 100%;
  }
  .board-col { flex: 0 0 480px; }
  .info-col { flex: 1; min-width: 320px; }
  #board { width: 480px; }
  h1 {
    font-size: 1.5rem;
    margin-bottom: 16px;
    color: #e94560;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .panel {
    background: #16213e;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
    border: 1px solid #0f3460;
  }
  .panel h2 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 10px;
  }
  .preset-badge {
    display: inline-block;
    background: #e94560;
    color: #fff;
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 700;
    font-size: 1.1rem;
  }
  .game-info { color: #aaa; font-size: 0.9rem; margin-top: 6px; }
  .tier-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #0f3460;
    font-size: 0.95rem;
  }
  .tier-row:last-child { border-bottom: none; }
  .tier-label {
    width: 70px;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
  }
  .tier-label.floor { color: #ff6b6b; }
  .tier-label.stretch { color: #ffd93d; }
  .tier-label.reach { color: #6bcb77; }
  .tier-move {
    font-family: 'Courier New', monospace;
    font-size: 1rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: #1a1a2e;
  }
  .tier-move.disagree {
    background: #e94560;
    color: #fff;
    font-weight: 700;
  }
  .tier-move.agree {
    background: #1b4332;
    color: #6bcb77;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-item {
    text-align: center;
    padding: 8px;
    background: #1a1a2e;
    border-radius: 6px;
  }
  .stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e94560;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    margin-top: 2px;
  }
  .move-log {
    max-height: 160px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #aaa;
    padding: 8px;
    background: #1a1a2e;
    border-radius: 6px;
  }
  .move-log .disagree-move { color: #e94560; font-weight: 700; }
  .summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  .summary-table th {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 2px solid #e94560;
    color: #e94560;
  }
  .summary-table td {
    padding: 5px 8px;
    border-bottom: 1px solid #0f3460;
  }
  #status {
    font-size: 0.85rem;
    color: #888;
    margin-top: 8px;
  }
  .highlight-sq {
    box-shadow: inset 0 0 0 3px #e94560;
  }
</style>
</head>
<body>
<div class="container">
  <div class="board-col">
    <h1>Stonefish Live Analysis</h1>
    <div id="board"></div>
    <div id="status">Connecting...</div>
  </div>
  <div class="info-col">
    <div class="panel" id="preset-panel">
      <h2>Current Preset</h2>
      <span class="preset-badge" id="elo-badge">--</span>
      <div class="game-info" id="game-info">Waiting...</div>
      <div class="game-info" id="tier-info"></div>
    </div>

    <div class="panel" id="tiers-panel">
      <h2>Tier Predictions</h2>
      <div class="tier-row">
        <span class="tier-label floor">Floor</span>
        <span class="tier-move" id="floor-move">--</span>
      </div>
      <div class="tier-row">
        <span class="tier-label stretch">Stretch</span>
        <span class="tier-move" id="stretch-move">--</span>
      </div>
      <div class="tier-row">
        <span class="tier-label reach">Reach</span>
        <span class="tier-move" id="reach-move">--</span>
      </div>
    </div>

    <div class="panel">
      <h2>Running Stats</h2>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="stat-value" id="stat-positions">0</div>
          <div class="stat-label">Positions</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-disagreements">0</div>
          <div class="stat-label">Disagreements</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-rate">0%</div>
          <div class="stat-label">Disagree Rate</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-move">0</div>
          <div class="stat-label">Move #</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Move Log</h2>
      <div class="move-log" id="move-log"></div>
    </div>

    <div class="panel" id="summary-panel" style="display:none">
      <h2>Final Summary</h2>
      <table class="summary-table" id="summary-table">
        <thead>
          <tr><th>ELO</th><th>Tiers</th><th>Positions</th><th>Disagree%</th><th>FvS%</th><th>FvR%</th><th>SvR%</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
var board = Chessboard('board', {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  appearSpeed: 150,
  moveSpeed: 150,
});

var moveLog = document.getElementById('move-log');

var source = new EventSource('/stream');
source.onopen = function() {
  document.getElementById('status').textContent = 'Connected — analysis running...';
};

source.onmessage = function(e) {
  var d = JSON.parse(e.data);

  if (d.type === 'preset_start') {
    document.getElementById('elo-badge').textContent = 'ELO ' + d.elo;
    document.getElementById('tier-info').textContent =
      'Floor=' + d.floor + '  Stretch=' + d.stretch + '  Reach=' + d.reach;
    moveLog.innerHTML = '';
    // Reset tier display
    ['floor-move','stretch-move','reach-move'].forEach(function(id) {
      document.getElementById(id).textContent = '--';
      document.getElementById(id).className = 'tier-move';
    });
  }

  if (d.type === 'game_start') {
    document.getElementById('game-info').textContent =
      'Game ' + d.game + ' / ' + d.total_games;
    board.position('start');
    moveLog.innerHTML = '';
  }

  if (d.type === 'move') {
    board.position(d.fen, true);

    document.getElementById('stat-move').textContent = d.move_num;

    // Stats
    if (d.stats) {
      document.getElementById('stat-positions').textContent = d.stats.positions;
      document.getElementById('stat-disagreements').textContent = d.stats.disagreements;
      document.getElementById('stat-rate').textContent = d.stats.rate.toFixed(1) + '%';
    }

    // Tiers
    if (d.tiers) {
      setTier('floor-move', d.tiers.floor_move, d.tiers.disagree && (d.tiers.fvs || d.tiers.fvr));
      setTier('stretch-move', d.tiers.stretch_move, d.tiers.disagree && (d.tiers.fvs || d.tiers.svr));
      setTier('reach-move', d.tiers.reach_move, d.tiers.disagree && (d.tiers.fvr || d.tiers.svr));
    } else {
      ['floor-move','stretch-move','reach-move'].forEach(function(id) {
        document.getElementById(id).textContent = '--';
        document.getElementById(id).className = 'tier-move';
      });
    }

    // Move log
    var cls = (d.tiers && d.tiers.disagree) ? 'disagree-move' : '';
    var moveText = d.san;
    if (d.tiers && d.tiers.disagree) {
      moveText += ' *';
    }
    var span = '<span class="' + cls + '">' + d.move_num + '. ' + moveText + '</span>  ';
    moveLog.innerHTML += span;
    moveLog.scrollTop = moveLog.scrollHeight;
  }

  if (d.type === 'game_end') {
    document.getElementById('status').textContent =
      'Game ' + d.game + ' finished (' + d.result + ', ' + d.moves + ' moves)';
  }

  if (d.type === 'analysis_complete') {
    document.getElementById('status').textContent = 'Analysis complete!';
    document.getElementById('summary-panel').style.display = 'block';
    var tbody = document.querySelector('#summary-table tbody');
    tbody.innerHTML = '';
    d.summary.forEach(function(row) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td>' + row.elo + '</td><td>' + row.tiers + '</td>' +
        '<td>' + row.positions + '</td><td>' + row.disagree_pct + '%</td>' +
        '<td>' + row.fvs_pct + '%</td><td>' + row.fvr_pct + '%</td>' +
        '<td>' + row.svr_pct + '%</td>';
      tbody.appendChild(tr);
    });
    source.close();
  }
};

source.onerror = function() {
  document.getElementById('status').textContent = 'Connection lost. Refresh to reconnect.';
};

function setTier(elemId, moveUci, isDisagreeing) {
  var el = document.getElementById(elemId);
  el.textContent = moveUci;
  el.className = 'tier-move ' + (isDisagreeing ? 'disagree' : 'agree');
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    port = 5000

    print("=" * 50)
    print("Stonefish Live Disagreement Viewer")
    print("=" * 50)
    print(f"Games per preset: {num_games}")
    print(f"Opening browser at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    # Start analysis in background thread
    t = threading.Thread(target=run_analysis, args=(num_games,), daemon=True)
    t.start()

    # Open browser after a short delay
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    # Run Flask (threaded=True so SSE doesn't block)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
