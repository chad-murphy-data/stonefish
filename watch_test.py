"""
Live Stonefish Self-Play Viewer
================================
Plays Stonefish vs PureStockfish (or Stonefish vs Stonefish) with full
puzzle detection, and shows the game live in a browser with:
  - The board (animated moves)
  - Puzzle alerts (type, eval gap, floor prob, disagreement)
  - Tier predictions per move
  - Running stats (puzzles found, solved, game result)

Run:  python watch_test.py [num_games] [--elo 1500] [--speed 0.5]
"""

import argparse
import chess
import chess.engine
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from flask import Flask, Response, render_template_string

# Force unbuffered stdout so prints from daemon thread show up immediately
if os.environ.get("PYTHONUNBUFFERED") != "1":
    sys.stdout.reconfigure(line_buffering=True)

from engine import STOCKFISH_PATH, PureStockfishBot
from stonefish.config import StonefishConfig
from stonefish.bot import StonefishBot
from stonefish.maia import MaiaEngine
from stonefish.presets import apply_preset, ELO_PRESETS

# ---------------------------------------------------------------------------
# Flask app + SSE
# ---------------------------------------------------------------------------

app = Flask(__name__)
event_queue = queue.Queue()


def sse_stream():
    while True:
        try:
            data = event_queue.get(timeout=30)
        except queue.Empty:
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


def emit(event_type, **kwargs):
    kwargs["type"] = event_type
    event_queue.put(kwargs)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def run_games(num_games, elo, move_delay):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 2, "Hash": 256})

    config = StonefishConfig(deep_depth=8, base_depth=5)
    if elo is not None:
        apply_preset(config, elo)
    # Speed caps for live viewing — keep moves under ~3s
    config.max_lookahead = 1
    config.rollout_depth = 2          # 1 full move rollout (fast but still catches compounding)
    config.deep_depth = 8             # Sufficient for puzzle validation
    config.num_candidates = 4         # Check fewer candidates

    # Fast Maia: cap simulated depth at 5 so tier predictions are ~3x faster
    fast_maia = MaiaEngine(stockfish_engine=engine, max_simulated_depth=5)

    stockfish_bot = PureStockfishBot(engine, depth=8)

    total_puzzles = 0
    total_solved = 0

    emit("session_start", elo=elo or config.floor_rating,
         floor=config.floor_rating,
         stretch=config.stretch_rating or "SF",
         reach=config.reach_rating or "SF",
         num_games=num_games)

    for game_num in range(num_games):
        sf_white = game_num % 2 == 0
        stonefish = StonefishBot(engine, config, maia=fast_maia)
        stonefish.start_game(chess.WHITE if sf_white else chess.BLACK)

        board = chess.Board()
        move_count = 0
        game_start_time = time.time()

        color_str = "White" if sf_white else "Black"
        print(f"\n{'-'*50}")
        print(f"Game {game_num + 1}/{num_games}  (Stonefish = {color_str})")
        print(f"{'-'*50}")

        emit("game_start", game=game_num + 1, total_games=num_games,
             sf_color="white" if sf_white else "black")

        for _ in range(300):
            if board.is_game_over():
                break

            is_sf_turn = (board.turn == chess.WHITE) == sf_white

            if is_sf_turn:
                # Stonefish's turn
                result = stonefish.choose_move_full(board)
                move = result.move
                san = board.san(move)

                puzzle_data = None
                if result.puzzle_found:
                    puzzle_data = {
                        "type": result.puzzle_type,
                        "eval_gap": round(result.puzzle_eval_gap, 2),
                        "floor_prob": round(result.puzzle_floor_prob * 100),
                        "disagreement": result.puzzle_disagreement,
                        "search_depth": result.puzzle_search_depth,
                        "better_move": result.puzzle_better_move_san,
                        "worse_move": result.puzzle_worse_move_san,
                        "is_mate": result.puzzle_is_mate,
                        "mate_dist": result.puzzle_mate_distance,
                    }

                board.push(move)
                move_count += 1

                # Console log: Stonefish move
                eval_str = f"{result.deep_eval:+.2f}"
                puzzle_tag = ""
                if result.puzzle_found:
                    pt = result.puzzle_type.upper()
                    if puzzle_data.get("is_mate"):
                        pt = f"MATE-IN-{puzzle_data['mate_dist']}"
                    puzzle_tag = (f"  ** {pt} PUZZLE ** "
                                 f"gap={puzzle_data['eval_gap']} "
                                 f"floor={puzzle_data['floor_prob']}% "
                                 f"tiers={puzzle_data['disagreement']}")
                print(f"  {board.fullmove_number:>3}. {san:<8} "
                      f"[rank {result.move_rank}] eval={eval_str}{puzzle_tag}")

                emit("sf_move", fen=board.fen(), move=move.uci(), san=san,
                     move_num=board.fullmove_number, half_move=move_count,
                     rank=result.move_rank, eval=result.deep_eval,
                     puzzle=puzzle_data,
                     candidates=result.candidate_evals[:4])

            else:
                # Opponent's turn (PureStockfish)
                # First notify stonefish about this position
                opp_move = stockfish_bot.choose_move(board)
                opp_san = board.san(opp_move)

                # Notify stonefish bot about opponent's move
                opp_result = stonefish.notify_opponent_move(board, opp_move)

                board.push(opp_move)
                move_count += 1

                opp_data = None
                if opp_result and opp_result.was_puzzle:
                    opp_data = {
                        "was_puzzle": True,
                        "found": opp_result.found_puzzle,
                        "rank": opp_result.move_rank,
                        "eval_cost": round(opp_result.eval_cost, 2),
                        "best_move": opp_result.best_response_san,
                        "puzzle_type": opp_result.puzzle_type,
                    }
                    total_puzzles += 1
                    if opp_result.found_puzzle:
                        total_solved += 1

                # Console log: Opponent move
                opp_tag = ""
                if opp_data and opp_data["was_puzzle"]:
                    if opp_data["found"]:
                        opp_tag = f"  >> FOUND (rank {opp_data['rank']})"
                    else:
                        opp_tag = (f"  >> MISSED (rank {opp_data['rank']}, "
                                   f"cost {opp_data['eval_cost']}p, "
                                   f"best={opp_data['best_move']})")
                print(f"       ...{opp_san:<8}{opp_tag}")

                emit("opp_move", fen=board.fen(), move=opp_move.uci(),
                     san=opp_san, move_num=board.fullmove_number,
                     half_move=move_count, opponent_data=opp_data)

            time.sleep(move_delay)

        # Game over
        state = stonefish.end_game(board)
        outcome = board.outcome()
        result_str = outcome.result() if outcome else "*"

        game_puzzles = 0
        game_solved = 0
        if state:
            game_puzzles = sum(1 for o in state.opponent_results if o.was_puzzle)
            game_solved = sum(1 for o in state.opponent_results
                              if o.was_puzzle and o.found_puzzle)

        sf_result = "?"
        if result_str == "1-0":
            sf_result = "WIN" if sf_white else "LOSS"
        elif result_str == "0-1":
            sf_result = "LOSS" if sf_white else "WIN"
        elif result_str in ("1/2-1/2", "*"):
            sf_result = "DRAW"

        game_elapsed = time.time() - game_start_time
        solve_str = f"{game_solved}/{game_puzzles}" if game_puzzles else "0/0"
        print(f"\n  Result: {sf_result} ({result_str}) in {move_count} moves "
              f"({game_elapsed:.0f}s)")
        print(f"  Puzzles: {game_puzzles} created, {solve_str} solved by opponent")

        emit("game_end", game=game_num + 1, result=result_str,
             sf_result=sf_result, moves=move_count,
             puzzles=game_puzzles, solved=game_solved,
             total_puzzles=total_puzzles, total_solved=total_solved)

        time.sleep(2)  # Pause between games

    # Session summary
    solve_rate = total_solved / total_puzzles * 100 if total_puzzles else 0
    print(f"\n{'='*50}")
    print(f"SESSION SUMMARY  ({num_games} games, ELO {elo})")
    print(f"{'='*50}")
    print(f"  Total puzzles created: {total_puzzles}")
    print(f"  Opponent solved:       {total_solved} ({solve_rate:.0f}%)")
    print(f"{'='*50}\n")

    engine.quit()
    emit("session_end", total_puzzles=total_puzzles, total_solved=total_solved,
         num_games=num_games)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stonefish Self-Play Viewer</title>
<link rel="stylesheet"
      href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    display: flex;
    justify-content: center;
    padding: 20px;
    min-height: 100vh;
  }
  .container {
    display: flex;
    gap: 24px;
    max-width: 1200px;
    width: 100%;
  }
  .board-col { flex: 0 0 480px; }
  .info-col { flex: 1; min-width: 340px; max-height: 95vh; overflow-y: auto; }
  #board { width: 480px; }
  h1 {
    font-size: 1.4rem;
    margin-bottom: 12px;
    color: #58a6ff;
    font-weight: 700;
  }
  .panel {
    background: #161b22;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid #30363d;
  }
  .panel h2 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
    margin-bottom: 8px;
  }
  .badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-weight: 700;
    font-size: 0.9rem;
  }
  .badge-elo { background: #58a6ff; color: #0d1117; }
  .badge-game { background: #238636; color: #fff; margin-left: 8px; }
  .badge-result { padding: 3px 12px; font-size: 1rem; }
  .badge-win { background: #238636; color: #fff; }
  .badge-loss { background: #da3633; color: #fff; }
  .badge-draw { background: #8b949e; color: #0d1117; }

  /* Puzzle alert */
  .puzzle-alert {
    background: #1c1917;
    border: 2px solid #f59e0b;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    animation: pulse 0.5s;
  }
  .puzzle-alert.negative { border-color: #ef4444; }
  .puzzle-alert.positive { border-color: #22c55e; }
  .puzzle-alert.mate { border-color: #a855f7; }
  .puzzle-alert h3 {
    font-size: 1rem;
    margin-bottom: 6px;
  }
  .puzzle-alert.negative h3 { color: #ef4444; }
  .puzzle-alert.positive h3 { color: #22c55e; }
  .puzzle-alert.mate h3 { color: #a855f7; }
  .puzzle-detail {
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.6;
  }
  .puzzle-detail strong { color: #c9d1d9; }
  @keyframes pulse {
    0% { transform: scale(1.02); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
  }

  /* Opponent response */
  .opp-response {
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    animation: fadeIn 0.3s;
  }
  .opp-found { background: #0d2818; border: 1px solid #238636; color: #3fb950; }
  .opp-missed { background: #2d1215; border: 1px solid #da3633; color: #f85149; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  /* Stats */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .stat-item {
    text-align: center;
    padding: 8px 4px;
    background: #0d1117;
    border-radius: 6px;
  }
  .stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #58a6ff;
  }
  .stat-label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    margin-top: 2px;
  }

  /* Move log */
  .move-log {
    max-height: 180px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #8b949e;
    padding: 8px;
    background: #0d1117;
    border-radius: 6px;
  }
  .move-log .sf-move { color: #58a6ff; }
  .move-log .opp-move { color: #c9d1d9; }
  .move-log .puzzle-move { color: #f59e0b; font-weight: 700; }

  /* Candidate list */
  .candidates {
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #8b949e;
  }
  .candidates .top { color: #58a6ff; font-weight: 700; }

  #status { font-size: 0.85rem; color: #8b949e; margin-top: 8px; }

  /* Game history */
  .game-history {
    font-size: 0.85rem;
    color: #8b949e;
  }
  .game-history-item {
    padding: 4px 0;
    border-bottom: 1px solid #21262d;
  }
</style>
</head>
<body>
<div class="container">
  <div class="board-col">
    <h1>Stonefish Self-Play</h1>
    <div id="board"></div>
    <div id="status">Connecting...</div>
  </div>
  <div class="info-col">
    <!-- Session info -->
    <div class="panel" id="session-panel">
      <h2>Session</h2>
      <span class="badge badge-elo" id="elo-badge">--</span>
      <span class="badge badge-game" id="game-badge">--</span>
      <div style="margin-top:6px;font-size:0.85rem;color:#8b949e" id="tier-info"></div>
    </div>

    <!-- Puzzle alert (dynamic) -->
    <div id="puzzle-container"></div>

    <!-- Opponent response (dynamic) -->
    <div id="opp-container"></div>

    <!-- Stats -->
    <div class="panel">
      <h2>Stats</h2>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="stat-value" id="stat-puzzles">0</div>
          <div class="stat-label">Puzzles</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-solved">0</div>
          <div class="stat-label">Solved</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-move">0</div>
          <div class="stat-label">Move</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" id="stat-eval">0.0</div>
          <div class="stat-label">Eval</div>
        </div>
      </div>
    </div>

    <!-- Candidates -->
    <div class="panel">
      <h2>Stonefish Candidates</h2>
      <div class="candidates" id="candidates">--</div>
    </div>

    <!-- Move log -->
    <div class="panel">
      <h2>Move Log</h2>
      <div class="move-log" id="move-log"></div>
    </div>

    <!-- Game history -->
    <div class="panel">
      <h2>Game Results</h2>
      <div class="game-history" id="game-history"></div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
var board = Chessboard('board', {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  appearSpeed: 200,
  moveSpeed: 200,
});

var moveLog = document.getElementById('move-log');
var puzzleContainer = document.getElementById('puzzle-container');
var oppContainer = document.getElementById('opp-container');
var totalPuzzles = 0;
var totalSolved = 0;

var source = new EventSource('/stream');
source.onopen = function() {
  document.getElementById('status').textContent = 'Connected';
};

source.onmessage = function(e) {
  var d = JSON.parse(e.data);

  if (d.type === 'session_start') {
    document.getElementById('elo-badge').textContent = 'ELO ' + d.elo;
    document.getElementById('tier-info').textContent =
      'Floor=' + d.floor + ' / Stretch=' + d.stretch + ' / Reach=' + d.reach;
  }

  if (d.type === 'game_start') {
    document.getElementById('game-badge').textContent =
      'Game ' + d.game + '/' + d.total_games + ' (SF=' + d.sf_color + ')';
    board.position('start');
    moveLog.innerHTML = '';
    puzzleContainer.innerHTML = '';
    oppContainer.innerHTML = '';
    document.getElementById('status').textContent =
      'Game ' + d.game + ' in progress...';
  }

  if (d.type === 'sf_move') {
    board.position(d.fen, true);
    document.getElementById('stat-move').textContent = d.move_num;
    document.getElementById('stat-eval').textContent =
      (d.eval >= 0 ? '+' : '') + d.eval.toFixed(1);

    // Candidates
    if (d.candidates) {
      var html = d.candidates.map(function(c, i) {
        var cls = i === 0 ? 'top' : '';
        return '<span class="' + cls + '">' + c[0] + ' (' + c[1].toFixed(2) + ')</span>';
      }).join(' &nbsp; ');
      document.getElementById('candidates').innerHTML = html;
    }

    // Puzzle alert
    oppContainer.innerHTML = '';
    if (d.puzzle) {
      var pclass = d.puzzle.type;
      var label = d.puzzle.type.charAt(0).toUpperCase() + d.puzzle.type.slice(1);
      if (d.puzzle.is_mate) label = 'Mate in ' + d.puzzle.mate_dist;

      var detail = '<strong>Eval gap:</strong> ' + d.puzzle.eval_gap + ' pawns';
      detail += ' &nbsp; <strong>Floor prob:</strong> ' + d.puzzle.floor_prob + '%';
      detail += '<br><strong>Tiers:</strong> ' + d.puzzle.disagreement;
      detail += ' &nbsp; <strong>Depth:</strong> ' + d.puzzle.search_depth;
      if (d.puzzle.better_move) {
        detail += '<br><strong>Correct:</strong> ' + d.puzzle.better_move;
        detail += ' &nbsp; <strong>Predicted:</strong> ' + d.puzzle.worse_move;
      }

      puzzleContainer.innerHTML =
        '<div class="puzzle-alert ' + pclass + '">' +
        '<h3>' + label + ' Puzzle</h3>' +
        '<div class="puzzle-detail">' + detail + '</div>' +
        '</div>';
    } else {
      puzzleContainer.innerHTML = '';
    }

    // Move log
    var cls = d.puzzle ? 'puzzle-move' : 'sf-move';
    var tag = d.puzzle ? ' [' + d.puzzle.type.toUpperCase() + ']' : '';
    moveLog.innerHTML += '<span class="' + cls + '">' +
      d.move_num + '. ' + d.san + tag + '</span> ';
    moveLog.scrollTop = moveLog.scrollHeight;
  }

  if (d.type === 'opp_move') {
    board.position(d.fen, true);

    // Opponent response to puzzle
    if (d.opponent_data && d.opponent_data.was_puzzle) {
      totalPuzzles++;
      if (d.opponent_data.found) {
        totalSolved++;
        oppContainer.innerHTML =
          '<div class="opp-response opp-found">Found it! Played ' +
          d.san + ' (rank ' + d.opponent_data.rank + ')</div>';
      } else {
        oppContainer.innerHTML =
          '<div class="opp-response opp-missed">Missed! Played ' +
          d.san + ' (rank ' + d.opponent_data.rank +
          ', cost ' + d.opponent_data.eval_cost +
          'p) &mdash; best was ' + d.opponent_data.best_move + '</div>';
      }
      document.getElementById('stat-puzzles').textContent = totalPuzzles;
      document.getElementById('stat-solved').textContent = totalSolved;
    } else {
      oppContainer.innerHTML = '';
      puzzleContainer.innerHTML = '';
    }

    // Move log
    moveLog.innerHTML += '<span class="opp-move">' + d.san + '</span> ';
    moveLog.scrollTop = moveLog.scrollHeight;
  }

  if (d.type === 'game_end') {
    var cls = 'badge badge-result ';
    if (d.sf_result === 'WIN') cls += 'badge-win';
    else if (d.sf_result === 'LOSS') cls += 'badge-loss';
    else cls += 'badge-draw';

    document.getElementById('status').textContent =
      'Game ' + d.game + ': ' + d.sf_result + ' (' + d.result + ') - ' +
      d.puzzles + ' puzzles, ' + d.solved + ' solved';

    var history = document.getElementById('game-history');
    history.innerHTML +=
      '<div class="game-history-item">' +
      'Game ' + d.game + ': <span class="' + cls + '">' + d.sf_result + '</span> ' +
      d.result + ' (' + d.moves + ' moves, ' +
      d.puzzles + ' puzzles, ' + d.solved + ' solved)</div>';

    totalPuzzles = d.total_puzzles;
    totalSolved = d.total_solved;
    document.getElementById('stat-puzzles').textContent = totalPuzzles;
    document.getElementById('stat-solved').textContent = totalSolved;
  }

  if (d.type === 'session_end') {
    document.getElementById('status').textContent =
      'Done! ' + d.num_games + ' games, ' +
      d.total_puzzles + ' total puzzles, ' + d.total_solved + ' solved';
    source.close();
  }
};

source.onerror = function() {
  document.getElementById('status').textContent = 'Connection lost.';
};
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stonefish Self-Play Viewer")
    parser.add_argument("games", type=int, nargs="?", default=2,
                        help="Number of games (default: 2)")
    parser.add_argument("--elo", type=int, default=1500,
                        help="ELO preset to use (default: 1500)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Delay between moves in seconds (default: 0.5)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Web server port (default: 5001)")
    args = parser.parse_args()

    print("=" * 50)
    print("Stonefish Self-Play Viewer")
    print("=" * 50)
    print(f"  Games: {args.games}")
    print(f"  ELO preset: {args.elo}")
    print(f"  Move delay: {args.speed}s")
    print(f"  URL: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    t = threading.Thread(target=run_games,
                         args=(args.games, args.elo, args.speed),
                         daemon=True)
    t.start()

    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
