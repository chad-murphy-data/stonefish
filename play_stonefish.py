"""
Play Against Stonefish
======================
A browser-based interface to play chess against Stonefish with live
puzzle detection and chat alerts.

Run:  python play_stonefish.py [--elo 500] [--color white] [--port 5002]
"""

import argparse
import chess
import chess.engine
import json
import os
import sys
import threading
import time
import webbrowser
from flask import Flask, Response, request, jsonify, render_template_string

# Force unbuffered stdout
if os.environ.get("PYTHONUNBUFFERED") != "1":
    sys.stdout.reconfigure(line_buffering=True)

from engine import STOCKFISH_PATH
from stonefish.config import StonefishConfig
from stonefish.bot import StonefishBot
from stonefish.chat import ChatEngine
from stonefish.maia import MaiaEngine
from stonefish.presets import apply_preset

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Global game state (single-player, single-game server)
game_lock = threading.Lock()
game = None  # will be a dict with all game state


class GameSession:
    """Holds one game session."""

    def __init__(self, elo, human_color_str):
        self.elo = elo
        self.human_is_white = human_color_str == "white"
        self.sf_color = chess.BLACK if self.human_is_white else chess.WHITE

        # Engine
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.engine.configure({"Threads": 2, "Hash": 256})

        # Config
        self.config = StonefishConfig(deep_depth=8, base_depth=5)
        apply_preset(self.config, elo)
        self.config.max_lookahead = 1
        self.config.rollout_depth = 2
        self.config.deep_depth = 8
        self.config.num_candidates = 4
        self.config.chat_enabled = True

        # Maia + Bot + Chat
        self.maia = MaiaEngine(stockfish_engine=self.engine, max_simulated_depth=5)
        self.bot = StonefishBot(self.engine, self.config, maia=self.maia)
        self.bot.start_game(self.sf_color)
        self.chat = ChatEngine(self.config)

        # Board
        self.board = chess.Board()
        self.messages = []  # chat messages: {sender, text}
        self.game_over = False
        self.result_str = ""

        # Onboarding
        msg = self.chat.on_game_start()
        if msg:
            self.messages.append({"sender": "stonefish", "text": msg})

        print(f"\nNew game: Human={human_color_str}, ELO={elo}")
        print(f"  Floor={self.config.floor_rating}, "
              f"Stretch={self.config.stretch_rating or 'SF'}, "
              f"Reach={self.config.reach_rating or 'SF'}")

    def is_human_turn(self):
        if self.game_over:
            return False
        return (self.board.turn == chess.WHITE) == self.human_is_white

    def make_human_move(self, uci_str):
        """Process a human move. Returns dict with game state update."""
        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            return {"error": "Invalid move format"}

        if move not in self.board.legal_moves:
            return {"error": "Illegal move"}

        san = self.board.san(move)

        # Notify stonefish about human's move
        opp_result = self.bot.notify_opponent_move(self.board, move)

        self.board.push(move)

        # Chat response to human's move
        new_messages = []
        if opp_result:
            chat_msg = self.chat.on_opponent_move(opp_result)
            if chat_msg:
                new_messages.append({"sender": "stonefish", "text": chat_msg})
                self.messages.append(new_messages[-1])

        print(f"  Human: {san}")
        if opp_result and opp_result.was_puzzle:
            tag = "FOUND" if opp_result.found_puzzle else "MISSED"
            print(f"    >> {tag} (rank {opp_result.move_rank}, "
                  f"cost {opp_result.eval_cost:.2f}p)")
        for m in new_messages:
            print(f"    Chat: {m['text']}")

        result = {
            "fen": self.board.fen(),
            "san": san,
            "messages": new_messages,
        }

        # Check game over
        if self.board.is_game_over():
            result.update(self._handle_game_over())
        else:
            # Stonefish replies
            sf_data = self._make_sf_move()
            result["sf_move"] = sf_data

        return result

    def make_sf_first_move(self):
        """If Stonefish is white, make its first move."""
        if self.human_is_white:
            return None
        return self._make_sf_move()

    def _make_sf_move(self):
        """Stonefish thinks and plays."""
        move_result = self.bot.choose_move_full(self.board)
        move = move_result.move
        san = self.board.san(move)
        self.board.push(move)

        # Chat about puzzle
        new_messages = []
        chat_msg = self.chat.on_our_move(move_result)
        if chat_msg:
            new_messages.append({"sender": "stonefish", "text": chat_msg})
            self.messages.append(new_messages[-1])

        # Console log
        puzzle_tag = ""
        if move_result.puzzle_found:
            pt = move_result.puzzle_type.upper()
            if move_result.puzzle_is_mate:
                pt = f"MATE-IN-{move_result.puzzle_mate_distance}"
            puzzle_tag = (f"  ** {pt} PUZZLE ** "
                         f"gap={move_result.puzzle_eval_gap:.2f} "
                         f"floor={move_result.puzzle_floor_prob*100:.0f}%")
        print(f"  Stonefish: {san} [rank {move_result.move_rank}] "
              f"eval={move_result.deep_eval:+.2f}{puzzle_tag}")
        for m in new_messages:
            print(f"    Chat: {m['text']}")

        data = {
            "fen": self.board.fen(),
            "move": move.uci(),
            "san": san,
            "eval": round(move_result.deep_eval, 2),
            "rank": move_result.move_rank,
            "messages": new_messages,
            "puzzle": None,
        }

        if move_result.puzzle_found:
            data["puzzle"] = {
                "type": move_result.puzzle_type,
                "eval_gap": round(move_result.puzzle_eval_gap, 2),
                "floor_prob": round(move_result.puzzle_floor_prob * 100),
                "is_mate": move_result.puzzle_is_mate,
            }

        # Check game over after SF move
        if self.board.is_game_over():
            data["game_over"] = self._handle_game_over()

        return data

    def _handle_game_over(self):
        self.game_over = True
        outcome = self.board.outcome()
        self.result_str = outcome.result() if outcome else "*"

        # Post-game summary
        state = self.bot.end_game(self.board)
        summary_messages = []
        if state:
            summaries = self.chat.generate_post_game_summary(state)
            for s in summaries:
                msg = {"sender": "stonefish", "text": s}
                summary_messages.append(msg)
                self.messages.append(msg)

        puzzles = 0
        solved = 0
        if state:
            puzzles = state.total_puzzles_created
            solved = state.total_puzzles_solved

        who_won = "draw"
        if self.result_str == "1-0":
            who_won = "white"
        elif self.result_str == "0-1":
            who_won = "black"

        human_won = (who_won == "white" and self.human_is_white) or \
                    (who_won == "black" and not self.human_is_white)

        print(f"\n  Game over: {self.result_str}")
        print(f"  Puzzles: {puzzles} created, {solved} solved")
        for m in summary_messages:
            print(f"    Chat: {m['text']}")

        return {
            "game_over": True,
            "result": self.result_str,
            "human_won": human_won,
            "draw": who_won == "draw",
            "puzzles_total": puzzles,
            "puzzles_solved": solved,
            "messages": summary_messages,
        }

    def get_legal_moves(self):
        """Return legal moves for current position."""
        return [m.uci() for m in self.board.legal_moves]

    def shutdown(self):
        try:
            self.bot.end_game()
        except Exception:
            pass
        try:
            self.engine.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global game
    data = request.json or {}
    elo = data.get("elo", 500)
    color = data.get("color", "white")

    with game_lock:
        if game is not None:
            game.shutdown()
        game = GameSession(elo, color)

        result = {
            "fen": game.board.fen(),
            "human_is_white": game.human_is_white,
            "messages": game.messages,
            "elo": elo,
        }

        # If Stonefish goes first
        if not game.human_is_white:
            sf_data = game.make_sf_first_move()
            if sf_data:
                result["sf_move"] = sf_data
                result["fen"] = sf_data["fen"]

    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def make_move():
    global game
    data = request.json or {}
    uci = data.get("move", "")

    with game_lock:
        if game is None:
            return jsonify({"error": "No active game"}), 400
        if game.game_over:
            return jsonify({"error": "Game is over"}), 400
        if not game.is_human_turn():
            return jsonify({"error": "Not your turn"}), 400

        result = game.make_human_move(uci)

    return jsonify(result)


@app.route("/api/legal_moves")
def legal_moves():
    with game_lock:
        if game is None:
            return jsonify({"moves": []})
        return jsonify({"moves": game.get_legal_moves()})


@app.route("/api/state")
def get_state():
    with game_lock:
        if game is None:
            return jsonify({"active": False})
        return jsonify({
            "active": True,
            "fen": game.board.fen(),
            "human_is_white": game.human_is_white,
            "is_human_turn": game.is_human_turn(),
            "game_over": game.game_over,
            "messages": game.messages,
        })


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Play Against Stonefish</title>
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
    max-width: 1100px;
    width: 100%;
  }
  .board-col { flex: 0 0 480px; }
  .info-col { flex: 1; min-width: 320px; display: flex; flex-direction: column; gap: 12px; max-height: 95vh; }
  #board { width: 480px; }

  h1 { font-size: 1.4rem; color: #58a6ff; font-weight: 700; margin-bottom: 8px; }

  .panel {
    background: #161b22;
    border-radius: 8px;
    padding: 14px;
    border: 1px solid #30363d;
  }
  .panel h2 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b949e;
    margin-bottom: 8px;
  }

  /* Setup */
  .setup-row {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .setup-row label { font-size: 0.85rem; color: #8b949e; }
  .setup-row select, .setup-row button {
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #0d1117;
    color: #c9d1d9;
    font-size: 0.9rem;
    cursor: pointer;
  }
  .setup-row button {
    background: #238636;
    border-color: #238636;
    color: #fff;
    font-weight: 700;
  }
  .setup-row button:hover { background: #2ea043; }

  /* Chat */
  .chat-box {
    flex: 1;
    min-height: 200px;
    max-height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px;
    background: #0d1117;
    border-radius: 6px;
  }
  .chat-msg {
    padding: 8px 12px;
    border-radius: 12px;
    max-width: 90%;
    font-size: 0.9rem;
    line-height: 1.4;
    animation: fadeIn 0.3s;
  }
  .chat-msg.stonefish {
    background: #1c2333;
    border: 1px solid #30363d;
    color: #c9d1d9;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  .chat-msg.stonefish.puzzle-alert {
    border-color: #f59e0b;
    background: #1c1917;
  }
  .chat-msg.stonefish.found-alert {
    border-color: #22c55e;
    background: #0d2818;
  }
  .chat-msg.stonefish.missed-alert {
    border-color: #ef4444;
    background: #2d1215;
  }
  .chat-msg.system {
    color: #8b949e;
    font-size: 0.82rem;
    font-style: italic;
    align-self: center;
    text-align: center;
  }
  .chat-sender {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
    color: #58a6ff;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  /* Stats */
  .stat-row {
    display: flex;
    gap: 16px;
    font-size: 0.9rem;
  }
  .stat-row .stat { color: #58a6ff; font-weight: 700; }

  /* Status */
  #status {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 4px;
    min-height: 1.2em;
  }
  #thinking {
    display: none;
    color: #f59e0b;
    font-size: 0.85rem;
    margin-top: 4px;
  }

  /* Highlight squares */
  .highlight-legal {
    background: radial-gradient(circle, rgba(88,166,255,0.35) 25%, transparent 25%);
  }
  .highlight-check {
    background: radial-gradient(circle, rgba(239,68,68,0.5) 60%, transparent 60%);
  }
</style>
</head>
<body>
<div class="container">
  <div class="board-col">
    <h1>Play Against Stonefish</h1>
    <div id="board"></div>
    <div id="status">Set up a new game to start.</div>
    <div id="thinking">Stonefish is thinking...</div>
  </div>
  <div class="info-col">
    <!-- Setup -->
    <div class="panel">
      <h2>New Game</h2>
      <div class="setup-row">
        <label>ELO:</label>
        <select id="elo-select">
          <option value="500" selected>500</option>
          <option value="750">750</option>
          <option value="1000">1000</option>
          <option value="1250">1250</option>
          <option value="1500">1500</option>
          <option value="1750">1750</option>
          <option value="2000">2000</option>
        </select>
        <label>Play as:</label>
        <select id="color-select">
          <option value="white" selected>White</option>
          <option value="black">Black</option>
          <option value="random">Random</option>
        </select>
        <button onclick="startGame()">New Game</button>
      </div>
    </div>

    <!-- Stats -->
    <div class="panel">
      <h2>Game Stats</h2>
      <div class="stat-row">
        <div>Puzzles: <span class="stat" id="stat-puzzles">0</span></div>
        <div>Solved: <span class="stat" id="stat-solved">0</span></div>
        <div>Move: <span class="stat" id="stat-move">0</span></div>
      </div>
    </div>

    <!-- Chat -->
    <div class="panel" style="flex:1; display:flex; flex-direction:column;">
      <h2>Stonefish Chat</h2>
      <div class="chat-box" id="chat-box"></div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var gameActive = false;
var humanIsWhite = true;
var isHumanTurn = false;
var selectedSquare = null;
var legalMoves = [];
var totalPuzzles = 0;
var totalSolved = 0;

// ---------------------------------------------------------------------------
// Board setup
// ---------------------------------------------------------------------------
var boardConfig = {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  draggable: true,
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
  onMouseoutSquare: onMouseoutSquare,
  onMouseoverSquare: onMouseoverSquare,
  appearSpeed: 200,
  moveSpeed: 200,
};
var board = Chessboard('board', boardConfig);

// ---------------------------------------------------------------------------
// Game management
// ---------------------------------------------------------------------------
function startGame() {
  var elo = parseInt(document.getElementById('elo-select').value);
  var color = document.getElementById('color-select').value;
  if (color === 'random') {
    color = Math.random() < 0.5 ? 'white' : 'black';
  }

  totalPuzzles = 0;
  totalSolved = 0;
  updateStats(1);
  document.getElementById('chat-box').innerHTML = '';
  document.getElementById('status').textContent = 'Starting new game...';
  document.getElementById('thinking').style.display = 'none';

  $.ajax({
    url: '/api/new_game',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({elo: elo, color: color}),
    success: function(data) {
      gameActive = true;
      humanIsWhite = data.human_is_white;
      board.orientation(humanIsWhite ? 'white' : 'black');
      board.position(data.fen, false);

      // Show initial messages
      if (data.messages) {
        data.messages.forEach(function(m) { addChatMessage(m.sender, m.text); });
      }

      // If SF moved first
      if (data.sf_move) {
        if (data.sf_move.messages) {
          data.sf_move.messages.forEach(function(m) { addChatMessage(m.sender, m.text, data.sf_move.puzzle); });
        }
        if (data.sf_move.game_over) {
          handleGameOver(data.sf_move.game_over);
          return;
        }
      }

      isHumanTurn = true;
      fetchLegalMoves();
      document.getElementById('status').textContent = 'Your turn!';
    },
    error: function() {
      document.getElementById('status').textContent = 'Error starting game.';
    }
  });
}

function fetchLegalMoves() {
  $.get('/api/legal_moves', function(data) {
    legalMoves = data.moves || [];
  });
}

// ---------------------------------------------------------------------------
// Move handling
// ---------------------------------------------------------------------------
function onDragStart(source, piece) {
  if (!gameActive || !isHumanTurn) return false;
  // Only drag our own pieces
  if (humanIsWhite && piece.search(/^b/) !== -1) return false;
  if (!humanIsWhite && piece.search(/^w/) !== -1) return false;
  return true;
}

function onDrop(source, target, piece) {
  if (!gameActive || !isHumanTurn) return 'snapback';

  // Build UCI move
  var uci = source + target;
  // Check for promotion
  if (piece === 'wP' && target[1] === '8') uci += 'q';
  if (piece === 'bP' && target[1] === '1') uci += 'q';

  // Check if move is in legal moves list
  var isLegal = legalMoves.some(function(m) {
    return m === uci || m.substring(0, 4) === uci.substring(0, 4);
  });
  if (!isLegal) return 'snapback';

  // Use exact legal move UCI (handles promotion correctly)
  var exactMove = legalMoves.find(function(m) {
    return m.substring(0, 4) === uci.substring(0, 4);
  });
  if (exactMove && exactMove.length > 4 && uci.length <= 4) {
    uci = exactMove;  // Use the legal move's promotion piece
  }

  sendMove(uci);
  return undefined;  // let it drop
}

function onSnapEnd() {
  // Board position is updated after animation
}

function onMouseoverSquare(square) {
  if (!gameActive || !isHumanTurn) return;
  // Highlight legal destinations from this square
  var targets = legalMoves.filter(function(m) { return m.substring(0, 2) === square; });
  if (targets.length === 0) return;
  targets.forEach(function(m) {
    var dest = m.substring(2, 4);
    $('#board .square-' + dest).addClass('highlight-legal');
  });
}

function onMouseoutSquare() {
  $('#board .square-55d63').removeClass('highlight-legal');
}

function sendMove(uci) {
  isHumanTurn = false;
  clearHighlights();
  document.getElementById('status').textContent = '';
  document.getElementById('thinking').style.display = 'block';

  $.ajax({
    url: '/api/move',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({move: uci}),
    success: function(data) {
      document.getElementById('thinking').style.display = 'none';

      if (data.error) {
        document.getElementById('status').textContent = 'Error: ' + data.error;
        isHumanTurn = true;
        return;
      }

      // Update board after human move
      board.position(data.fen, true);
      updateStats(data.fen);

      // Human move chat messages (found/missed puzzle alerts)
      if (data.messages) {
        data.messages.forEach(function(m) {
          var cssClass = '';
          if (m.text.match(/Nice find|well played|You got it|Exactly|Checkmate.*Great|You found|Beautiful|That's the mate/i)) cssClass = 'found';
          else if (m.text.match(/Not quite|slipped|better move|got away|off the hook|Lucky escape|moment passed|survived/i)) cssClass = 'missed';
          addChatMessage(m.sender, m.text, null, cssClass);
        });
      }

      // Check if game ended after human move
      if (data.game_over) {
        handleGameOver(data);
        return;
      }

      // Process SF response
      if (data.sf_move) {
        board.position(data.sf_move.fen, true);
        updateStats(data.sf_move.fen);

        if (data.sf_move.messages) {
          data.sf_move.messages.forEach(function(m) {
            addChatMessage(m.sender, m.text, data.sf_move.puzzle);
          });
        }

        if (data.sf_move.game_over) {
          handleGameOver(data.sf_move.game_over);
          return;
        }

        isHumanTurn = true;
        fetchLegalMoves();
        document.getElementById('status').textContent = 'Your turn!';
      }
    },
    error: function() {
      document.getElementById('thinking').style.display = 'none';
      document.getElementById('status').textContent = 'Error sending move.';
      isHumanTurn = true;
    }
  });
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function addChatMessage(sender, text, puzzleData, overrideClass) {
  var box = document.getElementById('chat-box');
  var div = document.createElement('div');
  div.className = 'chat-msg ' + sender;

  if (overrideClass === 'found') {
    div.className += ' found-alert';
  } else if (overrideClass === 'missed') {
    div.className += ' missed-alert';
  } else if (puzzleData) {
    div.className += ' puzzle-alert';
  }

  var label = document.createElement('div');
  label.className = 'chat-sender';
  label.textContent = sender === 'stonefish' ? 'Stonefish' : 'System';
  div.appendChild(label);

  var body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);

  if (puzzleData) {
    var meta = document.createElement('div');
    meta.style.cssText = 'font-size:0.75rem;color:#8b949e;margin-top:4px;';
    var ptype = puzzleData.type;
    if (puzzleData.is_mate) ptype = 'mate';
    meta.textContent = ptype.toUpperCase() + ' | gap: ' + puzzleData.eval_gap +
      'p | floor: ' + puzzleData.floor_prob + '%';
    div.appendChild(meta);
    // Track puzzles
    totalPuzzles++;
    document.getElementById('stat-puzzles').textContent = totalPuzzles;
  }

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------------------
// Game over
// ---------------------------------------------------------------------------
function handleGameOver(data) {
  gameActive = false;
  isHumanTurn = false;
  document.getElementById('thinking').style.display = 'none';

  var msg = 'Game over: ' + data.result;
  if (data.human_won) msg = 'You won! ' + data.result;
  else if (data.draw) msg = 'Draw! ' + data.result;
  else msg = 'You lost. ' + data.result;
  msg += ' | Puzzles: ' + data.puzzles_solved + '/' + data.puzzles_total + ' solved';

  document.getElementById('status').textContent = msg;
  totalSolved = data.puzzles_solved || 0;
  document.getElementById('stat-solved').textContent = totalSolved;

  // Post-game messages
  if (data.messages) {
    data.messages.forEach(function(m) {
      addChatMessage(m.sender, m.text);
    });
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function updateStats(fen) {
  // Extract move number from FEN
  if (typeof fen === 'string') {
    var parts = fen.split(' ');
    if (parts.length >= 6) {
      document.getElementById('stat-move').textContent = parts[5];
    }
  }
  document.getElementById('stat-puzzles').textContent = totalPuzzles;
  document.getElementById('stat-solved').textContent = totalSolved;
}

function clearHighlights() {
  $('#board .square-55d63').removeClass('highlight-legal highlight-check');
}

// Start a game automatically on load
$(function() {
  startGame();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play Against Stonefish")
    parser.add_argument("--elo", type=int, default=500,
                        help="ELO preset (default: 500)")
    parser.add_argument("--color", choices=["white", "black", "random"],
                        default="white",
                        help="Your color (default: white)")
    parser.add_argument("--port", type=int, default=5002,
                        help="Web server port (default: 5002)")
    args = parser.parse_args()

    print("=" * 50)
    print("Play Against Stonefish")
    print("=" * 50)
    print(f"  Default ELO: {args.elo}")
    print(f"  Default color: {args.color}")
    print(f"  URL: http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    threading.Timer(1.5, lambda: webbrowser.open(
        f"http://localhost:{args.port}")).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
