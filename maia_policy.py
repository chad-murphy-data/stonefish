"""
MaiaPolicyEngine -- extract Maia's full policy distribution per position
========================================================================
lc0 with `--verbose-move-stats=true` emits one `info string <uci_move> ... (P: XX.XX%) ...`
line per legal move during a search. python-chess's SimpleEngine drops these
intermediate lines, so we drive lc0 via subprocess and parse stdout ourselves.

Each `predict_policy(board)` returns a dict mapping chess.Move -> probability
in [0, 1], summing roughly to 1.0 across all legal moves.
"""

import subprocess
import re
import os
import fcntl
import time
import chess
from typing import Dict, Optional, List

# Examples we need to parse:
#   info string e2e4  (322 ) N:      51 (+ 0) (P: 44.24%) (WL:  0.02707) ...
#   info string node  (  20) N:      90 (+ 0) (P: 85.20%) ...
# We skip the `node` summary line and keep the per-move lines. The 3rd token
# is the move UCI; the policy percentage is in `(P: XX.XX%)`.
_MOVE_LINE = re.compile(
    r"^info string\s+(?P<move>[a-h][1-8][a-h][1-8][qrbnQRBN]?)\s.*?\(P:\s*(?P<p>[\d.]+)%\)"
)


class MaiaPolicyEngine:
    """Drives lc0 directly to get per-move policy probabilities."""

    def __init__(self, weights_path: str, lc0_path: str = "lc0",
                 backend: str = "eigen", threads: int = 1,
                 rating: int = 1900, nodes: int = 1):
        self.weights_path = weights_path
        self.rating = rating
        self.nodes = nodes
        self.proc = subprocess.Popen(
            [lc0_path, f"--weights={weights_path}", f"--backend={backend}",
             f"--threads={threads}", "--verbose-move-stats=true"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        # Set stdout fd to non-blocking so we can poll for available bytes
        fd = self.proc.stdout.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._buf = ""
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        if self.proc.stdin is None or self.proc.poll() is not None:
            raise RuntimeError("lc0 process is dead")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, marker: str, timeout: float = 30.0) -> List[str]:
        """Read stdout until a line starting with `marker` arrives. Returns all lines."""
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("lc0 exited unexpectedly")
            # Drain whatever bytes are available without blocking
            try:
                chunk = self.proc.stdout.read(4096)
            except (BlockingIOError, TypeError):
                chunk = None
            if chunk:
                self._buf += chunk
            else:
                time.sleep(0.01)
            # Pull complete lines out of the buffer
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                lines.append(line)
                if line.startswith(marker):
                    return lines
        raise TimeoutError(f"timed out waiting for {marker!r} (got {len(lines)} lines)")

    def predict_policy(self, board: chess.Board) -> Dict[chess.Move, float]:
        """Return Maia's policy distribution for the current position.

        Returns: dict mapping chess.Move -> probability (summing ~1.0).
        """
        fen = board.fen()
        self._send(f"position fen {fen}")
        self._send(f"go nodes {self.nodes}")

        lines = self._wait_for("bestmove")
        legal_uci_to_move = {m.uci(): m for m in board.legal_moves}
        policy: Dict[chess.Move, float] = {}
        for line in lines:
            m = _MOVE_LINE.match(line)
            if not m:
                continue
            uci = m.group("move")
            if uci not in legal_uci_to_move:
                continue  # skip the `node` summary line and any spurious matches
            policy[legal_uci_to_move[uci]] = float(m.group("p")) / 100.0
        return policy

    def predict_top(self, board: chess.Board) -> chess.Move:
        """Convenience: highest-policy move."""
        policy = self.predict_policy(board)
        if not policy:
            return next(iter(board.legal_moves))
        return max(policy, key=policy.get)

    def prob_of(self, board: chess.Board, move: chess.Move) -> float:
        """Convenience: probability of a specific move."""
        return self.predict_policy(board).get(move, 0.0)

    def sample_move(self, board: chess.Board, temperature: float = 1.0,
                    rng=None) -> chess.Move:
        """Sample a move from Maia's policy at the given temperature.

        temperature = 1.0: sample directly from the trained policy
                         (matches how Maia is meant to model humans).
        temperature < 1.0: concentrates toward the argmax (more peaky).
        temperature <= 0:  deterministic argmax (same as predict_top).
        """
        policy = self.predict_policy(board)
        if not policy:
            return next(iter(board.legal_moves))
        if temperature <= 0:
            return max(policy, key=policy.get)
        import random as _random
        rng = rng or _random
        moves = list(policy.keys())
        # Apply temperature: p_i ** (1/T), then renormalize.
        weights = [max(p, 1e-12) ** (1.0 / temperature) for p in policy.values()]
        total = sum(weights)
        if total <= 0:
            return max(policy, key=policy.get)
        weights = [w / total for w in weights]
        return rng.choices(moves, weights=weights, k=1)[0]

    def quit(self):
        try:
            if self.proc.poll() is None:
                self._send("quit")
                self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def __del__(self):
        self.quit()


if __name__ == "__main__":
    # Smoke test
    import sys
    weights = sys.argv[1] if len(sys.argv) > 1 else "/home/user/.maia/maia-1900.pb.gz"
    eng = MaiaPolicyEngine(weights)
    b = chess.Board()
    policy = eng.predict_policy(b)
    print(f"Maia 1900 policy from starting position ({len(policy)} legal moves):")
    sorted_moves = sorted(policy.items(), key=lambda kv: -kv[1])
    for move, p in sorted_moves[:10]:
        print(f"  {b.san(move):>6}  {p*100:6.2f}%")
    total = sum(policy.values())
    print(f"  (sum of all moves: {total*100:.1f}%)")
    eng.quit()
