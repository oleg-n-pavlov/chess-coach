"""Endgame tablebase lookup via Lichess Syzygy API.

For positions with 7 or fewer pieces, returns exact game-theoretic result
(win/draw/loss) and the optimal move.
"""

import time

import chess
import requests


SYZYGY_URL = "https://tablebase.lichess.ovh/standard"
_last_request_time = 0.0


class Tablebase:
    """Probe Syzygy endgame tablebases via Lichess API."""

    def __init__(self):
        self._cache: dict[str, dict | None] = {}

    def probe(self, board: chess.Board) -> dict | None:
        """Probe the tablebase for a position.

        Only queries positions with 7 or fewer pieces on the board.

        Args:
            board: Position to probe.

        Returns:
            Dict with result (win/draw/loss), DTZ, best move, or None.
        """
        # Only probe with <=7 pieces
        piece_count = len(board.piece_map())
        if piece_count > 7:
            return None

        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]

        result = self._query_syzygy(fen)
        self._cache[fen] = result
        return result

    def _query_syzygy(self, fen: str) -> dict | None:
        """Query Lichess Syzygy tablebase API."""
        global _last_request_time

        # Rate limiting
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            resp = requests.get(
                SYZYGY_URL,
                params={"fen": fen},
                timeout=5,
            )
            _last_request_time = time.monotonic()

            if resp.status_code != 200:
                return None

            data = resp.json()
        except (requests.RequestException, ValueError):
            return None

        category = data.get("category")
        if not category:
            return None

        # Map category to human-readable result
        # Categories: "win", "maybe-win", "cursed-win", "draw",
        #             "blessed-loss", "maybe-loss", "loss"
        if category in ("win", "maybe-win", "cursed-win"):
            result = "win"
        elif category == "draw":
            result = "draw"
        else:
            result = "loss"

        # DTZ (distance to zeroing move — capture or pawn push)
        dtz = data.get("dtz")

        # Best move
        best_move_uci = None
        moves = data.get("moves", [])
        if moves:
            # First move in the list is the best
            best_move_uci = moves[0].get("uci")

        return {
            "result": result,
            "category": category,
            "dtz": dtz,
            "best_move": best_move_uci,
        }
