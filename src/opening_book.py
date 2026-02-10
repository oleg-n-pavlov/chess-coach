"""Opening book lookup via Lichess Opening Explorer API.

Returns opening name, statistics, and popular continuations for a position.
"""

import time
from functools import lru_cache

import chess
import requests


LICHESS_OPENING_URL = "https://explorer.lichess.ovh/masters"
# Rate limit: 1 req/s for masters DB
_last_request_time = 0.0


class OpeningBook:
    """Lookup opening information for chess positions."""

    def __init__(self, min_games: int = 5):
        self.min_games = min_games
        self._cache: dict[str, dict | None] = {}

    def lookup(self, board: chess.Board) -> dict | None:
        """Look up a position in the opening book.

        Args:
            board: Position to look up.

        Returns:
            Dict with opening name, stats, and top moves, or None if not in book.
        """
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]

        result = self._query_lichess(fen)
        self._cache[fen] = result
        return result

    def is_book_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move is in the opening book."""
        info = self.lookup(board)
        if not info:
            return False
        move_uci = move.uci()
        return any(m["uci"] == move_uci for m in info.get("top_moves", []))

    def _query_lichess(self, fen: str) -> dict | None:
        """Query Lichess masters opening explorer."""
        global _last_request_time

        # Rate limiting
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            resp = requests.get(
                LICHESS_OPENING_URL,
                params={"fen": fen, "topGames": 0, "recentGames": 0},
                timeout=5,
            )
            _last_request_time = time.monotonic()

            if resp.status_code != 200:
                return None

            data = resp.json()
        except (requests.RequestException, ValueError):
            return None

        # Check if position is in the book
        total_games = data.get("white", 0) + data.get("draws", 0) + data.get("black", 0)
        if total_games < self.min_games:
            return None

        # Opening name
        opening = data.get("opening")
        opening_name = None
        eco = None
        if opening:
            opening_name = opening.get("name")
            eco = opening.get("eco")

        # Top moves with stats
        top_moves = []
        for m in data.get("moves", [])[:5]:
            games = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
            if games < 1:
                continue
            white_pct = m["white"] / games * 100 if games else 0
            draw_pct = m["draws"] / games * 100 if games else 0
            black_pct = m["black"] / games * 100 if games else 0
            top_moves.append({
                "san": m.get("san", ""),
                "uci": m.get("uci", ""),
                "games": games,
                "white_pct": round(white_pct, 1),
                "draw_pct": round(draw_pct, 1),
                "black_pct": round(black_pct, 1),
            })

        return {
            "name": opening_name,
            "eco": eco,
            "total_games": total_games,
            "white_wins": data.get("white", 0),
            "draws": data.get("draws", 0),
            "black_wins": data.get("black", 0),
            "top_moves": top_moves,
        }
