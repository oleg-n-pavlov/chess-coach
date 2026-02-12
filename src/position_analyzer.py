"""Analyze chess positions using Stockfish 17 + Leela concept vectors.

Combines:
  - Stockfish 17: best move, eval, top alternatives
  - Leela concepts: 22 human-interpretable features via SVM
  - Opening book (Lichess explorer API)
  - Endgame tablebases (Lichess Syzygy API)
"""

from pathlib import Path

import chess
import chess.engine

from concept_extractor import LeelaConceptExtractor
from opening_book import OpeningBook
from tablebase import Tablebase
from tactical_detector import (
    analyze_pv_tactics,
    detect_current_tactics,
    filter_significant_tactics,
)


# Piece values for material counting (standard)
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _count_material(board: chess.Board, color: chess.Color) -> int:
    """Count total material value for a side."""
    total = 0
    for piece_type in PIECE_VALUES:
        total += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
    return total


def _is_sacrifice(board: chess.Board, move: chess.Move) -> bool:
    """Check if a move sacrifices material (captures less valuable piece or moves to attacked square)."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        return False

    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0)

    # Check if the destination square is attacked by opponent
    opponent = not board.turn
    if board.is_attacked_by(opponent, move.to_square):
        # Piece moves to an attacked square
        captured = board.piece_at(move.to_square)
        captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        # It's a sacrifice if we're giving up more than we capture
        if moving_value > captured_value + 1:
            return True

    return False


class PositionAnalyzer:
    """Analyze positions with Stockfish + Leela concepts."""

    def __init__(
        self,
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
        leela_model: str | Path = "models/t78_with_intermediate.onnx",
        svm_dir: str | Path = "models/svm",
        depth: int = 18,
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        self.concept_extractor = LeelaConceptExtractor(leela_model, svm_dir)
        self.opening_book = OpeningBook()
        self.tablebase = Tablebase()

    def analyze(
        self,
        board: chess.Board,
        played_move: chess.Move | None = None,
        history: list[chess.Board] | None = None,
        move_index: int = 0,
    ) -> dict:
        """Full analysis of a position.

        Args:
            board: Position to analyze (BEFORE the move).
            played_move: The move that was played (optional).
            history: Previous board states for Leela encoding.
            move_index: Ply index (0-based) for opening book lookup.

        Returns:
            Dict with eval, best move, concepts, and move classification.
        """
        # Opening book lookup
        opening_info = self.opening_book.lookup(board)

        # Endgame tablebase lookup (≤7 pieces)
        tb_info = self.tablebase.probe(board)

        # Stockfish analysis
        sf_info = self.engine.analyse(
            board, chess.engine.Limit(depth=self.depth), multipv=3
        )

        best_moves = []
        for info in sf_info:
            pv = info.get("pv", [])
            score = info.get("score")
            if pv and score:
                # Convert PV to SAN by replaying moves on a copy
                pv_san = []
                pv_board = board.copy()
                for m in pv[:5]:
                    try:
                        pv_san.append(pv_board.san(m))
                        pv_board.push(m)
                    except (AssertionError, ValueError):
                        break
                best_moves.append({
                    "move": pv[0],
                    "san": board.san(pv[0]),
                    "eval": _score_to_cp(score, board.turn),
                    "mate_in": _score_to_mate(score, board.turn),
                    "pv": pv_san if pv_san else [board.san(pv[0])],
                    "pv_moves": list(pv[:6]),  # chess.Move objects for tactical analysis
                })

        # Concept analysis (before move)
        concepts_before = self.concept_extractor.get_concepts(board, history)

        result = {
            "fen": board.fen(),
            "best_moves": best_moves,
            "concepts_before": concepts_before,
        }

        # Add opening info if available
        if opening_info:
            result["opening"] = opening_info

        # Add tablebase info if available
        if tb_info:
            result["tablebase"] = tb_info

        # If a move was played, analyze the difference
        if played_move:
            played_san = board.san(played_move)
            played_eval = None
            played_mate_in = None
            played_pv_moves = []  # PV for played move (chess.Move objects)
            is_best = False

            # Find eval of played move
            for bm in best_moves:
                if bm["move"] == played_move:
                    played_eval = bm["eval"]
                    played_mate_in = bm["mate_in"]
                    played_pv_moves = bm.get("pv_moves", [])
                    is_best = (bm == best_moves[0])
                    break

            if played_eval is None:
                # Played move not in top 3 — analyze it separately
                board_after = board.copy()
                board_after.push(played_move)
                info = self.engine.analyse(board_after, chess.engine.Limit(depth=self.depth))
                score = info.get("score")
                if score:
                    played_eval = _score_to_cp(score, board_after.turn)
                    played_mate_in = _score_to_mate_after(score, board_after.turn)
                # Get PV for played move: played_move + continuation
                pv_after = info.get("pv", [])
                played_pv_moves = [played_move] + list(pv_after[:5])

            # Concept diff
            board_after = board.copy()
            board_after.push(played_move)
            history_after = [board.copy()] + (history or [])
            concepts_after = self.concept_extractor.get_concepts(board_after, history_after)

            # Compute concept diff
            concept_diff = {}
            if "raw_vector" not in concepts_before:
                for key in concepts_before:
                    if key in concepts_after:
                        concept_diff[key] = concepts_after[key] - concepts_before[key]

            # Classify move quality
            best_eval = best_moves[0]["eval"] if best_moves else 0
            eval_loss = (best_eval - played_eval) if played_eval is not None else 0

            move_quality = _classify_move(
                board, played_move, eval_loss, is_best,
                best_moves, opening_info, played_mate_in,
            )

            # Tactical analysis
            best_pv_moves = best_moves[0].get("pv_moves", []) if best_moves else []
            tactics_best = filter_significant_tactics(
                analyze_pv_tactics(board, best_pv_moves)
            ) if best_pv_moves else []
            tactics_played = filter_significant_tactics(
                analyze_pv_tactics(board, played_pv_moves)
            ) if played_pv_moves else []
            tactics_current = filter_significant_tactics(
                detect_current_tactics(board)
            )

            result.update({
                "played_move": played_san,
                "played_eval": played_eval,
                "played_mate_in": played_mate_in,
                "best_eval": best_eval,
                "eval_loss": eval_loss,
                "move_quality": move_quality,
                "is_best": is_best,
                "concepts_after": concepts_after,
                "concept_diff": concept_diff,
                # Top 3 most changed concepts
                "key_concepts": _top_concept_changes(concept_diff),
                # Tactical motifs
                "tactics_in_best_line": tactics_best,
                "tactics_in_played_line": tactics_played,
                "tactics_current": tactics_current,
            })

        return result

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """Convert Stockfish score to centipawns from white's perspective."""
    relative = score.relative
    if relative.is_mate():
        mate_in = relative.mate()
        return 10000 if mate_in > 0 else -10000
    cp = relative.score()
    return cp if turn == chess.WHITE else -cp


def _score_to_mate(score: chess.engine.PovScore, turn: chess.Color) -> int | None:
    """Extract mate-in-N from score (from white's perspective).

    Returns positive N if white mates in N, negative if black mates in N, None if no mate.
    """
    relative = score.relative
    if not relative.is_mate():
        return None
    mate_in = relative.mate()
    return mate_in if turn == chess.WHITE else -mate_in


def _score_to_mate_after(score: chess.engine.PovScore, turn: chess.Color) -> int | None:
    """Extract mate-in-N from the perspective BEFORE the move was played.

    When we analyze position after a move, the score is from the opponent's POV.
    We negate to get it from the mover's perspective.
    """
    relative = score.relative
    if not relative.is_mate():
        return None
    mate_in = relative.mate()
    # Negate because this is opponent's score after our move
    negated = -mate_in
    return negated if turn == chess.WHITE else -negated


def _classify_move(
    board: chess.Board,
    move: chess.Move,
    eval_loss: int,
    is_best: bool,
    best_moves: list[dict],
    opening_info: dict | None,
    played_mate_in: int | None,
) -> str:
    """Classify a move into categories: brilliant, great, best, good, book,
    inaccuracy, mistake, blunder, forced.

    Categories (highest priority first):
    - book: move is in the opening book
    - forced: only 1 legal move or all alternatives lose badly
    - brilliant: best/near-best + involves a material sacrifice
    - great: best/near-best + was the only non-losing move
    - best: eval_loss <= 10
    - good: eval_loss <= 30
    - inaccuracy: eval_loss 30-80
    - mistake: eval_loss 80-200
    - blunder: eval_loss > 200
    """
    # Book move
    if opening_info:
        move_uci = move.uci()
        for bm in opening_info.get("top_moves", []):
            if bm["uci"] == move_uci:
                return "book"

    # Forced move (only 1 legal move)
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 1:
        return "forced"

    # Forced: all other top moves lose badly (>300cp worse than best)
    if is_best and len(best_moves) >= 2:
        if abs(best_moves[0]["eval"] - best_moves[1]["eval"]) > 300:
            return "forced"

    # Brilliant: near-best move that sacrifices material
    if eval_loss <= 10 and _is_sacrifice(board, move):
        return "brilliant"

    # Great: near-best and only non-losing move (others are >150cp worse)
    if eval_loss <= 10 and len(best_moves) >= 2:
        if abs(best_moves[0]["eval"] - best_moves[1]["eval"]) > 150:
            return "great"

    # Standard classifications
    if eval_loss <= 10:
        return "best"
    elif eval_loss <= 30:
        return "good"
    elif eval_loss <= 80:
        return "inaccuracy"
    elif eval_loss <= 200:
        return "mistake"
    else:
        return "blunder"


def _top_concept_changes(diff: dict[str, float], n: int = 3) -> list[dict]:
    """Get the top N most changed concepts."""
    if not diff:
        return []
    sorted_concepts = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)
    return [{"concept": k, "change": v} for k, v in sorted_concepts[:n]]
