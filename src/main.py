"""Chess Coach: AI-тренер для разбора шахматных партий.

Анализирует PGN-партию и генерирует комментарии на русском языке,
используя Stockfish 17 + Leela concept vectors + Claude.
"""

import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent))

from commentary_generator import CommentaryGenerator, format_eval
from position_analyzer import PositionAnalyzer


def analyze_game(
    pgn_path: str | Path,
    stockfish_path: str = "/opt/homebrew/bin/stockfish",
    leela_model: str | Path = "models/t78_with_intermediate.onnx",
    svm_dir: str | Path = "models/svm",
    depth: int = 18,
    model: str = "claude-haiku-4-5-20251001",
    comment_threshold: int = 10,  # minimum eval loss (cp) to generate comment
) -> None:
    """Analyze a chess game and print annotated commentary.

    Args:
        pgn_path: Path to PGN file.
        stockfish_path: Path to Stockfish binary.
        leela_model: Path to Leela ONNX model.
        svm_dir: Path to trained SVM directory.
        depth: Stockfish analysis depth.
        model: Claude model to use.
        comment_threshold: Minimum eval loss to trigger a comment.
    """
    # Load game
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print("Error: Could not read PGN file")
        return

    # Print game info
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    event = game.headers.get("Event", "?")
    date = game.headers.get("Date", "?")
    result = game.headers.get("Result", "?")

    print(f"{'='*60}")
    print(f"  {white} — {black}")
    print(f"  {event}, {date}")
    print(f"  Результат: {result}")
    print(f"{'='*60}\n")

    # Initialize analyzers
    with PositionAnalyzer(stockfish_path, leela_model, svm_dir, depth) as analyzer:
        commentator = CommentaryGenerator(model=model)

        board = game.board()
        moves = list(game.mainline_moves())
        history: list[chess.Board] = []

        for i, move in enumerate(moves):
            move_number = (i // 2) + 1
            is_white = (i % 2 == 0)
            prefix = f"{move_number}." if is_white else f"{move_number}..."
            san = board.san(move)

            # Analyze position before the move
            analysis = analyzer.analyze(board, move, history=history[-7:] if history else None)

            eval_loss = analysis.get("eval_loss", 0)
            quality = analysis.get("move_quality", "")
            played_eval = analysis.get("played_eval")

            # Print move
            eval_str = ""
            if played_eval is not None:
                if played_eval > 0:
                    eval_str = f" [+{played_eval/100:.2f}]"
                else:
                    eval_str = f" [{played_eval/100:.2f}]"

            # Quality markers for display
            QUALITY_MARKERS = {
                "brilliant": "!!",
                "great": "!",
                "best": "",
                "good": "",
                "book": "",
                "forced": "",
                "inaccuracy": "?!",
                "mistake": "?",
                "blunder": "??",
            }
            QUALITY_LABELS = {
                "brilliant": " [BRILLIANT]",
                "great": " [GREAT]",
                "book": " [BOOK]",
                "forced": " [FORCED]",
            }
            quality_marker = QUALITY_MARKERS.get(quality, "")
            quality_label = QUALITY_LABELS.get(quality, "")

            # Mate info
            mate_in = analysis.get("played_mate_in")
            mate_str = ""
            if mate_in is not None:
                if mate_in > 0:
                    mate_str = f" [#M{mate_in}]"
                elif mate_in < 0:
                    mate_str = f" [#M{-mate_in}]"

            print(f"{prefix} {san}{quality_marker}{eval_str}{mate_str}{quality_label}")

            # Opening info (print once when opening name appears/changes)
            opening = analysis.get("opening")
            if opening and opening.get("name"):
                if not hasattr(analyze_game, '_last_opening') or analyze_game._last_opening != opening["name"]:
                    analyze_game._last_opening = opening["name"]
                    eco = opening.get("eco", "")
                    print(f"   [{eco}] {opening['name']}")

            # Tablebase info
            tb = analysis.get("tablebase")
            if tb:
                tb_labels = {"win": "выигрыш", "draw": "ничья", "loss": "проигрыш"}
                print(f"   [TB] {tb_labels.get(tb['result'], tb['result'])}, DTZ={tb.get('dtz', '?')}")

            # Generate commentary for interesting moves
            should_comment = False
            if quality in ("inaccuracy", "mistake", "blunder") and eval_loss >= comment_threshold:
                should_comment = True
            elif quality in ("brilliant", "great"):
                should_comment = True
            elif quality == "best" and eval_loss <= 0:
                key_concepts = analysis.get("key_concepts", [])
                if key_concepts and abs(key_concepts[0]["change"]) > 2:
                    should_comment = True

            if should_comment:
                comment = commentator.comment_move(analysis, move_number)
                if comment:
                    for line in _wrap_text(comment, 60):
                        print(f"   {line}")
                    print()

            # Update board and history
            history.append(board.copy())
            board.push(move)

    # Final eval
    print(f"\n{'='*60}")
    print(f"  Результат: {result}")
    print(f"{'='*60}")


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}" if current else word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess Coach: AI-тренер")
    parser.add_argument("pgn", help="Path to PGN file")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish depth")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model")
    parser.add_argument("--threshold", type=int, default=10, help="Min eval loss for comment")
    args = parser.parse_args()

    analyze_game(args.pgn, depth=args.depth, model=args.model, comment_threshold=args.threshold)
