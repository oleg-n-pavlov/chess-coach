"""Demo: analyze a game showing Stockfish eval + concept analysis.

Outputs structured analysis that can be fed to any LLM for commentary.
"""

import json
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent))

from position_analyzer import PositionAnalyzer


CONCEPT_NAMES_RU = {
    "king_safety_white": "безоп. бел. короля",
    "king_safety_black": "безоп. чёрн. короля",
    "mobility_white": "мобильность белых",
    "mobility_black": "мобильность чёрных",
    "space_white": "пространство белых",
    "space_black": "пространство чёрных",
    "threats_white": "угрозы белых",
    "threats_black": "угрозы чёрных",
    "passed_pawns_white": "проходные белых",
    "passed_pawns_black": "проходные чёрных",
    "bishops_white": "слоны белых",
    "bishops_black": "слоны чёрных",
    "knights_white": "кони белых",
    "knights_black": "кони чёрных",
    "rooks_white": "ладьи белых",
    "rooks_black": "ладьи чёрных",
    "queens_white": "ферзь белых",
    "queens_black": "ферзь чёрных",
}


def analyze_game_demo(pgn_path: str, depth: int = 16) -> list[dict]:
    """Analyze a game and return structured data."""
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print("Error: Could not read PGN")
        return []

    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    event = game.headers.get("Event", "?")
    result = game.headers.get("Result", "?")

    print(f"\n{'='*70}")
    print(f"  {white} — {black}")
    print(f"  {event}")
    print(f"  Результат: {result}")
    print(f"{'='*70}\n")

    results = []

    with PositionAnalyzer(
        stockfish_path="/opt/homebrew/bin/stockfish",
        leela_model="models/t78_with_intermediate.onnx",
        svm_dir="models/svm",
        depth=depth,
    ) as analyzer:
        board = game.board()
        moves = list(game.mainline_moves())
        history: list[chess.Board] = []

        for i, move in enumerate(moves):
            move_number = (i // 2) + 1
            is_white = (i % 2 == 0)
            prefix = f"{move_number}." if is_white else f"{move_number}..."
            san = board.san(move)

            analysis = analyzer.analyze(board, move, history=history[-7:] if history else None)

            eval_loss = analysis.get("eval_loss", 0)
            quality = analysis.get("move_quality", "")
            played_eval = analysis.get("played_eval")

            # Format eval
            if played_eval is not None:
                eval_str = f"{played_eval/100:+.2f}"
            else:
                eval_str = "?"

            # Quality marker
            marker = {"blunder": "??", "mistake": "?", "inaccuracy": "?!"}.get(quality, "")

            # Print basic move info
            print(f"{prefix:8s} {san}{marker:3s}  [{eval_str}]", end="")

            if quality in ("inaccuracy", "mistake", "blunder"):
                best = analysis["best_moves"][0] if analysis.get("best_moves") else None
                if best:
                    print(f"  (лучше: {best['san']} [{best['eval']/100:+.2f}])", end="")

            print()

            # Show concept changes for interesting moves
            key_concepts = analysis.get("key_concepts", [])
            if key_concepts and (eval_loss >= 15 or any(abs(c["change"]) > 1.5 for c in key_concepts)):
                for kc in key_concepts[:3]:
                    name = CONCEPT_NAMES_RU.get(kc["concept"], kc["concept"])
                    change = kc["change"]
                    arrow = "↑" if change > 0 else "↓"
                    print(f"         {arrow} {name}: {change:+.2f}")
                print()

            results.append({
                "move_number": move_number,
                "side": "white" if is_white else "black",
                "san": san,
                "quality": quality,
                "eval": played_eval,
                "eval_loss": eval_loss,
                "key_concepts": key_concepts,
            })

            history.append(board.copy())
            board.push(move)

    return results


if __name__ == "__main__":
    pgn = sys.argv[1] if len(sys.argv) > 1 else "examples/immortal_game.pgn"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    results = analyze_game_demo(pgn, depth)

    # Save for later LLM processing
    with open("data/last_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nАнализ сохранён в data/last_analysis.json")
