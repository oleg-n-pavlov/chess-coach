"""Chess Coach web server.

FastAPI backend with WebSocket for real-time engine analysis.
Serves the frontend and provides API for:
- Position analysis (Stockfish + Leela concepts)
- Opening book lookup
- Tablebase probe
- Commentary generation (Claude Haiku)
- Game import from Lichess/Chess.com
- Play against computer (Stockfish with Skill Level)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import chess
import chess.engine
import chess.pgn
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from commentary_generator import CommentaryGenerator, format_eval, format_concept_changes
from concept_extractor import LeelaConceptExtractor
from opening_book import OpeningBook
from tablebase import Tablebase
from position_analyzer import (
    _score_to_cp, _score_to_mate, _classify_move, _is_sacrifice,
    _top_concept_changes, PIECE_VALUES,
)

app = FastAPI(title="Chess Coach")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global engine instances (initialized on first connection)
_engine: chess.engine.SimpleEngine | None = None
_concept_extractor: LeelaConceptExtractor | None = None
_opening_book = OpeningBook()
_tablebase = Tablebase()
_commentator: CommentaryGenerator | None = None

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
LEELA_MODEL = Path(__file__).parent.parent / "models" / "t78_with_intermediate.onnx"
SVM_DIR = Path(__file__).parent.parent / "models" / "svm"


def _get_engine():
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    return _engine


def _get_concept_extractor():
    global _concept_extractor
    if _concept_extractor is None:
        _concept_extractor = LeelaConceptExtractor(str(LEELA_MODEL), str(SVM_DIR))
    return _concept_extractor


def _get_commentator(model: str = "claude-haiku-4-5-20251001"):
    global _commentator
    if _commentator is None:
        _commentator = CommentaryGenerator(model=model)
    return _commentator


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "analyze":
                result = await asyncio.to_thread(_analyze_position, data)
                # Pass through bulk_index for bulk analysis routing
                if "bulk_index" in data:
                    result["_bulk_index"] = data["bulk_index"]
                await websocket.send_json({"type": "analysis", "data": result})

            elif action == "comment":
                try:
                    result = await asyncio.to_thread(_generate_comment, data)
                except Exception as exc:
                    result = {"comment": f"[Error: {exc}]", "move_number": data.get("move_number", 1)}
                await websocket.send_json({"type": "comment", "data": result})

            elif action == "opening":
                result = await asyncio.to_thread(_lookup_opening, data)
                await websocket.send_json({"type": "opening", "data": result})

            elif action == "tablebase":
                result = await asyncio.to_thread(_probe_tablebase, data)
                await websocket.send_json({"type": "tablebase", "data": result})

            elif action == "bot_move":
                result = await asyncio.to_thread(_get_bot_move, data)
                await websocket.send_json({"type": "bot_move", "data": result})

            elif action == "import_lichess":
                result = await asyncio.to_thread(_import_lichess, data)
                await websocket.send_json({"type": "import", "data": result})

            elif action == "import_chesscom":
                result = await asyncio.to_thread(_import_chesscom, data)
                await websocket.send_json({"type": "import", "data": result})

    except WebSocketDisconnect:
        pass


def _analyze_position(data: dict) -> dict:
    """Analyze a position: Stockfish eval + concepts."""
    fen = data.get("fen", chess.STARTING_FEN)
    played_uci = data.get("move")  # UCI format, e.g. "e2e4"
    history_fens = data.get("history", [])
    depth = data.get("depth", 18)

    board = chess.Board(fen)
    engine = _get_engine()

    # Build history
    history = []
    for h_fen in history_fens[-7:]:
        history.append(chess.Board(h_fen))

    # Stockfish analysis
    sf_info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)

    best_moves = []
    for info in sf_info:
        pv = info.get("pv", [])
        score = info.get("score")
        if pv and score:
            pv_san = []
            pv_board = board.copy()
            for m in pv[:5]:
                try:
                    pv_san.append(pv_board.san(m))
                    pv_board.push(m)
                except (AssertionError, ValueError):
                    break
            best_moves.append({
                "move": pv[0].uci(),
                "san": board.san(pv[0]),
                "eval": _score_to_cp(score, board.turn),
                "mate_in": _score_to_mate(score, board.turn),
                "pv": pv_san if pv_san else [board.san(pv[0])],
            })

    # Concepts
    extractor = _get_concept_extractor()
    concepts = extractor.get_concepts(board, history if history else None)
    # Remove raw_vector if present
    if "raw_vector" in concepts:
        concepts = {}

    result = {
        "fen": fen,
        "best_moves": best_moves,
        "concepts": concepts,
        "eval": best_moves[0]["eval"] if best_moves else 0,
        "mate_in": best_moves[0]["mate_in"] if best_moves else None,
    }

    # If a move was played, classify it
    if played_uci:
        played_move = chess.Move.from_uci(played_uci)
        played_san = board.san(played_move)

        played_eval = None
        played_mate = None
        is_best = False

        for bm in best_moves:
            if bm["move"] == played_uci:
                played_eval = bm["eval"]
                played_mate = bm["mate_in"]
                is_best = (bm == best_moves[0])
                break

        if played_eval is None:
            board_after = board.copy()
            board_after.push(played_move)
            info = engine.analyse(board_after, chess.engine.Limit(depth=depth))
            score = info.get("score")
            if score:
                played_eval = _score_to_cp(score, board_after.turn)

        best_eval = best_moves[0]["eval"] if best_moves else 0
        eval_loss = (best_eval - played_eval) if played_eval is not None else 0

        # Concepts after
        board_after = board.copy()
        board_after.push(played_move)
        concepts_after = extractor.get_concepts(board_after, [board.copy()] + history[:6])
        if "raw_vector" in concepts_after:
            concepts_after = {}

        concept_diff = {}
        for key in concepts:
            if key in concepts_after:
                concept_diff[key] = round(concepts_after[key] - concepts[key], 3)

        # Opening info for classification
        opening_info = _opening_book.lookup(board)

        move_quality = _classify_move(
            board, played_move, eval_loss, is_best,
            # Convert best_moves back to internal format for _classify_move
            [{"move": chess.Move.from_uci(bm["move"]), "eval": bm["eval"]} for bm in best_moves],
            opening_info, played_mate,
        )

        result.update({
            "played_move": played_san,
            "played_eval": played_eval,
            "played_mate_in": played_mate,
            "eval_loss": eval_loss,
            "move_quality": move_quality,
            "is_best": is_best,
            "concepts_after": concepts_after,
            "concept_diff": concept_diff,
            "key_concepts": _top_concept_changes(concept_diff),
        })

    # Opening
    opening = _opening_book.lookup(board)
    if opening:
        result["opening"] = opening

    # Tablebase
    tb = _tablebase.probe(board)
    if tb:
        result["tablebase"] = tb

    return result


def _generate_comment(data: dict) -> dict:
    """Generate AI commentary for a move."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"comment": "", "error": "ANTHROPIC_API_KEY not set", "move_number": data.get("move_number", 1)}

    analysis = data.get("analysis", {})
    move_number = data.get("move_number", 1)
    lang = data.get("lang", "ru")

    commentator = _get_commentator()
    comment = commentator.comment_move(analysis, move_number, lang=lang, force=True)
    return {"comment": comment, "move_number": move_number}


def _lookup_opening(data: dict) -> dict:
    """Look up opening information."""
    fen = data.get("fen", chess.STARTING_FEN)
    board = chess.Board(fen)
    info = _opening_book.lookup(board)
    return info or {}


def _probe_tablebase(data: dict) -> dict:
    """Probe endgame tablebase."""
    fen = data.get("fen", chess.STARTING_FEN)
    board = chess.Board(fen)
    info = _tablebase.probe(board)
    return info or {}


def _get_bot_move(data: dict) -> dict:
    """Get a computer move at a given skill level."""
    fen = data.get("fen", chess.STARTING_FEN)
    skill = data.get("skill", 10)  # 0-20
    time_limit = data.get("time", 1.0)

    board = chess.Board(fen)
    engine = _get_engine()

    # Set skill level
    engine.configure({"Skill Level": skill})

    result = engine.play(board, chess.engine.Limit(time=time_limit))

    # Reset skill level
    engine.configure({"Skill Level": 20})

    move = result.move
    return {
        "move": move.uci(),
        "san": board.san(move),
    }


def _import_lichess(data: dict) -> dict:
    """Import games from Lichess for a user."""
    username = data.get("username", "")
    max_games = data.get("max", 20)

    if not username:
        return {"error": "Username required", "games": []}

    try:
        resp = requests.get(
            f"https://lichess.org/api/games/user/{username}",
            params={"max": max_games, "pgnInJson": True, "opening": True},
            headers={"Accept": "application/x-ndjson"},
            timeout=15,
        )
        if resp.status_code != 200:
            return {"error": f"Lichess API error: {resp.status_code}", "games": []}

        games = []
        for line in resp.text.strip().split("\n"):
            if line:
                game = json.loads(line)
                games.append({
                    "id": game.get("id", ""),
                    "white": game.get("players", {}).get("white", {}).get("user", {}).get("name", "?"),
                    "black": game.get("players", {}).get("black", {}).get("user", {}).get("name", "?"),
                    "white_rating": game.get("players", {}).get("white", {}).get("rating"),
                    "black_rating": game.get("players", {}).get("black", {}).get("rating"),
                    "result": game.get("winner", "draw"),
                    "pgn": game.get("pgn", ""),
                    "opening": game.get("opening", {}).get("name", ""),
                    "speed": game.get("speed", ""),
                    "date": game.get("createdAt", ""),
                })
        return {"games": games}
    except Exception as e:
        return {"error": str(e), "games": []}


def _import_chesscom(data: dict) -> dict:
    """Import games from Chess.com for a user."""
    username = data.get("username", "")
    max_games = data.get("max", 20)

    if not username:
        return {"error": "Username required", "games": []}

    try:
        # Get archives list
        resp = requests.get(
            f"https://api.chess.com/pub/player/{username}/games/archives",
            headers={"User-Agent": "ChessCoach/1.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return {"error": f"Chess.com API error: {resp.status_code}", "games": []}

        archives = resp.json().get("archives", [])
        if not archives:
            return {"error": "No games found", "games": []}

        # Get latest archive
        games = []
        for archive_url in reversed(archives[-3:]):  # Last 3 months
            resp = requests.get(
                archive_url,
                headers={"User-Agent": "ChessCoach/1.0"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            for game in resp.json().get("games", []):
                pgn = game.get("pgn", "")
                games.append({
                    "id": game.get("url", "").split("/")[-1] if game.get("url") else "",
                    "white": game.get("white", {}).get("username", "?"),
                    "black": game.get("black", {}).get("username", "?"),
                    "white_rating": game.get("white", {}).get("rating"),
                    "black_rating": game.get("black", {}).get("rating"),
                    "result": game.get("white", {}).get("result", ""),
                    "pgn": pgn,
                    "speed": game.get("time_class", ""),
                    "date": game.get("end_time", ""),
                })
                if len(games) >= max_games:
                    break
            if len(games) >= max_games:
                break

        return {"games": games[:max_games]}
    except Exception as e:
        return {"error": str(e), "games": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
