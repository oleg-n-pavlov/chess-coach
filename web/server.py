"""Chess Coach web server.

FastAPI backend with WebSocket for real-time engine analysis.
Serves the frontend and provides API for:
- Position analysis (Stockfish + Leela concepts)
- Opening book lookup
- Tablebase probe
- Commentary generation (Claude Haiku)
- Game import from Lichess/Chess.com
- Play against computer (Stockfish with Skill Level)
- Save/load analysis results
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
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

from commentary_generator import CommentaryGenerator
from position_analyzer import PositionAnalyzer

app = FastAPI(title="Chess Coach")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Saved analyses directory
ANALYSES_DIR = Path(__file__).parent.parent / "data" / "analyses"
ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

# Global instances (initialized on first connection)
_analyzer: PositionAnalyzer | None = None
_commentator: CommentaryGenerator | None = None

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
LEELA_MODEL = Path(__file__).parent.parent / "models" / "t78_with_intermediate.onnx"
SVM_DIR = Path(__file__).parent.parent / "models" / "svm"


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = PositionAnalyzer(
            stockfish_path=STOCKFISH_PATH,
            leela_model=str(LEELA_MODEL),
            svm_dir=str(SVM_DIR),
        )
    return _analyzer


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

            elif action == "bot_move":
                result = await asyncio.to_thread(_get_bot_move, data)
                await websocket.send_json({"type": "bot_move", "data": result})

            elif action == "import_lichess":
                result = await asyncio.to_thread(_import_lichess, data)
                await websocket.send_json({"type": "import", "data": result})

            elif action == "import_chesscom":
                result = await asyncio.to_thread(_import_chesscom, data)
                await websocket.send_json({"type": "import", "data": result})

            elif action == "save_analysis":
                result = await asyncio.to_thread(_save_analysis, data)
                await websocket.send_json({"type": "saved", "data": result})

            elif action == "load_analysis":
                result = await asyncio.to_thread(_load_analysis, data)
                await websocket.send_json({"type": "loaded", "data": result})

            elif action == "list_analyses":
                result = await asyncio.to_thread(_list_analyses)
                await websocket.send_json({"type": "analysis_list", "data": result})

    except WebSocketDisconnect:
        pass


def _analyze_position(data: dict) -> dict:
    """Analyze a position using PositionAnalyzer (Stockfish + Leela concepts)."""
    fen = data.get("fen", chess.STARTING_FEN)
    played_uci = data.get("move")  # UCI format, e.g. "e2e4"
    history_fens = data.get("history", [])

    board = chess.Board(fen)
    analyzer = _get_analyzer()

    # Build history
    history = [chess.Board(h) for h in history_fens[-7:]] if history_fens else None

    # Parse played move
    played_move = chess.Move.from_uci(played_uci) if played_uci else None

    # Run analysis
    raw_result = analyzer.analyze(board, played_move, history)

    # Convert to JSON-serializable format
    result = _serialize_analysis(raw_result)
    return result


def _serialize_analysis(raw: dict) -> dict:
    """Convert PositionAnalyzer output to JSON-serializable dict.

    The main issue is that best_moves[i]["move"] is a chess.Move object,
    and concepts_before may contain numpy arrays.
    """
    result = {
        "fen": raw.get("fen", ""),
    }

    # Best moves: convert chess.Move to UCI string
    best_moves = []
    for bm in raw.get("best_moves", []):
        best_moves.append({
            "move": bm["move"].uci() if hasattr(bm["move"], "uci") else str(bm["move"]),
            "san": bm.get("san", ""),
            "eval": bm.get("eval", 0),
            "mate_in": bm.get("mate_in"),
            "pv": bm.get("pv", []),
        })
    result["best_moves"] = best_moves
    result["eval"] = best_moves[0]["eval"] if best_moves else 0
    result["mate_in"] = best_moves[0]["mate_in"] if best_moves else None

    # Concepts (before move)
    concepts = raw.get("concepts_before", {})
    if "raw_vector" in concepts:
        concepts = {}
    result["concepts"] = {k: round(float(v), 3) for k, v in concepts.items()}

    # Opening info
    if raw.get("opening"):
        result["opening"] = raw["opening"]

    # Tablebase
    if raw.get("tablebase"):
        result["tablebase"] = raw["tablebase"]

    # Played move data
    if "played_move" in raw:
        concepts_after = raw.get("concepts_after", {})
        if "raw_vector" in concepts_after:
            concepts_after = {}

        concept_diff = raw.get("concept_diff", {})
        if "raw_vector_diff" in concept_diff:
            concept_diff = {}

        result.update({
            "played_move": raw["played_move"],
            "played_eval": raw.get("played_eval"),
            "played_mate_in": raw.get("played_mate_in"),
            "best_eval": raw.get("best_eval", 0),
            "eval_loss": raw.get("eval_loss", 0),
            "move_quality": raw.get("move_quality", ""),
            "is_best": raw.get("is_best", False),
            "concepts_after": {k: round(float(v), 3) for k, v in concepts_after.items()},
            "concept_diff": {k: round(float(v), 3) for k, v in concept_diff.items()},
            "key_concepts": raw.get("key_concepts", []),
            # Tactical motifs
            "tactics_in_best_line": raw.get("tactics_in_best_line", []),
            "tactics_in_played_line": raw.get("tactics_in_played_line", []),
            "tactics_current": raw.get("tactics_current", []),
        })

    return result


def _generate_comment(data: dict) -> dict:
    """Generate AI commentary for a move."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"comment": "", "error": "ANTHROPIC_API_KEY not set", "move_number": data.get("move_number", 1)}

    analysis = data.get("analysis", {})
    move_number = data.get("move_number", 1)
    lang = data.get("lang", "ru")

    commentator = _get_commentator()
    result = commentator.comment_move(analysis, move_number, lang=lang, force=True)
    # comment_move now returns {"comment": str, "prompt": str}
    return {
        "comment": result["comment"],
        "prompt": result["prompt"],
        "move_number": move_number,
    }


def _get_bot_move(data: dict) -> dict:
    """Get a computer move at a given skill level."""
    fen = data.get("fen", chess.STARTING_FEN)
    skill = data.get("skill", 10)  # 0-20
    time_limit = data.get("time", 1.0)

    board = chess.Board(fen)
    analyzer = _get_analyzer()
    engine = analyzer.engine

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
        return {"games": games, "username": username}
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

        return {"games": games[:max_games], "username": username}
    except Exception as e:
        return {"error": str(e), "games": []}


# ─── Save/Load Analysis ─────────────────────────────────────────

def _save_analysis(data: dict) -> dict:
    """Save analysis results to disk."""
    game_data = data.get("game_data", {})
    if not game_data:
        return {"error": "No game data to save"}

    analysis_id = str(uuid.uuid4())[:8]
    metadata = game_data.get("metadata", {})

    save_data = {
        "id": analysis_id,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata,
        "moves": game_data.get("moves", []),
    }

    filepath = ANALYSES_DIR / f"{analysis_id}.json"
    with open(filepath, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    return {
        "id": analysis_id,
        "title": _analysis_title(metadata),
        "saved_at": save_data["saved_at"],
    }


def _load_analysis(data: dict) -> dict:
    """Load a saved analysis from disk."""
    analysis_id = data.get("id", "")
    filepath = ANALYSES_DIR / f"{analysis_id}.json"

    if not filepath.exists():
        return {"error": f"Analysis {analysis_id} not found"}

    with open(filepath) as f:
        return json.load(f)


def _list_analyses() -> dict:
    """List all saved analyses."""
    analyses = []
    for filepath in sorted(ANALYSES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            analyses.append({
                "id": data.get("id", filepath.stem),
                "title": _analysis_title(metadata),
                "saved_at": data.get("saved_at", ""),
                "white": metadata.get("white", "?"),
                "black": metadata.get("black", "?"),
                "result": metadata.get("result", ""),
                "num_moves": len(data.get("moves", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return {"analyses": analyses}


def _analysis_title(metadata: dict) -> str:
    """Generate a title for a saved analysis."""
    white = metadata.get("white", "?")
    black = metadata.get("black", "?")
    result = metadata.get("result", "")
    return f"{white} vs {black}" + (f" ({result})" if result else "")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
