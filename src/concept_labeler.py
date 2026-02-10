"""Label chess positions with concept scores using Stockfish 8 eval trace.

Stockfish 8 is the last version with classical eval that provides
detailed breakdown of evaluation into individual terms:
  Material, Imbalance, Pawns, Knights, Bishops, Rooks, Queens,
  Mobility, King safety, Threats, Passed pawns, Space.

Each term has White/Black values for MG (middlegame) and EG (endgame).
We use this as ground truth labels to train SVM classifiers that map
Leela's internal representations to human-interpretable concepts.
"""

import re
import subprocess
from pathlib import Path

import chess

SF8_PATH = Path(__file__).parent.parent / "models" / "stockfish8"

# Concept names that map to Stockfish 8 eval terms
# For each SF8 term we create white and black concept labels
EVAL_TERMS = [
    "Material",
    "Imbalance",
    "Pawns",
    "Knights",
    "Bishop",
    "Rooks",
    "Queens",
    "Mobility",
    "King safety",
    "Threats",
    "Passed pawns",
    "Space",
]

# We use a combined MG+EG score (weighted average)
# Phase is approximated from material on the board


def _phase(board: chess.Board) -> float:
    """Estimate game phase (0 = endgame, 1 = middlegame).

    Based on remaining material (excluding pawns and kings).
    """
    phase_total = 24  # 2N + 2B + 4R + 2Q = 2*1 + 2*1 + 4*2 + 2*4 = 16... using SF weights
    # SF8 phase weights: knight=1, bishop=1, rook=2, queen=4
    phase = 0
    for color in [chess.WHITE, chess.BLACK]:
        phase += len(board.pieces(chess.KNIGHT, color)) * 1
        phase += len(board.pieces(chess.BISHOP, color)) * 1
        phase += len(board.pieces(chess.ROOK, color)) * 2
        phase += len(board.pieces(chess.QUEEN, color)) * 4
    return min(phase / phase_total, 1.0)


def _parse_eval_line(line: str) -> dict | None:
    """Parse a single line of Stockfish 8 eval trace output.

    Example line:
    '       Material |   ---   --- |   ---   --- |  0.00  0.00'
    '        Knights |  0.13  0.00 |  0.13  0.00 |  0.00  0.00'
    """
    # Match lines with the eval table format
    match = re.match(
        r"\s+([\w\s]+?)\s*\|"
        r"\s*([-\d.]+|---)\s+([-\d.]+|---)\s*\|"
        r"\s*([-\d.]+|---)\s+([-\d.]+|---)\s*\|"
        r"\s*([-\d.]+|---)\s+([-\d.]+|---)",
        line,
    )
    if not match:
        return None

    term = match.group(1).strip()
    if term not in EVAL_TERMS and term != "Total":
        return None

    def parse_val(s: str) -> float:
        if s == "---":
            return 0.0
        return float(s)

    return {
        "term": term,
        "white_mg": parse_val(match.group(2)),
        "white_eg": parse_val(match.group(3)),
        "black_mg": parse_val(match.group(4)),
        "black_eg": parse_val(match.group(5)),
        "total_mg": parse_val(match.group(6)),
        "total_eg": parse_val(match.group(7)),
    }


def get_sf8_eval_trace(fen: str, sf8_path: str | Path = SF8_PATH) -> dict[str, dict]:
    """Run Stockfish 8 eval trace on a position and parse the results.

    Returns dict mapping term name to {white_mg, white_eg, black_mg, black_eg, ...}
    """
    cmd = f'echo "position fen {fen}\neval\nquit" | {sf8_path}'
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=10
    )
    output = result.stdout

    terms = {}
    for line in output.split("\n"):
        parsed = _parse_eval_line(line)
        if parsed and parsed["term"] != "Total":
            terms[parsed["term"]] = parsed

    return terms


def label_position(board: chess.Board, sf8_path: str | Path = SF8_PATH) -> dict[str, float]:
    """Compute concept labels for a position using Stockfish 8 eval trace.

    Returns dict with 22 concept scores (11 per side).
    Each score is a phase-weighted combination of MG and EG values.
    """
    fen = board.fen()
    terms = get_sf8_eval_trace(fen, sf8_path)
    phase = _phase(board)

    labels = {}
    term_to_concept = {
        "Material": "material",
        "Imbalance": "imbalance",
        "Pawns": "pawns",
        "Knights": "knights",
        "Bishop": "bishops",
        "Rooks": "rooks",
        "Queens": "queens",
        "Mobility": "mobility",
        "King safety": "king_safety",
        "Threats": "threats",
        "Passed pawns": "passed_pawns",
        "Space": "space",
    }

    for sf_term, concept_name in term_to_concept.items():
        if sf_term in terms:
            t = terms[sf_term]
            # Phase-weighted combination of MG and EG
            white_score = t["white_mg"] * phase + t["white_eg"] * (1 - phase)
            black_score = t["black_mg"] * phase + t["black_eg"] * (1 - phase)
            labels[f"{concept_name}_white"] = white_score
            labels[f"{concept_name}_black"] = black_score
        else:
            labels[f"{concept_name}_white"] = 0.0
            labels[f"{concept_name}_black"] = 0.0

    return labels


def label_position_raw(board: chess.Board, sf8_path: str | Path = SF8_PATH) -> dict[str, dict]:
    """Get raw MG/EG scores for all terms (for more detailed analysis)."""
    return get_sf8_eval_trace(board.fen(), sf8_path)
