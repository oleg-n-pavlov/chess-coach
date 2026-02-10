"""Encode chess positions into Leela Chess Zero's 112-plane input format.

Leela's input encoding (INPUT_CLASSICAL_112_PLANE):
  - 13 planes per position × 8 positions (current + 7 history) = 104 planes
  - 8 auxiliary planes = 8 planes
  - Total: 112 planes, each 8×8

Per position (13 planes):
  Planes 0-5:   Our pieces (pawn, knight, bishop, rook, queen, king)
  Planes 6-11:  Their pieces (pawn, knight, bishop, rook, queen, king)
  Plane 12:     Repetition count (1 if position repeated, 0 otherwise)

Auxiliary planes (8 planes):
  Plane 104:    Castling us queenside
  Plane 105:    Castling us kingside
  Plane 106:    Castling them queenside
  Plane 107:    Castling them kingside
  Plane 108:    Side to move (all 1s if black to move, for flipping)
  Plane 109:    50-move rule counter (normalized to [0,1])
  Plane 110:    Zeros (unused)
  Plane 111:    Ones (unused)

IMPORTANT: Leela always sees the board from the perspective of the side to move.
If it's black's turn, the board is flipped vertically.
"""

import chess
import numpy as np


PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def _board_to_planes(board: chess.Board) -> np.ndarray:
    """Convert a single board state to 13 planes (current position only).

    Board is always from the perspective of the side to move.
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us
    flip = us == chess.BLACK

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Determine position (flip if black to move)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if flip:
            rank = 7 - rank

        plane_offset = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == us:
            planes[plane_offset, rank, file] = 1.0
        else:
            planes[6 + plane_offset, rank, file] = 1.0

    # Plane 12: repetitions
    # Count how many times this position has occurred
    if board.is_repetition(2):
        planes[12, :, :] = 1.0

    return planes


def encode_position(
    board: chess.Board,
    history: list[chess.Board] | None = None,
) -> np.ndarray:
    """Encode a chess position with history into Leela's 112-plane format.

    Args:
        board: Current board state.
        history: List of up to 7 previous board states (most recent first).
                 If None or shorter than 7, missing positions are zero-filled.

    Returns:
        numpy array of shape (1, 112, 8, 8) ready for ONNX inference.
    """
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # Current position: planes 0-12
    planes[0:13] = _board_to_planes(board)

    # History positions: planes 13-103 (7 previous positions, 13 planes each)
    if history:
        for i, hist_board in enumerate(history[:7]):
            offset = 13 * (i + 1)
            planes[offset : offset + 13] = _board_to_planes(hist_board)

    # Auxiliary planes
    us = board.turn
    flip = us == chess.BLACK

    # Castling rights (from our perspective)
    if us == chess.WHITE:
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[104, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[105, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[106, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[107, :, :] = 1.0
    else:
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[104, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[105, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[106, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[107, :, :] = 1.0

    # Side to move (all 1s if black - but Leela always sees from side-to-move,
    # so this plane encodes whether the board was flipped)
    if flip:
        planes[108, :, :] = 1.0

    # 50-move rule counter (normalized)
    planes[109, :, :] = board.halfmove_clock / 100.0

    # Plane 110: zeros (already zero)
    # Plane 111: ones
    planes[111, :, :] = 1.0

    return planes[np.newaxis, :, :, :]  # Add batch dimension


def boards_from_game(game_moves: list[chess.Move]) -> list[chess.Board]:
    """Given a list of moves, return list of board states after each move.

    Returns boards[0] = starting position, boards[1] = after first move, etc.
    """
    boards = []
    board = chess.Board()
    boards.append(board.copy())
    for move in game_moves:
        board.push(move)
        boards.append(board.copy())
    return boards
