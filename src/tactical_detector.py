"""Detect tactical motifs in chess positions and PV lines.

Algorithmically detects:
  - Fork: piece attacks ≥2 valuable enemy pieces simultaneously
  - Pin: long-range piece attacks a piece that shields a more valuable piece/king
  - Skewer: long-range piece attacks a valuable piece with a less valuable piece behind
  - Discovered attack: piece moves off a line, revealing attack from allied long-range piece

Analyzes both the current position and future positions along PV lines
(typically 4-6 half-moves deep).
"""

import chess

# Piece values for tactical evaluation
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,  # king is infinitely valuable for tactics
}

# Minimum value for a target to be "interesting" in a fork
FORK_MIN_VALUE = 3  # knight/bishop or higher

# Long-range piece types (can pin/skewer)
LONG_RANGE = {chess.BISHOP, chess.ROOK, chess.QUEEN}

# Ray directions for each long-range piece type
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS

PIECE_DIRS = {
    chess.BISHOP: BISHOP_DIRS,
    chess.ROOK: ROOK_DIRS,
    chess.QUEEN: QUEEN_DIRS,
}


def _square_name(sq: int) -> str:
    return chess.square_name(sq)


def _piece_name(board: chess.Board, sq: int) -> str:
    """Get piece name like 'knight d3' or 'king e1'."""
    piece = board.piece_at(sq)
    if piece is None:
        return _square_name(sq)
    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    return f"{names.get(piece.piece_type, '?')} {_square_name(sq)}"


def _ray_scan(board: chess.Board, from_sq: int, direction: tuple[int, int]) -> list[int]:
    """Walk along a ray from a square, return occupied squares encountered (in order)."""
    file, rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    df, dr = direction
    result = []
    f, r = file + df, rank + dr
    while 0 <= f <= 7 and 0 <= r <= 7:
        sq = chess.square(f, r)
        if board.piece_at(sq) is not None:
            result.append(sq)
        f += df
        r += dr
    return result


# ─── Fork Detection ──────────────────────────────────────────────


def detect_forks(
    board: chess.Board,
    move: chess.Move | None = None,
    side: chess.Color | None = None,
) -> list[dict]:
    """Detect forks after a move (or in current position if move is None).

    A fork is when a single piece attacks ≥2 enemy pieces of value ≥ FORK_MIN_VALUE.

    Args:
        board: Position to check.
        move: If given, push this move first.
        side: Which side's pieces to check for forks. If None, checks both sides.
    """
    if move is not None:
        board = board.copy()
        board.push(move)

    forks = []
    colors_to_check = [side] if side is not None else [chess.WHITE, chess.BLACK]

    for attacker_color in colors_to_check:
        for sq in board.piece_map():
            piece = board.piece_at(sq)
            if piece is None or piece.color != attacker_color:
                continue

            # Get all squares this piece attacks
            attacks = board.attacks(sq)
            targets = []
            for target_sq in attacks:
                target_piece = board.piece_at(target_sq)
                if target_piece is None or target_piece.color == attacker_color:
                    continue
                val = PIECE_VALUE.get(target_piece.piece_type, 0)
                if val >= FORK_MIN_VALUE:
                    targets.append({
                        "square": _square_name(target_sq),
                        "piece": _piece_name(board, target_sq),
                        "value": val,
                    })

            if len(targets) >= 2:
                # It's a fork! Prioritize by combined target value
                targets.sort(key=lambda t: t["value"], reverse=True)
                forks.append({
                    "type": "fork",
                    "attacker": _piece_name(board, sq),
                    "attacker_square": _square_name(sq),
                    "targets": targets,
                })

    return forks


# ─── Pin Detection ────────────────────────────────────────────────


def detect_pins(board: chess.Board) -> list[dict]:
    """Detect pins in the current position.

    A pin: long-range piece attacks enemy piece A, and behind A on the same ray
    sits a more valuable piece B (or king). A cannot move without exposing B.
    """
    pins = []

    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type not in LONG_RANGE:
            continue

        pinner_color = piece.color
        directions = PIECE_DIRS[piece.piece_type]

        for d in directions:
            hits = _ray_scan(board, sq, d)
            if len(hits) < 2:
                continue

            first = board.piece_at(hits[0])
            second = board.piece_at(hits[1])
            if first is None or second is None:
                continue

            # Both must be enemy pieces
            if first.color == pinner_color or second.color == pinner_color:
                continue

            first_val = PIECE_VALUE.get(first.piece_type, 0)
            second_val = PIECE_VALUE.get(second.piece_type, 0)

            # Pin: second piece is more valuable (or is king)
            if second_val > first_val:
                pins.append({
                    "type": "pin",
                    "pinner": _piece_name(board, sq),
                    "pinner_square": _square_name(sq),
                    "pinned": _piece_name(board, hits[0]),
                    "pinned_square": _square_name(hits[0]),
                    "behind": _piece_name(board, hits[1]),
                    "behind_square": _square_name(hits[1]),
                })

    return pins


# ─── Skewer Detection ─────────────────────────────────────────────


def detect_skewers(board: chess.Board) -> list[dict]:
    """Detect skewers in the current position.

    A skewer: long-range piece attacks valuable enemy piece A, and behind A
    sits a less valuable piece B. A is forced to move, exposing B.
    """
    skewers = []

    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type not in LONG_RANGE:
            continue

        attacker_color = piece.color
        directions = PIECE_DIRS[piece.piece_type]

        for d in directions:
            hits = _ray_scan(board, sq, d)
            if len(hits) < 2:
                continue

            first = board.piece_at(hits[0])
            second = board.piece_at(hits[1])
            if first is None or second is None:
                continue

            # Both must be enemy
            if first.color == attacker_color or second.color == attacker_color:
                continue

            first_val = PIECE_VALUE.get(first.piece_type, 0)
            second_val = PIECE_VALUE.get(second.piece_type, 0)

            # Skewer: first piece is more valuable, second is less
            # (opposite of pin — the valuable piece is in front)
            if first_val > second_val and first_val >= FORK_MIN_VALUE:
                skewers.append({
                    "type": "skewer",
                    "attacker": _piece_name(board, sq),
                    "attacker_square": _square_name(sq),
                    "front": _piece_name(board, hits[0]),
                    "front_square": _square_name(hits[0]),
                    "behind": _piece_name(board, hits[1]),
                    "behind_square": _square_name(hits[1]),
                })

    return skewers


# ─── Discovered Attack Detection ──────────────────────────────────


def detect_discovered_attacks(board_before: chess.Board, move: chess.Move) -> list[dict]:
    """Detect discovered attacks created by a move.

    A discovered attack: piece X moves off a line between allied long-range
    piece Y and enemy valuable piece Z. Now Y attacks Z directly.
    """
    discoveries = []
    mover_color = board_before.turn
    from_sq = move.from_square

    board_after = board_before.copy()
    board_after.push(move)

    # For each friendly long-range piece, check if `from_sq` was blocking a ray
    for sq in board_before.piece_map():
        piece = board_before.piece_at(sq)
        if piece is None or piece.color != mover_color:
            continue
        if piece.piece_type not in LONG_RANGE:
            continue
        if sq == from_sq:
            continue  # the mover itself

        directions = PIECE_DIRS[piece.piece_type]

        for d in directions:
            # Check if from_sq is on this ray
            hits_before = _ray_scan(board_before, sq, d)
            if from_sq not in hits_before:
                continue

            # from_sq was the FIRST piece on this ray (blocking)
            if hits_before[0] != from_sq:
                continue

            # After the move, check what's now on this ray
            hits_after = _ray_scan(board_after, sq, d)
            if not hits_after:
                continue

            target = board_after.piece_at(hits_after[0])
            if target is None or target.color == mover_color:
                continue

            target_val = PIECE_VALUE.get(target.piece_type, 0)
            if target_val < FORK_MIN_VALUE:
                continue

            # Check if this is a discovered CHECK
            is_check = (target.piece_type == chess.KING)

            discoveries.append({
                "type": "discovered_check" if is_check else "discovered_attack",
                "mover": _piece_name(board_before, from_sq),
                "mover_from": _square_name(from_sq),
                "mover_to": _square_name(move.to_square),
                "revealer": _piece_name(board_after, sq),
                "revealer_square": _square_name(sq),
                "target": _piece_name(board_after, hits_after[0]),
                "target_square": _square_name(hits_after[0]),
            })

    return discoveries


# ─── Analyze PV Line ──────────────────────────────────────────────


def analyze_pv_tactics(
    board: chess.Board,
    pv_moves: list[chess.Move],
    max_depth: int = 6,
) -> list[dict]:
    """Analyze a PV line for tactical motifs at each step.

    Returns list of tactics found, each annotated with the ply depth
    at which it occurs.
    """
    all_tactics = []
    sim_board = board.copy()

    for ply, move in enumerate(pv_moves[:max_depth]):
        if move not in sim_board.legal_moves:
            break

        # Detect discovered attacks BEFORE pushing
        discoveries = detect_discovered_attacks(sim_board, move)
        for t in discoveries:
            t["ply"] = ply + 1
            all_tactics.append(t)

        # Push the move
        sim_board.push(move)

        # Detect forks (the piece that just moved creates a fork)
        forks = detect_forks(sim_board)
        for t in forks:
            t["ply"] = ply + 1
            all_tactics.append(t)

        # Detect pins and skewers in the new position
        pins = detect_pins(sim_board)
        for t in pins:
            t["ply"] = ply + 1
            all_tactics.append(t)

        skewers = detect_skewers(sim_board)
        for t in skewers:
            t["ply"] = ply + 1
            all_tactics.append(t)

    return all_tactics


def detect_current_tactics(board: chess.Board) -> list[dict]:
    """Detect all tactical motifs in the current position (no PV needed)."""
    tactics = []

    # Forks — check all pieces of side to move's OPPONENT
    # (the opponent just moved, so their pieces might be forking)
    forks = detect_forks(board)
    for t in forks:
        t["ply"] = 0
        tactics.append(t)

    pins = detect_pins(board)
    for t in pins:
        t["ply"] = 0
        tactics.append(t)

    skewers = detect_skewers(board)
    for t in skewers:
        t["ply"] = 0
        tactics.append(t)

    return tactics


def filter_significant_tactics(tactics: list[dict]) -> list[dict]:
    """Filter out trivial or redundant tactics, keep the most interesting ones.

    Priority:
    1. Earlier ply > later ply
    2. Higher combined target value
    3. Discovered checks > discovered attacks > forks > skewers > pins
    """
    if not tactics:
        return []

    type_priority = {
        "discovered_check": 5,
        "discovered_attack": 4,
        "fork": 3,
        "skewer": 2,
        "pin": 1,
    }

    def score(t: dict) -> float:
        ply_penalty = t.get("ply", 0) * 10
        type_bonus = type_priority.get(t["type"], 0) * 5

        # Value of targets
        target_value = 0
        if "targets" in t:  # fork
            target_value = sum(tgt["value"] for tgt in t["targets"])
        elif "behind" in t:  # pin/skewer
            # Use piece name to infer value roughly
            behind = t.get("behind", "")
            for pname, val in [("queen", 9), ("rook", 5), ("king", 100),
                               ("bishop", 3), ("knight", 3)]:
                if pname in behind:
                    target_value = val
                    break
        elif "target" in t:  # discovered
            target = t.get("target", "")
            for pname, val in [("queen", 9), ("rook", 5), ("king", 100),
                               ("bishop", 3), ("knight", 3)]:
                if pname in target:
                    target_value = val
                    break

        return type_bonus + target_value - ply_penalty

    scored = [(score(t), t) for t in tactics]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate: same type + same key squares = same tactic (keep earliest ply)
    seen = set()
    result = []
    for _, t in scored:
        # Key ignores ply — so same pin at ply 1, 2, 3 is deduplicated
        key = (
            t["type"],
            t.get("attacker_square", ""),
            t.get("pinner_square", ""),
            t.get("revealer_square", ""),
            # For forks, include sorted target squares
            tuple(sorted(tgt["square"] for tgt in t.get("targets", []))),
            # For pins/skewers, include pinned and behind
            t.get("pinned_square", ""),
            t.get("behind_square", ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(t)

    # Return top 3 most significant
    return result[:3]
