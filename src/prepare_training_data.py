"""Prepare training data for SVM concept classifiers.

Pipeline:
1. Generate diverse chess positions (from random games or Lichess database)
2. For each position: get Leela vector (512-dim) and SF8 concept labels (24 values)
3. Save as numpy arrays for SVM training

For proof of concept, we generate positions by playing random games.
For production, use Lichess eval database positions.
"""

import json
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from concept_extractor import LeelaConceptExtractor
from concept_labeler import label_position


def generate_random_positions(n_positions: int, seed: int = 42) -> list[chess.Board]:
    """Generate diverse positions by playing semi-random games.

    Uses a mix of random moves and popular opening moves to get
    realistic positions across all game phases.
    """
    random.seed(seed)
    positions = []

    while len(positions) < n_positions:
        board = chess.Board()
        move_count = 0
        # Play a random game of 10-80 moves
        target_moves = random.randint(10, 80)

        while not board.is_game_over() and move_count < target_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Bias toward captures and checks for more interesting positions
            captures = [m for m in legal_moves if board.is_capture(m)]
            checks = [m for m in legal_moves if board.gives_check(m)]

            if random.random() < 0.3 and captures:
                move = random.choice(captures)
            elif random.random() < 0.1 and checks:
                move = random.choice(checks)
            else:
                move = random.choice(legal_moves)

            board.push(move)
            move_count += 1

            # Save position every few moves (skip very early game)
            if move_count >= 6 and random.random() < 0.3:
                positions.append(board.copy())
                if len(positions) >= n_positions:
                    break

        if len(positions) % 100 == 0 and len(positions) > 0:
            print(f"  Generated {len(positions)}/{n_positions} positions...")

    return positions[:n_positions]


def positions_from_pgn(pgn_path: str | Path, n_positions: int) -> list[chess.Board]:
    """Extract positions from a PGN file."""
    positions = []
    with open(pgn_path) as f:
        while len(positions) < n_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            moves = list(game.mainline_moves())
            for i, move in enumerate(moves):
                board.push(move)
                if i >= 6 and random.random() < 0.2:
                    positions.append(board.copy())
                    if len(positions) >= n_positions:
                        break

    return positions[:n_positions]


def prepare_data(
    n_positions: int = 5000,
    output_dir: str | Path = "data",
    pgn_path: str | Path | None = None,
) -> None:
    """Prepare training data: Leela vectors + SF8 labels.

    Args:
        n_positions: Number of positions to process.
        output_dir: Directory to save output files.
        pgn_path: Optional PGN file to extract positions from.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate positions
    print(f"Step 1: Generating {n_positions} positions...")
    if pgn_path:
        positions = positions_from_pgn(pgn_path, n_positions)
        print(f"  Got {len(positions)} positions from PGN")
    else:
        positions = generate_random_positions(n_positions)
    print(f"  Done: {len(positions)} positions")

    # Step 2: Extract Leela vectors
    print("Step 2: Extracting Leela vectors...")
    extractor = LeelaConceptExtractor(
        model_path=Path(__file__).parent.parent / "models" / "t78_with_intermediate.onnx"
    )

    vectors = []
    t0 = time.time()
    for i, board in enumerate(positions):
        vec = extractor.get_raw_vector(board)
        vectors.append(vec)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(positions) - i - 1) / rate
            print(f"  [{i + 1}/{len(positions)}] {rate:.1f} pos/s, ~{remaining:.0f}s remaining")

    vectors = np.array(vectors)  # (N, 512)
    print(f"  Done: vectors shape = {vectors.shape}")

    # Step 3: Get SF8 concept labels
    print("Step 3: Getting Stockfish 8 concept labels...")
    all_labels = []
    label_names = None
    t0 = time.time()

    for i, board in enumerate(positions):
        labels = label_position(board)
        if label_names is None:
            label_names = sorted(labels.keys())
        label_vec = [labels[k] for k in label_names]
        all_labels.append(label_vec)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(positions) - i - 1) / rate
            print(f"  [{i + 1}/{len(positions)}] {rate:.1f} pos/s, ~{remaining:.0f}s remaining")

    labels_array = np.array(all_labels)  # (N, 24)
    print(f"  Done: labels shape = {labels_array.shape}")

    # Step 4: Save
    np.save(output_dir / "leela_vectors.npy", vectors)
    np.save(output_dir / "sf8_labels.npy", labels_array)
    with open(output_dir / "label_names.json", "w") as f:
        json.dump(label_names, f)

    # Also save FENs for debugging
    fens = [board.fen() for board in positions]
    with open(output_dir / "positions.json", "w") as f:
        json.dump(fens, f)

    print(f"\nSaved to {output_dir}/:")
    print(f"  leela_vectors.npy: {vectors.shape}")
    print(f"  sf8_labels.npy: {labels_array.shape}")
    print(f"  label_names.json: {len(label_names)} concepts")
    print(f"  positions.json: {len(fens)} FENs")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-positions", type=int, default=5000)
    parser.add_argument("-o", "--output-dir", default="data")
    parser.add_argument("--pgn", default=None, help="Optional PGN file for positions")
    args = parser.parse_args()

    prepare_data(args.n_positions, args.output_dir, args.pgn)
