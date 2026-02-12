"""Prepare training data for SVM concept classifiers.

Pipeline:
1. Generate diverse chess positions (from Lichess games, PGN files, or random)
2. For each position: get Leela vector (512-dim) and SF8 concept labels (24 values)
3. Save as numpy arrays for SVM training

Recommended: Use Lichess API to get real game positions for best quality.
"""

import json
import random
import sys
import time
from io import StringIO
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from concept_extractor import LeelaConceptExtractor
from concept_labeler import label_position


def positions_from_lichess(
    n_positions: int,
    min_rating: int = 2000,
    players: list[str] | None = None,
    seed: int = 42,
) -> list[chess.Board]:
    """Download games from Lichess API and extract positions.

    Uses top players' rated games to get realistic positions
    across all game phases.

    Args:
        n_positions: Target number of positions.
        min_rating: Minimum player rating filter.
        players: List of Lichess usernames. If None, uses default top players.
        seed: Random seed for position sampling.
    """
    random.seed(seed)

    if players is None:
        # Mix of strong players with different styles
        players = [
            "DrNykterstein",   # Magnus Carlsen
            "nihalsarin",      # Nihal Sarin
            "Zhigalko_Sergei", # GM Sergei Zhigalko
            "lance5500",       # GM Alireza Firouzja
            "penguingm1",      # GM Andrew Tang (bullet specialist)
            "opperwezen",      # GM Anish Giri
            "Vladimirovich9000",  # various GM
            "FeegLood",        # GM Daniil Dubov
        ]

    positions = []
    games_per_player = max(50, n_positions // (len(players) * 8) + 10)

    for player in players:
        if len(positions) >= n_positions:
            break

        print(f"  Downloading games from {player}...")
        try:
            resp = requests.get(
                f"https://lichess.org/api/games/user/{player}",
                params={
                    "max": games_per_player,
                    "rated": "true",
                    "perfType": "blitz,rapid,classical",
                },
                headers={"Accept": "application/x-chess-pgn"},
                timeout=30,
                stream=True,
            )
            if resp.status_code != 200:
                print(f"    Error: {resp.status_code}")
                continue

            pgn_text = resp.text
            pgn_io = StringIO(pgn_text)

            game_count = 0
            while len(positions) < n_positions:
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    break

                game_count += 1
                board = game.board()
                moves = list(game.mainline_moves())

                for i, move in enumerate(moves):
                    board.push(move)
                    # Sample positions: skip early opening, sample every ~4th move
                    if i >= 8 and random.random() < 0.25:
                        positions.append(board.copy())
                        if len(positions) >= n_positions:
                            break

            print(f"    Got {game_count} games, {len(positions)} positions total")

        except Exception as e:
            print(f"    Error downloading from {player}: {e}")
            continue

        # Rate limit: Lichess asks for max 1 request per second
        time.sleep(1.5)

    random.shuffle(positions)
    return positions[:n_positions]


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
    source: str = "random",
) -> None:
    """Prepare training data: Leela vectors + SF8 labels.

    Args:
        n_positions: Number of positions to process.
        output_dir: Directory to save output files.
        pgn_path: Optional PGN file to extract positions from.
        source: Position source: "random", "lichess", or "pgn".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate positions
    print(f"Step 1: Getting {n_positions} positions (source: {source})...")
    if source == "lichess":
        positions = positions_from_lichess(n_positions)
    elif source == "pgn" or pgn_path:
        if not pgn_path:
            print("Error: --pgn path required for pgn source")
            return
        positions = positions_from_pgn(pgn_path, n_positions)
        print(f"  Got {len(positions)} positions from PGN")
    else:
        positions = generate_random_positions(n_positions)
    print(f"  Done: {len(positions)} positions")

    if len(positions) == 0:
        print("No positions generated!")
        return

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
    failed = 0

    for i, board in enumerate(positions):
        try:
            labels = label_position(board)
            if label_names is None:
                label_names = sorted(labels.keys())
            label_vec = [labels[k] for k in label_names]
            all_labels.append(label_vec)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"    Warning: failed to label position {i}: {e}")
            # Use zeros for failed positions
            if label_names:
                all_labels.append([0.0] * len(label_names))
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(positions) - i - 1) / rate
            print(f"  [{i + 1}/{len(positions)}] {rate:.1f} pos/s, ~{remaining:.0f}s remaining")

    if failed > 0:
        print(f"  Warning: {failed} positions failed SF8 labeling")

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

    parser = argparse.ArgumentParser(description="Prepare training data for SVM concept classifiers")
    parser.add_argument("-n", "--n-positions", type=int, default=5000)
    parser.add_argument("-o", "--output-dir", default="data")
    parser.add_argument("--pgn", default=None, help="PGN file for positions (with --source pgn)")
    parser.add_argument(
        "--source", default="random", choices=["random", "lichess", "pgn"],
        help="Position source: random (semi-random games), lichess (top player games via API), pgn (from file)"
    )
    args = parser.parse_args()

    prepare_data(args.n_positions, args.output_dir, args.pgn, args.source)
