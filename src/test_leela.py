"""Test: encode a position and run it through the modified Leela ONNX model."""

import chess
import numpy as np
import onnxruntime as ort

from leela_encoder import encode_position


def main():
    model_path = "/Users/ol/Documents/chessy/models/t78_with_intermediate.onnx"

    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path)

    print("Input:", session.get_inputs()[0].name, session.get_inputs()[0].shape)
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Encode starting position
    board = chess.Board()
    input_tensor = encode_position(board)
    print(f"\nInput tensor shape: {input_tensor.shape}")
    print(f"Non-zero planes: {np.count_nonzero(input_tensor.sum(axis=(2, 3)))}/112")

    # Run inference
    print("\nRunning inference...")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    policy, wdl, mlh, intermediate = outputs

    print(f"\nPolicy shape: {policy.shape}")
    print(f"WDL (win/draw/loss): {wdl[0]}")
    print(f"MLH (moves left): {mlh[0]}")
    print(f"Intermediate layer shape: {intermediate.shape}")

    # The intermediate vector stats
    vec = intermediate[0]  # Remove batch dim
    print(f"\nIntermediate vector stats:")
    print(f"  Shape: {vec.shape}")
    print(f"  Mean: {vec.mean():.4f}")
    print(f"  Std:  {vec.std():.4f}")
    print(f"  Min:  {vec.min():.4f}")
    print(f"  Max:  {vec.max():.4f}")

    # Now try after 1.e4
    print("\n--- After 1.e4 ---")
    board_before = board.copy()
    board.push_san("e4")
    input_tensor2 = encode_position(board, history=[board_before])
    outputs2 = session.run(None, {input_name: input_tensor2})
    policy2, wdl2, mlh2, intermediate2 = outputs2
    print(f"WDL: {wdl2[0]}")

    vec2 = intermediate2[0]
    diff = vec2 - vec
    print(f"Vector diff norm: {np.linalg.norm(diff):.4f}")
    print(f"Max change per channel: {np.abs(diff).max():.4f}")

    # Global average pooling (what CCC does before SVM)
    # Reduce spatial dims: (512, 8, 8) -> (512,)
    pooled1 = vec.mean(axis=(1, 2))
    pooled2 = vec2.mean(axis=(1, 2))
    print(f"\nPooled vector (for SVM): shape={pooled1.shape}")
    print(f"Pooled diff norm: {np.linalg.norm(pooled2 - pooled1):.4f}")


if __name__ == "__main__":
    main()
