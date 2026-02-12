"""Extract concept vectors from Leela Chess Zero's intermediate layer.

Uses the modified ONNX model with intermediate output from block39.
After SVM classifiers are trained, maps raw vectors to 22 chess concepts.
"""

from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from leela_encoder import encode_position


# 22 concepts from CCC paper (Stockfish 8 eval terms)
# For each concept there are white and black variants
CONCEPTS = [
    "material_white",
    "material_black",
    "pawns_white",
    "pawns_black",
    "knights_white",
    "knights_black",
    "bishops_white",
    "bishops_black",
    "rooks_white",
    "rooks_black",
    "queens_white",
    "queens_black",
    "king_safety_white",
    "king_safety_black",
    "threats_white",
    "threats_black",
    "passed_pawns_white",
    "passed_pawns_black",
    "space_white",
    "space_black",
    "mobility_white",
    "mobility_black",
]


class LeelaConceptExtractor:
    """Extract concept vectors from chess positions using Leela ONNX model."""

    def __init__(
        self,
        model_path: str | Path = "models/t78_with_intermediate.onnx",
        svm_dir: str | Path | None = None,
    ):
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

        # Load SVM classifiers if available
        self.svms: dict[str, object] | None = None
        if svm_dir and Path(svm_dir).exists():
            self._load_svms(Path(svm_dir))

    def _load_svms(self, svm_dir: Path) -> None:
        """Load pre-trained SVM classifiers and scaler for concept scoring."""
        import joblib

        self.svms = {}
        self.scaler = None

        # Load the StandardScaler used during training
        scaler_path = svm_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        for concept in CONCEPTS:
            path = svm_dir / f"{concept}.joblib"
            if path.exists():
                self.svms[concept] = joblib.load(path)

    def get_raw_vector(
        self,
        board: chess.Board,
        history: list[chess.Board] | None = None,
    ) -> np.ndarray:
        """Get the raw 512-dim vector from Leela's layer 40.

        Returns global-average-pooled vector: (512,)
        """
        input_tensor = encode_position(board, history)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # outputs[3] is the intermediate layer: (1, 512, 8, 8)
        raw = outputs[3][0]  # Remove batch dim -> (512, 8, 8)

        # Global average pooling -> (512,)
        pooled = raw.mean(axis=(1, 2))
        return pooled

    def get_leela_eval(
        self,
        board: chess.Board,
        history: list[chess.Board] | None = None,
    ) -> dict:
        """Get Leela's WDL evaluation alongside the raw vector."""
        input_tensor = encode_position(board, history)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        wdl = outputs[1][0]  # (3,) = [win, draw, loss]
        mlh = outputs[2][0]  # (1,) = moves left head
        raw = outputs[3][0]  # (512, 8, 8)
        pooled = raw.mean(axis=(1, 2))

        return {
            "vector": pooled,
            "wdl": {"win": float(wdl[0]), "draw": float(wdl[1]), "loss": float(wdl[2])},
            "mlh": float(mlh[0]),
        }

    def get_concepts(
        self,
        board: chess.Board,
        history: list[chess.Board] | None = None,
    ) -> dict[str, float]:
        """Get concept scores for a position.

        If SVMs are trained: returns concept scores (distance from SVM boundary).
        If not: returns raw vector (for training).
        """
        vec = self.get_raw_vector(board, history)

        if self.svms:
            # Apply the same scaling used during training
            vec_input = vec.reshape(1, -1)
            if self.scaler is not None:
                vec_input = self.scaler.transform(vec_input)

            scores = {}
            for concept in CONCEPTS:
                if concept in self.svms:
                    # decision_function returns distance from boundary
                    # positive = more of this concept, negative = less
                    score = self.svms[concept].decision_function(vec_input)[0]
                    scores[concept] = float(score)
            return scores
        else:
            # Without SVMs, return the raw vector as a dict
            return {"raw_vector": vec}

    def get_concept_diff(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        history_before: list[chess.Board] | None = None,
        history_after: list[chess.Board] | None = None,
    ) -> dict[str, float]:
        """Get the change in concepts between two positions (before/after a move).

        Positive values mean the concept increased for the side that just moved.
        """
        concepts_before = self.get_concepts(board_before, history_before)
        concepts_after = self.get_concepts(board_after, history_after)

        if "raw_vector" in concepts_before:
            # SVMs not trained yet - return vector diff
            diff_vec = concepts_after["raw_vector"] - concepts_before["raw_vector"]
            return {"raw_vector_diff": diff_vec, "diff_norm": float(np.linalg.norm(diff_vec))}

        diff = {}
        for concept in CONCEPTS:
            if concept in concepts_before and concept in concepts_after:
                diff[concept] = concepts_after[concept] - concepts_before[concept]

        return diff
