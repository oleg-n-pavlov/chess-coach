"""Train SVM classifiers to map Leela vectors to chess concepts.

For each concept, we train a Linear SVM:
  - Input: 512-dim Leela vector (from layer 40, global avg pooled)
  - Output: concept score (distance from SVM boundary)

Training approach (following CCC paper):
  - For each concept, take top 5% and bottom 5% positions by SF8 score
  - Train binary SVM: top 5% = positive, bottom 5% = negative
  - The SVM normal vector IS the concept vector
  - decision_function gives continuous concept score for any position
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def train_concept_svms(
    data_dir: str | Path = "data",
    output_dir: str | Path = "models/svm",
    percentile: float = 5.0,
) -> dict[str, float]:
    """Train SVM classifiers for all concepts.

    Args:
        data_dir: Directory with training data.
        output_dir: Directory to save trained models.
        percentile: Top/bottom percentile for positive/negative labels.

    Returns:
        Dict mapping concept name to test accuracy.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    vectors = np.load(data_dir / "leela_vectors.npy")  # (N, 512)
    labels = np.load(data_dir / "sf8_labels.npy")  # (N, n_concepts)
    with open(data_dir / "label_names.json") as f:
        label_names = json.load(f)

    print(f"Loaded {vectors.shape[0]} positions, {len(label_names)} concepts")
    print(f"Vector dim: {vectors.shape[1]}")

    # Normalize vectors
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    results = {}
    for i, concept_name in enumerate(label_names):
        concept_scores = labels[:, i]

        # Skip concepts with zero variance (e.g., all positions have same value)
        if concept_scores.std() < 1e-6:
            print(f"  {concept_name}: SKIPPED (zero variance)")
            continue

        # Top and bottom percentile
        high_threshold = np.percentile(concept_scores, 100 - percentile)
        low_threshold = np.percentile(concept_scores, percentile)

        # Need distinct thresholds
        if high_threshold <= low_threshold:
            print(f"  {concept_name}: SKIPPED (thresholds overlap)")
            continue

        high_mask = concept_scores >= high_threshold
        low_mask = concept_scores <= low_threshold

        X_high = vectors_scaled[high_mask]
        X_low = vectors_scaled[low_mask]

        if len(X_high) < 10 or len(X_low) < 10:
            print(f"  {concept_name}: SKIPPED (too few samples: {len(X_high)}/{len(X_low)})")
            continue

        # Create binary labels
        X = np.vstack([X_high, X_low])
        y = np.array([1] * len(X_high) + [0] * len(X_low))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Linear SVM
        svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
        svm.fit(X_train, y_train)

        # Evaluate
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[concept_name] = acc

        # Save model
        joblib.dump(svm, output_dir / f"{concept_name}.joblib")

        n_total = len(X)
        print(f"  {concept_name}: accuracy={acc:.3f} (n={n_total}, train={len(X_train)}, test={len(X_test)})")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Trained {len(results)}/{len(label_names)} concepts")
    if results:
        accs = list(results.values())
        print(f"Mean accuracy: {np.mean(accs):.3f}")
        print(f"Min accuracy:  {min(accs):.3f} ({min(results, key=results.get)})")
        print(f"Max accuracy:  {max(accs):.3f} ({max(results, key=results.get)})")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"label_names": label_names, "accuracies": results}, f, indent=2)

    return results


if __name__ == "__main__":
    train_concept_svms()
