from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.inference.predictor import DDoSPredictor
from src.inference.evidence_engine import TemporalEvidenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEQUENCE_DIR = DATA_DIR / "sequences"
EXPERIMENT_DIR = DATA_DIR / "experiments"
MODEL_DIR = PROJECT_ROOT / "models"


def check_file(path: Path, description: str) -> None:
    assert path.exists(), f"Missing: {path}"
    print(f"[PASS] {description}")


def main() -> None:

    print("=" * 70)
    print("FINAL OFFLINE PROJECT VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset / preprocessing
    # ------------------------------------------------------

    print("\n[1] DATA & PREPROCESSING")

    required_files = [
        (
            PROCESSED_DIR / "train.csv",
            "Training dataset",
        ),
        (
            PROCESSED_DIR / "test.csv",
            "Test dataset",
        ),
        (
            PROCESSED_DIR / "scaler.joblib",
            "Feature scaler",
        ),
        (
            PROCESSED_DIR / "label_encoder.joblib",
            "Label encoder",
        ),
    ]

    for path, description in required_files:
        check_file(path, description)

    train = pd.read_csv(
        PROCESSED_DIR / "train.csv"
    )

    test = pd.read_csv(
        PROCESSED_DIR / "test.csv"
    )

    assert len(train) > 0
    assert len(test) > 0

    print(
        f"[PASS] Train rows: {len(train):,}"
    )

    print(
        f"[PASS] Test rows: {len(test):,}"
    )

    # ------------------------------------------------------
    # Temporal sequences
    # ------------------------------------------------------

    print("\n[2] TEMPORAL SEQUENCES")

    sequence_files = [
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "feature_columns.txt",
    ]

    for filename in sequence_files:
        check_file(
            SEQUENCE_DIR / filename,
            filename,
        )

    X_train = np.load(
        SEQUENCE_DIR / "X_train.npy"
    )

    X_test = np.load(
        SEQUENCE_DIR / "X_test.npy"
    )

    y_train = np.load(
        SEQUENCE_DIR / "y_train.npy"
    )

    y_test = np.load(
        SEQUENCE_DIR / "y_test.npy"
    )

    assert X_train.ndim == 3
    assert X_test.ndim == 3

    assert X_train.shape[1:] == (
        10,
        15,
    )

    assert X_test.shape[1:] == (
        10,
        15,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print(
        f"[PASS] X_train: {X_train.shape}"
    )

    print(
        f"[PASS] X_test:  {X_test.shape}"
    )

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------

    print("\n[3] TRAINED MODELS")

    model_files = [
        "random_forest.joblib",
        "cnn_baseline.keras",
        "lstm_baseline.keras",
        "cnn_lstm.keras",
    ]

    for filename in model_files:
        check_file(
            MODEL_DIR / filename,
            filename,
        )

    # Load neural models to verify they are valid.
    for filename in [
        "cnn_baseline.keras",
        "lstm_baseline.keras",
        "cnn_lstm.keras",
    ]:

        model = tf.keras.models.load_model(
            MODEL_DIR / filename
        )

        print(
            f"[PASS] Loaded {filename} "
            f"({model.count_params():,} parameters)"
        )

    # ------------------------------------------------------
    # Inference components
    # ------------------------------------------------------

    print("\n[4] REAL-TIME INFERENCE COMPONENTS")

    predictor = DDoSPredictor()

    evidence = TemporalEvidenceEngine()

    assert predictor.sequence_length == 10
    assert evidence.current_action == "OBSERVE"

    print(
        "[PASS] CNN-LSTM predictor initialized"
    )

    print(
        "[PASS] Temporal evidence engine initialized"
    )

    # ------------------------------------------------------
    # Evidence lifecycle
    # ------------------------------------------------------

    print("\n[5] EVIDENCE LIFECYCLE")

    actions = []

    for confidence in [
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]:

        state = evidence.update(
            "SYN_FLOOD",
            confidence,
        )

        actions.append(
            state.action
        )

    assert actions == [
        "OBSERVE",
        "OBSERVE",
        "RATE_LIMIT",
        "RATE_LIMIT",
        "BLOCK",
    ]

    print(
        "[PASS] OBSERVE → RATE_LIMIT → BLOCK"
    )

    recovery = []

    for _ in range(3):

        state = evidence.update(
            "BENIGN",
            0.99,
        )

        recovery.append(
            state.action
        )

    assert recovery == [
        "OBSERVE",
        "OBSERVE",
        "RECOVER",
    ]

    print(
        "[PASS] BLOCK → OBSERVE → RECOVER"
    )

    # ------------------------------------------------------
    # Experiment outputs
    # ------------------------------------------------------

    print("\n[6] EXPERIMENT OUTPUTS")

    index_path = (
        EXPERIMENT_DIR
        / "experiment_index.csv"
    )

    check_file(
        index_path,
        "Experiment index",
    )

    experiment_index = pd.read_csv(
        index_path
    )

    assert len(experiment_index) >= 20

    print(
        f"[PASS] "
        f"{len(experiment_index)} experiment outputs indexed"
    )

    # ------------------------------------------------------
    # Inference benchmark
    # ------------------------------------------------------

    print("\n[7] PERFORMANCE BENCHMARK")

    benchmark_path = (
        EXPERIMENT_DIR
        / "inference_benchmark.csv"
    )

    check_file(
        benchmark_path,
        "Inference benchmark",
    )

    benchmark = pd.read_csv(
        benchmark_path
    ).iloc[0]

    assert benchmark[
        "mean_model_inference_ms"
    ] > 0

    assert benchmark[
        "mean_pipeline_ms"
    ] > 0

    print(
        f"[PASS] Mean model inference: "
        f"{benchmark['mean_model_inference_ms']:.2f} ms"
    )

    print(
        f"[PASS] Mean pipeline decision: "
        f"{benchmark['mean_pipeline_ms']:.2f} ms"
    )

    # ------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------

    print("\n[8] DASHBOARD DATA")

    dashboard_files = [
        (
            "temporal_horizons/"
            "temporal_horizon_results.csv"
        ),
        (
            "mitigation/"
            "mitigation_policy_summary.csv"
        ),
        (
            "shap/"
            "overall_feature_importance.csv"
        ),
    ]

    for relative_path in dashboard_files:

        check_file(
            EXPERIMENT_DIR / relative_path,
            relative_path,
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "[PASS] FINAL OFFLINE VALIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nThe offline framework is ready for "
        "Mininet-Ryu integration."
    )


if __name__ == "__main__":
    main()