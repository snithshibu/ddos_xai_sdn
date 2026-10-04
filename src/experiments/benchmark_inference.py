from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.inference.predictor import DDoSPredictor
from src.inference.evidence_engine import TemporalEvidenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "X_test.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "inference_benchmark.csv"
)


def main() -> None:

    print("=" * 70)
    print("CNN-LSTM INFERENCE BENCHMARK")
    print("=" * 70)

    X_test = np.load(X_TEST_PATH)

    print(
        f"\nTest sequences: {len(X_test)}"
    )

    predictor = DDoSPredictor()

    # ------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------

    print("\nWarming up model...")

    for _ in range(5):

        predictor.model.predict(
            X_test[:1],
            verbose=0,
        )

    # ------------------------------------------------------
    # Direct model inference
    # ------------------------------------------------------

    repetitions = min(
        100,
        len(X_test),
    )

    print(
        f"Benchmarking {repetitions} "
        "individual predictions..."
    )

    inference_times = []

    for index in range(repetitions):

        sample = X_test[
            index:index + 1
        ]

        start = time.perf_counter()

        predictor.model.predict(
            sample,
            verbose=0,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        inference_times.append(
            elapsed * 1000
        )

    inference_times = np.asarray(
        inference_times
    )

    # ------------------------------------------------------
    # Full pipeline benchmark
    # ------------------------------------------------------

    print(
        "\nBenchmarking complete "
        "prediction + evidence pipeline..."
    )

    pipeline_predictor = DDoSPredictor()

    evidence = TemporalEvidenceEngine()

    pipeline_times = []

    # Use one already-scaled sequence and feed
    # observations individually through the predictor.

    for sequence_index in range(
        min(20, len(X_test))
    ):

        pipeline_predictor.reset()
        evidence.reset()

        sequence = X_test[
            sequence_index
        ]

        start = time.perf_counter()

        result = None

        for observation in sequence:

            # Predictor expects raw features.
            # For this benchmark, use the already-scaled
            # observation directly through the model so
            # preprocessing is not measured twice.
            pipeline_predictor.buffer.append(
                observation
            )

        model_input = np.expand_dims(
            np.asarray(
                pipeline_predictor.buffer,
                dtype=np.float32,
            ),
            axis=0,
        )

        probabilities = (
            pipeline_predictor.model.predict(
                model_input,
                verbose=0,
            )[0]
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = (
            pipeline_predictor
            .label_encoder
            .inverse_transform(
                [predicted_index]
            )[0]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        evidence.update(
            predicted_label=predicted_label,
            confidence=confidence,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        pipeline_times.append(
            elapsed * 1000
        )

    pipeline_times = np.asarray(
        pipeline_times
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    mean_inference = float(
        np.mean(inference_times)
    )

    median_inference = float(
        np.median(inference_times)
    )

    p95_inference = float(
        np.percentile(
            inference_times,
            95,
        )
    )

    mean_pipeline = float(
        np.mean(pipeline_times)
    )

    median_pipeline = float(
        np.median(pipeline_times)
    )

    p95_pipeline = float(
        np.percentile(
            pipeline_times,
            95,
        )
    )

    predictions_per_second = (
        1000.0 / mean_inference
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"Mean model inference     : "
        f"{mean_inference:.3f} ms"
    )

    print(
        f"Median model inference   : "
        f"{median_inference:.3f} ms"
    )

    print(
        f"P95 model inference      : "
        f"{p95_inference:.3f} ms"
    )

    print(
        f"Mean pipeline decision   : "
        f"{mean_pipeline:.3f} ms"
    )

    print(
        f"Median pipeline decision : "
        f"{median_pipeline:.3f} ms"
    )

    print(
        f"P95 pipeline decision    : "
        f"{p95_pipeline:.3f} ms"
    )

    print(
        f"Model predictions/sec    : "
        f"{predictions_per_second:.2f}"
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import pandas as pd

    results = pd.DataFrame(
        [
            {
                "mean_model_inference_ms":
                    mean_inference,
                "median_model_inference_ms":
                    median_inference,
                "p95_model_inference_ms":
                    p95_inference,
                "mean_pipeline_ms":
                    mean_pipeline,
                "median_pipeline_ms":
                    median_pipeline,
                "p95_pipeline_ms":
                    p95_pipeline,
                "predictions_per_second":
                    predictions_per_second,
            }
        ]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        "\n[PASS] Inference benchmark completed."
    )


if __name__ == "__main__":
    main()