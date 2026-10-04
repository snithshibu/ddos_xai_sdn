from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import tensorflow as tf

from .feature_builder import (
    FEATURE_COLUMNS,
    build_scaled_feature_vector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn_lstm.keras"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "label_encoder.joblib"
)

SEQUENCE_LENGTH = 10


class DDoSPredictor:
    """
    Real-time CNN-LSTM inference engine.

    Maintains a rolling 10-observation temporal buffer
    and produces a prediction whenever the buffer is full.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        label_encoder_path: Path = LABEL_ENCODER_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.sequence_length = sequence_length

        print(f"Loading model: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        print(
            f"Loading label encoder: "
            f"{label_encoder_path}"
        )
        self.label_encoder = joblib.load(
            label_encoder_path
        )

        self.buffer = deque(
            maxlen=self.sequence_length
        )

    def reset(self) -> None:
        """
        Clear the temporal observation buffer.
        """

        self.buffer.clear()

    def add_observation(
        self,
        features: Mapping[str, float],
    ) -> dict | None:
        """
        Add one network observation.

        Returns None until enough observations have
        accumulated to form a complete sequence.

        Once the buffer contains 10 observations,
        returns the CNN-LSTM prediction.
        """

        scaled = build_scaled_feature_vector(
            features
        )

        # scaled shape: (1, 15)
        observation = scaled[0]

        self.buffer.append(observation)

        if len(self.buffer) < self.sequence_length:
            return None

        sequence = np.asarray(
            self.buffer,
            dtype=np.float32,
        )

        # Add batch dimension:
        # (10, 15) -> (1, 10, 15)
        model_input = np.expand_dims(
            sequence,
            axis=0,
        )

        probabilities = self.model.predict(
            model_input,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = (
            self.label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        class_probabilities = {
            label: float(probability)
            for label, probability in zip(
                self.label_encoder.classes_,
                probabilities,
            )
        }

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "sequence_length": len(self.buffer),
        }


def load_predictor() -> DDoSPredictor:
    """
    Convenience function for external modules.
    """

    return DDoSPredictor()


if __name__ == "__main__":

    import pandas as pd

    TEST_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test.csv"
    )

    print("=" * 70)
    print("CNN-LSTM REAL-TIME PREDICTOR SANITY TEST")
    print("=" * 70)

    df = pd.read_csv(TEST_PATH)

    predictor = DDoSPredictor()

    # Feed the first 10 observations.
    # These are raw feature rows from the processed test set.
    predictions = []

    for index in range(SEQUENCE_LENGTH):

        row = df.iloc[index]

        features = {
            column: row[column]
            for column in FEATURE_COLUMNS
        }

        result = predictor.add_observation(
            features
        )

        print(
            f"Observation {index + 1:02d}/"
            f"{SEQUENCE_LENGTH}: ",
            "prediction ready"
            if result is not None
            else "buffering",
        )

        if result is not None:
            predictions.append(result)

    print("\n" + "-" * 70)

    if not predictions:
        raise RuntimeError(
            "No prediction was produced."
        )

    result = predictions[-1]

    print(
        f"Predicted class : "
        f"{result['predicted_label']}"
    )

    print(
        f"Confidence      : "
        f"{result['confidence']:.6f}"
    )

    print("\nClass probabilities:")

    for label, probability in (
        result["class_probabilities"].items()
    ):
        print(
            f"  {label:<15} "
            f"{probability:.6f}"
        )

    print(
        f"\nSequence length : "
        f"{result['sequence_length']}"
    )

    assert result["sequence_length"] == 10
    assert result["predicted_label"] in predictor.label_encoder.classes_

    print("\n[PASS] CNN-LSTM predictor is ready.")


