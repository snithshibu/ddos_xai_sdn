from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd


# IMPORTANT:
# This order MUST remain identical to the order used during training.
FEATURE_COLUMNS = [
    "ip_proto",
    "tp_dst",
    "duration_sec",
    "packet_count",
    "byte_count",
    "pkt_delta",
    "byte_delta",
    "pkts_per_sec",
    "bytes_per_sec",
    "avg_pkt_size",
    "active_flows",
    "inter_arrival_mean",
    "packet_rate_std",
    "byte_rate_std",
    "flow_persistence_sec",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaler.joblib"
)


def validate_features(features: Mapping[str, float]) -> None:
    """
    Verify that all features required by the trained model exist.
    """

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )


def build_feature_vector(
    features: Mapping[str, float],
) -> np.ndarray:
    """
    Convert a network-statistics mapping into the exact
    feature ordering expected by the trained model.

    Returns:
        Array with shape (1, 15).
    """

    validate_features(features)

    values = [
        float(features[column])
        for column in FEATURE_COLUMNS
    ]

    return np.asarray(
        values,
        dtype=np.float32,
    ).reshape(1, -1)


def build_feature_dataframe(
    features: Mapping[str, float],
) -> pd.DataFrame:
    """
    Convert network statistics into a one-row DataFrame.

    Using a DataFrame preserves feature names when applying
    the sklearn scaler.
    """

    validate_features(features)

    return pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def scale_features(
    features: Mapping[str, float],
) -> np.ndarray:
    """
    Build and scale one feature vector using the scaler
    fitted exclusively on the training data.
    """

    dataframe = build_feature_dataframe(features)

    scaler = joblib.load(SCALER_PATH)

    scaled = scaler.transform(dataframe)

    return scaled.astype(np.float32)


def build_scaled_feature_vector(
    features: Mapping[str, float],
) -> np.ndarray:
    """
    Convenience function used by real-time inference.
    """

    return scale_features(features)


if __name__ == "__main__":

    # Small sanity test using one row from the processed
    # test dataset.

    test_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test.csv"
    )

    df = pd.read_csv(test_path)

    sample = df.iloc[0]

    sample_features = {
        column: sample[column]
        for column in FEATURE_COLUMNS
    }

    raw_vector = build_feature_vector(
        sample_features
    )

    scaled_vector = build_scaled_feature_vector(
        sample_features
    )

    print("=" * 60)
    print("FEATURE BUILDER SANITY TEST")
    print("=" * 60)

    print(f"Feature count : {len(FEATURE_COLUMNS)}")
    print(f"Raw shape     : {raw_vector.shape}")
    print(f"Scaled shape  : {scaled_vector.shape}")

    print("\nFeature order:")
    for index, column in enumerate(FEATURE_COLUMNS):
        print(f"{index:2d}: {column}")

    print("\nRaw vector:")
    print(raw_vector)

    print("\nScaled vector:")
    print(scaled_vector)

    assert raw_vector.shape == (1, 15)
    assert scaled_vector.shape == (1, 15)

    print("\n[PASS] Feature builder is ready.")
