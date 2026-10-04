from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn_lstm.keras"
)

X_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "X_test.npy"
)

Y_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "y_test.npy"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "feature_columns.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "shap"
)


def load_features() -> list[str]:
    with open(FEATURE_PATH, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def main() -> None:

    print("=" * 70)
    print("CNN-LSTM SHAP EXPLAINABILITY")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Load model and data
    # ------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    feature_names = load_features()

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"Feature count: {len(feature_names)}"
    )

    assert X_test.shape[2] == len(feature_names)

    # ------------------------------------------------------
    # Select representative background data
    # ------------------------------------------------------

    background_size = min(
        50,
        len(X_test),
    )

    explanation_size = min(
        20,
        len(X_test),
    )

    background = X_test[
        :background_size
    ]

    samples = X_test[
        :explanation_size
    ]

    print(
        f"\nBackground samples: "
        f"{len(background)}"
    )

    print(
        f"Explanation samples: "
        f"{len(samples)}"
    )

    # ------------------------------------------------------
    # Create SHAP explainer
    # ------------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.GradientExplainer(
        model,
        background,
    )

    print("Computing SHAP values...")

    shap_values = explainer.shap_values(
        samples
    )

    # ------------------------------------------------------
    # Handle SHAP output format
    # ------------------------------------------------------

    if isinstance(shap_values, list):

        # Older SHAP format:
        # list[class] -> samples x time x features

        shap_array = np.stack(
            shap_values,
            axis=-1,
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

    print(
        f"SHAP output shape: "
        f"{shap_array.shape}"
    )

    # ------------------------------------------------------
    # Aggregate across time
    # ------------------------------------------------------

    # We want feature-level importance.
    #
    # Original model input:
    #
    # samples × time × features
    #
    # Aggregate absolute SHAP values over:
    #
    # samples + time

    if shap_array.ndim == 4:

        # samples × time × features × classes

        feature_importance = np.mean(
            np.abs(shap_array),
            axis=(0, 1),
        )

        # Shape:
        # features × classes

    elif shap_array.ndim == 3:

        # samples × time × features

        feature_importance = np.mean(
            np.abs(shap_array),
            axis=(0, 1),
        )

        feature_importance = (
            feature_importance[:, np.newaxis]
        )

    else:

        raise RuntimeError(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )

    # ------------------------------------------------------
    # Save global feature importance
    # ------------------------------------------------------

    labels = [
        "BENIGN",
        "ICMP_FLOOD",
        "SLOWLORIS",
        "SYN_FLOOD",
        "UDP_FLOOD",
    ]

    rows = []

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        for class_index in range(
            feature_importance.shape[1]
        ):

            rows.append(
                {
                    "feature": feature_name,
                    "class": labels[class_index],
                    "mean_abs_shap": float(
                        feature_importance[
                            feature_index,
                            class_index,
                        ]
                    ),
                }
            )

    results = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # ------------------------------------------------------
    # Overall feature importance
    # ------------------------------------------------------

    overall = (
        results
        .groupby("feature")[
            "mean_abs_shap"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    overall_path = (
        OUTPUT_DIR
        / "overall_feature_importance.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print("\nTOP FEATURES")
    print("-" * 70)

    for _, row in overall.head(10).iterrows():

        print(
            f"{row['feature']:<25} "
            f"{row['mean_abs_shap']:.6f}"
        )

    print("\n" + "=" * 70)

    print(
        "Saved:"
        f"\n  {output_path}"
        f"\n  {overall_path}"
    )

    print(
        "\n[PASS] SHAP explainability experiment completed."
    )


if __name__ == "__main__":
    main()