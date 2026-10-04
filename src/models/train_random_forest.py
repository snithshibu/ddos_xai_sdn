from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"

LABEL_ENCODER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "label_encoder.joblib"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_OUT = MODEL_DIR / "random_forest.joblib"


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

TARGET_COLUMN = "label"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training data...")
    train = pd.read_csv(TRAIN_DATA)

    print("Loading testing data...")
    test = pd.read_csv(TEST_DATA)

    print(f"\nTraining rows: {len(train):,}")
    print(f"Testing rows : {len(test):,}")

    # --------------------------------------------------------
    # Prepare features and labels
    # --------------------------------------------------------

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # --------------------------------------------------------
    # Load label names
    # --------------------------------------------------------

    label_encoder = joblib.load(LABEL_ENCODER)

    class_names = list(label_encoder.classes_)

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("[PASS] Model training complete")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RANDOM FOREST RESULTS")
    print("=" * 70)

    print(f"\nAccuracy      : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nTop Feature Importances:")

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    for feature, importance in importances.items():
        print(
            f"  {feature:25s} : {importance:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_OUT
    )

    print("\n[PASS] Model saved:")
    print(MODEL_OUT)

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()