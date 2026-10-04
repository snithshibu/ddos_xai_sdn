from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "synthetic_ddos_sdn_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_OUT = PROCESSED_DIR / "train.csv"
TEST_OUT = PROCESSED_DIR / "test.csv"
SCALER_OUT = PROCESSED_DIR / "scaler.joblib"
LABEL_ENCODER_OUT = PROCESSED_DIR / "label_encoder.joblib"


# ============================================================
# FEATURES
# ============================================================

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
GROUP_COLUMN = "run_id"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DDoS DATASET PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(f"\nLoading dataset:")
    print(RAW_DATA)

    df = pd.read_csv(RAW_DATA)

    print(f"Loaded {len(df):,} rows.")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, GROUP_COLUMN]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Select required data
    # --------------------------------------------------------

    data = df[
        FEATURE_COLUMNS
        + [TARGET_COLUMN, GROUP_COLUMN, "flow_id", "timestamp"]
    ].copy()

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = data.isnull().sum().sum()

    if missing_values != 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    print("\n[PASS] No missing values")

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    data[TARGET_COLUMN] = label_encoder.fit_transform(
        data[TARGET_COLUMN]
    )

    print("\nLabel mapping:")

    for index, label in enumerate(label_encoder.classes_):
        print(f"  {index}: {label}")

    # --------------------------------------------------------
    # Grouped train/test split
    # --------------------------------------------------------

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    groups = data[GROUP_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    # --------------------------------------------------------
    # Verify group separation
    # --------------------------------------------------------

    train_runs = set(train[GROUP_COLUMN])
    test_runs = set(test[GROUP_COLUMN])

    overlap = train_runs.intersection(test_runs)

    if overlap:
        raise RuntimeError(
            f"Run leakage detected: {overlap}"
        )

    print("\n[PASS] No run_id overlap between train and test")

    # --------------------------------------------------------
    # Scale numerical features
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # --------------------------------------------------------

    scaler = StandardScaler()

    train[FEATURE_COLUMNS] = scaler.fit_transform(
        train[FEATURE_COLUMNS]
    )

    test[FEATURE_COLUMNS] = scaler.transform(
        test[FEATURE_COLUMNS]
    )

    print("[PASS] Feature scaler fitted only on training data")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    joblib.dump(
        scaler,
        SCALER_OUT
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_OUT
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"\nTraining rows : {len(train):,}")
    print(f"Testing rows  : {len(test):,}")

    print(f"\nTraining runs : {len(train_runs)}")
    print(f"Testing runs  : {len(test_runs)}")

    print("\nTraining class distribution:")
    print(
        train[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTesting class distribution:")
    print(
        test[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print(f"  {TRAIN_OUT}")
    print(f"  {TEST_OUT}")
    print(f"  {SCALER_OUT}")
    print(f"  {LABEL_ENCODER_OUT}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()