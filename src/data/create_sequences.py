from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"

SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"

TRAIN_X_OUT = SEQUENCE_DIR / "X_train.npy"
TRAIN_Y_OUT = SEQUENCE_DIR / "y_train.npy"

TEST_X_OUT = SEQUENCE_DIR / "X_test.npy"
TEST_Y_OUT = SEQUENCE_DIR / "y_test.npy"


# Number of consecutive one-second network observations
SEQUENCE_LENGTH = 10


# ============================================================
# MODEL FEATURES
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


# ============================================================
# LABEL PRIORITY
# ============================================================
#
# If multiple traffic classes are present during the same
# network observation, the most security-relevant active
# attack class gets priority.
#
# 4 = UDP_FLOOD
# 3 = SYN_FLOOD
# 2 = SLOWLORIS
# 1 = ICMP_FLOOD
# 0 = BENIGN
#
# We construct this dynamically from the actual labels.
# ============================================================

ATTACK_PRIORITY = {
    4: 4,  # UDP_FLOOD
    3: 3,  # SYN_FLOOD
    2: 2,  # SLOWLORIS
    1: 1,  # ICMP_FLOOD
    0: 0,  # BENIGN
}


# ============================================================
# NETWORK-LEVEL AGGREGATION
# ============================================================

def aggregate_network_observations(df):
    """
    Convert flow-level observations into network-level
    one-second observations.

    Each row in the returned dataframe represents:

        one run + one timestamp

    rather than one individual flow.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["run_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Numeric aggregation
    # --------------------------------------------------------

    grouped = (
        df.groupby(
            ["run_id", "timestamp"],
            sort=False
        )
        .agg(
            ip_proto=("ip_proto", "mean"),
            tp_dst=("tp_dst", "mean"),

            duration_sec=("duration_sec", "mean"),
            packet_count=("packet_count", "sum"),
            byte_count=("byte_count", "sum"),

            pkt_delta=("pkt_delta", "sum"),
            byte_delta=("byte_delta", "sum"),

            pkts_per_sec=("pkts_per_sec", "sum"),
            bytes_per_sec=("bytes_per_sec", "sum"),

            avg_pkt_size=("avg_pkt_size", "mean"),

            active_flows=("flow_id", "nunique"),

            inter_arrival_mean=(
                "inter_arrival_mean",
                "mean"
            ),

            packet_rate_std=(
                "packet_rate_std",
                "mean"
            ),

            byte_rate_std=(
                "byte_rate_std",
                "mean"
            ),

            flow_persistence_sec=(
                "flow_persistence_sec",
                "mean"
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Determine network-level label
    # --------------------------------------------------------
    #
    # If an attack is present at a timestamp, assign the
    # highest-priority active attack label.
    #
    # Otherwise the observation is BENIGN.
    # --------------------------------------------------------

    label_by_timestamp = (
        df.groupby(
            ["run_id", "timestamp"]
        )[TARGET_COLUMN]
        .apply(
            lambda labels: max(
                labels,
                key=lambda x: ATTACK_PRIORITY[int(x)]
            )
        )
        .reset_index(name=TARGET_COLUMN)
    )

    grouped = grouped.merge(
        label_by_timestamp,
        on=["run_id", "timestamp"],
        how="left",
    )

    return grouped


# ============================================================
# TEMPORAL SEQUENCE CREATION
# ============================================================

def create_sequences(df):

    network_df = aggregate_network_observations(df)

    X_sequences = []
    y_sequences = []

    # --------------------------------------------------------
    # Process every run independently
    # --------------------------------------------------------

    for run_id, run in network_df.groupby(
        "run_id",
        sort=False
    ):

        run = run.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Ensure we have consecutive observations
        # ----------------------------------------------------

        if len(run) < SEQUENCE_LENGTH:
            continue

        features = run[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        labels = run[
            TARGET_COLUMN
        ].to_numpy(
            dtype=np.int64
        )

        timestamps = run[
            "timestamp"
        ].to_numpy()

        # ----------------------------------------------------
        # Sliding temporal windows
        # ----------------------------------------------------

        for start in range(
            len(run) - SEQUENCE_LENGTH + 1
        ):

            end = start + SEQUENCE_LENGTH

            # ------------------------------------------------
            # Verify one-second temporal continuity
            # ------------------------------------------------

            window_times = timestamps[start:end]

            deltas = (
                np.diff(
                    window_times
                )
                / np.timedelta64(1, "s")
            )

            if not np.allclose(
                deltas,
                1.0
            ):
                continue

            sequence = features[start:end]

            # Final timestamp determines the sequence label.
            label = labels[end - 1]

            X_sequences.append(sequence)
            y_sequences.append(label)

    X = np.asarray(
        X_sequences,
        dtype=np.float32
    )

    y = np.asarray(
        y_sequences,
        dtype=np.int64
    )

    return X, y


# ============================================================
# REPORT
# ============================================================

def report_dataset(name, X, y):

    print(f"\n{name}:")

    print(
        f"  Sequences : {len(X):,}"
    )

    print(
        f"  Shape     : {X.shape}"
    )

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print("  Classes:")

    class_names = {
        0: "BENIGN",
        1: "ICMP_FLOOD",
        2: "SLOWLORIS",
        3: "SYN_FLOOD",
        4: "UDP_FLOOD",
    }

    for label in range(5):

        count = counts[
            np.where(
                unique == label
            )[0][0]
        ] if label in unique else 0

        print(
            f"    {label}: "
            f"{class_names[label]:12s} "
            f"{count:,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NETWORK-LEVEL TEMPORAL SEQUENCE CONSTRUCTION")
    print("=" * 70)

    print(
        f"\nSequence length: "
        f"{SEQUENCE_LENGTH} seconds"
    )

    print(
        "\nTemporal unit: "
        "run_id + timestamp"
    )

    print(
        "Sequences never cross run boundaries."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading training data...")
    train = pd.read_csv(TRAIN_DATA)

    print("Loading testing data...")
    test = pd.read_csv(TEST_DATA)

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Testing rows : {len(test):,}"
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    print(
        "\nAggregating training traffic "
        "into network-level observations..."
    )

    X_train, y_train = create_sequences(
        train
    )

    print(
        "Aggregating testing traffic "
        "into network-level observations..."
    )

    X_test, y_test = create_sequences(
        test
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    expected_features = len(
        FEATURE_COLUMNS
    )

    assert X_train.ndim == 3
    assert X_test.ndim == 3

    assert (
        X_train.shape[1]
        == SEQUENCE_LENGTH
    )

    assert (
        X_test.shape[1]
        == SEQUENCE_LENGTH
    )

    assert (
        X_train.shape[2]
        == expected_features
    )

    assert (
        X_test.shape[2]
        == expected_features
    )

    assert (
        len(X_train)
        == len(y_train)
    )

    assert (
        len(X_test)
        == len(y_test)
    )

    print(
        "\n[PASS] Sequence dimensions valid"
    )

    print(
        "[PASS] Feature dimensions valid"
    )

    print(
        "[PASS] Sequence/label counts match"
    )

    # --------------------------------------------------------
    # Check all classes
    # --------------------------------------------------------

    train_classes = set(
        np.unique(y_train)
    )

    test_classes = set(
        np.unique(y_test)
    )

    expected_classes = {
        0, 1, 2, 3, 4
    }

    if train_classes != expected_classes:
        print(
            "\n[WARNING] Training set does not "
            "contain every class."
        )

    else:
        print(
            "[PASS] All five classes present "
            "in training sequences"
        )

    if test_classes != expected_classes:
        print(
            "[WARNING] Testing set does not "
            "contain every class."
        )

    else:
        print(
            "[PASS] All five classes present "
            "in testing sequences"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    SEQUENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        TRAIN_X_OUT,
        X_train
    )

    np.save(
        TRAIN_Y_OUT,
        y_train
    )

    np.save(
        TEST_X_OUT,
        X_test
    )

    np.save(
        TEST_Y_OUT,
        y_test
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    report_dataset(
        "TRAINING SEQUENCES",
        X_train,
        y_train
    )

    report_dataset(
        "TESTING SEQUENCES",
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save feature metadata
    # --------------------------------------------------------

    feature_file = (
        SEQUENCE_DIR
        / "feature_columns.txt"
    )

    with open(
        feature_file,
        "w"
    ) as f:

        for feature in FEATURE_COLUMNS:
            f.write(
                feature + "\n"
            )

    print("\nSaved:")
    print(f"  {TRAIN_X_OUT}")
    print(f"  {TRAIN_Y_OUT}")
    print(f"  {TEST_X_OUT}")
    print(f"  {TEST_Y_OUT}")
    print(f"  {feature_file}")

    print("\n" + "=" * 70)
    print(
        "NETWORK-LEVEL SEQUENCE "
        "CONSTRUCTION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()