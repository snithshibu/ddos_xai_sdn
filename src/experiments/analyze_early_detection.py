from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test.csv"


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


# ============================================================
# LABELS
# ============================================================

CLASS_NAMES = {
    0: "BENIGN",
    1: "ICMP_FLOOD",
    2: "SLOWLORIS",
    3: "SYN_FLOOD",
    4: "UDP_FLOOD",
}


ATTACK_CLASSES = {
    1: "ICMP_FLOOD",
    2: "SLOWLORIS",
    3: "SYN_FLOOD",
    4: "UDP_FLOOD",
}


# Same priority used by create_sequences.py
ATTACK_PRIORITY = {
    4: 4,
    3: 3,
    2: 2,
    1: 1,
    0: 0,
}


# ============================================================
# NETWORK-LEVEL AGGREGATION
# ============================================================

def aggregate_network_observations(df):
    """
    Convert flow-level observations into one-second
    network-level observations.

    This mirrors the aggregation used in
    src/data/create_sequences.py.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        ["run_id", "timestamp"]
    ).reset_index(drop=True)

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

    # Determine network-level label
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
# CONTIGUOUS WINDOWS
# ============================================================

def get_contiguous_windows(run, window_lengths):
    """
    Find contiguous temporal windows of different lengths.

    Returns statistics rather than creating model inputs yet.
    """

    results = []

    run = run.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    timestamps = run["timestamp"].to_numpy()
    labels = run[TARGET_COLUMN].to_numpy()

    for length in window_lengths:

        if len(run) < length:
            continue

        for start in range(
            len(run) - length + 1
        ):

            end = start + length

            window_times = timestamps[start:end]

            deltas = (
                np.diff(window_times)
                / np.timedelta64(1, "s")
            )

            # Only accept truly consecutive
            # one-second observations.
            if not np.allclose(
                deltas,
                1.0
            ):
                continue

            window_labels = labels[start:end]

            final_label = int(
                window_labels[-1]
            )

            attack_seconds = int(
                np.sum(window_labels != 0)
            )

            results.append(
                {
                    "run_id": run["run_id"].iloc[0],
                    "start_time": window_times[0],
                    "end_time": window_times[-1],
                    "window_seconds": length,
                    "final_label": final_label,
                    "final_class": CLASS_NAMES[
                        final_label
                    ],
                    "attack_seconds": attack_seconds,
                    "all_benign": bool(
                        np.all(
                            window_labels == 0
                        )
                    ),
                    "contains_attack": bool(
                        np.any(
                            window_labels != 0
                        )
                    ),
                }
            )

    return results


# ============================================================
# ATTACK TRANSITION ANALYSIS
# ============================================================

def analyze_attack_transitions(network_df):
    """
    Find the first timestamp at which each attack class
    becomes active within each run.
    """

    transitions = []

    for run_id, run in network_df.groupby(
        "run_id",
        sort=False
    ):

        run = run.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        previous_label = 0

        for _, row in run.iterrows():

            current_label = int(
                row[TARGET_COLUMN]
            )

            # Detect transition from benign to attack
            if (
                previous_label == 0
                and current_label != 0
            ):

                transitions.append(
                    {
                        "run_id": run_id,
                        "attack_class": CLASS_NAMES[
                            current_label
                        ],
                        "attack_start": row[
                            "timestamp"
                        ],
                    }
                )

            previous_label = current_label

    return pd.DataFrame(transitions)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("EARLY DETECTION DATASET ANALYSIS")
    print("=" * 70)

    print("\nLoading test data...")

    df = pd.read_csv(
        TEST_DATA
    )

    print(
        f"Test rows: {len(df):,}"
    )

    print(
        "\nAggregating flow traffic "
        "into network-level observations..."
    )

    network_df = aggregate_network_observations(
        df
    )

    print(
        f"Network observations: "
        f"{len(network_df):,}"
    )

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NETWORK OBSERVATION SUMMARY")
    print("=" * 70)

    print(
        f"Runs: "
        f"{network_df['run_id'].nunique():,}"
    )

    print(
        f"Time range: "
        f"{network_df['timestamp'].min()} "
        f"→ "
        f"{network_df['timestamp'].max()}"
    )

    print("\nClass distribution:")

    counts = (
        network_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    for label, name in CLASS_NAMES.items():

        print(
            f"  {label}: "
            f"{name:12s} "
            f"{counts.get(label, 0):,}"
        )

    # --------------------------------------------------------
    # Window feasibility
    # --------------------------------------------------------

    WINDOW_LENGTHS = [
        1,
        3,
        5,
        10,
    ]

    print("\n" + "=" * 70)
    print("EARLY-DETECTION WINDOW FEASIBILITY")
    print("=" * 70)

    all_windows = []

    for run_id, run in network_df.groupby(
        "run_id",
        sort=False
    ):

        windows = get_contiguous_windows(
            run,
            WINDOW_LENGTHS
        )

        all_windows.extend(
            windows
        )

    windows_df = pd.DataFrame(
        all_windows
    )

    if windows_df.empty:

        print(
            "\n[ERROR] No contiguous windows found."
        )

        return

    for length in WINDOW_LENGTHS:

        subset = windows_df[
            windows_df["window_seconds"]
            == length
        ]

        print(
            f"\n{length:2d}-second windows:"
        )

        print(
            f"  Total windows : "
            f"{len(subset):,}"
        )

        for label, name in CLASS_NAMES.items():

            count = np.sum(
                subset["final_label"]
                == label
            )

            print(
                f"  {name:12s}: "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Attack-containing windows
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK-CONTAINING WINDOW ANALYSIS")
    print("=" * 70)

    for length in WINDOW_LENGTHS:

        subset = windows_df[
            windows_df["window_seconds"]
            == length
        ]

        print(
            f"\n{length}-second observation:"
        )

        for label, name in ATTACK_CLASSES.items():

            attack_subset = subset[
                subset["final_label"]
                == label
            ]

            if len(attack_subset) == 0:

                print(
                    f"  {name:12s}: "
                    f"NO WINDOWS"
                )

                continue

            attack_seconds = (
                attack_subset[
                    "attack_seconds"
                ]
            )

            print(
                f"  {name:12s}: "
                f"{len(attack_subset):,} windows | "
                f"mean attack evidence = "
                f"{attack_seconds.mean():.2f}s | "
                f"min = {attack_seconds.min()}s | "
                f"max = {attack_seconds.max()}s"
            )

    # --------------------------------------------------------
    # Transition analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK TRANSITION ANALYSIS")
    print("=" * 70)

    transitions = analyze_attack_transitions(
        network_df
    )

    if transitions.empty:

        print(
            "\nNo attack transitions detected."
        )

    else:

        print(
            f"\nDetected attack transitions: "
            f"{len(transitions):,}"
        )

        for name in ATTACK_CLASSES.values():

            subset = transitions[
                transitions["attack_class"]
                == name
            ]

            print(
                f"  {name:12s}: "
                f"{len(subset):,} transitions"
            )

    # --------------------------------------------------------
    # Save analysis data
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "experiments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    windows_output = (
        output_dir
        / "early_detection_windows.csv"
    )

    windows_df.to_csv(
        windows_output,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {windows_output}"
    )

    if not transitions.empty:

        transition_output = (
            output_dir
            / "attack_transitions.csv"
        )

        transitions.to_csv(
            transition_output,
            index=False
        )

        print(
            f"  {transition_output}"
        )

    print("\n" + "=" * 70)
    print(
        "EARLY DETECTION ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()