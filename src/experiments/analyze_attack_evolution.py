from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_ddos_sdn_dataset.csv"
)

PROCESSED_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "experiments"
)


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


ATTACK_SCENARIOS = {
    "ICMP_FLOOD": 1,
    "SLOWLORIS": 2,
    "SYN_FLOOD": 3,
    "UDP_FLOOD": 4,
}


# Same priority as create_sequences.py
ATTACK_PRIORITY = {
    "UDP_FLOOD": 4,
    "SYN_FLOOD": 3,
    "SLOWLORIS": 2,
    "ICMP_FLOOD": 1,
    "BENIGN": 0,
}


# ============================================================
# NETWORK-LEVEL AGGREGATION
# ============================================================

def aggregate_network_observations(df):
    """
    Convert flow-level data into one network-level
    observation per run_id + timestamp.

    The aggregation mirrors create_sequences.py,
    while retaining phase and scenario metadata.
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

    # --------------------------------------------------------
    # Network-level attack label
    # --------------------------------------------------------

    label_by_timestamp = (
        df.groupby(
            ["run_id", "timestamp"]
        )["label"]
        .apply(
            lambda labels: max(
                labels,
                key=lambda x: ATTACK_PRIORITY[str(x)]
            )
        )
        .reset_index(name="label")
    )

    grouped = grouped.merge(
        label_by_timestamp,
        on=["run_id", "timestamp"],
        how="left",
    )

    # --------------------------------------------------------
    # Preserve experimental metadata
    #
    # At each timestamp, retain the scenario and phase.
    # These are NOT model features.
    # --------------------------------------------------------

    metadata = (
        df.groupby(
            ["run_id", "timestamp"]
        )
        .agg(
            scenario=("scenario", "first"),
            phase=("phase", "first"),
        )
        .reset_index()
    )

    grouped = grouped.merge(
        metadata,
        on=["run_id", "timestamp"],
        how="left",
    )

    return grouped


# ============================================================
# CONTINUITY
# ============================================================

def is_contiguous(timestamps):
    """
    Check whether observations are exactly one second apart.
    """

    if len(timestamps) <= 1:
        return True

    deltas = (
        np.diff(timestamps)
        / np.timedelta64(1, "s")
    )

    return bool(
        np.allclose(
            deltas,
            1.0
        )
    )


# ============================================================
# ATTACK EVOLUTION
# ============================================================

def analyze_attack_run(run):
    """
    Analyze one attack run.

    We identify:

        NORMAL
        EARLY_CHANGE
        ESCALATION
        SUSTAINED
        DECAY
        RECOVERY

    and calculate the number of network observations
    belonging to each stage.
    """

    run = (
        run.sort_values("timestamp")
        .reset_index(drop=True)
    )

    scenario_values = (
        run["scenario"]
        .dropna()
        .unique()
        .tolist()
    )

    scenario = (
        scenario_values[0]
        if scenario_values
        else "UNKNOWN"
    )

    # Ignore normal and flash-crowd scenarios here.
    if scenario not in ATTACK_SCENARIOS:
        return None

    result = {
        "run_id": run["run_id"].iloc[0],
        "scenario": scenario,
    }

    # --------------------------------------------------------
    # Phase statistics
    # --------------------------------------------------------

    phase_order = [
        "NORMAL",
        "EARLY_CHANGE",
        "ESCALATION",
        "SUSTAINED",
        "DECAY",
        "RECOVERY",
    ]

    for phase in phase_order:

        subset = run[
            run["phase"] == phase
        ]

        result[
            f"{phase.lower()}_seconds"
        ] = len(subset)

        # Attack observations inside each phase
        attack_subset = subset[
            subset["label"] != 0
        ]

        result[
            f"{phase.lower()}_attack_seconds"
        ] = len(attack_subset)

    # --------------------------------------------------------
    # First attack observation
    # --------------------------------------------------------

    attack_rows = run[
        run["label"] != "BENIGN"
    ]

    if not attack_rows.empty:

        first_attack = attack_rows.iloc[0]

        result["first_attack_time"] = (
            first_attack["timestamp"]
        )

        result["first_attack_phase"] = (
            first_attack["phase"]
        )

        result["first_attack_label"] = (
            str(first_attack["label"])
        )

    else:

        result["first_attack_time"] = None
        result["first_attack_phase"] = None
        result["first_attack_label"] = None

    # --------------------------------------------------------
    # First attack per phase
    # --------------------------------------------------------

    for phase in [
        "EARLY_CHANGE",
        "ESCALATION",
        "SUSTAINED",
    ]:

        subset = run[
            (run["phase"] == phase)
            & (run["label"] != "BENIGN")
        ]

        if not subset.empty:

            result[
                f"{phase.lower()}_first_attack"
            ] = subset.iloc[0]["timestamp"]

            result[
                f"{phase.lower()}_attack_class"
            ] = str(subset.iloc[0]["label"])

        else:

            result[
                f"{phase.lower()}_first_attack"
            ] = None

            result[
                f"{phase.lower()}_attack_class"
            ] = None

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ATTACK EVOLUTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load exact test split
    # --------------------------------------------------------

    print("\nLoading processed test split...")

    processed_test = pd.read_csv(
        PROCESSED_TEST
    )

    exact_test_runs = set(
        processed_test["run_id"]
        .unique()
    )

    print(
        f"Exact test runs: "
        f"{len(exact_test_runs)}"
    )

    print(
        f"Expected test rows: "
        f"{len(processed_test):,}"
    )

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    print("\nLoading raw dataset...")

    raw = pd.read_csv(
        RAW_DATA
    )

    # --------------------------------------------------------
    # Recover exact test population
    # --------------------------------------------------------

    test = raw[
        raw["run_id"].isin(
            exact_test_runs
        )
    ].copy()

    print(
        f"Matched raw rows: "
        f"{len(test):,}"
    )

    if len(test) != len(processed_test):

        raise RuntimeError(
            "Exact test population mismatch!"
        )

    print(
        "[PASS] Exact test population confirmed"
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print(
        "\nAggregating into network-level "
        "one-second observations..."
    )

    network_df = (
        aggregate_network_observations(
            test
        )
    )

    print(
        f"Network observations: "
        f"{len(network_df):,}"
    )

    # --------------------------------------------------------
    # Analyze attack runs
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK RUN ANALYSIS")
    print("=" * 70)

    results = []

    for run_id, run in network_df.groupby(
        "run_id",
        sort=True
    ):

        result = analyze_attack_run(
            run
        )

        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print(
            "\n[ERROR] No attack runs found."
        )

        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nAttack runs found: "
        f"{len(results_df)}"
    )

    print("\nRun summary:")

    summary_columns = [
        "run_id",
        "scenario",
        "normal_seconds",
        "early_change_seconds",
        "escalation_seconds",
        "sustained_seconds",
        "decay_seconds",
        "recovery_seconds",
        "first_attack_phase",
        "first_attack_label",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Aggregate phase statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE STATISTICS BY ATTACK TYPE")
    print("=" * 70)

    for scenario in ATTACK_SCENARIOS:

        subset = results_df[
            results_df["scenario"]
            == scenario
        ]

        if subset.empty:
            continue

        print(
            f"\n{scenario}"
        )

        for phase in [
            "normal",
            "early_change",
            "escalation",
            "sustained",
            "decay",
            "recovery",
        ]:

            column = (
                f"{phase}_seconds"
            )

            attack_column = (
                f"{phase}_attack_seconds"
            )

            if column not in subset.columns:
                continue

            print(
                f"  {phase:13s}: "
                f"mean={subset[column].mean():.2f}s | "
                f"min={subset[column].min()}s | "
                f"max={subset[column].max()}s | "
                f"attack={subset[attack_column].sum()}s"
            )

    # --------------------------------------------------------
    # Attack onset analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK ONSET ANALYSIS")
    print("=" * 70)

    onset_columns = [
        "run_id",
        "scenario",
        "first_attack_time",
        "first_attack_phase",
        "first_attack_label",
    ]

    print(
        results_df[
            onset_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "attack_evolution_analysis.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {output_file}"
    )

    print("\n" + "=" * 70)
    print(
        "ATTACK EVOLUTION ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()