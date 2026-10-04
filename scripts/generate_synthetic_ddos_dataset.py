import argparse
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

VICTIM_IP = "10.0.0.1"
HOST_IPS = [f"10.0.0.{i}" for i in range(2, 13)]
DPID = 1

OBSERVATION_INTERVAL = 1.0
RUN_DURATION = 120

DEFAULT_RUNS = 150


# ============================================================
# FLOW OBJECT
# ============================================================

class Flow:
    """
    Represents one synthetic SDN/OpenFlow-like flow.

    Each flow has its own:
    - source/destination
    - protocol
    - ports
    - start time
    - duration
    - packet/byte counters
    - short-term rate history
    """

    def __init__(
        self,
        flow_id,
        ip_src,
        ip_dst,
        ip_proto,
        tp_src,
        tp_dst,
        start_time,
        scenario,
        label,
    ):
        self.flow_id = flow_id

        self.ip_src = ip_src
        self.ip_dst = ip_dst

        self.ip_proto = ip_proto
        self.tp_src = tp_src
        self.tp_dst = tp_dst

        self.start_time = start_time

        self.duration_sec = 0
        self.packet_count = 0
        self.byte_count = 0

        self.pkt_rate_history = deque(maxlen=5)
        self.byte_rate_history = deque(maxlen=5)

        self.scenario = scenario
        self.label = label
        self.label_bin = 0 if label == "BENIGN" else 1

        self.is_active = True

    def update(
        self,
        current_time,
        pkt_delta,
        avg_pkt_size,
        persistence_sec,
        phase,
        active_flows,
    ):
        """
        Update the flow for one observation interval.
        """

        self.duration_sec = max(
            0,
            int((current_time - self.start_time).total_seconds())
        )

        pkt_delta = max(0, int(pkt_delta))

        if pkt_delta > 0:
            avg_pkt_size = max(1.0, float(avg_pkt_size))
            byte_delta = max(
                0,
                int(pkt_delta * avg_pkt_size)
            )
        else:
            byte_delta = 0

        self.packet_count += pkt_delta
        self.byte_count += byte_delta

        pkts_per_sec = pkt_delta / OBSERVATION_INTERVAL
        bytes_per_sec = byte_delta / OBSERVATION_INTERVAL

        self.pkt_rate_history.append(pkts_per_sec)
        self.byte_rate_history.append(bytes_per_sec)

        packet_rate_std = float(
            np.std(self.pkt_rate_history)
        )

        byte_rate_std = float(
            np.std(self.byte_rate_history)
        )

        if pkt_delta > 0:
            inter_arrival_mean = (
                OBSERVATION_INTERVAL / pkt_delta
            )
        else:
            # No packets observed during this window.
            # Use the observation interval as a bounded
            # representation rather than introducing NaN.
            inter_arrival_mean = OBSERVATION_INTERVAL

        if pkt_delta > 0:
            actual_avg_pkt_size = (
                byte_delta / pkt_delta
            )
        else:
            actual_avg_pkt_size = 0.0

        return {
            "timestamp": current_time,
            "flow_id": self.flow_id,

            "ip_src": self.ip_src,
            "ip_dst": self.ip_dst,

            "ip_proto": self.ip_proto,
            "tp_src": self.tp_src,
            "tp_dst": self.tp_dst,

            "duration_sec": self.duration_sec,
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,

            "pkt_delta": pkt_delta,
            "byte_delta": byte_delta,

            "pkts_per_sec": pkts_per_sec,
            "bytes_per_sec": bytes_per_sec,

            "avg_pkt_size": actual_avg_pkt_size,

            "active_flows": active_flows,

            "inter_arrival_mean": inter_arrival_mean,

            "packet_rate_std": packet_rate_std,
            "byte_rate_std": byte_rate_std,

            "flow_persistence_sec": persistence_sec,

            "scenario": self.scenario,
            "phase": phase,

            "label": self.label,
            "label_bin": self.label_bin,
        }


# ============================================================
# TEMPORAL PHASE
# ============================================================

def get_phase(t):
    """
    Returns the temporal phase and normalized attack intensity.
    """

    if t < 20:
        return "NORMAL", 0.0

    if 20 <= t < 30:
        intensity = (t - 20) / 10.0
        return "EARLY_CHANGE", intensity

    if 30 <= t < 60:
        intensity = (t - 30) / 30.0
        return "ESCALATION", intensity

    if 60 <= t < 90:
        return "SUSTAINED", 1.0

    if 90 <= t < 110:
        intensity = 1.0 - ((t - 90) / 20.0)
        return "DECAY", max(0.0, intensity)

    return "RECOVERY", 0.0


# ============================================================
# TRAFFIC PROFILE
# ============================================================

def get_traffic_profile(
    scenario,
    is_attacker,
    intensity,
    phase,
):
    """
    Returns a traffic profile for the current source/flow.

    Returns:
        packet_rate
        average_packet_size
        protocol
        destination_port
        new_flow_probability
        kill_probability
        label
    """

    # --------------------------------------------------------
    # DEFAULT BENIGN PROFILE
    # --------------------------------------------------------

    packet_rate = random.uniform(1, 15)

    avg_packet_size = random.uniform(250, 1000)

    protocol = random.choices(
        [6, 17, 1],
        weights=[0.75, 0.20, 0.05],
        k=1,
    )[0]

    if protocol == 6:
        destination_port = random.choice([80, 443])
    elif protocol == 17:
        destination_port = random.choice([53, 123])
    else:
        destination_port = 0

    new_flow_probability = 0.10
    kill_probability = 0.20

    label = "BENIGN"

    # --------------------------------------------------------
    # LEGITIMATE FLASH CROWD
    # --------------------------------------------------------

    if scenario == "FLASH_CROWD" and is_attacker:

        # Legitimate traffic increases progressively.
        packet_rate = (
            10
            + (90 * intensity)
            + random.uniform(-5, 10)
        )

        avg_packet_size = random.uniform(
            500,
            1200,
        )

        protocol = random.choices(
            [6, 17],
            weights=[0.90, 0.10],
            k=1,
        )[0]

        if protocol == 6:
            destination_port = random.choice(
                [80, 443]
            )
        else:
            destination_port = random.choice(
                [53, 123]
            )

        new_flow_probability = (
            0.08 + (0.25 * intensity)
        )

        kill_probability = 0.12

        label = "BENIGN"

    # --------------------------------------------------------
    # SYN FLOOD
    # --------------------------------------------------------

    elif scenario == "SYN_FLOOD" and is_attacker:

        label = "SYN_FLOOD"

        # Vary maximum intensity between runs.
        attack_scale = random.uniform(
            180,
            320,
        )

        packet_rate = (
            5
            + attack_scale * intensity
        )

        avg_packet_size = random.uniform(
            56,
            72,
        )

        protocol = 6
        destination_port = random.choice(
            [80, 443]
        )

        # SYN floods create many short-lived flows.
        new_flow_probability = (
            0.20 + (0.75 * intensity)
        )

        kill_probability = 0.65

    # --------------------------------------------------------
    # UDP FLOOD
    # --------------------------------------------------------

    elif scenario == "UDP_FLOOD" and is_attacker:

        label = "UDP_FLOOD"

        attack_scale = random.uniform(
            250,
            500,
        )

        packet_rate = (
            10
            + attack_scale * intensity
        )

        avg_packet_size = random.choice(
            [
                random.uniform(64, 128),
                random.uniform(256, 600),
                random.uniform(800, 1200),
            ]
        )

        protocol = 17

        destination_port = random.randint(
            1024,
            65535,
        )

        new_flow_probability = (
            0.10 + (0.35 * intensity)
        )

        kill_probability = 0.30

    # --------------------------------------------------------
    # ICMP FLOOD
    # --------------------------------------------------------

    elif scenario == "ICMP_FLOOD" and is_attacker:

        label = "ICMP_FLOOD"

        attack_scale = random.uniform(
            180,
            380,
        )

        packet_rate = (
            10
            + attack_scale * intensity
        )

        avg_packet_size = random.uniform(
            70,
            100,
        )

        protocol = 1
        destination_port = 0

        new_flow_probability = (
            0.08 + (0.20 * intensity)
        )

        kill_probability = 0.20

    # --------------------------------------------------------
    # SLOWLORIS
    # --------------------------------------------------------

    elif scenario == "SLOWLORIS" and is_attacker:

        label = "SLOWLORIS"

        # Intentionally overlaps with legitimate
        # low-rate persistent HTTP traffic.
        packet_rate = random.uniform(
            0.5,
            2.5,
        )

        avg_packet_size = random.uniform(
            70,
            160,
        )

        protocol = 6
        destination_port = random.choice(
            [80, 443]
        )

        # Slow accumulation of long-lived connections.
        new_flow_probability = (
            0.04 + (0.20 * intensity)
        )

        # Very low termination probability.
        kill_probability = 0.015

    return {
        "packet_rate": max(0.0, packet_rate),
        "avg_packet_size": max(1.0, avg_packet_size),
        "protocol": protocol,
        "destination_port": destination_port,
        "new_flow_probability": new_flow_probability,
        "kill_probability": kill_probability,
        "label": label,
    }


# ============================================================
# DATASET GENERATION
# ============================================================

def simulate_dataset(
    seed=42,
    num_runs=DEFAULT_RUNS,
):
    np.random.seed(seed)
    random.seed(seed)

    # --------------------------------------------------------
    # Scenario distribution
    # --------------------------------------------------------

    scenarios = (
        ["NORMAL"] * 75
        + ["FLASH_CROWD"] * 15
        + ["SYN_FLOOD"] * 15
        + ["UDP_FLOOD"] * 15
        + ["ICMP_FLOOD"] * 15
        + ["SLOWLORIS"] * 15
    )

    if num_runs < len(scenarios):
        scenarios = scenarios[:num_runs]

    random.shuffle(scenarios)

    all_rows = []

    base_time = datetime(
        2026,
        10,
        3,
        20,
        0,
        0,
    )

    # --------------------------------------------------------
    # Run loop
    # --------------------------------------------------------

    for run_idx, scenario in enumerate(scenarios):

        run_id = f"RUN_{run_idx + 1:03d}"

        # Randomly select hosts participating in this run.
        run_hosts = random.sample(
            HOST_IPS,
            k=random.randint(5, 9),
        )

        # Random attacker subset.
        if scenario in [
            "SYN_FLOOD",
            "UDP_FLOOD",
            "ICMP_FLOOD",
            "SLOWLORIS",
        ]:
            attackers = set(
                random.sample(
                    run_hosts,
                    k=random.randint(2, 4),
                )
            )

        elif scenario == "FLASH_CROWD":
            # Legitimate users participating in the burst.
            attackers = set(
                random.sample(
                    run_hosts,
                    k=random.randint(3, len(run_hosts)),
                )
            )

        else:
            attackers = set()

        # ----------------------------------------------------
        # Per-source behavioural persistence
        # ----------------------------------------------------

        source_persistence = {
            ip: 0
            for ip in run_hosts
        }

        # Active synthetic flows.
        flows = []

        flow_counter = 0

        # ----------------------------------------------------
        # Time loop
        # ----------------------------------------------------

        for t in range(RUN_DURATION):

            current_time = (
                base_time
                + timedelta(
                    seconds=(
                        run_idx
                        * RUN_DURATION
                        + t
                    )
                )
            )

            phase, intensity = get_phase(t)

            # ------------------------------------------------
            # Process each participating source
            # ------------------------------------------------

            for ip in run_hosts:

                is_attacker = ip in attackers

                # --------------------------------------------
                # Persistence tracking
                # --------------------------------------------

                if (
                    is_attacker
                    and scenario not in [
                        "NORMAL",
                        "FLASH_CROWD",
                    ]
                    and intensity > 0
                ):
                    source_persistence[ip] += 1

                elif phase == "RECOVERY":
                    source_persistence[ip] = max(
                        0,
                        source_persistence[ip] - 2,
                    )

                elif phase == "NORMAL":
                    source_persistence[ip] = 0

                # --------------------------------------------
                # Traffic profile
                # --------------------------------------------

                profile = get_traffic_profile(
                    scenario=scenario,
                    is_attacker=is_attacker,
                    intensity=intensity,
                    phase=phase,
                )

                # --------------------------------------------
                # Check active flows for this source
                # --------------------------------------------

                source_flows = [
                    f
                    for f in flows
                    if f.ip_src == ip
                    and f.is_active
                ]

                # --------------------------------------------
                # Create a flow when needed
                # --------------------------------------------

                should_create = (
                    len(source_flows) == 0
                    or random.random()
                    < profile["new_flow_probability"]
                )

                if should_create:

                    flow_counter += 1

                    if profile["protocol"] == 1:
                        source_port = 0
                    else:
                        source_port = random.randint(
                            1024,
                            65535,
                        )

                    new_flow = Flow(
                        flow_id=(
                            f"{run_id}_FLOW_"
                            f"{flow_counter:05d}"
                        ),
                        ip_src=ip,
                        ip_dst=VICTIM_IP,
                        ip_proto=profile["protocol"],
                        tp_src=source_port,
                        tp_dst=profile["destination_port"],
                        start_time=current_time,
                        scenario=scenario,
                        label=profile["label"],
                    )

                    flows.append(new_flow)

                    source_flows.append(
                        new_flow
                    )

                # --------------------------------------------
                # Active-flow count for this source
                # --------------------------------------------

                active_count = len(source_flows)

                # --------------------------------------------
                # Update every active flow
                # --------------------------------------------

                for flow in list(source_flows):

                    # ----------------------------------------
                    # Slight per-flow variation
                    # ----------------------------------------

                    rate = profile["packet_rate"]

                    # Independent flow-level variation.
                    rate *= random.uniform(
                        0.80,
                        1.20,
                    )

                    # ----------------------------------------
                    # Slowloris intermittent transmission
                    # ----------------------------------------

                    if (
                        flow.label == "SLOWLORIS"
                        and random.random() < 0.35
                    ):
                        rate = 0.0

                    # ----------------------------------------
                    # Generate packet delta
                    # ----------------------------------------

                    if rate <= 0:
                        pkt_delta = 0

                    else:
                        pkt_delta = max(
                            0,
                            int(
                                np.random.normal(
                                    rate,
                                    max(
                                        0.5,
                                        rate * 0.15,
                                    ),
                                )
                            ),
                        )

                    # ----------------------------------------
                    # Update flow
                    # ----------------------------------------

                    row = flow.update(
                        current_time=current_time,
                        pkt_delta=pkt_delta,
                        avg_pkt_size=profile[
                            "avg_packet_size"
                        ],
                        persistence_sec=source_persistence[
                            ip
                        ],
                        phase=phase,
                        active_flows=active_count,
                    )

                    row["run_id"] = run_id
                    row["dpid"] = DPID

                    all_rows.append(row)

                    # ----------------------------------------
                    # Flow termination
                    # ----------------------------------------

                    if (
                        flow.duration_sec > 2
                        and random.random()
                        < profile["kill_probability"]
                    ):
                        flow.is_active = False

        # ----------------------------------------------------
        # Move to next run
        # ----------------------------------------------------

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(all_rows)

    column_order = [
        "timestamp",
        "run_id",
        "flow_id",

        "dpid",

        "ip_src",
        "ip_dst",

        "ip_proto",
        "tp_src",
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

        "scenario",
        "phase",

        "label",
        "label_bin",
    ]

    df = df[column_order]

    # Chronological ordering while preserving flow identity.
    df = (
        df.sort_values(
            by=[
                "run_id",
                "timestamp",
                "flow_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(df):

    print("\n" + "=" * 65)
    print("TEMPORAL / DATA INTEGRITY SANITY CHECKS")
    print("=" * 65)

    results = []

    # --------------------------------------------------------
    # 1. No NaNs
    # --------------------------------------------------------

    results.append(
        (
            "No missing values",
            df.isna().sum().sum() == 0,
        )
    )

    # --------------------------------------------------------
    # 2. No duplicate rows
    # --------------------------------------------------------

    results.append(
        (
            "No duplicate rows",
            df.duplicated().sum() == 0,
        )
    )

    # --------------------------------------------------------
    # 3. Timestamp ordering
    # --------------------------------------------------------

    timestamp_ok = (
        df.groupby("run_id")["timestamp"]
        .apply(lambda x: x.is_monotonic_increasing)
        .all()
    )

    results.append(
        (
            "Timestamps ordered within runs",
            bool(timestamp_ok),
        )
    )

    # --------------------------------------------------------
    # 4. Non-negative values
    # --------------------------------------------------------

    non_negative_columns = [
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

    non_negative_ok = (
        df[non_negative_columns] >= 0
    ).all().all()

    results.append(
        (
            "No negative traffic statistics",
            bool(non_negative_ok),
        )
    )

    # --------------------------------------------------------
    # 5. Mathematical consistency
    # --------------------------------------------------------

    bytes_consistent = np.allclose(
        df["byte_delta"],
        df["pkt_delta"]
        * df["avg_pkt_size"],
        atol=1.0,
    )

    results.append(
        (
            "Byte/packet mathematical consistency",
            bool(bytes_consistent),
        )
    )

    # --------------------------------------------------------
    # 6. Flash crowd is benign
    # --------------------------------------------------------

    flash_rows = df[
        df["scenario"] == "FLASH_CROWD"
    ]

    flash_ok = (
        len(flash_rows) > 0
        and flash_rows["label"].eq(
            "BENIGN"
        ).all()
    )

    results.append(
        (
            "Flash crowd remains BENIGN",
            bool(flash_ok),
        )
    )

    # --------------------------------------------------------
    # 7. Valid binary labels
    # --------------------------------------------------------

    binary_ok = set(
        df["label_bin"].unique()
    ).issubset({0, 1})

    results.append(
        (
            "Binary labels are valid",
            bool(binary_ok),
        )
    )

    # --------------------------------------------------------
    # 8. Valid multiclass labels
    # --------------------------------------------------------

    expected_labels = {
        "BENIGN",
        "SYN_FLOOD",
        "UDP_FLOOD",
        "ICMP_FLOOD",
        "SLOWLORIS",
    }

    labels_ok = set(
        df["label"].unique()
    ).issubset(expected_labels)

    results.append(
        (
            "Multiclass labels are valid",
            bool(labels_ok),
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for name, passed in results:

        status = "PASS" if passed else "FAIL"

        print(
            f"[{status}] {name}"
        )

    print("=" * 65)

    all_passed = all(
        passed
        for _, passed in results
    )

    if all_passed:
        print(
            "ALL SANITY CHECKS PASSED"
        )
    else:
        print(
            "WARNING: ONE OR MORE CHECKS FAILED"
        )

    print("=" * 65)

    return all_passed


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def print_data_quality_report(df):

    print("\n" + "=" * 65)
    print("DATASET QUALITY REPORT")
    print("=" * 65)

    print(
        f"Total rows:          {len(df):,}"
    )

    print(
        f"Unique runs:         {df['run_id'].nunique():,}"
    )

    print(
        f"Unique flows:        {df['flow_id'].nunique():,}"
    )

    print(
        f"Unique source IPs:   {df['ip_src'].nunique():,}"
    )

    print(
        f"Missing values:      "
        f"{df.isna().sum().sum():,}"
    )

    print(
        f"Duplicate rows:      "
        f"{df.duplicated().sum():,}"
    )

    print(
        f"Time range:          "
        f"{df['timestamp'].min()} "
        f"to "
        f"{df['timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nCLASS DISTRIBUTION:")

    class_distribution = (
        df["label"]
        .value_counts(normalize=True)
        * 100
    )

    for label, percentage in (
        class_distribution.items()
    ):
        count = (
            df["label"]
            .value_counts()
            .loc[label]
        )

        print(
            f"  {label:<15}"
            f"{count:>8,} rows "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Scenario distribution
    # --------------------------------------------------------

    print("\nSCENARIO DISTRIBUTION:")

    scenario_distribution = (
        df["scenario"]
        .value_counts()
    )

    for scenario, count in (
        scenario_distribution.items()
    ):
        print(
            f"  {scenario:<15}"
            f"{count:>8,}"
        )

    # --------------------------------------------------------
    # Phase distribution
    # --------------------------------------------------------

    print("\nPHASE DISTRIBUTION:")

    phase_distribution = (
        df["phase"]
        .value_counts()
    )

    for phase, count in (
        phase_distribution.items()
    ):
        print(
            f"  {phase:<15}"
            f"{count:>8,}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nFEATURE STATISTICS BY CLASS"
        " (Mean ± Standard Deviation):"
    )

    statistical_columns = [
        "duration_sec",
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

    grouped = (
        df.groupby("label")[
            statistical_columns
        ]
        .agg(["mean", "std"])
    )

    for label in grouped.index:

        print(
            f"\n[{label}]"
        )

        for column in statistical_columns:

            mean_value = grouped.loc[
                label,
                (column, "mean"),
            ]

            std_value = grouped.loc[
                label,
                (column, "std"),
            ]

            print(
                f"  {column:<24}"
                f"{mean_value:>10.2f} ± "
                f"{std_value:>10.2f}"
            )

    print("=" * 65)


# ============================================================
# README
# ============================================================

def generate_readme():

    return """
# Synthetic SDN DDoS Network Flow-Statistics Dataset

## Project

AN EXPLAINABLE AI FRAMEWORK FOR DDoS NETWORK TRAFFIC CLASSIFICATION

## Important Disclaimer

This is a 100% SYNTHETIC dataset.

It does not represent real network captures and was not collected from
physical or virtual network hardware.

It is designed for:

- ML pipeline development
- temporal sequence modelling
- controlled DDoS experiments
- early detection experiments
- explainability experiments
- adaptive mitigation methodology development

Real Mininet + Open vSwitch + Ryu traffic may later be used for
external validation.

## Classes

BENIGN
SYN_FLOOD
UDP_FLOOD
ICMP_FLOOD
SLOWLORIS

FLASH_CROWD is a scenario but remains BENIGN.

## Temporal Design

Each run contains a temporal lifecycle:

NORMAL
EARLY_CHANGE
ESCALATION
SUSTAINED
DECAY
RECOVERY

Individual flows have:

- flow start time
- duration
- cumulative packet count
- cumulative byte count
- packet/byte deltas
- protocol
- ports
- short-term traffic statistics

Flows may be created and terminated during a run.

## Important Metadata Columns

The following columns must NOT be used as predictive ML features:

run_id
flow_id
scenario
phase
ip_src
ip_dst
label
label_bin

These are metadata or targets.

## Recommended Dataset Splitting

Split by run_id.

Example:

TRAIN:
RUN_001 - RUN_100

VALIDATION:
RUN_101 - RUN_125

TEST:
RUN_126 - RUN_150

Do NOT randomly shuffle individual rows across train and test.

This would cause temporal leakage.

## Synthetic Data Limitation

High performance on this dataset does not prove equivalent performance
on real-world network traffic.

Real SDN traffic should be used for additional validation whenever
available.
"""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic temporal SDN "
            "DDoS flow-statistics dataset"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of independent runs",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="synthetic_ddos_sdn_dataset.csv",
        help="Output CSV filename",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(
        "SYNTHETIC SDN DDoS DATASET GENERATOR"
    )
    print("=" * 65)

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Runs: {args.runs}"
    )

    print(
        "Generating traffic..."
    )

    df = simulate_dataset(
        seed=args.seed,
        num_runs=args.runs,
    )

    print(
        f"Generated {len(df):,} rows."
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        args.out,
        index=False,
    )

    print(
        f"Saved dataset: {args.out}"
    )

    # --------------------------------------------------------
    # Save README
    # --------------------------------------------------------

    with open(
        "README.md",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            generate_readme()
        )

    print(
        "Saved README.md"
    )

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    sanity_ok = run_sanity_checks(
        df
    )

    print_data_quality_report(
        df
    )

    if sanity_ok:

        print(
            "\nDATASET GENERATION COMPLETE."
        )

    else:

        print(
            "\nDATASET GENERATED, "
            "BUT SANITY CHECKS REQUIRE ATTENTION."
        )


if __name__ == "__main__":
    main()