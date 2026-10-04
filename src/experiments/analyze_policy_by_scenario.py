from pathlib import Path

import pandas as pd
import numpy as np


INPUT_DIR = Path("data/experiments/mitigation")
OUTPUT_DIR = Path("data/experiments/mitigation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


POLICIES = {
    "IMMEDIATE_BLOCK": "immediate_block_timeline.csv",
    "PERSISTENCE_3": "persistence_3_timeline.csv",
    "TEMPORAL_EVIDENCE": "temporal_evidence_timeline.csv",
}

ATTACK_CLASSES = {
    "ICMP_FLOOD",
    "SLOWLORIS",
    "SYN_FLOOD",
    "UDP_FLOOD",
}


def scenario_name(row):
    """
    Recover the traffic scenario from the original
    prediction timeline.

    The policy timeline inherits scenario information
    from the CNN-LSTM temporal prediction file.
    """

    return row["scenario"]


def build_scenario_summary(df, policy_name):
    rows = []

    for scenario, group in df.groupby("scenario", sort=True):

        benign = group["true_label"] == "BENIGN"
        attack = group["true_label"].isin(ATTACK_CLASSES)

        mitigated = group["mitigated"]

        false_mitigation = (
            benign & mitigated
        )

        attack_mitigation = (
            attack & mitigated
        )

        rows.append({
            "policy": policy_name,
            "scenario": scenario,
            "observations": len(group),

            "attack_observations": int(attack.sum()),
            "benign_observations": int(benign.sum()),

            "attack_mitigation_rate": (
                attack_mitigation.mean()
                if attack.any()
                else np.nan
            ),

            "false_mitigation_rate": (
                false_mitigation.mean()
                if benign.any()
                else np.nan
            ),

            "observe_fraction": (
                (group["action"] == "OBSERVE").mean()
            ),

            "rate_limit_fraction": (
                (group["action"] == "RATE_LIMIT").mean()
            ),

            "block_fraction": (
                (group["action"] == "BLOCK").mean()
            ),
        })

    return pd.DataFrame(rows)


def build_phase_summary(df, policy_name):
    rows = []

    for scenario, scenario_group in df.groupby(
        "scenario",
        sort=True,
    ):

        for phase, group in scenario_group.groupby(
            "phase",
            sort=False,
        ):

            rows.append({
                "policy": policy_name,
                "scenario": scenario,
                "phase": phase,
                "observations": len(group),

                "observe_fraction": (
                    (group["action"] == "OBSERVE").mean()
                ),

                "rate_limit_fraction": (
                    (group["action"] == "RATE_LIMIT").mean()
                ),

                "block_fraction": (
                    (group["action"] == "BLOCK").mean()
                ),

                "mitigation_fraction": (
                    group["mitigated"].mean()
                ),
            })

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print("SCENARIO-LEVEL MITIGATION ANALYSIS")
    print("=" * 70)

    scenario_results = []
    phase_results = []

    for policy_name, filename in POLICIES.items():

        path = INPUT_DIR / filename

        print(f"\nLoading {policy_name}:")
        print(path)

        df = pd.read_csv(path)

        if "scenario" not in df.columns:
            raise ValueError(
                f"'scenario' column missing from {filename}"
            )

        if "phase" not in df.columns:
            raise ValueError(
                f"'phase' column missing from {filename}"
            )

        if "action" not in df.columns:
            raise ValueError(
                f"'action' column missing from {filename}"
            )

        # Recalculate these explicitly so the analysis
        # cannot depend on stale columns.
        df["mitigated"] = df["action"].isin(
            ["RATE_LIMIT", "BLOCK"]
        )

        scenario_results.append(
            build_scenario_summary(
                df,
                policy_name,
            )
        )

        phase_results.append(
            build_phase_summary(
                df,
                policy_name,
            )
        )

    scenario_summary = pd.concat(
        scenario_results,
        ignore_index=True,
    )

    phase_summary = pd.concat(
        phase_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    scenario_path = (
        OUTPUT_DIR /
        "policy_scenario_summary.csv"
    )

    phase_path = (
        OUTPUT_DIR /
        "policy_phase_summary.csv"
    )

    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )

    phase_summary.to_csv(
        phase_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print scenario comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    display_columns = [
        "policy",
        "scenario",
        "observations",
        "attack_mitigation_rate",
        "false_mitigation_rate",
        "observe_fraction",
        "rate_limit_fraction",
        "block_fraction",
    ]

    print(
        scenario_summary[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Print phase behaviour
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE-WISE POLICY BEHAVIOUR")
    print("=" * 70)

    print(
        phase_summary[
            [
                "policy",
                "scenario",
                "phase",
                "observations",
                "observe_fraction",
                "rate_limit_fraction",
                "block_fraction",
                "mitigation_fraction",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Flash crowd focused analysis
    # --------------------------------------------------------

    flash = scenario_summary[
        scenario_summary["scenario"] == "FLASH_CROWD"
    ].copy()

    print("\n" + "=" * 70)
    print("FLASH-CROWD LEGITIMATE-STRESS TEST")
    print("=" * 70)

    if flash.empty:
        print("No FLASH_CROWD observations found.")
    else:
        print(
            flash[
                [
                    "policy",
                    "observations",
                    "false_mitigation_rate",
                    "observe_fraction",
                    "rate_limit_fraction",
                    "block_fraction",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Slowloris focused analysis
    # --------------------------------------------------------

    slow = scenario_summary[
        scenario_summary["scenario"] == "SLOWLORIS"
    ].copy()

    print("\n" + "=" * 70)
    print("SLOWLORIS ANALYSIS")
    print("=" * 70)

    if slow.empty:
        print("No SLOWLORIS observations found.")
    else:
        print(
            slow[
                [
                    "policy",
                    "observations",
                    "attack_mitigation_rate",
                    "false_mitigation_rate",
                    "observe_fraction",
                    "rate_limit_fraction",
                    "block_fraction",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(scenario_path)
    print(phase_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
