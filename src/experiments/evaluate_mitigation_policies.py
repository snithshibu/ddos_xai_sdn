from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path(
    "data/experiments/cnn_lstm_temporal_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/experiments/mitigation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_CLASSES = {
    "ICMP_FLOOD",
    "SLOWLORIS",
    "SYN_FLOOD",
    "UDP_FLOOD",
}

# ------------------------------------------------------------
# Policy definitions
# ------------------------------------------------------------

def immediate_block(predictions):
    """
    Block whenever the classifier predicts an attack.
    Otherwise observe.
    """
    return predictions["predicted_label"].apply(
        lambda label: "BLOCK"
        if label in ATTACK_CLASSES
        else "OBSERVE"
    )


def persistence_block(predictions, required_consecutive=3):
    """
    Block after N consecutive attack predictions.
    Counter resets when BENIGN is predicted.
    """
    decisions = []

    for _, run in predictions.groupby("run_id", sort=False):

        counter = 0

        for _, row in run.sort_values("final_timestamp").iterrows():

            if row["predicted_label"] in ATTACK_CLASSES:
                counter += 1
            else:
                counter = 0

            decisions.append(
                "BLOCK"
                if counter >= required_consecutive
                else "OBSERVE"
            )

    return pd.Series(
        decisions,
        index=predictions.index,
    )


def temporal_evidence_policy(
    predictions,
    confidence_threshold=0.90,
    rate_limit_consecutive=3,
    block_consecutive=5,
):
    """
    Simple evidence-aware policy.

    OBSERVE:
        insufficient evidence

    RATE_LIMIT:
        attack prediction + sufficient confidence +
        persistent evidence

    BLOCK:
        stronger persistent evidence
    """

    decisions = []

    for _, run in predictions.groupby("run_id", sort=False):

        counter = 0

        for _, row in run.sort_values("final_timestamp").iterrows():

            predicted_attack = (
                row["predicted_label"] in ATTACK_CLASSES
            )

            confident = (
                row["confidence"] >= confidence_threshold
            )

            if predicted_attack and confident:
                counter += 1
            else:
                counter = 0

            if counter >= block_consecutive:
                action = "BLOCK"

            elif counter >= rate_limit_consecutive:
                action = "RATE_LIMIT"

            else:
                action = "OBSERVE"

            decisions.append(action)

    return pd.Series(
        decisions,
        index=predictions.index,
    )


# ------------------------------------------------------------
# Evaluate a policy
# ------------------------------------------------------------

def evaluate_policy(df, policy_name, actions):

    result = df.copy()
    result["action"] = actions.values

    # A mitigation is considered active for RATE_LIMIT or BLOCK.
    result["mitigated"] = result["action"].isin(
        ["RATE_LIMIT", "BLOCK"]
    )

    # Incorrect mitigation = benign traffic acted upon.
    result["false_mitigation"] = (
        (result["true_label"] == "BENIGN")
        & result["mitigated"]
    )

    # Attack correctly acted upon.
    result["attack_mitigated"] = (
        result["true_label"].isin(ATTACK_CLASSES)
        & result["mitigated"]
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    total = len(result)

    benign = result["true_label"] == "BENIGN"
    attack = result["true_label"].isin(ATTACK_CLASSES)

    false_mitigation_rate = (
        result.loc[benign, "false_mitigation"].mean()
        if benign.any()
        else 0.0
    )

    attack_mitigation_rate = (
        result.loc[attack, "attack_mitigated"].mean()
        if attack.any()
        else 0.0
    )

    observe_fraction = (
        (result["action"] == "OBSERVE").mean()
    )

    rate_limit_fraction = (
        (result["action"] == "RATE_LIMIT").mean()
    )

    block_fraction = (
        (result["action"] == "BLOCK").mean()
    )

    # --------------------------------------------------------
    # First mitigation time for each attack run
    # --------------------------------------------------------

    attack_delays = []

    attack_runs = (
        result.loc[
            result["true_label"].isin(ATTACK_CLASSES),
            "run_id",
        ]
        .unique()
    )

    for run_id in attack_runs:

        run = result[
            result["run_id"] == run_id
        ].sort_values("final_timestamp")

        # First timestamp where the run has actual attack
        attack_rows = run[
            run["true_label"].isin(ATTACK_CLASSES)
        ]

        if attack_rows.empty:
            continue

        attack_start = attack_rows.iloc[0]["final_timestamp"]

        mitigation_rows = run[
            (run["final_timestamp"] >= attack_start)
            & run["mitigated"]
        ]

        if mitigation_rows.empty:
            continue

        first_mitigation = mitigation_rows.iloc[0][
            "final_timestamp"
        ]

        delay = (
            first_mitigation - attack_start
        ).total_seconds()

        attack_delays.append(delay)

    mean_delay = (
        np.mean(attack_delays)
        if attack_delays
        else np.nan
    )

    median_delay = (
        np.median(attack_delays)
        if attack_delays
        else np.nan
    )

    return {
        "policy": policy_name,
        "total_observations": total,
        "attack_observations": int(attack.sum()),
        "benign_observations": int(benign.sum()),
        "attack_mitigation_rate": attack_mitigation_rate,
        "false_mitigation_rate": false_mitigation_rate,
        "observe_fraction": observe_fraction,
        "rate_limit_fraction": rate_limit_fraction,
        "block_fraction": block_fraction,
        "mean_mitigation_delay_sec": mean_delay,
        "median_mitigation_delay_sec": median_delay,
    }, result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("MITIGATION POLICY EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    df["final_timestamp"] = pd.to_datetime(
        df["final_timestamp"]
    )

    # Make sure global ordering is deterministic.
    df = df.sort_values(
        ["run_id", "final_timestamp"]
    ).reset_index(drop=True)

    print(f"Input observations: {len(df):,}")
    print(f"Runs: {df['run_id'].nunique()}")

    policies = {}

    # --------------------------------------------------------
    # Policy A
    # --------------------------------------------------------

    policies["IMMEDIATE_BLOCK"] = immediate_block(df)

    # --------------------------------------------------------
    # Policy B
    # --------------------------------------------------------

    policies["PERSISTENCE_3"] = persistence_block(
        df,
        required_consecutive=3,
    )

    # --------------------------------------------------------
    # Policy C
    # --------------------------------------------------------

    policies["TEMPORAL_EVIDENCE"] = (
        temporal_evidence_policy(
            df,
            confidence_threshold=0.90,
            rate_limit_consecutive=3,
            block_consecutive=5,
        )
    )

    all_results = []
    detailed_results = {}

    for name, actions in policies.items():

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        summary, detailed = evaluate_policy(
            df,
            name,
            actions,
        )

        all_results.append(summary)
        detailed_results[name] = detailed

        print(
            f"Attack mitigation rate : "
            f"{summary['attack_mitigation_rate']:.4f}"
        )

        print(
            f"False mitigation rate   : "
            f"{summary['false_mitigation_rate']:.4f}"
        )

        print(
            f"Mean mitigation delay   : "
            f"{summary['mean_mitigation_delay_sec']:.2f} s"
        )

        print(
            f"Median mitigation delay : "
            f"{summary['median_mitigation_delay_sec']:.2f} s"
        )

        print(
            f"OBSERVE                 : "
            f"{summary['observe_fraction']:.4f}"
        )

        print(
            f"RATE_LIMIT              : "
            f"{summary['rate_limit_fraction']:.4f}"
        )

        print(
            f"BLOCK                   : "
            f"{summary['block_fraction']:.4f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_results)

    summary_path = (
        OUTPUT_DIR /
        "mitigation_policy_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save detailed policy timelines
    # --------------------------------------------------------

    for name, detailed in detailed_results.items():

        detailed.to_csv(
            OUTPUT_DIR /
            f"{name.lower()}_timeline.csv",
            index=False,
        )

    print("\n" + "=" * 70)
    print("FINAL POLICY COMPARISON")
    print("=" * 70)

    print(
        summary_df.to_string(index=False)
    )

    print("\nSaved:")
    print(summary_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
