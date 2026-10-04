from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.predictor import DDoSPredictor
from src.inference.evidence_engine import TemporalEvidenceEngine
from src.inference.feature_builder import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)


def build_features(row: pd.Series) -> dict:
    """Extract the model's 15 input features from one row."""

    return {
        column: row[column]
        for column in FEATURE_COLUMNS
    }


def run_lifecycle_test() -> None:

    print("=" * 70)
    print("REAL-TIME ATTACK LIFECYCLE INTEGRATION TEST")
    print("=" * 70)

    df = pd.read_csv(TEST_PATH)

    predictor = DDoSPredictor()

    evidence = TemporalEvidenceEngine(
        confidence_threshold=0.90,
        rate_limit_threshold=3,
        block_threshold=5,
        recovery_threshold=3,
    )

    # ------------------------------------------------------
    # Phase 1: Fill predictor's temporal buffer
    # ------------------------------------------------------

    print("\nPHASE 1 — TEMPORAL BUFFER")

    for index in range(10):

        row = df.iloc[index]

        prediction = predictor.add_observation(
            build_features(row)
        )

        print(
            f"{index + 1:02d}/10 → "
            + (
                "prediction ready"
                if prediction is not None
                else "buffering"
            )
        )

    assert prediction is not None

    print(
        f"\nInitial prediction: "
        f"{prediction['predicted_label']} "
        f"({prediction['confidence']:.4f})"
    )

    # ------------------------------------------------------
    # Phase 2: Controlled attack sequence
    #
    # We use the evidence engine directly here because
    # the purpose of this phase is to validate the complete
    # mitigation state machine.
    # ------------------------------------------------------

    print("\nPHASE 2 — ATTACK ESCALATION")

    attack_sequence = [
        ("SYN_FLOOD", 0.95),
        ("SYN_FLOOD", 0.96),
        ("SYN_FLOOD", 0.97),
        ("SYN_FLOOD", 0.98),
        ("SYN_FLOOD", 0.99),
    ]

    attack_actions = []

    for index, (label, confidence) in enumerate(
        attack_sequence,
        start=1,
    ):

        state = evidence.update(
            predicted_label=label,
            confidence=confidence,
        )

        attack_actions.append(state.action)

        print(
            f"{index}: "
            f"{label:<12} "
            f"confidence={confidence:.2f} "
            f"attack_count={state.consecutive_attack} "
            f"action={state.action:<11} "
            f"evidence={state.attack_evidence:.3f}"
        )

    assert attack_actions == [
        "OBSERVE",
        "OBSERVE",
        "RATE_LIMIT",
        "RATE_LIMIT",
        "BLOCK",
    ]

    assert evidence.mitigation_active is True

    # ------------------------------------------------------
    # Phase 3: Recovery
    # ------------------------------------------------------

    print("\nPHASE 3 — RECOVERY")

    recovery_actions = []

    for index in range(3):

        state = evidence.update(
            predicted_label="BENIGN",
            confidence=0.99,
        )

        recovery_actions.append(state.action)

        print(
            f"{index + 1}: "
            f"BENIGN       "
            f"benign_count={state.consecutive_benign} "
            f"action={state.action}"
        )

    assert recovery_actions == [
        "OBSERVE",
        "OBSERVE",
        "RECOVER",
    ]

    assert evidence.mitigation_active is False

    print("\n" + "=" * 70)
    print("[PASS] Full attack lifecycle validated.")
    print("=" * 70)

    print(
        "\nLifecycle:"
        "\n  OBSERVE"
        "\n      ↓"
        "\n  OBSERVE"
        "\n      ↓"
        "\n  RATE_LIMIT"
        "\n      ↓"
        "\n  RATE_LIMIT"
        "\n      ↓"
        "\n  BLOCK"
        "\n      ↓"
        "\n  OBSERVE"
        "\n      ↓"
        "\n  OBSERVE"
        "\n      ↓"
        "\n  RECOVER"
    )


if __name__ == "__main__":
    run_lifecycle_test()