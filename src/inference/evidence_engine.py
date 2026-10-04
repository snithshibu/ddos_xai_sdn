from __future__ import annotations

from collections import deque
from dataclasses import dataclass


ATTACK_CLASSES = {
    "ICMP_FLOOD",
    "SLOWLORIS",
    "SYN_FLOOD",
    "UDP_FLOOD",
}


@dataclass
class EvidenceState:
    """Current state of the temporal evidence engine."""

    action: str
    predicted_label: str
    confidence: float
    consecutive_attack: int
    consecutive_benign: int
    attack_evidence: float
    reason: str


class TemporalEvidenceEngine:
    """
    Stateful decision layer between the CNN-LSTM predictor
    and the mitigation system.

    States:
        OBSERVE
        RATE_LIMIT
        BLOCK
        RECOVER
    """

    def __init__(
        self,
        confidence_threshold: float = 0.90,
        rate_limit_threshold: int = 3,
        block_threshold: int = 5,
        recovery_threshold: int = 3,
        history_size: int = 10,
    ):
        self.confidence_threshold = confidence_threshold
        self.rate_limit_threshold = rate_limit_threshold
        self.block_threshold = block_threshold
        self.recovery_threshold = recovery_threshold

        self.history = deque(maxlen=history_size)

        self.consecutive_attack = 0
        self.consecutive_benign = 0

        self.current_action = "OBSERVE"

        # Remembers whether mitigation was actually activated.
        self.mitigation_active = False

    def reset(self) -> None:
        """Reset the complete evidence state."""

        self.history.clear()

        self.consecutive_attack = 0
        self.consecutive_benign = 0

        self.current_action = "OBSERVE"
        self.mitigation_active = False

    def update(
        self,
        predicted_label: str,
        confidence: float,
    ) -> EvidenceState:
        """
        Add one CNN-LSTM prediction and update the
        temporal evidence state.
        """

        is_attack = predicted_label in ATTACK_CLASSES

        confident_attack = (
            is_attack
            and confidence >= self.confidence_threshold
        )

        # ----------------------------------------------------
        # Update temporal counters
        # ----------------------------------------------------

        if confident_attack:
            self.consecutive_attack += 1
            self.consecutive_benign = 0
        else:
            self.consecutive_benign += 1
            self.consecutive_attack = 0

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        self.history.append(
            {
                "label": predicted_label,
                "confidence": float(confidence),
                "attack": is_attack,
                "confident_attack": confident_attack,
            }
        )

        # ----------------------------------------------------
        # Calculate temporal evidence
        # ----------------------------------------------------

        attack_evidence = self._calculate_attack_evidence()

        # ----------------------------------------------------
        # Decide action
        # ----------------------------------------------------

        if (
            self.mitigation_active
            and self.consecutive_benign
            >= self.recovery_threshold
        ):
            # Actual recovery from an active mitigation state.
            action = "RECOVER"

            reason = (
                "Sustained benign evidence supports recovery."
            )

            self.mitigation_active = False

        elif self.consecutive_attack >= self.block_threshold:
            action = "BLOCK"

            reason = (
                "Strong persistent attack evidence "
                "exceeded block threshold."
            )

            self.mitigation_active = True

        elif self.consecutive_attack >= self.rate_limit_threshold:
            action = "RATE_LIMIT"

            reason = (
                "Persistent attack evidence "
                "exceeded rate-limit threshold."
            )

            self.mitigation_active = True

        else:
            action = "OBSERVE"

            if not is_attack:
                reason = "No attack prediction."

            elif confidence < self.confidence_threshold:
                reason = (
                    "Attack prediction lacks "
                    "sufficient confidence."
                )

            else:
                reason = (
                    "Attack evidence is still "
                    "insufficient for mitigation."
                )

        self.current_action = action

        return EvidenceState(
            action=action,
            predicted_label=predicted_label,
            confidence=float(confidence),
            consecutive_attack=self.consecutive_attack,
            consecutive_benign=self.consecutive_benign,
            attack_evidence=attack_evidence,
            reason=reason,
        )

    def _calculate_attack_evidence(self) -> float:
        """
        Calculate temporal attack evidence from recent
        predictions.

        Evidence =
            attack prediction fraction
            × average confidence of attack predictions
        """

        if not self.history:
            return 0.0

        attack_entries = [
            item
            for item in self.history
            if item["attack"]
        ]

        if not attack_entries:
            return 0.0

        attack_fraction = (
            len(attack_entries)
            / len(self.history)
        )

        average_confidence = (
            sum(
                item["confidence"]
                for item in attack_entries
            )
            / len(attack_entries)
        )

        evidence = (
            attack_fraction
            * average_confidence
        )

        return float(
            max(0.0, min(1.0, evidence))
        )


def run_sanity_test() -> None:
    """Run a complete normal → attack → recovery test."""

    print("=" * 70)
    print("TEMPORAL EVIDENCE ENGINE SANITY TEST")
    print("=" * 70)

    engine = TemporalEvidenceEngine(
        confidence_threshold=0.90,
        rate_limit_threshold=3,
        block_threshold=5,
        recovery_threshold=3,
    )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    print("\nNORMAL TRAFFIC")

    for i in range(3):
        state = engine.update(
            "BENIGN",
            0.99,
        )

        print(
            f"{i + 1}: "
            f"{state.action:<11} "
            f"attack_count={state.consecutive_attack} "
            f"benign_count={state.consecutive_benign} "
            f"evidence={state.attack_evidence:.3f}"
        )

    # --------------------------------------------------------
    # ATTACK
    # --------------------------------------------------------

    print("\nATTACK BEGINS")

    attack_predictions = [
        ("SYN_FLOOD", 0.72),
        ("SYN_FLOOD", 0.94),
        ("SYN_FLOOD", 0.97),
        ("SYN_FLOOD", 0.98),
        ("SYN_FLOOD", 0.99),
        ("SYN_FLOOD", 0.99),
    ]

    for i, (label, confidence) in enumerate(
        attack_predictions,
        start=1,
    ):
        state = engine.update(
            label,
            confidence,
        )

        print(
            f"{i}: "
            f"{state.action:<11} "
            f"confidence={confidence:.2f} "
            f"attack_count={state.consecutive_attack} "
            f"evidence={state.attack_evidence:.3f}"
        )

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    print("\nRECOVERY")

    for i in range(4):
        state = engine.update(
            "BENIGN",
            0.99,
        )

        print(
            f"{i + 1}: "
            f"{state.action:<11} "
            f"benign_count={state.consecutive_benign} "
            f"evidence={state.attack_evidence:.3f}"
        )

    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    # Test escalation.
    engine.reset()

    states = [
        engine.update("SYN_FLOOD", 0.95),
        engine.update("SYN_FLOOD", 0.96),
        engine.update("SYN_FLOOD", 0.97),
        engine.update("SYN_FLOOD", 0.98),
        engine.update("SYN_FLOOD", 0.99),
    ]

    assert states[0].action == "OBSERVE"
    assert states[1].action == "OBSERVE"
    assert states[2].action == "RATE_LIMIT"
    assert states[3].action == "RATE_LIMIT"
    assert states[4].action == "BLOCK"

    # Test normal traffic does NOT trigger recovery.
    engine.reset()

    normal_states = [
        engine.update("BENIGN", 0.99)
        for _ in range(3)
    ]

    assert all(
        state.action == "OBSERVE"
        for state in normal_states
    )

    # Test actual recovery.
    engine.reset()

    for confidence in [0.95, 0.96, 0.97, 0.98, 0.99]:
        engine.update(
            "SYN_FLOOD",
            confidence,
        )

    recovery_states = [
        engine.update("BENIGN", 0.99)
        for _ in range(3)
    ]

    assert recovery_states[0].action == "OBSERVE"
    assert recovery_states[1].action == "OBSERVE"
    assert recovery_states[2].action == "RECOVER"

    # After recovery, another benign observation should
    # return to normal OBSERVE state.
    final_state = engine.update(
        "BENIGN",
        0.99,
    )

    assert final_state.action == "OBSERVE"

    print(
        "\n[PASS] Temporal evidence engine is ready."
    )


if __name__ == "__main__":
    run_sanity_test()
