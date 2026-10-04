from __future__ import annotations

from src.inference.predictor import DDoSPredictor
from src.inference.evidence_engine import TemporalEvidenceEngine


class RealtimeInferencePipeline:
    """
    End-to-end real-time DDoS detection pipeline.

    Network observation
        ↓
    CNN-LSTM predictor
        ↓
    Temporal evidence engine
        ↓
    Mitigation action
    """

    def __init__(self):
        self.predictor = DDoSPredictor()

        self.evidence_engine = TemporalEvidenceEngine(
            confidence_threshold=0.90,
            rate_limit_threshold=3,
            block_threshold=5,
            recovery_threshold=3,
        )

    def process_observation(
        self,
        observation: dict,
    ) -> dict:
        """
        Process one network observation.

        The observation must contain the 15 features
        used during CNN-LSTM training.
        """

        # --------------------------------------------------
        # 1. CNN-LSTM prediction
        # --------------------------------------------------

        prediction = self.predictor.add_observation(
            observation
        )

        # The predictor needs 10 observations before
        # producing its first prediction.
        if prediction is None:
            return {
                "status": "BUFFERING",
                "action": "OBSERVE",
                "sequence_length": (
                    len(self.predictor.buffer)
                ),
            }

        # --------------------------------------------------
        # 2. Temporal evidence decision
        # --------------------------------------------------

        evidence_state = self.evidence_engine.update(
            predicted_label=prediction["predicted_label"],
            confidence=prediction["confidence"],
        )

        # --------------------------------------------------
        # 3. Return complete decision
        # --------------------------------------------------

        return {
            "status": "PREDICTION_READY",
            "predicted_label": prediction[
                "predicted_label"
            ],
            "confidence": prediction[
                "confidence"
            ],
            "class_probabilities": prediction[
                "class_probabilities"
            ],
            "sequence_length": prediction[
                "sequence_length"
            ],
            "action": evidence_state.action,
            "consecutive_attack": (
                evidence_state.consecutive_attack
            ),
            "consecutive_benign": (
                evidence_state.consecutive_benign
            ),
            "attack_evidence": (
                evidence_state.attack_evidence
            ),
            "reason": evidence_state.reason,
        }

    def reset(self) -> None:
        """
        Reset both the CNN-LSTM temporal buffer
        and the evidence engine.
        """

        self.predictor.reset()
        self.evidence_engine.reset()


def run_sanity_test() -> None:
    """
    Verify that the complete real-time inference
    pipeline can process network observations.
    """

    print("=" * 70)
    print("REAL-TIME INFERENCE PIPELINE SANITY TEST")
    print("=" * 70)

    pipeline = RealtimeInferencePipeline()

    # --------------------------------------------------
    # Test observation
    # --------------------------------------------------

    observation = {
        "ip_proto": 6,
        "tp_dst": 80,
        "duration_sec": 2.0,
        "packet_count": 200,
        "byte_count": 12000,
        "pkt_delta": 20,
        "byte_delta": 1200,
        "pkts_per_sec": 100.0,
        "bytes_per_sec": 6000.0,
        "avg_pkt_size": 60.0,
        "active_flows": 3,
        "inter_arrival_mean": 0.01,
        "packet_rate_std": 10.0,
        "byte_rate_std": 500.0,
        "flow_persistence_sec": 5.0,
    }

    # --------------------------------------------------
    # Feed observations
    # --------------------------------------------------

    for index in range(10):

        result = pipeline.process_observation(
            observation
        )

        print(
            f"\nObservation "
            f"{index + 1:02d}/10"
        )

        print(
            f"Status          : "
            f"{result['status']}"
        )

        print(
            f"Sequence buffer : "
            f"{result['sequence_length']}/10"
        )

        if result["status"] == "PREDICTION_READY":

            print(
                f"Prediction      : "
                f"{result['predicted_label']}"
            )

            print(
                f"Confidence      : "
                f"{result['confidence']:.6f}"
            )

            print(
                f"Action          : "
                f"{result['action']}"
            )

            print(
                f"Attack evidence : "
                f"{result['attack_evidence']:.6f}"
            )

            print(
                f"Attack count    : "
                f"{result['consecutive_attack']}"
            )

            print(
                f"Benign count    : "
                f"{result['consecutive_benign']}"
            )

            print(
                f"Reason          : "
                f"{result['reason']}"
            )

    print("\n" + "=" * 70)
    print(
        "[PASS] Real-time inference pipeline is operational."
    )
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_test()