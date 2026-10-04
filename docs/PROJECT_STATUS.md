# Project Status

## Project

**AN EXPLAINABLE AI FRAMEWORK FOR DDoS NETWORK TRAFFIC CLASSIFICATION**

---

## Phase 1 — Project Foundation

- [x] Project directory created
- [x] Folder structure created
- [x] Python virtual environment created
- [x] Core dependencies installed
- [x] Project configuration created
- [x] Class definitions created
- [x] Setup test completed
- [x] Git initialized

---

## Phase 2 — Data Pipeline

- [ ] Receive/verify Ubuntu SDN environment
- [ ] Verify Mininet-Ryu connectivity
- [ ] Verify OpenFlow 1.3
- [ ] Generate benign traffic
- [ ] Generate SYN flood traffic
- [ ] Generate UDP flood traffic
- [ ] Generate ICMP flood traffic
- [ ] Generate Slowloris traffic
- [ ] Collect flow statistics
- [ ] Verify labels
- [ ] Build cleaned dataset

---

## Phase 3 — Feature Engineering

- [ ] Define final feature set
- [ ] Add temporal features
- [ ] Add inter-arrival statistics
- [ ] Add flow persistence
- [ ] Add concurrent-flow features
- [ ] Clean categorical/network identifiers
- [ ] Normalize/encode features

---

## Phase 4 — Temporal Dataset

- [ ] Define sequence window
- [ ] Generate t-n ... t sequences
- [ ] Prevent temporal leakage
- [ ] Train/validation/test split
- [ ] Save sequence dataset

---

## Phase 5 — Model Development

- [ ] Random Forest baseline
- [ ] CNN
- [ ] LSTM
- [ ] CNN-LSTM
- [ ] Compare models
- [ ] Save trained models

---

## Phase 6 — Temporal Evidence Layer

- [ ] Prediction confidence
- [ ] Persistence tracking
- [ ] Evidence accumulation
- [ ] Observation state
- [ ] Mitigation state
- [ ] Recovery state

---

## Phase 7 — Adaptive Mitigation

- [ ] OBSERVE
- [ ] RATE-LIMIT
- [ ] BLOCK
- [ ] RECOVER
- [ ] Connect policy to Ryu
- [ ] Measure mitigation latency

---

## Phase 8 — Explainability

- [ ] SHAP
- [ ] Global feature importance
- [ ] Per-prediction explanation
- [ ] Attack-specific explanations

---

## Phase 9 — Evaluation

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Macro F1
- [ ] Per-class F1
- [ ] Confusion matrix
- [ ] False-positive rate
- [ ] Detection latency
- [ ] Mitigation latency
- [ ] Attack suppression
- [ ] Legitimate throughput
- [ ] Packet loss
- [ ] Latency
- [ ] Recovery time
- [ ] Controller overhead

---

## Phase 10 — Dashboard & Demo

- [ ] Streamlit dashboard
- [ ] Live traffic display
- [ ] Attack classification
- [ ] Confidence
- [ ] SHAP explanation
- [ ] Mitigation state
- [ ] Network impact
- [ ] Recovery

---

## Current Research Direction

Temporal-Evidence-Aware DDoS Detection and Adaptive Mitigation.

Core pipeline:

Mininet
→ Open vSwitch
→ Ryu
→ Flow Statistics
→ Feature Engineering
→ CNN-LSTM
→ Prediction + Confidence
→ Temporal Evidence
→ Adaptive Mitigation
→ Network Feedback
→ SHAP Explanation
→ Dashboard