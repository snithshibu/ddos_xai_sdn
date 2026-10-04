
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
# ddos_xai_sdn
# ddos_xai_sdn
