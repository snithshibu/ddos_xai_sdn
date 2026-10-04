from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "experiments"


st.set_page_config(
    page_title="DDoS Detection & Mitigation",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================
# Helpers
# ==========================================================

@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    path = EXPERIMENT_DIR / relative_path
    return pd.read_csv(path)


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


# ==========================================================
# Header
# ==========================================================

st.title(
    "🛡️ Explainable AI Framework for DDoS Network Traffic Classification"
)

st.caption(
    "Offline experimental dashboard — "
    "live Mininet/Ryu integration will be connected later."
)


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Navigation")

section = st.sidebar.radio(
    "Select section",
    [
        "Overview",
        "Model Performance",
        "Temporal Analysis",
        "Mitigation",
        "Explainability",
        "Inference Performance",
    ],
)


# ==========================================================
# Overview
# ==========================================================

if section == "Overview":

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Traffic Classes",
        "5",
    )

    col2.metric(
        "Temporal Window",
        "10 s",
    )

    col3.metric(
        "Test Sequences",
        "3,330",
    )

    col4.metric(
        "CNN-LSTM Accuracy",
        "99.70%",
    )

    st.divider()

    st.subheader("Detection Pipeline")

    st.code(
        """
Network Traffic
      ↓
Feature Engineering
      ↓
Temporal Sequence Construction
      ↓
CNN-LSTM
      ↓
Prediction + Confidence
      ↓
Temporal Evidence Engine
      ↓
OBSERVE / RATE_LIMIT / BLOCK / RECOVER
      ↓
SDN Mitigation
        """,
        language="text",
    )

    st.subheader("Attack Classes")

    classes = pd.DataFrame(
        {
            "Class": [
                "BENIGN",
                "ICMP_FLOOD",
                "SLOWLORIS",
                "SYN_FLOOD",
                "UDP_FLOOD",
            ],
            "Type": [
                "Legitimate",
                "Volumetric",
                "Low-rate / Application-layer",
                "Volumetric",
                "Volumetric",
            ],
        }
    )

    st.dataframe(
        classes,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# Model Performance
# ==========================================================

elif section == "Model Performance":

    st.header("Model Performance")

    model_results = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "CNN",
                "LSTM",
                "CNN-LSTM",
            ],
            "Accuracy": [
                0.9992,
                0.9880,
                0.9949,
                0.9970,
            ],
            "Macro F1": [
                0.9991,
                0.9610,
                0.9922,
                0.9951,
            ],
            "Weighted F1": [
                0.9992,
                0.9882,
                0.9949,
                0.9970,
            ],
        }
    )

    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.2%}",
                "Macro F1": "{:.2%}",
                "Weighted F1": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Accuracy")

    st.bar_chart(
        model_results.set_index("Model")[
            "Accuracy"
        ]
    )

    st.subheader("Macro F1")

    st.bar_chart(
        model_results.set_index("Model")[
            "Macro F1"
        ]
    )


# ==========================================================
# Temporal Analysis
# ==========================================================

elif section == "Temporal Analysis":

    st.header("Temporal Horizon Analysis")

    results = load_csv(
        "temporal_horizons/temporal_horizon_results.csv"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Accuracy vs Temporal Observation Horizon"
    )

    horizon_chart = results.copy()

    horizon_chart["horizon"] = (
        horizon_chart["horizon_seconds"]
        .astype(str)
        + " s"
    )

    st.line_chart(
        horizon_chart.set_index("horizon")[
            "accuracy"
        ]
    )

    st.info(
        "Longer temporal context improves overall "
        "classification performance in the current "
        "synthetic evaluation, with the largest "
        "improvement observed for SYN_FLOOD."
    )


# ==========================================================
# Mitigation
# ==========================================================

elif section == "Mitigation":

    st.header(
        "Adaptive Mitigation Policy Evaluation"
    )

    summary = load_csv(
        "mitigation/mitigation_policy_summary.csv"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Policy Comparison"
    )

    numeric_columns = [
        column
        for column in summary.columns
        if column not in ["policy"]
    ]

    if numeric_columns:

        default_metric = "attack_mitigation_rate"

        if default_metric in numeric_columns:
            default_index = numeric_columns.index(
                default_metric
            )
        else:
            default_index = 0

        selected_metric = st.selectbox(
            "Metric",
            numeric_columns,
            index=default_index,
        )

        chart_data = summary.set_index(
            "policy"
        )[selected_metric]

        st.bar_chart(chart_data)

    st.warning(
        "These results are offline synthetic-dataset "
        "policy simulations. Network-level mitigation "
        "performance will be evaluated using Mininet/Ryu."
    )


# ==========================================================
# Explainability
# ==========================================================

elif section == "Explainability":

    st.header("Explainable AI")

    st.warning(
        "SHAP execution is currently functional, "
        "but attribution validation is pending. "
        "Zero-valued preliminary attributions are "
        "not treated as final results."
    )

    shap_path = (
        EXPERIMENT_DIR
        / "shap"
        / "overall_feature_importance.csv"
    )

    if shap_path.exists():

        shap_results = pd.read_csv(
            shap_path
        )

        st.subheader(
            "Preliminary Feature Importance"
        )

        st.dataframe(
            shap_results,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "feature" in shap_results.columns
            and "mean_abs_shap"
            in shap_results.columns
        ):

            st.bar_chart(
                shap_results.set_index(
                    "feature"
                )["mean_abs_shap"]
            )


# ==========================================================
# Inference Performance
# ==========================================================

elif section == "Inference Performance":

    st.header(
        "Inference Performance"
    )

    benchmark = load_csv(
        "inference_benchmark.csv"
    ).iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Model Inference",
        f"{benchmark['mean_model_inference_ms']:.2f} ms",
    )

    col2.metric(
        "Mean Pipeline Decision",
        f"{benchmark['mean_pipeline_ms']:.2f} ms",
    )

    col3.metric(
        "Model Throughput",
        f"{benchmark['predictions_per_second']:.2f}/s",
    )

    st.subheader(
        "Latency Statistics"
    )

    latency = pd.DataFrame(
        {
            "Metric": [
                "Mean model inference",
                "Median model inference",
                "P95 model inference",
                "Mean pipeline",
                "Median pipeline",
                "P95 pipeline",
            ],
            "Latency (ms)": [
                benchmark[
                    "mean_model_inference_ms"
                ],
                benchmark[
                    "median_model_inference_ms"
                ],
                benchmark[
                    "p95_model_inference_ms"
                ],
                benchmark[
                    "mean_pipeline_ms"
                ],
                benchmark[
                    "median_pipeline_ms"
                ],
                benchmark[
                    "p95_pipeline_ms"
                ],
            ],
        }
    )

    st.dataframe(
        latency,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "AN EXPLAINABLE AI FRAMEWORK FOR DDoS NETWORK TRAFFIC CLASSIFICATION"
)