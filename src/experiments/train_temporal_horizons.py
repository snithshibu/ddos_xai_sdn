from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Configuration
# ============================================================

RAW = Path("data/raw/synthetic_ddos_sdn_dataset.csv")
TRAIN = Path("data/processed/train.csv")
TEST = Path("data/processed/test.csv")

OUTPUT_DIR = Path("data/experiments/temporal_horizons")
MODEL_DIR = Path("models/temporal_horizons")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 5, 10]
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "ip_proto",
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
]

LABEL_MAP = {
    "BENIGN": 0,
    "ICMP_FLOOD": 1,
    "SLOWLORIS": 2,
    "SYN_FLOOD": 3,
    "UDP_FLOOD": 4,
}

CLASS_NAMES = [
    "BENIGN",
    "ICMP_FLOOD",
    "SLOWLORIS",
    "SYN_FLOOD",
    "UDP_FLOOD",
]


# ============================================================
# Reconstruct network-level observations
# ============================================================

def build_network_observations(df):
    agg = (
        df.groupby(["run_id", "timestamp"], as_index=False)
        .agg({
            "ip_proto": "mean",
            "tp_dst": "mean",
            "duration_sec": "mean",
            "packet_count": "sum",
            "byte_count": "sum",
            "pkt_delta": "sum",
            "byte_delta": "sum",
            "pkts_per_sec": "sum",
            "bytes_per_sec": "sum",
            "avg_pkt_size": "mean",
            "active_flows": lambda x: x.nunique(),
            "inter_arrival_mean": "mean",
            "packet_rate_std": "mean",
            "byte_rate_std": "mean",
            "flow_persistence_sec": "mean",
        })
    )

    # Recover the network-level label using attack priority.
    priority = {
        "UDP_FLOOD": 4,
        "SYN_FLOOD": 3,
        "SLOWLORIS": 2,
        "ICMP_FLOOD": 1,
        "BENIGN": 0,
    }

    def network_label(values):
        return max(values, key=lambda x: priority[str(x)])

    labels = (
        df.groupby(["run_id", "timestamp"])["label"]
        .agg(network_label)
        .reset_index()
    )

    network = agg.merge(
        labels,
        on=["run_id", "timestamp"],
        how="left",
    )

    network["timestamp"] = pd.to_datetime(network["timestamp"])
    network["label_id"] = network["label"].map(LABEL_MAP)

    return network


# ============================================================
# Create temporal sequences
# ============================================================

def create_sequences(network, horizon):
    X = []
    y = []
    metadata = []

    for run_id, run in network.groupby("run_id"):
        run = run.sort_values("timestamp").reset_index(drop=True)

        for start in range(len(run) - horizon + 1):

            window = run.iloc[start:start + horizon]

            # Require consecutive one-second observations.
            deltas = (
                window["timestamp"]
                .diff()
                .dropna()
                .dt.total_seconds()
            )

            if not (deltas == 1).all():
                continue

            X.append(
                window[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            )

            y.append(int(window.iloc[-1]["label_id"]))

            metadata.append({
                "run_id": run_id,
                "timestamp": window.iloc[-1]["timestamp"],
            })

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        pd.DataFrame(metadata),
    )


# ============================================================
# Model
# ============================================================

def build_model(horizon):
    model = keras.Sequential([
        layers.Input(shape=(horizon, len(FEATURE_COLUMNS))),

        layers.Conv1D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),

        layers.Conv1D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        layers.BatchNormalization(),

        layers.LSTM(
            64,
            return_sequences=False,
        ),

        layers.Dropout(0.3),

        layers.Dense(
            64,
            activation="relu",
        ),

        layers.Dropout(0.3),

        layers.Dense(
            len(CLASS_NAMES),
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TEMPORAL HORIZON EXPERIMENT")
    print("=" * 70)

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("\nLoading exact grouped train/test data...")

    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)

    train_runs = set(train_df["run_id"].unique())
    test_runs = set(test_df["run_id"].unique())

    print(f"Train runs: {len(train_runs)}")
    print(f"Test runs : {len(test_runs)}")

    # Raw dataset is required for phase/scenario metadata,
    # but the actual model input remains the same 15 features.
    raw = pd.read_csv(RAW)

    train_raw = raw[raw["run_id"].isin(train_runs)].copy()
    test_raw = raw[raw["run_id"].isin(test_runs)].copy()

    print(f"Train raw rows: {len(train_raw):,}")
    print(f"Test raw rows : {len(test_raw):,}")

    # Build network observations.
    train_network = build_network_observations(train_raw)
    test_network = build_network_observations(test_raw)

    print(
        f"Train network observations: {len(train_network):,}"
    )
    print(
        f"Test network observations : {len(test_network):,}"
    )

    # Load scaler fitted only on the original training set.
    scaler = joblib.load(
        "data/processed/scaler.joblib"
    )

    results = []

    for horizon in HORIZONS:

        print("\n" + "=" * 70)
        print(f"HORIZON = {horizon} SECOND(S)")
        print("=" * 70)

        X_train, y_train, train_meta = create_sequences(
            train_network,
            horizon,
        )

        X_test, y_test, test_meta = create_sequences(
            test_network,
            horizon,
        )

        print(f"X_train: {X_train.shape}")
        print(f"X_test : {X_test.shape}")

        # Scale features using the already-fitted training scaler.
        original_shape = X_train.shape

        X_train_2d = X_train.reshape(-1, len(FEATURE_COLUMNS))
        X_test_2d = X_test.reshape(-1, len(FEATURE_COLUMNS))

        X_train = scaler.transform(X_train_2d).reshape(
            original_shape
        )

        X_test = scaler.transform(X_test_2d).reshape(
            X_test.shape
        )

        model = build_model(horizon)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
        )

        probabilities = model.predict(
            X_test,
            verbose=0,
        )

        predictions = probabilities.argmax(axis=1)

        accuracy = float(
            np.mean(predictions == y_test)
        )

        print(
            f"\nTest accuracy: {accuracy:.4f}"
        )

        # Per-class results
        class_results = {}

        for class_id, class_name in enumerate(CLASS_NAMES):

            mask = y_test == class_id

            if mask.sum() == 0:
                continue

            class_accuracy = float(
                np.mean(
                    predictions[mask] == y_test[mask]
                )
            )

            class_results[class_name] = class_accuracy

            print(
                f"{class_name:12s}: "
                f"n={mask.sum():5d} | "
                f"accuracy={class_accuracy:.4f}"
            )

        # Save model
        model_path = (
            MODEL_DIR /
            f"cnn_lstm_{horizon}s.keras"
        )

        model.save(model_path)

        # Save predictions
        prediction_df = test_meta.copy()

        prediction_df["true_label"] = [
            CLASS_NAMES[i]
            for i in y_test
        ]

        prediction_df["predicted_label"] = [
            CLASS_NAMES[i]
            for i in predictions
        ]

        prediction_df["confidence"] = (
            probabilities.max(axis=1)
        )

        prediction_df["correct"] = (
            predictions == y_test
        )

        prediction_df.to_csv(
            OUTPUT_DIR /
            f"predictions_{horizon}s.csv",
            index=False,
        )

        row = {
            "horizon_seconds": horizon,
            "train_sequences": len(X_train),
            "test_sequences": len(X_test),
            "accuracy": accuracy,
        }

        for class_name, value in class_results.items():
            row[f"{class_name}_accuracy"] = value

        results.append(row)

    # ========================================================
    # Save summary
    # ========================================================

    results_df = pd.DataFrame(results)

    output = (
        OUTPUT_DIR /
        "temporal_horizon_results.csv"
    )

    results_df.to_csv(output, index=False)

    print("\n" + "=" * 70)
    print("FINAL TEMPORAL HORIZON RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved:")
    print(output)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
