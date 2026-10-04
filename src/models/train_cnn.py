from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"
MODEL_DIR = PROJECT_ROOT / "models"

X_TRAIN_PATH = SEQUENCE_DIR / "X_train.npy"
Y_TRAIN_PATH = SEQUENCE_DIR / "y_train.npy"

X_TEST_PATH = SEQUENCE_DIR / "X_test.npy"
Y_TEST_PATH = SEQUENCE_DIR / "y_test.npy"

MODEL_OUT = MODEL_DIR / "cnn_baseline.keras"

RANDOM_SEED = 42

EPOCHS = 30
BATCH_SIZE = 64


CLASS_NAMES = [
    "BENIGN",
    "ICMP_FLOOD",
    "SLOWLORIS",
    "SYN_FLOOD",
    "UDP_FLOOD",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# MODEL
# ============================================================

def build_cnn(input_shape, num_classes):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),

            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling1D(
                pool_size=2
            ),

            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.GlobalAveragePooling1D(),

            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),

            tf.keras.layers.Dropout(
                0.30
            ),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CNN BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    print("\nLoading temporal sequences...")

    X_train = np.load(
        X_TRAIN_PATH
    )

    y_train = np.load(
        Y_TRAIN_PATH
    )

    X_test = np.load(
        X_TEST_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_shape = X_train.shape[1:]
    num_classes = len(CLASS_NAMES)

    model = build_cnn(
        input_shape,
        num_classes
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel architecture:")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(
        "\n[PASS] CNN training complete"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CNN RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall   : {recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : {weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(
        MODEL_OUT
    )

    print(
        "\n[PASS] Model saved:"
    )

    print(
        MODEL_OUT
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    print(
        "\nBest validation accuracy:"
        f" {best_val_accuracy:.4f}"
    )

    print(
        "Best validation loss:"
        f" "
        f"{best_val_loss:.6f}"
    )

    print("\n" + "=" * 70)
    print("CNN BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()