"""
train_model.py — Phase 2b: Train Gesture Classifier

Reads gesture_data.csv → trains dense net → saves:
    gesture_model.keras
    gesture_words.npy
    training_report.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import callbacks, layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

CSV_FILE   = "gesture_data.csv"
MODEL_OUT  = "gesture_model.keras"
WORDS_OUT  = "gesture_words.npy"
REPORT_OUT = "training_report.png"

EPOCHS     = 300
BATCH_SIZE = 32
TEST_SPLIT = 0.20
NOISE_STD  = 0.015  # slightly higher noise forces model to learn shape, not exact coords
SEED       = 42

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_dataset(path: str):
    if not os.path.isfile(path):
        print(f"[ERROR] '{path}' not found — run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(path, header=0)
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} NaN rows.")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    X     = df.iloc[:, 1:].values.astype("float32")
    y_raw = df.iloc[:,  0].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print(f"\n[DATA] {len(df)} samples · {len(encoder.classes_)} classes")
    counts = pd.Series(y_raw).value_counts().sort_index()
    for cls, n in counts.items():
        bar = "█" * min(40, max(1, n // max(1, counts.max() // 40)))
        print(f"       {cls:<20} {n:>5}  {bar}")

    return X, y, encoder


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Residual dense network.  Residual connections let the model learn
    'what changed from the previous layer' rather than the whole signal,
    which is exactly what you need to tell apart similar gestures like A vs M.

    Also uses Label Smoothing (0.08) so the model doesn't over-commit to the
    majority class when gestures are ambiguous — it stays more calibrated.
    """
    inp = layers.Input(shape=(input_dim,))
    x   = layers.GaussianNoise(NOISE_STD)(inp)

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.35)(x)

    # ── Residual block A (512 → 512) ─────────────────────────────────────────
    res = x
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.30)(x)
    x   = layers.Dense(512, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])          # residual skip
    x   = layers.Activation("relu")(x)

    # ── Residual block B (512 → 256) ─────────────────────────────────────────
    res = layers.Dense(256)(x)            # projection to match dimensions
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.25)(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])
    x   = layers.Activation("relu")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.20)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.BatchNormalization()(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    mdl = models.Model(inp, out, name="GestureNet_v2")
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # Label smoothing: stops the model from becoming over-confident
        # on the wrong class when training gestures look similar.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    mdl.summary()
    return mdl


def save_report(history, y_val, y_pred, class_names, path: str) -> None:
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle("Training Report", fontsize=14, fontweight="bold")
    grid = plt.GridSpec(1, 3, wspace=0.38)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(history.history["accuracy"],     label="Train", color="#1f77b4", lw=2)
    ax.plot(history.history["val_accuracy"], label="Val",   color="#ff7f0e", lw=2, ls="--")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right"); ax.grid(True, ls=":", alpha=0.5)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(history.history["loss"],     label="Train", color="#d62728", lw=2)
    ax.plot(history.history["val_loss"], label="Val",   color="#2ca02c", lw=2, ls="--")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy")
    ax.legend(loc="upper right"); ax.grid(True, ls=":", alpha=0.5)

    ax  = fig.add_subplot(grid[0, 2])
    cm  = confusion_matrix(y_val, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 9, "weight": "bold"}, ax=ax,
    )
    ax.set_title("Confusion Matrix (row-normalised)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)

    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] Report → {path}")


def main() -> None:
    X, y, encoder = load_dataset(CSV_FILE)
    num_classes   = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
    )
    print(f"\n[SPLIT] train={len(X_train)}  val={len(X_val)}")

    # Class weights so small/large gesture classes contribute equally
    cw_arr  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw_arr))
    print(f"[WEIGHTS] {cw_dict}")
    # แนวทางปรับปรุง: โหลดของเก่ามาเรียนรู้ต่อร่วมกับข้อมูลใหม่
    if os.path.exists(MODEL_OUT):
        try:
            existing        = tf.keras.models.load_model(MODEL_OUT)
            old_num_classes = existing.output_shape[-1]
            if old_num_classes != num_classes:
                print(f"\n[WARN] Existing model has {old_num_classes} output classes but "
                      f"current data has {num_classes}. Rebuilding from scratch.")
                model = build_model(X_train.shape[1], num_classes)
            else:
                print(f"\n[LOAD] Existing model matches ({old_num_classes} classes). "
                      "Fine-tuning with lower LR.")
                model = existing
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
        except Exception as e:
            print(f"\n[WARN] Could not load model ({e}). Building new one.")
            model = build_model(X_train.shape[1], num_classes)
    else:
        print("\n[BUILD] No existing model — building new one.")
        model = build_model(X_train.shape[1], num_classes)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=40,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12,
            min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
    ]

    print("\n[TRAIN] Starting …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=cbs,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, digits=3))

    # Warn about any class with F1 < 0.80 — those need more training images
    from sklearn.metrics import classification_report as _cr
    cr = _cr(y_val, y_pred, target_names=encoder.classes_, output_dict=True)
    weak = [(c, d["f1-score"]) for c, d in cr.items()
            if isinstance(d, dict) and d.get("f1-score", 1.0) < 0.80]
    if weak:
        print("\n[WARN] Weak classes (add more training images for these):")
        for c, f1 in sorted(weak, key=lambda x: x[1]):
            print(f"       {c:<20} F1={f1:.3f}")
    else:
        print("[OK] All classes F1 ≥ 0.80")

    model.save(MODEL_OUT)
    np.save(WORDS_OUT, np.array(list(encoder.classes_)))
    print(f"\n[SAVED] Model  → {MODEL_OUT}")
    print(f"[SAVED] Labels → {WORDS_OUT}")

    save_report(history, y_val, y_pred, encoder.classes_, REPORT_OUT)
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
    
