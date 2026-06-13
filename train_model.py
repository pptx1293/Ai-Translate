"""
Phase 2b — Model Training (Fixed Architecture Layout)
Reads gesture_data.csv → trains TensorFlow classifier → saves model + labels

Run AFTER collect_data.py:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
CSV_FILE    = "gesture_data.csv"
MODEL_OUT   = "gesture_model.keras"
LABELS_OUT  = "gesture_labels.txt"
REPORT_OUT  = "training_report.png"

EPOCHS      = 80
BATCH_SIZE  = 16
DROPOUT     = 0.4
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("=" * 50)
print("   Phase 2b — Gesture Model Training")
print("=" * 50)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"[ERROR] {CSV_FILE} not found. Run collect_data.py first.")

df_test = pd.read_csv(CSV_FILE, nrows=2)
first_cell = str(df_test.columns[0]).strip().lower()
has_header = "label" in first_cell or first_cell.startswith("f")

if has_header:
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
else:
    print("[System Note] No valid string header found. Injecting structural column map dynamically...")
    total_cols = df_test.shape[1]
    feature_length = total_cols - 1
    custom_headers = ["label"] + [f"f{i}" for i in range(feature_length)]
    df = pd.read_csv(CSV_FILE, header=None, names=custom_headers)

print(f"\n[Data] Loaded {len(df)} samples from {CSV_FILE}")
print(f"[Data] Columns : {df.shape[1]-1} features + 1 label")

print("\n[Data] Samples per gesture:")
counts = df["label"].value_counts()
for label, count in counts.items():
    bar = "█" * (count // 5) if (count // 5) > 0 else "▏"
    print(f"  {label:<15} {count:>4}  {bar}")

# ─── 2. Prepare features & labels ────────────────────────────────────────────
X = df.drop("label", axis=1).values.astype(np.float32)
y_raw = df["label"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
NUM_CLASSES = len(encoder.classes_)

print(f"\n[Labels] {NUM_CLASSES} gestures: {list(encoder.classes_)}")

with open(LABELS_OUT, "w") as f:
    for label in encoder.classes_:
        f.write(label + "\n")
print(f"[Labels] Saved to {LABELS_OUT}")

# ─── 3. Normalize features ────────────────────────────────────────────────────
X_min = X.min(axis=0)
X_max = X.max(axis=0)
denom = (X_max - X_min)
denom[denom == 0] = 1.0  
X_norm = (X - X_min) / denom

np.save("feature_min.npy", X_min)
np.save("feature_max.npy", X_max)
print("[Norm] Saved feature_min.npy / feature_max.npy")

# ─── 4. Train / validation split ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[Split] Train: {len(X_train)}  |  Val: {len(X_val)}")

# ─── 5. Build model ───────────────────────────────────────────────────────────
FEATURE_DIM = X_train.shape[1]

# ปรับการดึง Layers ผ่าน tf.keras โดยตรง เพื่อรองรับ TensorFlow 2.16+ อย่างสมบูรณ์
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURE_DIM,)),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(DROPOUT),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(DROPOUT),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(DROPOUT * 0.5),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
], name="gesture_classifier")

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ─── 6. Callbacks ─────────────────────────────────────────────────────────────
cb_early  = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
cb_reduce = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, verbose=1)
cb_ckpt   = tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1)

# ─── 7. Train ─────────────────────────────────────────────────────────────────
print("\n[Train] Starting...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_early, cb_reduce, cb_ckpt],
    verbose=1,
)

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
print("\n[Eval] Best model on validation set:")
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"  Accuracy : {acc*100:.1f}%")
print(f"  Loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\n[Classification Report]")
print(classification_report(y_val, y_pred, target_names=encoder.classes_))

# ─── 9. Plot training curves + confusion matrix ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Gesture Model — Training Report", fontsize=14, fontweight="bold")

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=axes[2],
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_,
            cmap="Blues")
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(REPORT_OUT, dpi=120)
print(f"\n[Plot] Saved training report to {REPORT_OUT}")

# ─── 10. Summary & Precise Word Export ────────────────────────────────────────
# บังคับบันทึกคำศัพท์โดยดึงอาเรย์มาจาก encoder โดยตรง ป้องกันคลาสสลับในระบบทำนายผล
trained_words = list(encoder.classes_)
np.save('gesture_words.npy', np.array(trained_words))

print("\n" + "=" * 50)
print("   Training complete!")
print(f"   Model  → {MODEL_OUT}")
print(f"   Labels → {LABELS_OUT}")
print(f"   Report → {REPORT_OUT}")
print(f"   Words  → gesture_words.npy (Locked Sequence)")
print(f"   Val accuracy: {acc*100:.1f}%")
print("=" * 50)
print(f"[SYSTEM] Exported unique array: {trained_words}")
print("\nNext: run phase3_inference.py to test live predictions.")