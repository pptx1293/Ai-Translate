"""
Phase 2b — Model Training for Arm and Hand Architecture
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
from keras import layers, models, callbacks

CSV_FILE    = "gesture_data.csv"
MODEL_OUT   = "gesture_model.keras"
REPORT_OUT  = "training_report.png"
EPOCHS      = 100
BATCH_SIZE  = 16
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"[ERROR] Run collect_data.py first to create {CSV_FILE}")

# --- NEW FIXED CODE ---
# Change header=None to header=0 to recognize the text headers automatically
df = pd.read_csv(CSV_FILE, header=0)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Using .iloc is still perfectly safe, but now it only captures the numerical data rows!
X = df.iloc[:, 1:].values.astype("float32")
y_raw = df.iloc[:, 0].values
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Restores the exact mathematically optimal weights cleanly
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_decay = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

print("\n[➔] Training Model on Arm & Hand Vectors...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop, lr_decay], verbose=1)

# CRITICAL FIX: Save components FIRST so weights are locked, or run prediction now 
# that .fit() has closed and restored the optimal weight states.
print("\n[➔] Evaluating model prediction states using restored optimal weights...")
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_val, y_pred, target_names=encoder.classes_))

# ==========================================
#        GENERATING TRAINING REPORT
# ==========================================
print(f"[➔] Compiling metrics map visual representation into {REPORT_OUT}...")

# Initialize a grid canvas for curves + confusion matrix layout
fig = plt.figure(figsize=(15, 5))
grid = plt.GridSpec(1, 3, wspace=0.3)

# 1. Accuracy Curve Subplot
ax_acc = fig.add_subplot(grid[0, 0])
ax_acc.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
ax_acc.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linestyle='--', linewidth=2)
ax_acc.set_title('Model Accuracy History', fontsize=12, fontweight='bold', pad=10)
ax_acc.set_xlabel('Epochs', fontsize=10)
ax_acc.set_ylabel('Accuracy Score', fontsize=10)
ax_acc.grid(True, linestyle=':', alpha=0.6)
ax_acc.legend(loc='lower right')

# 2. Loss Curve Subplot
ax_loss = fig.add_subplot(grid[0, 1])
ax_loss.plot(history.history['loss'], label='Train Loss', color='#d62728', linewidth=2)
ax_loss.plot(history.history['val_loss'], label='Val Loss', color='#2ca02c', linestyle='--', linewidth=2)
ax_loss.set_title('Model Loss History', fontsize=12, fontweight='bold', pad=10)
ax_loss.set_xlabel('Epochs', fontsize=10)
ax_loss.set_ylabel('Cross-Entropy Loss', fontsize=10)
ax_loss.grid(True, linestyle=':', alpha=0.6)
ax_loss.legend(loc='upper right')

# 3. Confusion Matrix Subplot (Now matches saved model exactly)
ax_cm = fig.add_subplot(grid[0, 2])
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            annot_kws={"size": 11, "weight": "bold"}, ax=ax_cm)
ax_cm.set_title('Confusion Matrix Heatmap', fontsize=12, fontweight='bold', pad=10)
ax_cm.set_xlabel('Predicted Gesture Label', fontsize=10, labelpad=8)
ax_cm.set_ylabel('True Gesture Label', fontsize=10, labelpad=8)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Save the unified visual analytics canvas
plt.savefig(REPORT_OUT, bbox_inches='tight', dpi=150)
plt.close()
print(f"[SUCCESS] Metrics report compiled and exported safely!")

# Save necessary translation assets
np.save('gesture_words.npy', np.array(list(encoder.classes_)))
model.save(MODEL_OUT)
print(f"[SUCCESS] Exported components to {MODEL_OUT} and gesture_words.npy")