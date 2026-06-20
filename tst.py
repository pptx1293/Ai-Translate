"""
tst.py — Inference Engine (hand-only mode)

Feature extraction imported from features.py.
No Pose model — classification uses hand landmarks only.
"""

# Protobuf compatibility shim (must run before TF import)
from google.protobuf import symbol_database as _symbol_database
_orig = _symbol_database.Default().GetPrototype
def _patched(descriptor):
    try:
        return _orig(descriptor)
    except Exception:
        from google.protobuf import message
        return message.Message
_symbol_database.Default().GetPrototype = _patched

import logging
import os
import sys

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from features import extract_features as _extract_features, FEATURE_LEN, _coords

MODEL_PATH = "gesture_model.keras"
WORDS_PATH = "gesture_words.npy"

ACTIONS: np.ndarray          = np.array([])
model:   tf.keras.Model|None = None

LETTER_GESTURES = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "hello",
    "backspace", "me", "thank you", "you", "space",
}

_SMOOTH_WINDOW = 8
_prob_buffer: list[np.ndarray] = []
_skip_next_frame: bool = False


# ── Custom objects for loading model ──────────────────────────────────────────
class RandomFeatureDropout(tf.keras.layers.Layer):
    def __init__(self, rate=0.08, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    def call(self, inputs, training=None):
        return inputs
    def get_config(self):
        cfg = super().get_config()
        cfg["rate"] = self.rate
        return cfg

def loss_fn(y_true, y_pred):
    return y_pred


# ── Public API ────────────────────────────────────────────────────────────────

def init_engine() -> bool:
    global ACTIONS, model
    ok = True

    if os.path.exists(WORDS_PATH):
        ACTIONS = np.load(WORDS_PATH, allow_pickle=True)
    else:
        print(f"[tst] WARNING: label file not found → {WORDS_PATH}")
        ok = False

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'RandomFeatureDropout': RandomFeatureDropout,
                    'loss_fn': loss_fn
                }
            )
            # Warm-up
            model.predict(np.zeros((1, FEATURE_LEN), dtype=np.float32), verbose=0)
            print(f"[tst] Model ready — {len(ACTIONS)} classes: {list(ACTIONS)}")
        except Exception as e:
            print(f"[tst] ERROR loading model: {e}")
            ok = False
    else:
        print(f"[tst] WARNING: model not found → {MODEL_PATH}")
        ok = False

    return ok


def translate_frame(results) -> tuple[str, float]:
    global _prob_buffer
    if model is None or len(ACTIONS) == 0:
        return "Engine Not Initialized", 0.0
    try:
        features = _extract_features(results)

        probs = model(
            tf.expand_dims(tf.constant(features), 0), training=False
        ).numpy()[0]

        _prob_buffer.append(probs)
        if len(_prob_buffer) > _SMOOTH_WINDOW:
            _prob_buffer.pop(0)

        avg_probs = np.mean(_prob_buffer, axis=0)

        idx   = int(np.argmax(avg_probs))
        label = str(ACTIONS[idx])
        conf  = float(avg_probs[idx])

        override = _shape_override(results, label, conf)
        if override:
            print(f"[tst DEBUG] Model: {label} ({conf:.2f}) -> OVERRIDDEN TO -> {override[0]} ({override[1]:.2f})")
        else:
            # Only log if a hand is detected (i.e. results has hands)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                print(f"[tst DEBUG] Model: {label} ({conf:.2f}) -> NO OVERRIDE")
        return override if override else (label, conf)
    except Exception as exc:
        _prob_buffer.clear()
        return f"Error: {exc}", 0.0


def reset_smooth_buffer() -> None:
    global _prob_buffer
    _prob_buffer.clear()


# ── Shape overrides ───────────────────────────────────────────────────────────

def _shape_override(results, label: str, conf: float) -> tuple[str, float] | None:
    if label.lower() not in LETTER_GESTURES:
        return None

    shape_best = None
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is None:
            continue
        candidate = _classify_letter_shape(hand)
        if candidate:
            print(f"[tst DEBUG] Hand shape detected: {candidate[0]} ({candidate[1]:.2f})")
        if candidate and (shape_best is None or candidate[1] > shape_best[1]):
            shape_best = candidate

    if shape_best is None:
        return None

    shape_label, shape_conf = shape_best

    CONFUSION_GROUPS: list[set[str]] = [
        {"a", "c", "e", "m", "n", "o", "s", "t"},
        {"g", "h"},
        {"b", "d", "hello"},
        {"u", "r", "v", "k"},
        {"i", "y", "j"},
        {"f", "w"},
        {"p", "q"},
        {"l"},
    ]
    model_group = next((g for g in CONFUSION_GROUPS if label.lower() in g), None)
    if model_group is None or shape_label not in model_group:
        if model_group is None:
            print(f"[tst DEBUG] Override rejected: Model predicted label '{label}' is not in any confusion group.")
        else:
            print(f"[tst DEBUG] Override rejected: Detected shape '{shape_label}' is not in the model's confusion group {model_group}.")
        return None

    if shape_conf >= 0.88:
        return shape_label, max(conf, shape_conf)
    return None


def _classify_letter_shape(hand_landmarks) -> tuple[str, float] | None:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist  = points[0]
    scale  = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in points)
    if scale < 1e-6:
        return None
    pts = [(pt - wrist) / scale for pt in points]

    def _up(pip_i: int, tip_i: int, thresh: float = 0.18) -> bool:
        pip, tip = pts[pip_i], pts[tip_i]
        return bool(
            np.linalg.norm(tip[:2]) > np.linalg.norm(pip[:2]) + thresh
            and tip[1] < pip[1] + 0.16
        )

    def _side(mcp_i: int, tip_i: int, thresh: float = 0.28) -> bool:
        mcp, tip = pts[mcp_i], pts[tip_i]
        return bool(abs(tip[0] - mcp[0]) > thresh)

    def _curled(mcp_i: int, tip_i: int, thresh: float = 0.25) -> bool:
        return bool(np.linalg.norm((pts[tip_i] - pts[mcp_i])[:2]) < thresh)

    # Unit vector of horizontal axis (pinky MCP to index MCP)
    x_dir = pts[5][:2] - pts[17][:2]
    x_norm = np.linalg.norm(x_dir)
    if x_norm > 1e-6:
        x_dir = x_dir / x_norm
    else:
        x_dir = np.array([1.0, 0.0], dtype=np.float32)

    idx_up  = _up(6,  8)
    mid_up  = _up(10, 12)
    rng_up  = _up(14, 16)
    pky_up  = _up(18, 20)
    idx_curl = _curled(5,  8)
    mid_curl = _curled(9,  12)
    rng_curl = _curled(13, 16)
    pky_curl = _curled(17, 20)

    # G vs H
    idx_side = _side(5, 8)
    mid_side = _side(9, 12)
    if idx_side and mid_side and rng_curl and pky_curl:
        return "h", 0.93
    thumb_ext = float(np.linalg.norm((pts[4] - pts[2])[:2]))
    if idx_side and not mid_side and mid_curl and rng_curl and pky_curl and thumb_ext > 0.26:
        return "g", 0.93

    # B vs HELLO
    if idx_up and mid_up and rng_up and pky_up:
        d_mid_pip = float(np.linalg.norm((pts[4] - pts[10])[:2]))
        d_pky_pip = float(np.linalg.norm((pts[4] - pts[18])[:2]))
        if d_mid_pip <= 0.25:
            return "hello", 0.96
        else:
            if d_pky_pip <= 0.11:
                return "hello", 0.96
            else:
                return "b", 0.96

    # D
    if idx_up and not mid_up and not rng_up and not pky_up and mid_curl:
        return "d", 0.96

    # L
    thumb_ext = float(np.linalg.norm((pts[4] - pts[2])[:2]))
    if idx_up and not mid_up and not rng_up and not pky_up and thumb_ext > 0.26:
        return "l", 0.94

    # Y vs I vs J
    if pky_up and not idx_up and not mid_up and not rng_up:
        d_thumb_pky_tip = float(np.linalg.norm((pts[4] - pts[20])[:2]))
        d_thumb_idx_mcp = float(np.linalg.norm((pts[4] - pts[5])[:2]))
        thumb_ext = float(np.linalg.norm((pts[4] - pts[2])[:2]))
        
        if d_thumb_pky_tip > 0.55 and d_thumb_idx_mcp > 0.20 and thumb_ext > 0.26:
            return "y", 0.93
        else:
            return "i", 0.93

    # P vs Q
    idx_down = pts[8][1] > pts[5][1] + 0.1
    mid_down = pts[12][1] > pts[9][1] + 0.1
    if idx_down and not idx_curl:
        d_mid_tip = float(np.linalg.norm((pts[12] - pts[9])[:2]))
        d_thumb_mid_pip = float(np.linalg.norm((pts[4] - pts[10])[:2]))
        thumb_x = float(np.dot(pts[4][:2], x_dir))
        
        if mid_down and d_mid_tip > 0.20 and d_thumb_mid_pip < 0.18 and thumb_x > 0.0:
            return "p", 0.95
        else:
            return "q", 0.95

    # Unified A / C / E / M / N / O / S / T (Fist & Rounded gestures)
    d_idx_curl = float(np.linalg.norm((pts[8] - pts[5])[:2]))
    if not idx_up and not mid_up and not rng_up and not pky_up and d_idx_curl <= 0.32:
        d_idx_mcp = float(np.linalg.norm((pts[4] - pts[5])[:2]))
        d_mid_mcp = float(np.linalg.norm((pts[4] - pts[9])[:2]))
        d_rng_mcp = float(np.linalg.norm((pts[4] - pts[13])[:2]))
        d_idx_pip = float(np.linalg.norm((pts[4] - pts[6])[:2]))
        d_mid_pip = float(np.linalg.norm((pts[4] - pts[10])[:2]))
        d_pky_pip = float(np.linalg.norm((pts[4] - pts[18])[:2]))
        thumb_ext = float(np.linalg.norm((pts[4] - pts[2])[:2]))
        d_thumb_idx_tip = float(np.linalg.norm((pts[4] - pts[8])[:2]))
        thumb_x = float(np.dot(pts[4][:2], x_dir))
        
        if d_mid_pip <= 0.19:
            if d_idx_pip <= 0.16:
                if d_thumb_idx_tip <= 0.22:
                    if d_idx_pip <= 0.12:
                        return "s", 0.95
                    else:
                        return "e", 0.95
                else:
                    return "t", 0.95
            else:
                if d_thumb_idx_tip <= 0.28:
                    return "t", 0.95
                else:
                    return "n", 0.95
        else:
            if d_mid_mcp <= 0.15:
                if d_idx_pip <= 0.30:
                    if thumb_x <= 0.25:
                        return "m", 0.95
                    else:
                        if d_thumb_idx_tip <= 0.20:
                            return "s", 0.95
                        else:
                            return "m", 0.95
                else:
                    if d_pky_pip <= 0.28:
                        return "e", 0.95
                    else:
                        if d_rng_mcp <= 0.21:
                            return "m", 0.95
                        else:
                            return "o", 0.95
            else:
                if d_thumb_idx_tip <= 0.11:
                    if d_mid_pip <= 0.28:
                        return "s", 0.95
                    else:
                        return "o", 0.95
                else:
                    if thumb_ext <= 0.41:
                        if d_mid_pip <= 0.35:
                            return "s", 0.95
                        else:
                            return "c", 0.95
                    else:
                        return "a", 0.96

    # O vs C fallback (if index finger is slightly extended up but others are down)
    if not mid_up and not rng_up and not pky_up:
        d_thumb_idx_tip = float(np.linalg.norm((pts[4] - pts[8])[:2]))
        if d_thumb_idx_tip < 0.25:
            return "o", 0.95
        elif d_thumb_idx_tip >= 0.25 and not idx_curl and not mid_curl:
            return "c", 0.95

    # U, R, V, K group
    if idx_up and mid_up and not rng_up and not pky_up:
        d_thumb_mid_pip = float(np.linalg.norm((pts[4] - pts[10])[:2]))
        tip_dist = float(np.linalg.norm((pts[8] - pts[12])[:2]))
        
        if tip_dist <= 0.09:
            dip_sep = float(np.linalg.norm((pts[7] - pts[11])[:2]))
            
            x_idx = float(np.dot(pts[8][:2], x_dir))
            x_mid = float(np.dot(pts[12][:2], x_dir))
            proj_diff = x_idx - x_mid
            
            if dip_sep <= 0.04:
                return "r", 0.91
            elif proj_diff <= -0.01:
                return "r", 0.91
            else:
                return "u", 0.91
        else: # tip_dist > 0.09
            if d_thumb_mid_pip <= 0.23:
                d_mid_tip_idx_pip = float(np.linalg.norm((pts[12] - pts[6])[:2]))
                if d_mid_tip_idx_pip <= 0.25:
                    return "r", 0.91
                else:
                    return "k", 0.91
            else:
                return "v", 0.91

    # W
    if idx_up and mid_up and rng_up and not pky_up:
        return "w", 0.90

    # F
    if mid_up and rng_up and pky_up and not idx_up:
        if float(np.linalg.norm((pts[8] - pts[4])[:2])) < 0.25:
            return "f", 0.91

    # Handled in the Unified A/C/E/M/N/O/S/T block above

    return None

init_engine()