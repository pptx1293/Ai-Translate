"""
tst.py — Inference Engine

Feature extraction here is IDENTICAL to collect_data.py extract_features().
Both files must stay in sync — if you change one, change the other.
"""

# ── Protobuf compatibility shim (must run before TF import) ──────────────────
from google.protobuf import symbol_database as _symbol_database
_orig = _symbol_database.Default().GetPrototype
def _patched(descriptor):
    try:
        return _orig(descriptor)
    except Exception:
        from google.protobuf import message
        return message.Message
_symbol_database.Default().GetPrototype = _patched
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import sys

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = "gesture_model.keras"
WORDS_PATH = "gesture_words.npy"

ARM_IDX     = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FEATURE_LEN = len(ARM_IDX) * 3 + 21 * 3 + 21 * 3   # 162

ACTIONS: np.ndarray          = np.array([])
model:   tf.keras.Model|None = None
LETTER_GESTURES = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
}

# Sliding window: accumulate raw softmax vectors over recent frames.
# The averaged probability is much more stable than any single frame,
# which is the core fix for "A reads as M" style single-frame flips.
_SMOOTH_WINDOW  = 8           # frames to average (≈ 270 ms at 30 fps)
_prob_buffer = []


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
        model = tf.keras.models.load_model(MODEL_PATH)
        # Warm-up: eliminates the ~200 ms spike on the first real prediction
        model.predict(np.zeros((1, FEATURE_LEN), dtype=np.float32), verbose=0)
        print(f"[tst] Model ready — {len(ACTIONS)} classes: {list(ACTIONS)}")
    else:
        print(f"[tst] WARNING: model not found → {MODEL_PATH}")
        ok = False

    return ok


def translate_frame(results) -> tuple[str, float]:
    """
    Accepts _Results from main.py (separate Hands + Pose, already flipped).
    Returns (label, confidence).

    Uses a sliding-window average of the last _SMOOTH_WINDOW softmax vectors
    before picking the winner.  This prevents a single noisy frame from
    flipping the prediction (e.g. A → M) while adding only ~5 frames of lag.
    """
    global _prob_buffer
    if model is None or len(ACTIONS) == 0:
        return "Engine Not Initialized", 0.0
    try:
        features = _extract(results)

        # model(x) is faster than model.predict(x) for single samples
        probs = model(
            tf.expand_dims(tf.constant(features), 0), training=False
        ).numpy()[0]

        # Accumulate into sliding window
        _prob_buffer.append(probs)
        if len(_prob_buffer) > _SMOOTH_WINDOW:
            _prob_buffer.pop(0)

        # Average probabilities across the window
        avg_probs = np.mean(_prob_buffer, axis=0)

        idx   = int(np.argmax(avg_probs))
        label = str(ACTIONS[idx])
        conf  = float(avg_probs[idx])

        override = _shape_override(results, label, conf)
        return override if override else (label, conf)
    except Exception as exc:
        _prob_buffer.clear()
        return f"Error: {exc}", 0.0


def reset_smooth_buffer() -> None:
    """Call this whenever the gesture buffer is cleared in main.py (e.g. after a word fires)."""
    global _prob_buffer
    _prob_buffer.clear()


# ── Feature extraction — IDENTICAL to collect_data.py extract_features() ─────

def _shape_override(results, label: str, conf: float) -> tuple[str, float] | None:
    """
    Geometric safety net — only fires when BOTH:
      1. The neural net predicted something in a known confusion group
      2. The geometric classifier is highly confident about a different label in that group
    This means it never overrides unambiguous predictions.
    """
    if label.lower() not in LETTER_GESTURES:
        return None

    shape_best = None
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is None:
            continue
        candidate = _classify_letter_shape(hand)
        if candidate and (shape_best is None or candidate[1] > shape_best[1]):
            shape_best = candidate

    if shape_best is None:
        return None

    shape_label, shape_conf = shape_best

    # Only override within the same confusion group
    CONFUSION_GROUPS: list[set[str]] = [
        {"a", "s", "m", "n", "e"},   # fist / near-fist variants
        {"g", "h"},                   # sideways pointing
        {"b", "d"},                   # all-up vs one-up
        {"u", "r"},                   # two-up parallel vs crossed
        {"v", "k"},                   # two-up spread vs thumb-out
        {"i", "y"},                   # pinky-only vs pinky+thumb
        {"f", "w"},                   # three-finger variants
        {"l"},
    ]
    model_group = next((g for g in CONFUSION_GROUPS if label.lower() in g), None)
    if model_group is None or shape_label not in model_group:
        return None

    if shape_conf >= 0.88:
        return shape_label, max(conf, shape_conf)
    return None


def _classify_letter_shape(hand_landmarks) -> tuple[str, float] | None:  # noqa: C901
    """
    Pure geometry on normalised landmarks.
    Returns (label, confidence) or None if no rule fires confidently.
    """
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist  = points[0]
    scale  = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in points)
    if scale < 1e-6:
        return None
    pts = [(pt - wrist) / scale for pt in points]

    # ── Helpers ───────────────────────────────────────────────────────────────
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

    def _thumb_extended() -> bool:
        return bool(np.linalg.norm((pts[4] - pts[2])[:2]) > 0.45)

    def _thumb_tucked() -> bool:
        return bool(np.linalg.norm((pts[4] - pts[5])[:2]) < 0.32)

    idx_up  = _up(6,  8)
    mid_up  = _up(10, 12)
    rng_up  = _up(14, 16)
    pky_up  = _up(18, 20)
    idx_curl = _curled(5,  8)
    mid_curl = _curled(9,  12)
    rng_curl = _curled(13, 16)
    pky_curl = _curled(17, 20)

    # ── G vs H (sideways hand) ───────────────────────────────────────────────
    idx_side = _side(5, 8)
    mid_side = _side(9, 12)
    if idx_side and mid_side and rng_curl and pky_curl:
        return "h", 0.93
    if idx_side and not mid_side and mid_curl and rng_curl and pky_curl and _thumb_extended():
        return "g", 0.93

    # ── B: all 4 up, thumb tucked ────────────────────────────────────────────
    if idx_up and mid_up and rng_up and pky_up and _thumb_tucked():
        return "b", 0.96

    # ── D: index up only ─────────────────────────────────────────────────────
    if idx_up and not mid_up and not rng_up and not pky_up and mid_curl:
        return "d", 0.96

    # ── L: index up + thumb out ──────────────────────────────────────────────
    if idx_up and not mid_up and not rng_up and not pky_up and _thumb_extended():
        return "l", 0.94

    # ── I: pinky only, thumb tucked ─────────────────────────────────────────
    if pky_up and not idx_up and not mid_up and not rng_up and not _thumb_extended():
        return "i", 0.93

    # ── Y: pinky + thumb extended ───────────────────────────────────────────
    if pky_up and not idx_up and not mid_up and not rng_up and _thumb_extended():
        return "y", 0.93

    # ── U vs R: index + middle up, ring + pinky curled ──────────────────────
    if idx_up and mid_up and not rng_up and not pky_up and not _thumb_extended():
        x_sep = abs(pts[8][0] - pts[12][0])
        if x_sep < 0.10:
            return "r", 0.91
        else:
            return "u", 0.91

    # ── V vs K ───────────────────────────────────────────────────────────────
    if idx_up and mid_up and not rng_up and not pky_up:
        tip_sep = abs(pts[8][0] - pts[12][0])
        if _thumb_extended() and tip_sep > 0.15:
            return "k", 0.91
        elif tip_sep > 0.20:
            return "v", 0.91

    # ── W: index + middle + ring up ─────────────────────────────────────────
    if idx_up and mid_up and rng_up and not pky_up:
        return "w", 0.90

    # ── F: middle/ring/pinky up, index touches thumb ─────────────────────────
    if mid_up and rng_up and pky_up and not idx_up:
        if float(np.linalg.norm((pts[8] - pts[4])[:2])) < 0.25:
            return "f", 0.91

    # ── A / S / M / N / E — fist variants ───────────────────────────────────
    all_curled = idx_curl and mid_curl and rng_curl and pky_curl
    if all_curled:
        thumb_tip_y  = pts[4][1]
        index_pip_y  = pts[6][1]

        # Distances from thumb tip to each finger's MCP — used to tell A
        # (thumb beside index, not tucked under anything) apart from
        # M/N (thumb tucked progressively deeper under the fingers).
        thumb_to_idx_mcp  = float(np.linalg.norm((pts[4] - pts[5])[:2]))
        thumb_to_mid_mcp  = float(np.linalg.norm((pts[4] - pts[9])[:2]))
        thumb_to_ring_mcp = float(np.linalg.norm((pts[4] - pts[13])[:2]))

        # Check tuck state FIRST. A tucked thumb can spuriously satisfy
        # _thumb_extended()'s raw 2D distance check if it's pressed flat
        # against the palm, so we must rule out M/N/E before falling
        # through to A — this was the source of A being misread.
        if _thumb_tucked():
            if thumb_to_ring_mcp < 0.30:
                return "m", 0.90
            elif thumb_to_mid_mcp < 0.30:
                return "n", 0.90
            else:
                return "e", 0.86   # fingers bent at PIP, thumb tucked under

        # A: thumb rests beside the index finger (not tucked under),
        # tip above the PIP line, and clearly NOT close to the middle/ring
        # MCPs (which would indicate a tucked thumb instead).
        if (
            thumb_tip_y < index_pip_y
            and _thumb_extended()
            and thumb_to_mid_mcp > 0.30
            and thumb_to_idx_mcp > 0.18
        ):
            return "a", 0.92

        # S: thumb crosses over the front of curled fingers — not
        # extended outward, not deeply tucked underneath.
        if not _thumb_extended():
            return "s", 0.87

    return None

def _coords(lm) -> np.ndarray:
    """Invert x to match the flipped-image coordinate space used in training."""
    return np.array([1.0 - lm.x, lm.y, lm.z], dtype=np.float32)


def _normalise_points(points: list[np.ndarray], origin: np.ndarray, scale: float) -> list[float]:
    if scale < 1e-6:
        scale = 1.0
    normalised: list[float] = []
    for pt in points:
        normalised.extend(((pt - origin) / scale).tolist())
    return normalised


def _normalised_hand(hand_landmarks) -> list[float]:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist = points[0]
    scale = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in points)
    return _normalise_points(points, wrist, scale)


def _extract(results) -> np.ndarray:
    """
    162-dim feature vector.
    
    Coordinates are normalised so gesture shape matters more than where the
    hand appears in the camera frame.
    """
    feats: list[float] = []

    # 1 ── Arm landmarks from Pose (index-pair swap + x inversion)
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        arm_points: list[np.ndarray] = []
        for i in ARM_IDX:
            j = (i + 1) if (i % 2 != 0) else (i - 1)
            arm_points.append(_coords(lms[j]) if j < len(lms) else np.zeros(3, dtype=np.float32))
        shoulder_mid = (arm_points[0] + arm_points[1]) / 2.0
        shoulder_width = float(np.linalg.norm((arm_points[0] - arm_points[1])[:2]))
        feats += _normalise_points(arm_points, shoulder_mid, shoulder_width)
    else:
        feats += [0.0] * (len(ARM_IDX) * 3)

    # 2 ── Left hand (user's physical left, corrected by _Results)
    if results.left_hand_landmarks:
        feats += _normalised_hand(results.left_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    # 3 ── Right hand (user's physical right, corrected by _Results)
    if results.right_hand_landmarks:
        feats += _normalised_hand(results.right_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    return np.array(feats, dtype=np.float32)

init_engine()