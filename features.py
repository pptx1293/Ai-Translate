"""
features.py — Single source of truth for feature extraction.

Both collect_data.py and tst.py import from here.
If you change the feature layout, change it ONLY here.

Hand-only mode: 126 features (left_hand×63 + right_hand×63).
Pose/arm landmarks removed — gestures are classified from hands alone.
"""

import numpy as np

FEATURE_LEN = 21 * 3 + 21 * 3 + 20 + 20   # 166  (left hand + right hand + relative thumb distances)


def _coords(lm) -> np.ndarray:
    """Invert x to match the flipped-image coordinate space used in training."""
    return np.array([1.0 - lm.x, lm.y, lm.z], dtype=np.float32)


def _normalise_points(
    points: list[np.ndarray], origin: np.ndarray, scale: float
) -> list[float]:
    if scale < 1e-6:
        scale = 1.0
    normalised: list[float] = []
    for pt in points:
        normalised.extend(((pt - origin) / scale).tolist())
    return normalised


def _normalised_hand(hand_landmarks) -> list[float]:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist  = points[0]
    scale  = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in points)
    return _normalise_points(points, wrist, scale)


def _thumb_distances(hand_landmarks) -> list[float]:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    thumb_tip = points[4]
    wrist = points[0]
    scale = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in points)
    if scale < 1e-6:
        scale = 1.0
    
    distances: list[float] = []
    for idx, pt in enumerate(points):
        if idx == 4:
            continue
        dist = float(np.linalg.norm(pt - thumb_tip)) / scale
        distances.append(dist)
    return distances


def extract_features(results) -> list[float]:
    """
    126-dim feature vector.
    Layout: [left_hand×63] [right_hand×63]

    'results' must expose:
        .left_hand_landmarks   (user's physical left)
        .right_hand_landmarks  (user's physical right)
    """
    feats: list[float] = []

    # Left hand (physical left — corrected by _Results swap in callers)
    if results.left_hand_landmarks:
        feats += _normalised_hand(results.left_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    # Right hand (physical right — corrected by _Results swap in callers)
    if results.right_hand_landmarks:
        feats += _normalised_hand(results.right_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    # Left hand thumb distances
    if results.left_hand_landmarks:
        feats += _thumb_distances(results.left_hand_landmarks)
    else:
        feats += [0.0] * 20

    # Right hand thumb distances
    if results.right_hand_landmarks:
        feats += _thumb_distances(results.right_hand_landmarks)
    else:
        feats += [0.0] * 20

    return feats


def feature_checksum(results) -> float:
    """
    Quick sanity check: call this on the first sample in both collect_data.py
    and tst.py startup and compare. If they differ, the feature contract is broken.
    """
    return float(np.sum(extract_features(results)))