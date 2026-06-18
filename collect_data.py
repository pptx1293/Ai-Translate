import csv
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR    = "picture_file"
CSV_FILE     = "gesture_data.csv"
TARGET_W     = 1280
TARGET_H     = 720

ARM_IDX      = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FEATURE_LEN  = len(ARM_IDX) * 3 + 21 * 3 + 21 * 3   # 162
CTRL_CLASSES = {"neutral", "idle"}
IMG_EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# ── MediaPipe — separate Hands + Pose, NO Holistic, NO face ──────────────────
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose

hands_detector = mp_hands.Hands(
    static_image_mode        = True,
    max_num_hands            = 2,
    model_complexity         = 1,
    min_detection_confidence = 0.4,
)
pose_detector = mp_pose.Pose(
    static_image_mode        = True,
    model_complexity         = 2,
    enable_segmentation      = False,
    min_detection_confidence = 0.4,
)


# ── Result wrapper — IDENTICAL to main.py _Results ───────────────────────────
class _Results:
    """
    After cv2.flip(frame, 1) MediaPipe Hands labels are mirror-swapped:
      MP "Left"  on flipped image = user's physical RIGHT hand
      MP "Right" on flipped image = user's physical LEFT  hand
    We store them swapped so attributes always reflect the user's real hand.
    Must match main.py _Results exactly.
    """
    def __init__(self, hands_res, pose_res):
        self.pose_landmarks       = pose_res.pose_landmarks if pose_res else None
        self.left_hand_landmarks  = None
        self.right_hand_landmarks = None
        if hands_res and hands_res.multi_hand_landmarks:
            for lms, handed in zip(
                hands_res.multi_hand_landmarks,
                hands_res.multi_handedness,
            ):
                mp_label = handed.classification[0].label
                if mp_label == "Left":
                    self.right_hand_landmarks = lms   # swapped
                else:
                    self.left_hand_landmarks  = lms   # swapped


# ── Feature extraction — IDENTICAL to tst.py _extract() ──────────────────────
def _coords(lm) -> np.ndarray:
    """Invert x to match flipped-image coordinate space."""
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


def extract_features(results) -> list[float]:
    """
    162-dim feature vector.
    MUST be byte-for-byte identical to tst.py _extract().
    Layout: [arm×36] [left_hand×63] [right_hand×63]
    """
    feats: list[float] = []

    # 1 ── Arm pose (index-pair swap + x inversion)
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        arm_points: list[np.ndarray] = []
        for i in ARM_IDX:
            j = (i + 1) if (i % 2 != 0) else (i - 1)
            arm_points.append(_coords(lms[j]) if j < len(lms) else np.zeros(3, dtype=np.float32))
        shoulder_mid   = (arm_points[0] + arm_points[1]) / 2.0
        shoulder_width = float(np.linalg.norm((arm_points[0] - arm_points[1])[:2]))
        feats += _normalise_points(arm_points, shoulder_mid, shoulder_width)
    else:
        feats += [0.0] * (len(ARM_IDX) * 3)

    # 2 ── Left hand (physical left — already corrected by _Results swap)
    if results.left_hand_landmarks:
        feats += _normalised_hand(results.left_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    # 3 ── Right hand (physical right — already corrected by _Results swap)
    if results.right_hand_landmarks:
        feats += _normalised_hand(results.right_hand_landmarks)
    else:
        feats += [0.0] * (21 * 3)

    return feats


# ── Processing ────────────────────────────────────────────────────────────────
def process_folder(folder: str, label: str, writer) -> tuple[int, int]:
    paths: list[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    if not paths:
        return 0, 0

    is_ctrl = label in CTRL_CLASSES
    success = 0

    for img_path in sorted(paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"   ⚠  Unreadable: {img_path}")
            continue

        # ── Must match main.py frame processing exactly ──
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        frame = cv2.flip(frame, 1)                      # ← same flip as main.py
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_res = hands_detector.process(rgb)
        pose_res  = pose_detector.process(rgb)
        results   = _Results(hands_res, pose_res)

        has_hand = results.left_hand_landmarks or results.right_hand_landmarks
        if not is_ctrl and not has_hand:
            continue

        writer.writerow([label] + extract_features(results))
        success += 1

    return success, len(paths)


def main() -> None:
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] Directory not found: '{IMAGE_DIR}'")
        sys.exit(1)

    subfolders = sorted(f.path for f in os.scandir(IMAGE_DIR) if f.is_dir())
    if not subfolders:
        print(f"[ERROR] No sub-folders in '{IMAGE_DIR}'")
        sys.exit(1)

    # Always rebuild CSV from scratch so no stale mismatched rows remain
    if os.path.isfile(CSV_FILE):
        os.remove(CSV_FILE)
        print(f"[CSV] Removed old '{CSV_FILE}' — rebuilding from scratch.")

    total_rows = 0
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_LEN)])
        print(f"[CSV] Created '{CSV_FILE}' ({FEATURE_LEN} features per row)\n")

        print(f"── {len(subfolders)} categories ──")
        for folder in subfolders:
            label = os.path.basename(folder).strip().lower()
            print(f"  ➔  '{label}' ...", end=" ", flush=True)
            ok, total = process_folder(folder, label, writer)
            print(f"{ok}/{total} rows exported")
            total_rows += ok
            fh.flush()

    hands_detector.close()
    pose_detector.close()
    print(f"\n[DONE] {total_rows} rows written to '{CSV_FILE}'")
    print("       → Now run train_model.py to rebuild the model.")


if __name__ == "__main__":
    main()