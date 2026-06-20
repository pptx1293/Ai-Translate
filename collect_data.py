"""
collect_data.py — Phase 1: Export hand landmark features to CSV.

Hand-only mode: no Pose detector, no arm/body landmarks.
Feature extraction imported from features.py (single source of truth).
"""

import csv
import glob
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from features import extract_features, FEATURE_LEN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR    = "picture_file"
CSV_FILE     = "gesture_data.csv"
TARGET_W     = 1280
TARGET_H     = 720

CTRL_CLASSES = {"neutral", "idle"}
IMG_EXTS     = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# ── MediaPipe — Hands only, no Pose ──────────────────────────────────────────
mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode        = True,
    max_num_hands            = 2,
    model_complexity         = 1,
    min_detection_confidence = 0.4,
)


# ── Result wrapper ────────────────────────────────────────────────────────────
class _Results:
    def __init__(self, hands_res):
        self.pose_landmarks       = None
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


# ── Processing ────────────────────────────────────────────────────────────────
_checksum_printed = False


def process_folder(folder: str, label: str, writer) -> tuple[int, int]:
    global _checksum_printed

    # Find all images recursively
    img_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                img_paths.append(os.path.join(root, file))
    
    # Find all videos recursively
    video_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.flv']:
                video_paths.append(os.path.join(root, file))

    is_ctrl = label in CTRL_CLASSES
    success = 0
    total_sources = len(img_paths) + len(video_paths)

    if total_sources == 0:
        return 0, 0

    # 1. Process Images
    for img_path in sorted(img_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"   WARNING  Unreadable image: {img_path}")
            continue

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_res = hands_detector.process(rgb)
        results   = _Results(hands_res)

        has_hand = results.left_hand_landmarks or results.right_hand_landmarks
        if not is_ctrl and not has_hand:
            continue

        feats = extract_features(results)

        if not _checksum_printed:
            print(f"[SYNC CHECK] First feature checksum = {sum(feats):.6f}")
            print(f"             Compare to tst.py startup checksum.")
            _checksum_printed = True

        writer.writerow([label] + feats)
        success += 1

    # 2. Process Videos
    for video_path in sorted(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   WARNING  Unreadable video: {video_path}")
            continue

        frame_idx = 0
        video_success = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_res = hands_detector.process(rgb)
            results   = _Results(hands_res)

            has_hand = results.left_hand_landmarks or results.right_hand_landmarks
            if not is_ctrl and not has_hand:
                continue

            feats = extract_features(results)

            if not _checksum_printed:
                print(f"[SYNC CHECK] First feature checksum = {sum(feats):.6f}")
                print(f"             Compare to tst.py startup checksum.")
                _checksum_printed = True

            writer.writerow([label] + feats)
            success += 1
            video_success += 1

        cap.release()
        if video_success > 0:
            print(f" (video '{os.path.basename(video_path)}': {video_success}/{frame_idx} frames exported)", end="", flush=True)

    return success, total_sources


def main() -> None:
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] Directory not found: '{IMAGE_DIR}'")
        sys.exit(1)

    subfolders = sorted(f.path for f in os.scandir(IMAGE_DIR) if f.is_dir())
    if not subfolders:
        print(f"[ERROR] No sub-folders in '{IMAGE_DIR}'")
        sys.exit(1)

    if os.path.isfile(CSV_FILE):
        os.remove(CSV_FILE)
        print(f"[CSV] Removed old '{CSV_FILE}' — rebuilding from scratch.")

    total_rows = 0
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_LEN)])
        print(f"[CSV] Created '{CSV_FILE}' ({FEATURE_LEN} features per row)\n")

        print(f"-- {len(subfolders)} categories --")
        for folder in subfolders:
            label = os.path.basename(folder).strip().lower()
            print(f"  ->  '{label}' ...", end=" ", flush=True)
            ok, total = process_folder(folder, label, writer)
            print(f"{ok}/{total} rows exported")
            total_rows += ok
            fh.flush()

    hands_detector.close()
    print(f"\n[DONE] {total_rows} rows written to '{CSV_FILE}'")
    print("       -> Now run train_model.py to rebuild the model.")


if __name__ == "__main__":
    main()