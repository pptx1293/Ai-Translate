"""
Phase 2a — Data Collection (Live Widescreen Camera Version)
Captures holistic landmarks from a live webcam stream with custom extraction and white skeletons.
Strictly prevents invalid zero-padding data and hand-switching artifacts from corrupting the dataset.

How to use:
    1. Run: python collect_data.py
    2. Type a gesture label (e.g. "yes", "no", "help") and press Enter
    3. Stand in front of the camera
    4. Press SPACE to record a sample landmark row vector
    5. Press N to switch to the next gesture label
    6. Press Q when done — data is cleanly appended to gesture_data.csv
"""

import cv2
 # Placeholder assuming internal media mapping dependencies
import mediapipe as mp
import csv
import os
import sys
import numpy as np

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CSV_FILE = "gesture_data.csv"

# EXACT FIXED SIZES MATCHING YOUR TRANSLATION CONFIGURATION SCRIPT
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode        = False,
    model_complexity         = 1,
    smooth_landmarks         = True,
    refine_face_landmarks    = True,
    min_detection_confidence = 0.6,
    min_tracking_confidence  = 0.5,
)

# ─── Upper-body pose indices & Face Keys ──────────────────────────────────────
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FACE_KEY = [1, 4, 9, 10, 33, 37, 40, 46, 52, 55,
            61, 70, 105, 133, 152, 234, 263, 362, 397, 468]

# ─── Custom Feature Extraction Engine ──────────────────────────────────────────
def extract_feature_vector(results):

    features = []

   

    # 1. Pose (Upper Body)

    if results.pose_landmarks:

        pose_lms = results.pose_landmarks.landmark

        for i in POSE_IDX:

            if i < len(pose_lms):

                features += [pose_lms[i].x, pose_lms[i].y, pose_lms[i].z]

            else:

                features += [0.0, 0.0, 0.0]

    else:

        features += [0.0] * (len(POSE_IDX) * 3)



    # 2. Left Hand

    if results.left_hand_landmarks:

        for lm in results.left_hand_landmarks.landmark:

            features += [lm.x, lm.y, lm.z]

    else:

        features += [0.0] * (21 * 3)



    # 3. Right Hand

    if results.right_hand_landmarks:

        for lm in results.right_hand_landmarks.landmark:

            features += [lm.x, lm.y, lm.z]

    else:

        features += [0.0] * (21 * 3)



    # 4. Face (Key 20)

    if results.face_landmarks:

        face_lms = results.face_landmarks.landmark

        for i in FACE_KEY:

            if i < len(face_lms):

                features += [face_lms[i].x, face_lms[i].y, face_lms[i].z]

            else:

                features += [0.0, 0.0, 0.0]

    else:

        features += [0.0] * (20 * 3)



    # คืนค่ากลับไปเป็น List เพื่อใช้ทำงานกับ csv_writer.writerow ได้โดยไม่ติดขัด

    return features

FEATURE_LENGTH = len(POSE_IDX)*3 + 21*3 + 21*3 + 20*3  # = 225

# ─── CSV Initializer Engine ───────────────────────────────────────────────────
file_exists = os.path.exists(CSV_FILE)
csv_file   = open(CSV_FILE, "a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    csv_writer.writerow(header)
    print(f"[New] Created dataset target file: {CSV_FILE}")
else:
    print(f"[Append] Appending data rows to: {CSV_FILE}")

# ─── Initialize Live Webcam Stream ───────────────────────────────────────────
cap = cv2.VideoCapture(0)

# Set base native widescreen resolution parameters
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

# ─── State Controls ───────────────────────────────────────────────────────────
current_label   = ""
sample_count    = 0
total_count     = 0
recording_flash = 0   

# ─── Prompt for the initial tracking label ────────────────────────────────────
current_label = input("Enter gesture label to start (e.g. 'yes'): ").strip().lower()
print(f"\n→ Extracting coordinates for target: '{current_label}'")
print("   SPACE = record sample | N = switch label | Q = save & quit\n")

# ─── Main Execution Live Loop ─────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("\n[ERROR] Failed to access live camera device stream.")
        break

    # Mirror horizontally for user interface comfort
    frame = cv2.flip(frame, 1)
    
    # Hard scale frame size to ensure consistency with your calculation matrix
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    h, w, _ = frame.shape

    # Process frame through MediaPipe Holistic pipeline
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    # ── Draw White Skeleton Indicators ────────────────────────────────────────
    # เปลี่ยนการวาดเส้นโครงสร้างทั้งหมดเป็นสีขาวและจุดขาวมินิมอลตามที่คุณรีเควส
    white_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # ── Flash dynamic screen border on save actions ───────────────────────────
    if recording_flash > 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 100), 8)
        recording_flash -= 1

    # Detect visible modules
    visible = []
    if results.left_hand_landmarks:  visible.append("Left hand")
    if results.right_hand_landmarks: visible.append("Right hand")
    if results.pose_landmarks:       visible.append("Pose")
    if results.face_landmarks:       visible.append("Face")
    vis_str = ", ".join(visible) if visible else "Nothing detected"

    # ── UI HUD Display Overlays ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (450, 135), (0,0,0), -1)
    cv2.putText(frame, "SOURCE: LIVE CAMERA FEED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Label     : {current_label}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, f"Samples   : {sample_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,255,150), 2)
    cv2.putText(frame, f"Total Row : {total_count}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1)
    cv2.putText(frame, f"Visible   : {vis_str}", (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,100), 1)
    cv2.putText(frame, "SPACE=record row   N=next label   Q=exit save profile", (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

    if sample_count < 100:
        cv2.putText(frame, f"Targeting Samples: {sample_count}/100+", (w-320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)
    else:
        cv2.putText(frame, "Sample count goal met!", (w-260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)

    cv2.imshow("Phase 2a — Data Collection (Live Widescreen)", frame)

    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q'):
        print("\n[Manual Quit] Closing active pipelines...")
        break

    elif key == ord(' '):
        # ตรวจสอบการใช้งานคลาสปกติ (Neutral) เพื่อป้องกันระบบสับสน
        is_neutral_class = "neutral" in current_label or "idle" in current_label
        
        # กฎเหล็ก: ถ้าไม่ใช่คลาส neutral ระบบต้องตรวจจับเจอ "มืออย่างน้อยหนึ่งข้าง" ถึงจะยอมให้เซฟลงไฟล์
        if not is_neutral_class and not (results.left_hand_landmarks or results.right_hand_landmarks):
            print("\r  [BLOCKED] Cannot record gesture! No hands detected in frame.     ", end="")
            continue
            
        feats = extract_feature_vector(results)
        csv_writer.writerow([current_label] + feats)
        csv_file.flush()
        sample_count    += 1
        total_count     += 1
        recording_flash  = 4
        print(f"\r  [{current_label}] {sample_count} samples added to csv (visible: {vis_str})   ", end="")

    elif key == ord('n'):
        print(f"\n  ✓ Wrapped up recording session for label: '{current_label}' ({sample_count} samples)")
        current_label = input("  Enter next gesture label: ").strip().lower()
        sample_count  = 0
        print(f"  → Now recording feature vectors for: '{current_label}'\n")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
csv_file.close()
cap.release()
cv2.destroyAllWindows()
holistic.close()
print(f"\nExecution Complete. Total global entries recorded: {total_count}")
print(f"Saved Target destination output file path: {CSV_FILE}")