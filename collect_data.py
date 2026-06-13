"""
Phase 2a (Sequential Image Version) — Data Collection from Frame-by-Frame Pictures
Scans folders inside 'dataset_images', extracts holistic landmark features, 
and strictly filters out empty frames (where hands are out of boundary).
"""

import cv2
import mediapipe as mp
import csv
import os
import glob
import sys

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
IMAGE_DATASET_DIR = "picture_file" 
CSV_FILE          = "gesture_data.csv"
TARGET_WIDTH      = 1280
TARGET_HEIGHT     = 720

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode        = True, # เปิดโหมดรูปภาพนิ่งเพื่อความแม่นยำขั้นสูงสุด
    model_complexity         = 2,    # เพิ่มความเข้มข้นโมเดลเป็นระดับ 2 เพื่อจับจุดท่าทางที่แสงสว่างจ้าได้คมชัดยิ่งขึ้น
    refine_face_landmarks    = True,
    min_detection_confidence = 0.4     # ลดเกณฑ์ลงมาเล็กน้อยเผื่อสแกนรูปภาพที่หลุดขอบเฟรม
)

POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FACE_KEY = [1, 4, 9, 10, 33, 37, 40, 46, 52, 55, 61, 70, 105, 133, 152, 234, 263, 362, 397, 468]

def extract_feature_vector(results):
    features = []
    # 1. Pose
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        for i in POSE_IDX:
            features += [pose_lms[i].x, pose_lms[i].y, pose_lms[i].z] if i < len(pose_lms) else [0.0, 0.0, 0.0]
    else:
        features += [0.0] * (len(POSE_IDX) * 3)

    # 2. Left Hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark: features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * (21 * 3)

    # 3. Right Hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark: features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * (21 * 3)

    # 4. Face
    if results.face_landmarks:
        face_lms = results.face_landmarks.landmark
        for i in FACE_KEY:
            features += [face_lms[i].x, face_lms[i].y, face_lms[i].z] if i < len(face_lms) else [0.0, 0.0, 0.0]
    else:
        features += [0.0] * (20 * 3)

    return features

FEATURE_LENGTH = len(POSE_IDX)*3 + 21*3 + 21*3 + 20*3 # 225 มิติ

# ─── Initialize CSV ───────────────────────────────────────────────────────────
file_exists = os.path.exists(CSV_FILE)
csv_file = open(CSV_FILE, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    csv_writer.writerow(header)

# ─── Main Batch Processing Engine ─────────────────────────────────────────────
if not os.path.exists(IMAGE_DATASET_DIR):
    print(f"[ERROR] Please create directory: '{IMAGE_DATASET_DIR}'")
    csv_file.close()
    sys.exit()

subfolders = [f.path for f in os.scandir(IMAGE_DATASET_DIR) if f.is_dir()]

print("\n--- Processing Frame-by-Frame Subfolders ---")
for folder_path in subfolders:
    gesture_label = os.path.basename(folder_path).strip().lower()
    
    # สนับสนุนทุกนามสกุลไฟล์ภาพ
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_paths:
        continue
        
    print(f"[➔] Scanning Category '{gesture_label}' ({len(image_paths)} images)...")
    success_rows = 0
    
    for img_path in sorted(image_paths): # เรียงลำดับชื่อไฟล์ภาพ 1.1, 1.2, 1.3 เพื่อความเป็นระเบียบ
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # ถอดค่าพิกัดดิบแบบมุมมองปกติ (ห้ามพลิกรูปภาพในโฟลเดอร์ เพื่อให้ตรงกับโครงสร้างของระบบดึงพิกัดใน main.py)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        # ─── EXTRACTION GUARD FOR SEQUENCE DATA ──────────────────────────────
        is_control_class = gesture_label in ["neutral", "idle"]
        
        # กฎเหล็กดักความสับสน: หากรูปภาพนั้นๆ ตรวจไม่เจอมือเลย (เช่น รูปภาพเริ่มต้นที่เพิ่งยกมือขึ้นมา)
        # ระบบจะข้ามทันทีเพื่อไม่ให้พิกัดว่างไปรบกวนข้อมูลคำศัพท์หลัก
        if not is_control_class and not (results.left_hand_landmarks or results.right_hand_landmarks):
            print(f"   [Skipped Frame] {os.path.basename(img_path)} -> No hand coordinates detected.")
            continue
            
        feats = extract_feature_vector(results)
        csv_writer.writerow([gesture_label] + feats)
        success_rows += 1

    print(f"   ✓ Done! Exported {success_rows}/{len(image_paths)} valid frames to training matrix.")
    csv_file.flush()

csv_file.close()
holistic.close()
print("\n[COMPLETE] All clean frame records appended successfully!")