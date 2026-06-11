"""
Phase 2a (Batch Version) — Data Collection from Image Folders
Scans a main directory containing subfolders named after gesture labels, 
extracts holistic landmark feature vectors, and saves them to gesture_data.csv.

Strictly filters out invalid frames to prevent background artifacts from ruining accuracy.
"""
import sys
import cv2
import mediapipe as mp
import csv
import os
import glob
import numpy as np

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
IMAGE_DATASET_DIR = "picture_file" 
CSV_FILE = "gesture_data.csv"

# EXACT FIXED SIZES MATCHING YOUR TRAINING CONFIGURATION MATRIX
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode        = True,  
    model_complexity         = 1,
    refine_face_landmarks    = True,
    min_detection_confidence = 0.5
)

# ─── Upper-body pose indices & Face Keys (Perfect 225-Dimension Alignment) ────
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

    return features

FEATURE_LENGTH = len(POSE_IDX)*3 + 21*3 + 21*3 + 20*3  # = 225

# ─── CSV Initializer Engine ───────────────────────────────────────────────────
file_exists = os.path.exists(CSV_FILE)
csv_file   = open(CSV_FILE, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    csv_writer.writerow(header)
    print(f"[New] Created dataset target file: {CSV_FILE}")
else:
    print(f"[Append] Appending data rows to: {CSV_FILE}")

# ─── Main Folder Batch Processing ─────────────────────────────────────────────
if not os.path.exists(IMAGE_DATASET_DIR):
    print(f"[ERROR] Main directory '{IMAGE_DATASET_DIR}' not found. Please create it first.")
    csv_file.close()
    sys.exit()

# ค้นหาโฟลเดอร์ย่อยทั้งหมดที่อยู่ในโฟลเดอร์หลัก (ชื่อโฟลเดอร์ย่อย = Label)
subfolders = [f.path for f in os.scandir(IMAGE_DATASET_DIR) if f.is_dir()]

if not subfolders:
    print(f"[Warning] No gesture label subfolders found inside '{IMAGE_DATASET_DIR}'.")
    csv_file.close()
    sys.exit()

print(f"\n--- Starting Automated Image Batch Processing ---")
print(f"Found {len(subfolders)} gesture categories to extract.\n")

global_success_count = 0

for folder_path in subfolders:
    gesture_label = os.path.basename(folder_path).strip().lower()
    
    # ดึงไฟล์รูปภาพทุกตระกูลที่อยู่ในโฟลเดอร์ย่อยนั้นๆ
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_paths:
        print(f"  [-] Class '{gesture_label}': No valid images found. Skipping.")
        continue
        
    print(f"  [➔] Processing Class '{gesture_label}' (Found {len(image_paths)} images)...")
    class_saved_count = 0
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"      [Skipped] Could not read image file: {os.path.basename(img_path)}")
            continue
            
        # ปรับสเกลภาพให้ได้สัดส่วนตรงกับสเปค 16:9 ก่อนส่งคำนวณ
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # ส่งข้อมูลเข้า MediaPipe Holistic Pipeline
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        # ─── DATA GUARD FILTERS ───────────────────────────────────────────────
        is_neutral_class = "neutral" in gesture_label or "idle" in gesture_label
        
        # กฎเหล็กป้องกันโมเดลสับสน: 
        # ถ้าไม่ใช่คลาสปล่อยมือเฉยๆ (neutral) แต่รูปนั้นตรวจจับไม่เจอมือข้างไหนเลย -> บล็อกทิ้งทันที!
        if not is_neutral_class and not (results.left_hand_landmarks or results.right_hand_landmarks):
            # ไม่บันทึกลง CSV เพื่อไม่ให้พิกัดว่างๆ ไปทำให้ AI จำผิดพลาด
            continue
            
        # ดึงฟีเจอร์เวกเตอร์ 225 มิติตามรูปแบบโครงสร้างของคุณ
        feats = extract_feature_vector(results)
        
        # บันทึกข้อมูลลงสู่ไฟล์ CSV ปลายทาง
        csv_writer.writerow([gesture_label] + feats)
        class_saved_count += 1
        global_success_count += 1

    print(f"      ✓ Done! Successfully exported {class_saved_count}/{len(image_paths)} frames to CSV.")
    csv_file.flush()

# ─── Cleanup ──────────────────────────────────────────────────────────────────
csv_file.close()
holistic.close()
print(f"\n=========================================================")
print(f"All processing tasks finished completely!")
print(f"Total new rows written into dataset: {global_success_count} entries.")
print(f"Target Output: {CSV_FILE}")
print(f"=========================================================")