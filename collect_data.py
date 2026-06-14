"""
Phase 2a (Sequential Image Version) — Arm and Hand Specific Data Collection
"""
import cv2
import mediapipe as mp
import csv
import os
import glob
import sys

IMAGE_DATASET_DIR = "picture_file" 
CSV_FILE          = "gesture_data.csv"
TARGET_WIDTH      = 1280
TARGET_HEIGHT     = 720

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode        = True, 
    model_complexity         = 2,    
    min_detection_confidence = 0.4     
)

# เลือกดึงเฉพาะพิกัดแขน: ไหล่(11,12), ข้อศอก(13,14), ข้อมือ(15,16), นิ้วก้อย(17,18), นิ้วชี้(19,20), นิ้วหัวแม่มือ(21,22)
ARM_IDX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def extract_feature_vector(results):
    features = []
    
    # 1. แขนซ้ายและแขนขวา (จาก Pose Landmark)
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        for i in ARM_IDX:
            features += [pose_lms[i].x, pose_lms[i].y, pose_lms[i].z] if i < len(pose_lms) else [0.0, 0.0, 0.0]
    else:
        features += [0.0] * (len(ARM_IDX) * 3)

    # 2. มือซ้าย (21 จุด)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark: 
            features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * (21 * 3)

    # 3. มือขวา (21 จุด)
    if append_rh := results.right_hand_landmarks:
        for lm in append_rh.landmark: 
            features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * (21 * 3)

    return features

# คำนวณมิติข้อมูลใหม่ทั้งหมด: (12 แขน * 3) + (21 มือซ้าย * 3) + (21 มือขวา * 3) = 162 มิติ
FEATURE_LENGTH = len(ARM_IDX)*3 + 21*3 + 21*3 

file_exists = os.path.exists(CSV_FILE)
csv_file = open(CSV_FILE, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"] + [f"f{i}" for i in range(FEATURE_LENGTH)]
    csv_writer.writerow(header)

if not os.path.exists(IMAGE_DATASET_DIR):
    print(f"[ERROR] Please create directory: '{IMAGE_DATASET_DIR}'")
    csv_file.close()
    sys.exit()

subfolders = [f.path for f in os.scandir(IMAGE_DATASET_DIR) if f.is_dir()]

print("\n--- Processing Arm and Hand Sequence Folders ---")
for folder_path in subfolders:
    gesture_label = os.path.basename(folder_path).strip().lower()
    
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_paths:
        continue
        
    print(f"[➔] Scanning Category '{gesture_label}' ({len(image_paths)} images)...")
    success_rows = 0
    
    for img_path in sorted(image_paths): 
        frame = cv2.imread(img_path)
        if frame is None: continue
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        is_control_class = gesture_label in ["neutral", "idle"]
        if not is_control_class and not (results.left_hand_landmarks or results.right_hand_landmarks):
            continue
            
        feats = extract_feature_vector(results)
        csv_writer.writerow([gesture_label] + feats)
        success_rows += 1

    print(f"   ✓ Done! Exported {success_rows}/{len(image_paths)} frames.")
    csv_file.flush()

csv_file.close()
holistic.close()
print("\n[COMPLETE] Extracted features appended successfully!")