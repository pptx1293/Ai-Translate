import sys
from google.protobuf import symbol_database as _symbol_database

_original_get_prototype = _symbol_database.Default().GetPrototype
def _patched_get_prototype(descriptor):
    try: return _original_get_prototype(descriptor)
    except Exception:
        from google.protobuf import message
        return message.Message
_symbol_database.Default().GetPrototype = _patched_get_prototype

import os
import numpy as np
import tensorflow as tf

MODEL_PATH = 'gesture_model.keras' 
WORDS_PATH = 'gesture_words.npy'
ARM_IDX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

ACTIONS = np.array([])
model = None

def init_engine():
    global ACTIONS, model
    if os.path.exists(WORDS_PATH):
        ACTIONS = np.load(WORDS_PATH, allow_pickle=True)
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

def extract_feature_vector(results, invert_x=True):
    features = []
    
    def get_coords(lm, invert):
        x = (1.0 - lm.x) if invert else lm.x
        return [x, lm.y, lm.z]

    # 1. พิกัดแขน (สลับโครงสร้างซ้ายขวาตามระนาบกระจก)
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        for i in ARM_IDX:
            # หาก invert_x เป็น True พิกัดแขนจะถูกสลับคู่ทางตรรกะกระจกโดยอัตโนมัติ
            target_idx = i
            if invert_x:
                if i % 2 != 0: target_idx = i + 1  # ขยับซ้ายไปขวา
                else: target_idx = i - 1           # ขยับขวาไปซ้าย
            
            if target_idx < len(pose_lms):
                features += get_coords(pose_lms[target_idx], invert_x)
            else:
                features += [0.0, 0.0, 0.0]
    else:
        features += [0.0] * (len(ARM_IDX) * 3)

    # 2. มือซ้าย (ดึงจากขวาถ้าส่องกระจก)
    lh_source = results.right_hand_landmarks if invert_x else results.left_hand_landmarks
    if lh_source:
        for lm in lh_source.landmark: features += get_coords(lm, invert_x)
    else:
        features += [0.0] * (21 * 3)

    # 3. มือขวา (ดึงจากซ้ายถ้าส่องกระจก)
    rh_source = results.left_hand_landmarks if invert_x else results.right_hand_landmarks
    if rh_source:
        for lm in rh_source.landmark: features += get_coords(lm, invert_x)
    else:
        features += [0.0] * (21 * 3)

    return np.array(features)

def translate_frame(mediapipe_results):
    global model, ACTIONS
    if model is None or len(ACTIONS) == 0:
        return "Engine Not Initialized", 0.0

    try:
        features = extract_feature_vector(mediapipe_results, invert_x=True)
        input_data = np.expand_dims(features, axis=0) 
        prediction = model.predict(input_data, verbose=0)[0]
        best_match_idx = np.argmax(prediction)
        return ACTIONS[best_match_idx], float(prediction[best_match_idx])
    except Exception as e:
        return f"Error: {str(e)}", 0.0

init_engine()