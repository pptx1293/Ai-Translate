# =========================================================================
# SYSTEM PATCH: FIXED PROTOBUF GENERATION FOR MEDIAPIPE + TENSORFLOW
# =========================================================================
import sys
from google.protobuf import symbol_database as _symbol_database

_original_get_prototype = _symbol_database.Default().GetPrototype

def _patched_get_prototype(descriptor):
    try:
        return _original_get_prototype(descriptor)
    except Exception:
        from google.protobuf import message
        return message.Message

_symbol_database.Default().GetPrototype = _patched_get_prototype

import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
MODEL_PATH = 'gesture_model.keras' 
WORDS_PATH = 'gesture_words.npy'

# ─── Indices mapping (Perfect 225 dimension configuration) ───────────────────
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
FACE_KEY = [1, 4, 9, 10, 33, 37, 40, 46, 52, 55, 61, 70, 105, 133, 152, 234, 263, 362, 397, 468]

# ─── Global Variables for Model and Labels ────────────────────────────────────
ACTIONS = np.array([])
model = None

# =========================================================================
# ─── INITIALIZE TRANSLATION ENGINE ───────────────────────────────────────
# =========================================================================
def init_engine():
    global ACTIONS, model
    # Load gesture words
    if os.path.exists(WORDS_PATH):
        try:
            ACTIONS = np.load(WORDS_PATH, allow_pickle=True)
            print(f"[SUCCESS] Engine loaded gesture labels: {ACTIONS}")
        except Exception as e:
            print(f"[ERROR] Engine failed reading {WORDS_PATH}: {e}")
    else:
        print(f"[CRITICAL] '{WORDS_PATH}' missing!")

    # Load trained Keras model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[SUCCESS] Engine loaded AI model: '{MODEL_PATH}'")
    except Exception as e:
        print(f"[ERROR] Engine could not load model file: {e}")

# =========================================================================
# ─── SYNCHRONIZED FEATURE EXTRACTION (225 CHANNELS) ──────────────────────
# =========================================================================
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

    return np.array(features)

# =========================================================================
# ─── CORE TRANSLATION FUNCTION ───────────────────────────────────────────
# =========================================================================
def translate_frame(mediapipe_results):
    """
    Accepts raw MediaPipe holistic process results from your other script,
    performs AI inference, and returns (predicted_word, confidence_percentage)
    """
    global model, ACTIONS
    
    if model is None or len(ACTIONS) == 0:
        return "Engine Not Initialized", 0.0

    try:
        # Extract features to perfectly match the 225-dimension training shape
        features = extract_feature_vector(mediapipe_results)
        input_data = np.expand_dims(features, axis=0) # Reshape to (1, 225)
        
        # Run prediction
        prediction = model.predict(input_data, verbose=0)[0]
        best_match_idx = np.argmax(prediction)
        confidence = prediction[best_match_idx]
        
        if best_match_idx < len(ACTIONS):
            if confidence > 0.45: # Standard target filtering threshold
                return ACTIONS[best_match_idx], float(confidence)
            else:
                return "Analyzing...", float(confidence)
        else:
            return "Index Error", 0.0
                
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Initialize immediately upon import/execution
init_engine()