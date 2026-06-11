import cv2
import mediapipe as mp
import time
import tst  # This imports your pure detection script

# ─── Configuration ───────────────────────────────────────────────────────────
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode        = False,
    model_complexity         = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5
)

# ─── Open Camera Stream ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)  # Opens webcam

# Set base native widescreen resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

print("[START] Running live 16:9 feed with dynamic gesture filter...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Mirror frame for intuitive placement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Hard scale frame layer to match configuration targets exactly
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    h, w, _ = frame.shape

    # Process frame through MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    # =========================================================================
    # ─── DYNAMIC HAND DETECTION FILTER ───────────────────────────────────────
    # =========================================================================
    # Only call the AI model if at least one hand is physically detected on screen
    # ─── เพิ่มตัวแปรสถานะก่อนเข้า Loop หลัก ───────────────────────────────────
    #sequence_predictions = []  # ใช้สะสมผลลัพธ์การทำนาย
    CONFIDENCE_THRESHOLD = 0.9 # ต้องมั่นใจเกิน 85% ถึงจะยอมรับว่าเป็นท่านั้นจริงๆ

# ─── ภายใน While Loop (จุดที่ทำนายผล) ────────────────────────────────────
    if results.left_hand_landmarks or results.right_hand_landmarks:
        predicted_gesture, confidence = tst.translate_frame(results)
        
        # กรองด้วยค่าความมั่นใจขั้นต่ำ
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_gesture = "Analyzing..."
    else:
        predicted_gesture = "No Gesture Detected"
        confidence = 0.0

    # แสดงผลบนหน้าจอตามปกติ...

    # =========================================================================
    # ─── RENDERING & OVERSIGHTS ──────────────────────────────────────────────
    # =========================================================================
    # Render standard visual skeletons
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Display translation overlay banner
    
    cv2.putText(frame, f"GESTURE: {predicted_gesture.upper()}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"CONFIDENCE: {confidence*100:.1f}%", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("16:9 Video Feed Translation Framework", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()