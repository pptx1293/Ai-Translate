import cv2
import mediapipe as mp
import numpy as np
import tst  # Pure detection engine

# ─── Configuration ───────────────────────────────────────────────────────────
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

show = []          # เก็บคำศัพท์ที่ต่อกันเป็นประโยค
active = False     # สถานะควบคุมการบันทึก (เริ่มทำงานเมื่อทำท่า START)
last_added_word = "" # ตัวแปรดักจับเพื่อป้องกันไม่ให้คำเดิมถูกบันทึกรัวๆ ในเฟรมติดกัน

# ปรับค่า Threshold ลงมาที่ 0.50 - 0.60 เพื่อให้ตอบสนองง่ายขึ้นและไม่หลุดพิกัด
CONFIDENCE_THRESHOLD = 0.85

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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

print("[START] Running live 16:9 feed with dynamic gesture filter...")
print(">> System State: Waiting for 'START' gesture to unlock translation...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    # แก้ไขจุดผิดพลาด: บังคับย่อสเกลภาพให้ตรงกับมิติ 16:9 ก่อนประมวลผล
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    h, w, _ = frame.shape
   ## frame = cv2.flip(frame, 1)  # กลับด้านภาพก่อนส่งให้ MediaPipe เพื่อให้พิกัดตรงกับรูปภาพในโฟลเดอร์ร้อยเปอร์เซ็นต์
    # 1. ส่งภาพดิบที่ยังไม่ได้กลับด้านให้ MediaPipe ตรวจพิกัด (เพื่อให้พิกัดตรงกับรูปภาพในโฟลเดอร์ร้อยเปอร์เซ็นต์)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    # 2. ค่อยสั่งกลับด้านหน้าจอ (Flip) เพื่อให้แสดงผลบนจอคอมพิวเตอร์เป็นกระจกเงาธรรมชาติ
    #--frame = cv2.flip(frame, 1)

    predicted_gesture = "No Hand Detected"
    current_conf = 0.0

    # ─── DYNAMIC HAND DETECTION FILTER ───────────────────────────────────────
    if results.left_hand_landmarks or results.right_hand_landmarks:
        # เรียกโมเดลทำนายคำศัพท์
        predicted_gesture, current_conf = tst.translate_frame(results)
        predicted_gesture = predicted_gesture.strip().upper()

        # ตรวจเช็คเกณฑ์ความมั่นใจ (Confidence)
        if current_conf >= CONFIDENCE_THRESHOLD:
            
            # ระบบสั่งการคุมสวิตช์เปิด/ปิด
            if predicted_gesture == "START":
                if not active:
                    active = True
                    print("[SYSTEM] Translation activated via START gesture.")
            elif predicted_gesture == "STOP":
                if active:
                    active = False
                    print("[SYSTEM] Translation paused via STOP gesture.")
            
            # ตรรกะร้อยเรียงคำให้เป็นประโยค (ทำงานเมื่อระบบ active และไม่ใช่คำคุมระบบ)
            elif active and predicted_gesture not in ["START", "STOP", "NEUTRAL", "IDLE"]:
                # ดักจับไม่ให้คำเดิมถูกบันทึกเบิ้ลติดๆ กันในแต่ละเฟรม
                if predicted_gesture != last_added_word:
                    show.append(predicted_gesture)
                    last_added_word = predicted_gesture
                    print(f"[SENTENCE BUILDING] Added: {predicted_gesture} -> Current: {' '.join(show)}")
        else:
            # หากความมั่นใจต่ำ ให้แสดงผลแจ้งเตือนว่ากำลังวิเคราะห์
            predicted_gesture = "Analyzing..."
            
    else:
        # หากเอามือลง ให้รีเซ็ตค่าสถานะคำล่าสุด เพื่อให้พร้อมสแกนคำต่อไปเมื่อยกมือขึ้นมาใหม่
        last_added_word = ""

    # ─── RENDERING & OVERLAYS ──────────────────────────────────────────────
    # วาดเส้นโครงกระดูกด้วยพิกัดที่สัมพันธ์กับการกลับด้านหน้าจอแล้ว
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # วาด HUD และแสดงผลข้อความประโยคที่แปลได้
    status_color = (0, 255, 0) if active else (0, 0, 255)
    status_text = "RECORDING ACTIVE" if active else "SYSTEM LOCKED (DO START)"
    
    cv2.rectangle(frame, (10, 10), (550, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {status_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"LIVE    : {predicted_gesture} ({current_conf*100:.1f}%)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"SENTENCE: {' '.join(show)}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("16:9 Video Feed Translation Framework", frame) 

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'): # กดปุ่ม 'C' บนคีย์บอร์ดเพื่อล้างประโยคทิ้ง (Clear Sentence)
        show = []
        last_added_word = ""
        print("[SYSTEM] Sentence cleared.")

cap.release()
cv2.destroyAllWindows()
holistic.close()