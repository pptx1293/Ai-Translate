"""
main.py — Live Gesture Recognition (Arm & Hand ONLY, No Face)

Press  Q = quit  |  C = clear sentence  |  Z = undo last word
"""

import collections
import time

import cv2
import mediapipe as mp
import tst

# ── Resolution ───────────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1280, 720

# ── Thresholds (tune to your environment) ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # min per-frame confidence to count a frame
STABILITY_FRAMES     = 10     # consecutive agreeing frames before gesture fires
COOLDOWN_SEC         = 1.2    # min seconds between word appends

# Gestures that are ALWAYS treated as control commands (never appended to sentence)
CONTROL_GESTURES = {"START", "STOP", "NEUTRAL", "IDLE"}

# ── Colours (BGR) ────────────────────────────────────────────────────────────
CLR_GREEN  = (  0, 220,  80)
CLR_RED    = (  0,  60, 220)
CLR_CYAN   = (220, 220,   0)
CLR_WHITE  = (255, 255, 255)
CLR_GRAY   = (130, 130, 130)
CLR_BLACK  = (  0,   0,   0)
CLR_ORANGE = (  0, 165, 255)

# ── Runtime state ────────────────────────────────────────────────────────────
sentence       : list[str]         = []
active         : bool              = False
last_word      : str               = ""
last_append_ts : float             = 0.0
gesture_buffer : collections.deque = collections.deque(maxlen=STABILITY_FRAMES)

# ── MediaPipe — separate Hands + Pose (face pipeline never runs) ─────────────
mp_hands   = mp.solutions.hands
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 2,
    model_complexity         = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)
pose_detector = mp_pose.Pose(
    static_image_mode        = False,
    model_complexity         = 1,
    smooth_landmarks         = True,
    enable_segmentation      = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)

draw_hand_pt   = mp_drawing.DrawingSpec(color=CLR_GREEN,  thickness=1, circle_radius=2)
draw_hand_ln   = mp_drawing.DrawingSpec(color=CLR_WHITE,  thickness=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[START] Face pipeline OFF — using separate Hands + Pose.")
print("        Q=quit  C=clear  Z=undo")


# ── Result wrapper ────────────────────────────────────────────────────────────
class _Results:
    """
    Wraps separate Hands + Pose detections into the same attribute shape
    that tst.translate_frame() expects.

    IMPORTANT — handedness after cv2.flip(frame, 1):
      The frame is flipped BEFORE being passed to MediaPipe.
      MediaPipe Hands sees the mirrored image and reports:
        "Left"  = physically the user's RIGHT hand
        "Right" = physically the user's LEFT hand
      So we store them SWAPPED relative to the MP label.
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
                mp_label = handed.classification[0].label  # MP's label on flipped image
                # Swap: MP "Left" on flipped image → user's right hand
                if mp_label == "Left":
                    self.right_hand_landmarks = lms   # ← swapped
                else:
                    self.left_hand_landmarks  = lms   # ← swapped


# ── HUD ──────────────────────────────────────────────────────────────────────
ARM_CONNECTIONS = [
    (11, 13), (13, 15),            # left:  shoulder→elbow→wrist
    (12, 14), (14, 16),            # right: shoulder→elbow→wrist
    (11, 12),                      # shoulder line
    (15, 17), (15, 19), (15, 21),  # left wrist tips
    (16, 18), (16, 20), (16, 22),  # right wrist tips
]


def _draw_skeleton(frame, results) -> None:
    """Draw hand meshes and arm skeleton only (no face, no legs)."""
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS, draw_hand_pt, draw_hand_ln)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS, draw_hand_pt, draw_hand_ln)

    if results.pose_landmarks:
        lms  = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        for a, b in ARM_CONNECTIONS:
            if lms[a].visibility > 0.4 and lms[b].visibility > 0.4:
                cv2.line(frame,
                         (int(lms[a].x * w), int(lms[a].y * h)),
                         (int(lms[b].x * w), int(lms[b].y * h)),
                         CLR_GRAY, 1)
        for i in range(11, 23):
            if lms[i].visibility > 0.4:
                cv2.circle(frame,
                           (int(lms[i].x * w), int(lms[i].y * h)),
                           3, CLR_ORANGE, -1)


def _draw_hud(frame, gesture: str, conf: float) -> None:
    buf_fill = len(gesture_buffer)
    agreeing = len(set(gesture_buffer)) <= 1
    bar_w    = int((buf_fill / STABILITY_FRAMES) * 180)
    bar_clr  = CLR_GREEN if agreeing else CLR_ORANGE

    cv2.rectangle(frame, (8, 8), (660, 148), CLR_BLACK, -1)
    cv2.rectangle(frame, (8, 8), (660, 148), CLR_GRAY,   1)

    status_clr  = CLR_GREEN if active else CLR_RED
    cv2.putText(frame, "● RECORDING" if active else "● LOCKED",
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_clr, 2)

    cd = max(0.0, COOLDOWN_SEC - (time.time() - last_append_ts))
    if cd > 0:
        cv2.putText(frame, f"cd {cd:.1f}s",
                    (240, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GRAY, 1)

    cv2.putText(frame, f"LIVE: {gesture}  ({conf * 100:.1f}%)",
                (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, CLR_WHITE, 1)

    words   = " ".join(sentence) or "(empty)"
    display = words if len(words) <= 55 else "…" + words[-54:]
    cv2.putText(frame, f"SENTENCE: {display}",
                (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_CYAN, 2)

    # Stability progress bar
    cv2.rectangle(frame, (18, 112), (198, 128), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (18, 112), (18 + bar_w, 128), bar_clr, -1)
    cv2.putText(frame, "STABILITY",
                (204, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_GRAY, 1)

    cv2.putText(frame, "Q:quit  C:clear  Z:undo",
                (12, TARGET_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_GRAY, 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    frame = cv2.flip(frame, 1)          # mirror so gestures feel natural

    rgb                 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_res           = hands_detector.process(rgb)
    pose_res            = pose_detector.process(rgb)
    rgb.flags.writeable = True

    results  = _Results(hands_res, pose_res)
    has_hand = results.left_hand_landmarks or results.right_hand_landmarks

    gesture = "No Hand Detected"
    conf    = 0.0
    stable  = None

    if has_hand:
        gesture, conf = tst.translate_frame(results)
        gesture       = gesture.strip().upper()

        if conf >= CONFIDENCE_THRESHOLD:
            gesture_buffer.append(gesture)
        else:
            gesture_buffer.clear()
            gesture = "Analyzing…"

        # A gesture is "stable" when the full buffer holds the same label
        if len(gesture_buffer) == STABILITY_FRAMES and len(set(gesture_buffer)) == 1:
            stable = gesture_buffer[0]
    else:
        gesture_buffer.clear()

    # ── State machine ──────────────────────────────────────────────────────
    now = time.time()
    if stable:
        if stable == "START" and not active:
            active = True
            gesture_buffer.clear()
            print("[STATE] Recording STARTED")

        elif stable == "STOP" and active:
            active = False
            gesture_buffer.clear()
            print("[STATE] Recording STOPPED")

        elif active and stable not in CONTROL_GESTURES:
            time_ok = (now - last_append_ts) >= COOLDOWN_SEC
            novel   = stable != last_word
            if time_ok and novel:
                sentence.append(stable)
                last_word      = stable
                last_append_ts = now
                gesture_buffer.clear()
                print(f"[WORD]  {stable}  →  {' '.join(sentence)}")

    _draw_skeleton(frame, results)
    _draw_hud(frame, gesture, conf)
    cv2.imshow("Gesture → Sentence", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence.clear()
        last_word = ""
        print("[CLEAR] Sentence cleared")
    elif key == ord("z") and sentence:
        removed = sentence.pop()
        last_word = sentence[-1] if sentence else ""
        print(f"[UNDO]  Removed '{removed}'")

cap.release()
cv2.destroyAllWindows()
hands_detector.close()
pose_detector.close()