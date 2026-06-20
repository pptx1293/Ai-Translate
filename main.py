import collections
import time

import cv2
import mediapipe as mp
import tst

# Safe wrapper — works even if tst.py is an older version without reset_smooth_buffer
def _reset_tst_buffer():
    if hasattr(tst, "reset_smooth_buffer"):
        tst.reset_smooth_buffer()
    elif hasattr(tst, "_prob_buffer"):
        tst._prob_buffer.clear()

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
mp_drawing = mp.solutions.drawing_utils

def make_hands_detector() -> mp_hands.Hands:
    return mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        model_complexity         = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

draw_hand_pt   = mp_drawing.DrawingSpec(color=CLR_GREEN,  thickness=1, circle_radius=2)
draw_hand_ln   = mp_drawing.DrawingSpec(color=CLR_WHITE,  thickness=1)

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


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
                mp_label = handed.classification[0].label  # MP's label on flipped image
                # Swap: MP "Left" on flipped image → user's right hand
                if mp_label == "Left":
                    self.right_hand_landmarks = lms   # ← swapped
                else:
                    self.left_hand_landmarks  = lms   # ← swapped


# ── HUD ──────────────────────────────────────────────────────────────────────


def _draw_skeleton(frame, results) -> None:
    """Draw hand meshes only."""
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS, draw_hand_pt, draw_hand_ln)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS, draw_hand_pt, draw_hand_ln)


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

    words   = "".join(sentence) or "(empty)"
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


# ── Helper functions for modular import ───────────────────────────────────────

def process_frame(frame, hands_det) -> tuple[_Results, bool]:
    rgb                 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_res           = hands_det.process(rgb)
    rgb.flags.writeable = True

    results  = _Results(hands_res)
    has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
    return results, has_hand


def detect_gesture(results, has_hand, gesture_buf) -> tuple[str, float, float, str | None]:
    gesture   = "No Hand Detected"
    conf      = 0.0
    threshold = CONFIDENCE_THRESHOLD
    stable    = None

    if has_hand:
        gesture, conf = tst.translate_frame(results)
        gesture       = gesture.strip().upper()
        if gesture == "SPACE":
            gesture = " "
        if conf >= CONFIDENCE_THRESHOLD:
            gesture_buf.append(gesture)
        else:
            gesture_buf.clear()
            gesture = "Analyzing…"

        # A gesture is "stable" when the full buffer holds the same label
        if len(gesture_buf) == STABILITY_FRAMES and len(set(gesture_buf)) == 1:
            stable = gesture_buf[0]
    else:
        gesture_buf.clear()
        _reset_tst_buffer()

    return gesture, conf, threshold, stable


def run_state_machine(
    stable, gesture, conf, threshold,
    sent, act, last_w, last_app_ts, gesture_buf
) -> tuple[list, bool, str, float, collections.deque]:
    now = time.time()
    if stable:
        if stable == "START" and not act:
            act = True
            gesture_buf.clear()
            _reset_tst_buffer()
            print("[STATE] Recording STARTED")

        elif stable == "STOP" and act:
            act = False
            gesture_buf.clear()
            _reset_tst_buffer()
            print("[STATE] Recording STOPPED")

        elif act and stable == "BACKSPACE":
            time_ok = (now - last_app_ts) >= COOLDOWN_SEC
            if time_ok:
                if sent:
                    removed = sent.pop()
                    print(f"[WORD]  Removed last word: '{removed}'  →  {' '.join(sent)}")
                else:
                    print("[WORD]  Sentence already empty, cannot backspace")
                last_w         = ""
                last_app_ts    = now
                gesture_buf.clear()
                _reset_tst_buffer()

        elif act and stable not in CONTROL_GESTURES:
            time_ok = (now - last_app_ts) >= COOLDOWN_SEC
            novel   = stable
            if time_ok and novel:
                sent.append(stable)
                last_w         = stable
                last_app_ts    = now
                gesture_buf.clear()
                _reset_tst_buffer()
                print(f"[WORD]  {stable}  →  {' '.join(sent)}")

    return sent, act, last_w, last_app_ts, gesture_buf


def draw_hands(frame, results) -> None:
    _draw_skeleton(frame, results)


def run_standalone() -> None:
    global sentence, active, last_word, last_append_ts, gesture_buffer

    hands_det = make_hands_detector()
    camera    = open_camera()

    print("[START] Hand-only mode — no body/pose overlay.")
    print("Q=quit  C=clear  Z=undo")

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        frame = cv2.flip(frame, 1)

        results, has_hand = process_frame(frame, hands_det)
        gesture, conf, threshold, stable = detect_gesture(results, has_hand, gesture_buffer)

        sentence, active, last_word, last_append_ts, gesture_buffer = run_state_machine(
            stable, gesture, conf, threshold,
            sentence, active, last_word, last_append_ts, gesture_buffer,
        )

        draw_hands(frame, results)
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

    camera.release()
    cv2.destroyAllWindows()
    hands_det.close()


if __name__ == "__main__":
    run_standalone()