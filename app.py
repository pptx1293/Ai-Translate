"""
app.py — Flask web server for Ai-Translate.

Imports functions from main.py to run gesture detection
in a background thread, and streams the camera via MJPEG.

Run:  python app.py
Open: http://127.0.0.1:5000
"""

import collections
import os
import threading
import time
import webbrowser

import cv2
from flask import Flask, Response, jsonify, render_template

import main   # ← all gesture logic lives here now
import tst

app = Flask(__name__)

# ── Shared state (background thread writes, Flask routes read) ────────────────
_lock  = threading.Lock()
_state = {
    "live_gesture": "No Hand Detected",
    "live_conf":    0.0,
    "sentence":     [],
    "active":       False,
    "status":       "starting",
    "error":        "",
    "buffer_fill":  0,      # how many frames currently agree (stability buffer)
    "buffer_max":   1,      # frames needed for a gesture to lock in (main.STABILITY_FRAMES)
    "agreeing":     True,   # whether the buffered frames all agree on one gesture
}

_frame_lock  = threading.Lock()
_latest_jpeg: bytes | None = None

# Flag set by the /clear route, consumed once by the background loop.
# We can't just clear _state["sentence"] directly from the Flask route,
# because the background loop overwrites _state["sentence"] from its own
# local `sentence` list every single frame — so a direct write would get
# stomped on almost instantly. Instead we set a flag here, and the loop
# clears its *own* local `sentence` list (and resets _state) the next
# time it checks.
_clear_requested = False

# Set by the /shutdown route. The gesture loop checks this every frame so
# it can break out cleanly, release the camera (cap.release()) and close
# the MediaPipe detector (hands_detector.close()) BEFORE the process exits.
# We don't just kill the process immediately on /shutdown, because that
# could leave the camera device locked until the OS notices the process
# died — better to let OpenCV release it itself first.
_stop_event = threading.Event()


def _get_state_snapshot() -> dict:
    with _lock:
        return {
            "live_gesture": _state["live_gesture"],
            "live_conf":    round(_state["live_conf"], 3),
            "sentence":     list(_state["sentence"]),
            "word":         "".join(_state["sentence"]),
            "active":       _state["active"],
            "status":       _state["status"],
            "error":        _state["error"],
            "buffer_fill":  _state["buffer_fill"],
            "buffer_max":   _state["buffer_max"],
            "agreeing":     _state["agreeing"],
        }


# ── Background thread — uses main.py functions directly ───────────────────────
def _gesture_loop() -> None:
    global _latest_jpeg

    # Use main.py's factory functions
    hands_detector = main.make_hands_detector()
    cap            = main.open_camera()

    if not cap.isOpened():
        with _lock:
            _state["status"] = "error"
            _state["error"]  = "Could not open camera (cv2.VideoCapture(0) failed)"
        hands_detector.close()
        return

    # Local loop state (same variables as main.py run_standalone)
    sentence       : list            = []
    active         : bool            = False
    last_word      : str             = ""
    last_append_ts : float           = 0.0
    gesture_buffer : collections.deque = collections.deque(maxlen=main.STABILITY_FRAMES)

    with _lock:
        _state["status"] = "running"

    print("[gesture_loop] Camera opened — detection running.")

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # ── Handle a pending /clear request from the website ───────────
            global _clear_requested
            with _lock:
                do_clear = _clear_requested
                _clear_requested = False
            if do_clear:
                sentence  = []
                last_word = ""
                gesture_buffer.clear()
                tst.reset_smooth_buffer()
                with _lock:
                    _state["sentence"] = []
                print("[CLEAR] Sentence cleared (from website)")

            # ── Same pre-processing as main.py ─────────────────────────────
            frame = cv2.resize(frame, (main.TARGET_W, main.TARGET_H))
            frame = cv2.flip(frame, 1)

            # ── main.py: process frame ─────────────────────────────────────
            results, has_hand = main.process_frame(frame, hands_detector)

            # ── main.py: detect gesture ────────────────────────────────────
            gesture, conf, threshold, stable = main.detect_gesture(
                results, has_hand, gesture_buffer
            )

            # ── main.py: state machine ─────────────────────────────────────
            sentence, active, last_word, last_append_ts, gesture_buffer = \
                main.run_state_machine(
                    stable, gesture, conf, threshold,
                    sentence, active, last_word, last_append_ts, gesture_buffer,
                )

            # ── main.py: draw overlays ─────────────────────────────────────
            main.draw_hands(frame, results)
            # NOTE: main.draw_hud(...) intentionally NOT called here.
            # The LOCKED/LIVE/SENTENCE/STABILITY box is drawn by main.py's
            # draw_hud(); the website already shows that info in its own
            # HTML/CSS via the /gesture endpoint, so we skip it on the
            # video frame itself. Standalone mode (python main.py) is
            # untouched and still shows the HUD in its local cv2 window.

            # ── Encode frame to JPEG for MJPEG stream ─────────────────────
            stream_frame = cv2.resize(frame, (854, 480))
            ok, jpeg = cv2.imencode(
                ".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if ok:
                with _frame_lock:
                    _latest_jpeg = jpeg.tobytes()

            # ── Push state to Flask ────────────────────────────────────────
            with _lock:
                _state["live_gesture"] = gesture
                _state["live_conf"]    = conf
                _state["sentence"]     = list(sentence)
                _state["active"]       = active
                _state["buffer_fill"]  = len(gesture_buffer)
                _state["buffer_max"]   = main.STABILITY_FRAMES
                _state["agreeing"]     = len(set(gesture_buffer)) <= 1

    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"]  = str(exc)
        print(f"[gesture_loop ERROR] {exc}")
    finally:
        cap.release()
        hands_detector.close()
        print("[gesture_loop] Camera released.")
        with _lock:
            _state["status"] = "stopped"
        if _stop_event.is_set():
            # Camera is safely released — now it's safe to kill the whole
            # process (including the Flask server). os._exit() is used
            # instead of sys.exit() because this runs in a background
            # thread, where sys.exit() would only stop that thread, not
            # the Flask server thread.
            print("[gesture_loop] Shutdown requested — exiting process.")
            time.sleep(0.3)  # tiny pause so the /shutdown HTTP response can flush
            os._exit(0)


# ── MJPEG generator ───────────────────────────────────────────────────────────
def _mjpeg_generator():
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.033)


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/gesture")
def gesture():
    return jsonify(_get_state_snapshot())


@app.route("/clear", methods=["POST"])
def clear():
    global _clear_requested
    with _lock:
        _clear_requested = True
    return jsonify({"ok": True})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    Called by the website's Stop button. Signals the background gesture
    loop to stop, which releases the camera and then kills the whole
    process (server included) — see _gesture_loop's finally block.
    """
    _stop_event.set()
    return jsonify({"ok": True, "message": "Shutting down..."})


# ── Entry point ───────────────────────────────────────────────────────────────
def _open_browser_when_ready() -> None:
    """
    Wait a moment for Flask to actually be listening, then open the
    website automatically — so running this one script both starts the
    server and opens the page, instead of you having to start app.py
    and then separately open http://127.0.0.1:5000 in a browser.
    """
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    t = threading.Thread(target=_gesture_loop, daemon=True, name="gesture-loop")
    t.start()

    threading.Thread(target=_open_browser_when_ready, daemon=True,
                      name="open-browser").start()

    # debug=False required — debug=True double-starts the background thread
    app.run(host="0.0.0.0", port=5000, debug=False)