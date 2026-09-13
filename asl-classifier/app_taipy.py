#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASL Alphabet Translator — Taipy GUI + MJPEG camera stream.

The camera feed is served as a standard MJPEG stream on port 5001,
embedded in the Taipy page as a plain <img> tag.  Prediction state
(sign, confidence) is pushed to Taipy via broadcast_callback at ~5 Hz
(no need to push every frame — the video already updates independently).

Run from the asl-classifier/ directory:
    python app_taipy.py

Open http://127.0.0.1:5000 in your browser.
"""

import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerResult
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest.landmark_utils import extract_static_row
from model.static_classifier.static_classifier import StaticClassifier
from utils.cvfpscalc import CvFpsCalc

from taipy.gui import Gui, State
from taipy.gui.gui_actions import broadcast_callback

logging.getLogger('taipy.gui').setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HAND_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'model', 'hand_landmarker.task')

# ── Classifier ────────────────────────────────────────────────────────────────

def _load_labels(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8-sig') as f:
        return [l.strip() for l in f if l.strip()]


static_clf    = StaticClassifier()
static_labels = _load_labels('model/static_classifier/static_labels.csv')
print(f'Static model ready: {static_clf.ready}  ({len(static_labels)} labels)')

# ── Shared state (written by camera thread, read by Taipy + MJPEG) ───────────
_lock          = threading.Lock()
_latest_jpeg   = b''          # raw JPEG bytes for MJPEG stream
_latest_sign   = '—'
_latest_conf   = 0
_latest_fps    = 0

# ── Taipy app state ───────────────────────────────────────────────────────────
current_sign      = '—'
confidence        = 0
fps_display       = '0'
status_msg        = '⏳ Starting camera...'
sentence          = ''
history           = pd.DataFrame(columns=['sign', 'confidence'])
static_threshold  = 60
auto_add          = True
camera_active     = True

_auto_add_cooldown = 0.0
_camera_running    = False
_camera_thread     = None


# ── MJPEG server ──────────────────────────────────────────────────────────────

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence access log

    def do_GET(self):
        if self.path != '/stream':
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            while True:
                with _lock:
                    jpg = _latest_jpeg
                if jpg:
                    self.wfile.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                    )
                    self.wfile.flush()
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _start_mjpeg_server(port: int = 5001) -> None:
    server = HTTPServer(('127.0.0.1', port), _MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f'MJPEG stream → http://127.0.0.1:{port}/stream')


# ── Camera + inference thread ─────────────────────────────────────────────────

def _camera_loop(gui: Gui, static_th: float) -> None:
    global _camera_running, _auto_add_cooldown
    global _latest_jpeg, _latest_sign, _latest_conf, _latest_fps

    _latest_landmarks = [None]

    def _on_result(result: HandLandmarkerResult,
                   output_image: mp.Image, timestamp_ms: int) -> None:
        _latest_landmarks[0] = result.hand_landmarks or []

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_HAND_MODEL),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        broadcast_callback(gui, _cb_status, ['❌ Camera not found.'])
        landmarker.close()
        return

    fps_calc     = CvFpsCalc(buffer_len=15)
    last_sign    = '—'
    last_conf    = 0
    stable_sign  = '—'
    stable_count = 0
    STABLE_NEEDED     = 4
    AUTO_ADD_COOLDOWN = 1.5
    frame_ts     = 0
    last_push    = 0.0   # last time we pushed sign state to Taipy

    while _camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        fps_val = int(fps_calc.get())

        # ── MediaPipe detection ───────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        landmarker.detect_async(mp_image, frame_ts)

        lms_list  = _latest_landmarks[0]
        pred_sign = '—'
        pred_conf = 0

        if lms_list and static_clf.ready:
            lms = lms_list[0]
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 150), -1)
            row = extract_static_row(lms)
            idx, cf = static_clf(row)
            if 0 <= idx < len(static_labels) and cf * 100 >= static_th:
                pred_sign = static_labels[idx]
                pred_conf = int(cf * 100)

        # ── Stability gating ──────────────────────────────────────────────────
        if pred_sign == stable_sign:
            stable_count += 1
        else:
            stable_sign  = pred_sign
            stable_count = 1

        if stable_count >= STABLE_NEEDED:
            last_sign = pred_sign
            last_conf = pred_conf

        # ── Overlay ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 52), (12, 12, 26), -1)
        cv2.rectangle(frame, (8, 6), (114, 46), (40, 180, 100), -1)
        cv2.putText(frame, 'LETTER', (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, last_sign, (128, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'{last_conf}%', (w - 66, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS {fps_val}', (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 160), 1, cv2.LINE_AA)

        # ── Encode JPEG for MJPEG stream ──────────────────────────────────────
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _latest_jpeg  = buf.tobytes()
                _latest_sign  = last_sign
                _latest_conf  = last_conf
                _latest_fps   = fps_val

        # ── Push sign state to Taipy at ~5 Hz (not every frame) ──────────────
        now = time.time()
        should_auto = (
            auto_add
            and last_sign != '—'
            and last_conf >= 90
            and stable_count >= 10
            and (now - _auto_add_cooldown) > AUTO_ADD_COOLDOWN
        )
        if now - last_push >= 0.2:
            last_push = now
            broadcast_callback(gui, _cb_update,
                               [last_sign, last_conf, str(fps_val), should_auto])

        time.sleep(0.03)

    cap.release()
    landmarker.close()


# ── Taipy callbacks ───────────────────────────────────────────────────────────

def _cb_update(state: State, sign, conf, fps, should_auto) -> None:
    state.current_sign = sign
    state.confidence   = conf
    state.fps_display  = fps
    state.status_msg   = '🟢 Camera active'

    if should_auto and sign != '—':
        global _auto_add_cooldown
        _auto_add_cooldown = time.time()
        state.sentence = (state.sentence + sign).lstrip()
        new_row = pd.DataFrame([{'sign': sign, 'confidence': conf}])
        state.history = pd.concat([new_row, state.history],
                                   ignore_index=True).head(20)


def _cb_status(state: State, msg: str) -> None:
    state.status_msg = msg


# ── Button callbacks ──────────────────────────────────────────────────────────

def on_add_sign(state: State) -> None:
    global _auto_add_cooldown
    if state.current_sign and state.current_sign != '—':
        state.sentence = (state.sentence + state.current_sign).lstrip()
        new_row = pd.DataFrame([{'sign': state.current_sign,
                                  'confidence': state.confidence}])
        state.history = pd.concat([new_row, state.history],
                                   ignore_index=True).head(20)
        _auto_add_cooldown = time.time()


def on_add_space(state: State) -> None:
    state.sentence = state.sentence + ' '


def on_clear_sentence(state: State) -> None:
    state.sentence = ''


def on_backspace(state: State) -> None:
    state.sentence = state.sentence[:-1]


def on_clear_history(state: State) -> None:
    state.history = pd.DataFrame(columns=['sign', 'confidence'])


def on_threshold_change(state: State, var: str, value) -> None:
    pass


def on_camera_toggle(state: State) -> None:
    global _camera_running, _camera_thread, _gui_ref
    if _camera_running:
        _camera_running = False
        state.camera_active = False
        state.status_msg = '⏸ Camera paused'
    else:
        _camera_running = True
        state.camera_active = True
        _camera_thread = threading.Thread(
            target=_camera_loop,
            args=[_gui_ref, state.static_threshold],
            daemon=True,
        )
        _camera_thread.start()
        state.status_msg = '🟢 Camera active'


# ── Page ──────────────────────────────────────────────────────────────────────
# Camera feed comes from the MJPEG server on port 5001 — plain <img> tag in HTML.

page = """
<main|part|class_name=app-container|

<header|part|class_name=app-header|
# 🤟 ASL Alphabet Translator
<|{status_msg}|text|class_name=status-pill|>
|header>

<cols|layout|columns=5 3|class_name=main-layout|

<cam|part|class_name=card camera-card|
<|http://127.0.0.1:5001/stream|image|width=100%|class_name=camera-feed|>
<cambtn|part|class_name=btn-row center|
<|{camera_toggle_label}|button|on_action=on_camera_toggle|class_name=btn-secondary|>
|cambtn>
|cam>

<right|part|class_name=right-col|

<pred|part|class_name=card prediction-card|
<|{current_sign}|text|class_name=sign-display|>
<badges|part|class_name=badge-row|
<|LETTER|text|class_name=badge-type|>
<|{confidence}%|text|class_name=badge-conf|>
|badges>
<|{confidence}|progress|class_name=conf-bar|>
<predbtn|part|class_name=btn-row|
<|➕ Add Letter|button|on_action=on_add_sign|class_name=btn-primary|>
<|⎵ Space|button|on_action=on_add_space|class_name=btn-secondary|>
<|⌫|button|on_action=on_backspace|class_name=btn-secondary|>
|predbtn>
|pred>

<sent|part|class_name=card sentence-card|
### 📝 Sentence Builder
<|{sentence if sentence else "(start signing...)"}|text|class_name=sentence-display|>
<sentbtn|part|class_name=btn-row|
<|🗑 Clear|button|on_action=on_clear_sentence|class_name=btn-danger|>
|sentbtn>
|sent>

<sett|part|class_name=card settings-card|
### ⚙ Settings
<slider|part|class_name=slider-row|
Letter confidence: **<|{static_threshold}|text|>%**
<|{static_threshold}|slider|min=30|max=95|on_change=on_threshold_change|class_name=thresh-slider|>
|slider>
<|Auto-add letters|toggle|value={auto_add}|class_name=toggle-auto|>
|sett>

<hist|part|class_name=card history-card|
### 📋 Recognition Log
<|{history}|table|show_all=True|class_name=history-table|>
<|Clear Log|button|on_action=on_clear_history|class_name=btn-secondary small|>
|hist>

|right>

|cols>

|main>
"""


# ── App init ──────────────────────────────────────────────────────────────────

_gui_ref: Gui | None = None


def on_init(state: State) -> None:
    global _camera_running, _camera_thread, _gui_ref
    state.camera_toggle_label = '⏸ Pause Camera'
    if _camera_running:
        return
    _camera_running = True
    _camera_thread = threading.Thread(
        target=_camera_loop,
        args=[_gui_ref, state.static_threshold],
        daemon=True,
    )
    _camera_thread.start()


if __name__ == '__main__':
    camera_toggle_label = '⏸ Pause Camera'

    _start_mjpeg_server(port=5001)

    gui = Gui(page=page, css_file='styles.css')
    _gui_ref = gui

    gui.run(
        host='127.0.0.1',
        port=5000,
        title='ASL Alphabet Translator',
        favicon='🤟',
        debug=False,
        use_reloader=False,
    )
