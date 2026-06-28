import glob
import os
import sys
import io
import json
from datetime import datetime

# Reconfigure stdout/stderr encoding on Windows to support Unicode characters
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import threading
import time
import base64
import yaml
import numpy as np
import cv2
import atexit
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_from_directory
from flask import session, redirect, url_for, render_template

# Add parent directory to path to allow importing modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from voice_commands import VoiceCommands
except Exception as e:
    print(f"[AUDIO] Could not import VoiceCommands: {e}")
    VoiceCommands = None

try:
    from navigation_freespace_rpi import (
        FreespaceInference,
        MaskSmoother,
        DirectionVoter,
        mask_to_direction,
    )
    HAS_FREESPACE = True
except Exception as e:
    print(f"[MODEL] Could not import navigation_freespace_rpi: {e}")
    HAS_FREESPACE = False

try:
    import mqtt_client
    HAS_MQTT = True
except Exception as e:
    print(f"[MQTT] Could not import mqtt_client: {e}")
    HAS_MQTT = False

try:
    import firebase_cloud
    HAS_FIREBASE = True
except Exception as e:
    print(f"[FIREBASE] Could not import firebase_cloud: {e}")
    HAS_FIREBASE = False

UI_FOLDER = os.path.join(BASE_DIR, 'iot-el', 'dashborad')
if not os.path.isdir(UI_FOLDER):
    UI_FOLDER = os.path.join(BASE_DIR, 'dashborad')

# Load settings from settings.yaml
CONFIG_PATH = Path(BASE_DIR) / "config" / "settings.yaml"
try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    HOST = config["dashboard"]["host"]
    PORT = config["dashboard"]["port"]
except Exception as e:
    print(f"[WARN] Failed to load settings.yaml: {e}. Using defaults.")
    HOST = "0.0.0.0"
    PORT = 5500

# Override host if running on RPi to allow external network connection
if HOST == "127.0.0.1":
    HOST = "0.0.0.0"

app = Flask(__name__, static_folder=UI_FOLDER, template_folder=UI_FOLDER)
app.secret_key = 'your-secret-key-here'

# ── Initialize Firebase cloud sync ────────────────────────────────────────────
if HAS_FIREBASE:
    firebase_cloud.initialize()

# ── User store (persisted to users.json) ─────────────────────────────────────
USERS_FILE = Path(BASE_DIR) / "config" / "users.json"

def _load_users():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Default built-in admin account
    return {"admin": {"password": "123", "full_name": "123"}}

def _save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        print(f"[AUTH] Could not save users file: {e}")

USERS = _load_users()

# ── Activity log (in-memory, last 200 entries) ────────────────────────────────
ACTIVITY_LOG = []
LOG_LOCK = threading.Lock()

def add_log(event_type: str, message: str, level: str = "info"):
    """Append a structured log entry. Levels: info | warn | danger"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": event_type,
        "message": message,
        "level": level
    }
    with LOG_LOCK:
        ACTIVITY_LOG.append(entry)
        if len(ACTIVITY_LOG) > 200:
            ACTIVITY_LOG.pop(0)
    print(f"[LOG] [{level.upper()}] [{event_type}] {message}")
    # Mirror warn/danger entries to Firebase for remote caretaker visibility
    if HAS_FIREBASE and level in ("warn", "danger"):
        firebase_cloud.push_log(entry)

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        user_rec = USERS.get(u)
        if user_rec and user_rec.get('password') == p:
            session['user'] = u
            add_log("AUTH", f"User '{u}' logged in successfully.", "info")
            return redirect(url_for('index'))
        add_log("AUTH", f"Failed login attempt for username '{u}'.", "warn")
        return redirect(url_for('login') + '?error=1')
    return send_from_directory(UI_FOLDER, 'login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return send_from_directory(UI_FOLDER, 'signup.html')

    # Accept JSON body (from fetch) or form data
    if request.is_json:
        data = request.get_json(force=True) or {}
    else:
        data = request.form.to_dict()

    username = data.get('username', '').strip()
    password = data.get('password', '')
    full_name = data.get('full_name', '').strip()

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required."}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({"success": False, "error": "Username must be 3–20 characters."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    global USERS
    USERS = _load_users()  # Refresh before writing
    if username in USERS:
        return jsonify({"success": False, "error": "Username already exists."}), 409

    USERS[username] = {"password": password, "full_name": full_name or username}
    _save_users(USERS)
    add_log("AUTH", f"New user registered: '{username}' ({full_name}).", "info")
    return jsonify({"success": True}), 201

@app.route('/logout')
def logout():
    user = session.get('user', 'unknown')
    add_log("AUTH", f"User '{user}' logged out.", "info")
    session.clear()
    return redirect(url_for('login'))
    
voice = VoiceCommands(cooldown=1.2) if VoiceCommands is not None else None

# Shared state between background thread and Flask routes
latest_jpeg_frame = None
latest_telemetry = {
    "direction": "FORWARD",
    "fps": 0.0,
    "obstacle_count": 0,
    "status": {
        "camera": "Offline",
        "model": "Offline",
        "server": "Active"
    },
    "obstacles": []
}
telemetry_lock = threading.Lock()
frame_lock = threading.Lock()
shutdown_event = threading.Event()

def open_camera():
    print("[CAMERA] Attempting to connect to local webcam...")
    for index in [0, 1, '/dev/video0', '/dev/video1']:
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"[CAMERA] ✓ Successfully connected to local webcam (index={index})!")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Always grab latest frame
                cap.set(cv2.CAP_PROP_FPS, 30)          # Request 30 FPS from camera
                return cap
        except Exception:
            pass
    print("[CAMERA] ⚠ Local webcam could not be opened. Falling back to simulation.")
    return None

def detect_obstacles_canny(frame):
    """Fallback Canny edge-based obstacle detector."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    h, w = frame.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            continue
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_h < 25 or box_w < 25:
            continue

        cx = (x + box_w / 2) / max(1, w)
        cy = (y + box_h / 2) / max(1, h)
        distance = max(0.4, min(8.0, 1.2 + 140.0 / (box_h + 1e-6)))
        label = "Obstacle"

        obstacles.append({
            "label": label,
            "distance": round(float(distance), 1),
            "x": int(x),
            "y": int(y),
            "w": int(box_w),
            "h": int(box_h),
            "center_x": round(float(cx), 2),
            "center_y": round(float(cy), 2),
        })

    obstacles = sorted(obstacles, key=lambda item: item['distance'])

    if not obstacles:
        return "FORWARD", [], 0

    closest = obstacles[0]
    if closest['distance'] < 1.8:
        direction = "STOP"
    elif closest['center_x'] < 0.33:
        direction = "SLIGHT LEFT"
    elif closest['center_x'] > 0.67:
        direction = "SLIGHT RIGHT"
    else:
        direction = "FORWARD"

    return direction, obstacles, len(obstacles)

def make_synthetic_frame(t):
    """Generates a synthetic street scene to feed the ONNX model when camera is simulated."""
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    # Sky: light blue
    frame[:160, :] = [200, 150, 100]
    # Ground: dark gray
    frame[160:, :] = [80, 80, 80]
    
    # Perspective road polygon
    road_pts = np.array([[110, 160], [210, 160], [300, 320], [20, 320]], dtype=np.int32)
    cv2.fillPoly(frame, [road_pts], [120, 120, 120])
    
    # Simulated moving obstacle (e.g., a moving person)
    cx = int(160 + 70 * np.sin(t))
    cy = int(220 + 15 * np.cos(t * 0.7))
    cv2.rectangle(frame, (cx - 15, cy - 35), (cx + 15, cy + 10), [50, 50, 200], -1)
    
    # Simulated static obstacle (e.g., roadside post)
    cv2.rectangle(frame, (50, 160), (70, 230), [50, 150, 50], -1)
    
    # Add minor noise
    noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame

def run_navigation_loop():
    global latest_jpeg_frame, latest_telemetry
    print("\n[BACKEND] Starting Navigation and Video Stream Loop...")

    # 1. Initialize Camera
    cap = open_camera()
    camera_active = (cap is not None)

    # 2. Initialize ONNX Inference Engine
    model_active = False
    engine = None
    smoother = None
    voter = None

    if HAS_FREESPACE:
        model_path = Path(__file__).parent / "models" / "freespace_int8.onnx"
        if not model_path.exists():
            model_path = Path(__file__).parent / "freespace_int8.onnx"
        if model_path.exists():
            try:
                engine = FreespaceInference(model_path)
                smoother = MaskSmoother(window=3)
                voter = DirectionVoter(window=5)
                model_active = True
                print("[BACKEND] ✓ Free-space ONNX model loaded successfully.")
            except Exception as e:
                print(f"[BACKEND] ⚠ Failed to load free-space model: {e}. Running in Canny edge mode.")
        else:
            print(f"[BACKEND] ⚠ Model not found at {model_path}. Running in Canny edge mode.")
    else:
        print("[BACKEND] ⚠ Free-space model libs not imported. Running in Canny edge mode.")

    # Simulation variables in case camera fails
    sim_t = 0.0
    frame_times = []
    
    # Pre-generate standard blank HUD in case of camera error
    blank_hud = np.zeros((320, 320, 3), dtype=np.uint8)
    for x in range(0, 320, 40):
        cv2.line(blank_hud, (x, 0), (x, 320), (20, 20, 40), 1)
    for y in range(0, 320, 40):
        cv2.line(blank_hud, (0, y), (320, y), (20, 20, 40), 1)
    cv2.line(blank_hud, (160, 160), (40, 320), (0, 100, 255), 1)
    cv2.line(blank_hud, (160, 160), (280, 320), (0, 100, 255), 1)

    while not shutdown_event.is_set():
        t_start = time.perf_counter()
        frame = None

        if camera_active:
            # Grab+retrieve pattern to always get the latest frame
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                print("[BACKEND] ⚠ Camera read failed. Releasing camera.")
                camera_active = False
                cap.release()
                
        # If camera is offline, fallback to simulation mode
        is_simulated = not camera_active
        if is_simulated:
            if model_active:
                sim_t += 0.05
                frame = make_synthetic_frame(sim_t)
            else:
                frame = blank_hud.copy()

        # Run detection
        direction = "FORWARD"
        obstacles = []
        obstacle_count = 0
        smoothed_mask = None

        if model_active and engine is not None:
            try:
                tensor = engine.preprocess(frame)
                mask = engine.infer(tensor)
                smoothed_mask = smoother.update(mask)
                raw_direction, zone_info = mask_to_direction(smoothed_mask)
                direction = voter.vote(raw_direction).upper()
                
                walkable_pct = float(zone_info.get('total_free', 0.0)) * 100.0
                
                obs_mask = (smoothed_mask == 0).astype(np.uint8) * 255
                obs_mask_bottom = np.zeros_like(obs_mask)
                obs_mask_bottom[160:, :] = obs_mask[160:, :]
                
                contours, _ = cv2.findContours(obs_mask_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 500:
                        continue
                    x, y, box_w, box_h = cv2.boundingRect(contour)
                    if box_h < 15 or box_w < 15:
                        continue
                    
                    cx = (x + box_w / 2) / 320.0
                    cy = (y + box_h / 2) / 320.0
                    distance = max(0.4, min(8.0, 1.2 + 140.0 / (box_h + 1e-6)))
                    
                    label = "Obstacle"
                        
                    obstacles.append({
                        "label": label,
                        "distance": round(float(distance), 1),
                        "x": int(x),
                        "y": int(y),
                        "w": int(box_w),
                        "h": int(box_h),
                        "center_x": round(float(cx), 2),
                        "center_y": round(float(cy), 2),
                    })
                
                obstacles = sorted(obstacles, key=lambda item: item['distance'])
                obstacle_count = len(obstacles)
                
                if not obstacles:
                    obstacles = [
                        {
                            "label": "Walkable",
                            "distance": round(max(0.0, 100.0 - walkable_pct) / 10.0, 1),
                            "x": 0, "y": 0, "w": 0, "h": 0,
                            "center_x": 0.5, "center_y": 0.5,
                            "conf": round(walkable_pct / 100.0, 2)
                        }
                    ]
                
                if voice is not None and direction != "FORWARD":
                    voice.speak(direction)
            except Exception as ex:
                print(f"[BACKEND] Free-space inference error: {ex}")
                direction, obstacles, obstacle_count = detect_obstacles_canny(frame)
        else:
            if is_simulated:
                sim_t += 0.05
                cy1 = 0.65 + 0.25 * np.sin(sim_t)
                cx1 = 0.25 + 0.05 * np.cos(sim_t)
                w1, h1 = 0.12, 0.28
                x1_min = int((cx1 - w1/2) * 320)
                x1_max = int((cx1 + w1/2) * 320)
                y1_min = int((cy1 - h1/2) * 320)
                y1_max = int((cy1 + h1/2) * 320)
                
                cy2 = 0.70 + 0.20 * np.cos(sim_t * 0.7)
                cx2 = 0.75 + 0.03 * np.sin(sim_t * 0.7)
                w2, h2 = 0.22, 0.18
                x2_min = int((cx2 - w2/2) * 320)
                x2_max = int((cx2 + w2/2) * 320)
                y2_min = int((cy2 - h2/2) * 320)
                y2_max = int((cy2 + h2/2) * 320)

                dist1 = max(0.5, 8.0 - (cy1 * 8.0))
                dist2 = max(0.5, 8.0 - (cy2 * 8.0))

                obstacles = [
                    {
                        "label": "Obstacle",
                        "distance": round(float(dist1), 1),
                        "x": x1_min, "y": y1_min, "w": (x1_max - x1_min), "h": (y1_max - y1_min),
                        "center_x": round(float(cx1), 2), "center_y": round(float(cy1), 2)
                    },
                    {
                        "label": "Obstacle",
                        "distance": round(float(dist2), 1),
                        "x": x2_min, "y": y2_min, "w": (x2_max - x2_min), "h": (y2_max - y2_min),
                        "center_x": round(float(cx2), 2), "center_y": round(float(cy2), 2)
                    }
                ]
                obstacles = sorted(obstacles, key=lambda item: item['distance'])
                obstacle_count = len(obstacles)
                
                closest = obstacles[0]
                if closest['distance'] < 1.8:
                    direction = "STOP"
                elif closest['center_x'] < 0.33:
                    direction = "SLIGHT LEFT"
                elif closest['center_x'] > 0.67:
                    direction = "SLIGHT RIGHT"
                else:
                    direction = "FORWARD"
            else:
                direction, obstacles, obstacle_count = detect_obstacles_canny(frame)

        vis = frame.copy()
        if vis.shape[0] != 320 or vis.shape[1] != 320:
            vis = cv2.resize(vis, (320, 320))

        if model_active and smoothed_mask is not None:
            mask_resized = cv2.resize(smoothed_mask, (320, 320), interpolation=cv2.INTER_NEAREST)
            walkable = mask_resized > 0
            vis[walkable] = (vis[walkable] * 0.6 + np.array([0, 220, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
            
            non_walkable = ~walkable
            bottom_half_mask = np.zeros((320, 320), dtype=bool)
            bottom_half_mask[160:, :] = True
            nw_bottom = non_walkable & bottom_half_mask
            vis[nw_bottom] = (vis[nw_bottom] * 0.8 + np.array([0, 0, 180], dtype=np.uint8) * 0.2).astype(np.uint8)

        cv2.line(vis, (int(320 * 0.33), 0), (int(320 * 0.33), 320), (50, 50, 100), 1)
        cv2.line(vis, (int(320 * 0.67), 0), (int(320 * 0.67), 320), (50, 50, 100), 1)

        for obs in obstacles:
            x, y, w, h = obs['x'], obs['y'], obs['w'], obs['h']
            if w == 0 or h == 0:
                continue
            lbl = obs['label']
            dst = obs['distance']
            if dst < 2.0:
                color = (0, 0, 255)
            elif dst < 3.5:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{lbl} {dst:.1f}m", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.putText(vis, f"DIR: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / np.mean(frame_times) if frame_times else 0.0

        _, buffer = cv2.imencode('.jpg', vis)
        jpeg_bytes = buffer.tobytes()

        with frame_lock:
            latest_jpeg_frame = jpeg_bytes

        firestore_frame = None
        if HAS_FIREBASE:
            try:
                small_vis = cv2.resize(vis, (240, 240))
                _, small_buffer = cv2.imencode('.jpg', small_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                firestore_frame = base64.b64encode(small_buffer).decode('utf-8')
            except Exception as e:
                print(f"[FIREBASE] Failed to encode Firestore frame: {e}")

        with telemetry_lock:
            latest_telemetry.update({
                "direction": direction,
                "fps": round(fps, 1),
                "obstacle_count": obstacle_count,
                "obstacles": obstacles[:5],
                "status": {
                    "camera": "Connected" if camera_active else "Simulated",
                    "model": "Free-space ONNX" if model_active else ("Canny Edge" if not is_simulated else "Simulated"),
                    "server": "Active"
                }
            })

        # Push frame and telemetry to Firebase independently
        if HAS_FIREBASE:
            # Fast frame push (separate document, ~5 FPS)
            if firestore_frame:
                firebase_cloud.push_frame(firestore_frame, fps=fps)
            # Slower telemetry push (direction, obstacles, status)
            with telemetry_lock:
                telem_snapshot = dict(latest_telemetry)
            firebase_cloud.push_telemetry(telem_snapshot)

        if voice is not None and obstacle_count > 0 and obstacles[0]['distance'] < 3.5:
            alert_text = f"Obstacle detected. Count of obstacles: {obstacle_count}."
            voice.speak(alert_text)
            level = "danger" if obstacles[0]['distance'] < 1.8 else "warn"
            add_log("NAV", f"DIR={direction} | Obstacles={obstacle_count} | Closest={obstacles[0]['distance']}m", level)

        if HAS_FIREBASE and obstacle_count > 0 and obstacles[0]['distance'] < 1.8:
            firebase_cloud.push_alert(
    alert_type="DANGER_OBSTACLE",
    free_ratio=0.0,  # Pass a numeric float here
    reason=f"Direction={direction} | Obstacles={obstacle_count} | Closest={obstacles[0]['distance']}m"
)


        if HAS_MQTT:
            try:
                threading.Thread(
                    target=mqtt_client.publish_status,
                    args=(direction,),
                    daemon=True
                ).start()

                if len(obstacles) > 0:
                    lbls = [o['label'] for o in obstacles]
                    threading.Thread(
                        target=mqtt_client.publish_navigation,
                        args=(direction, lbls, round(fps, 1)),
                        daemon=True
                    ).start()
            except Exception:
                pass

        # Throttle loop to ~30 FPS (33ms) for smooth streaming
        sleep_time = max(0.005, 0.033 - (time.perf_counter() - t_start))
        time.sleep(sleep_time)

    if camera_active:
        cap.release()
    print("[BACKEND] Navigation loop shutdown completed.")

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(UI_FOLDER, path)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/video')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.06)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def telemetry():
    with telemetry_lock:
        return jsonify(latest_telemetry)

@app.route('/api/mqtt-config')
def mqtt_config():
    if HAS_MQTT:
        return jsonify({
            "broker": mqtt_client.BROKER,
            "port": mqtt_client.PORT,
            "topic": mqtt_client.TOPIC,
            "status_topic": mqtt_client.STATUS_TOPIC,
            "alerts_topic": mqtt_client.ALERTS_TOPIC,
            "health_topic": mqtt_client.HEALTH_TOPIC
        })
    else:
        return jsonify({
            "broker": "127.0.0.1",
            "port": 8083,
            "topic": "iot/navigation",
            "status_topic": "iot/navigation/status",
            "alerts_topic": "iot/navigation/alerts",
            "health_topic": "iot/navigation/health"
        })

@app.route('/speak')
def speak():
    text = request.args.get('text', '').strip()
    if not text:
        return "No text provided", 400
    print(f"[AUDIO LOG] System Spoke: {text}")
    add_log("VOICE", f"TTS output: {text}", "info")
    if voice is not None:
        voice.speak(text)
    return "OK", 200

@app.route('/api/logs')
def get_logs():
    limit = min(int(request.args.get('limit', 50)), 200)
    with LOG_LOCK:
        entries = list(reversed(ACTIVITY_LOG[-limit:]))
    return jsonify(entries)

@app.route('/api/bt-status')
def bt_status():
    import subprocess
    try:
        # Check for the Hardware USB Bluetooth Transmitter via PipeWire
        wp_res = subprocess.run(['wpctl', 'status'], capture_output=True, text=True, timeout=3)
        if 'USB2.0 Device' in wp_res.stdout or 'Bluetooth' in wp_res.stdout:
            add_log("BT", "Hardware Bluetooth Audio Transmitter connected", "info")
            return jsonify({'connected': True, 'device': 'Hardware BT Transmitter (USB)'})
            
        # Fallback to standard native bluetooth
        bt_res = subprocess.run(['bluetoothctl', 'info'], capture_output=True, text=True, timeout=3)
        if 'Connected: yes' in bt_res.stdout:
            name = 'Native Bluetooth Device'
            for line in bt_res.stdout.splitlines():
                if line.strip().startswith('Name:'):
                    name = line.split('Name:', 1)[1].strip()
                    break
            add_log("BT", f"Native Bluetooth connected: {name}", "info")
            return jsonify({'connected': True, 'device': name})
            
        return jsonify({'connected': False, 'message': 'No device connected'})
    except Exception as e:
        return jsonify({'connected': False, 'message': str(e)})

def run_health_loop():
    import psutil
    
    def get_cpu_temp():
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            return 45.0
            
    print("[SERVER] Starting Device Health MQTT Publisher Loop...")
    while not shutdown_event.is_set():
        if HAS_MQTT:
            try:
                temp = get_cpu_temp()
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                mqtt_client.publish_health(temp, cpu, mem)
            except Exception as e:
                pass
        time.sleep(3.0)

@atexit.register
def cleanup():
    shutdown_event.set()

if __name__ == '__main__':
    nav_thread = threading.Thread(target=run_navigation_loop, daemon=True)
    nav_thread.start()

    health_thread = threading.Thread(target=run_health_loop, daemon=True)
    health_thread.start()

    print(f"\n[SERVER] Starting dashboard on http://{HOST}:{PORT}")
    try:
        app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        nav_thread.join(timeout=2.0)
        health_thread.join(timeout=2.0)
        print("[SERVER] Stopped.")