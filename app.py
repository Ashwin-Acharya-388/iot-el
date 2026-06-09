import glob
import os
import sys
import threading
import time
import base64
import yaml
import numpy as np
import cv2
import atexit
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_from_directory

from flask import session, redirect, url_for, request, render_template


# Add parent directory to path to allow importing modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from voice_commands import VoiceCommands
except Exception as e:
    print(f"[AUDIO] Could not import VoiceCommands: {e}")
    VoiceCommands = None

try:
    from navigation_system_rpi import (
        ONNXInference,
        SimpleTracker,
        TemporalSmoother,
        DirectionVoter,
        find_safe_direction,
        estimate_distance,
        CLASS_NAMES,
    )
    HAS_YOLO_LIBS = True
except Exception as e:
    print(f"[YOLO] Could not import navigation_system_rpi: {e}")
    HAS_YOLO_LIBS = False

try:
    import mqtt_client
    HAS_MQTT = True
except Exception as e:
    print(f"[MQTT] Could not import mqtt_client: {e}")
    HAS_MQTT = False

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

CREDENTIALS = {"admin": "blind2024"}
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if CREDENTIALS.get(u) == p:
            session['user'] = u
            return redirect(url_for('index'))
        return send_from_directory(UI_FOLDER, 'login.html')
    return send_from_directory(UI_FOLDER, 'login.html')

@app.route('/logout')
def logout():
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
    candidates = []
    # Try environment variables first
    cam_dev = os.getenv('CAMERA_DEVICE', '').strip()
    if cam_dev:
        candidates.append(cam_dev)
    
    existing_video_devices = sorted(glob.glob('/dev/video*'))
    candidates.extend(existing_video_devices)

    cam_idx = int(os.getenv('CAMERA_INDEX', '0'))
    candidates.extend([str(i) for i in range(0, 10)])
    candidates.append(cam_idx)
    candidates.append(0)

    backend_order = []
    for name in ('CAP_DSHOW', 'CAP_MSMF', 'CAP_V4L2'):
        backend = getattr(cv2, name, None)
        if backend is not None:
            backend_order.append(backend)
    if 0 not in backend_order:
        backend_order.append(0)

    seen = set()
    for candidate in candidates:
        key = candidate if isinstance(candidate, str) else f'index:{candidate}'
        if key in seen:
            continue
        seen.add(key)

        for backend in backend_order:
            try:
                if isinstance(candidate, str) and not candidate.isdigit():
                    cap = cv2.VideoCapture(candidate, backend)
                else:
                    cap = cv2.VideoCapture(int(candidate), backend)
                if cap.isOpened():
                    # Set RPi-optimized capture resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    ok, _ = cap.read()
                    if ok:
                        print(f"[CAMERA] Using camera source: {candidate} (backend={backend})")
                        return cap
                cap.release()
            except Exception as exc:
                print(f"[CAMERA] Failed candidate {candidate} backend {backend}: {exc}")

    print("[CAMERA] No working camera device found.")
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
        label = "Person" if box_h > 60 else "Obstacle"
        if box_w > 90 or box_h > 90:
            label = "Vehicle"

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

def run_navigation_loop():
    global latest_jpeg_frame, latest_telemetry
    print("\n[BACKEND] Starting Navigation and Video Stream Loop...")

    # 1. Initialize Camera
    cap = open_camera()
    camera_active = (cap is not None)

    # 2. Initialize ONNX Inference Engine
    model_active = False
    engine = None
    tracker = None
    smoother = None
    voter = None

    if HAS_YOLO_LIBS:
        model_path = Path(__file__).parent / "models" / "yolov8n_qat.onnx"
        if model_path.exists():
            try:
                engine = ONNXInference(model_path)
                tracker = SimpleTracker(max_age=5, min_hits=2)
                smoother = TemporalSmoother(window=3)
                voter = DirectionVoter(window=5)
                model_active = True
                print("[BACKEND] ✓ ONNX Model loaded successfully.")
            except Exception as e:
                print(f"[BACKEND] ⚠ Failed to load ONNX model: {e}. Running in Canny edge mode.")
        else:
            print(f"[BACKEND] ⚠ Model not found at {model_path}. Running in Canny edge mode.")
    else:
        print("[BACKEND] ⚠ YOLO libraries not imported. Running in Canny edge mode.")

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
            ret, frame = cap.read()
            if not ret:
                print("[BACKEND] ⚠ Camera read failed. Releasing camera.")
                camera_active = False
                cap.release()

        # If camera is offline, fallback to simulation mode
        is_simulated = not camera_active
        if is_simulated:
            frame = blank_hud.copy()

        # Run detection
        direction = "FORWARD"
        obstacles = []
        obstacle_count = 0

        # Case A: Real Camera + Real YOLO model
        if not is_simulated and model_active and engine is not None:
            try:
                # Preprocess (resizes to 320x320 internally)
                tensor = engine.preprocess(frame)
                output = engine.infer(tensor)
                dets = engine.postprocess(output, conf_thresh=0.45)
                
                tracked = tracker.update(dets)
                smoothed = smoother.update(tracked)
                
                raw_direction, closest_obs = find_safe_direction(smoothed)
                direction = voter.vote(raw_direction).upper()
                
                for det in smoothed:
                    x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    conf = det[4]
                    cls = int(det[5])
                    dist = estimate_distance(det)
                    label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "obstacle"
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    cx = (x1 + box_w / 2) / 320.0
                    cy = (y1 + box_h / 2) / 320.0
                    
                    obstacles.append({
                        "label": label.capitalize(),
                        "distance": round(float(dist), 1),
                        "x": x1,
                        "y": y1,
                        "w": box_w,
                        "h": box_h,
                        "center_x": round(float(cx), 2),
                        "center_y": round(float(cy), 2),
                        "conf": round(float(conf), 2)
                    })
                obstacles = sorted(obstacles, key=lambda item: item['distance'])
                obstacle_count = len(obstacles)
            except Exception as ex:
                print(f"[BACKEND] YOLO inference error: {ex}")
                # Fallback to Canny
                direction, obstacles, obstacle_count = detect_obstacles_canny(frame)
        else:
            # Case B: Camera online but no YOLO, or Simulation mode
            if is_simulated:
                # Generate simulated moving obstacles on HUD
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

                # Simulated distances
                dist1 = max(0.5, 8.0 - (cy1 * 8.0))
                dist2 = max(0.5, 8.0 - (cy2 * 8.0))

                obstacles = [
                    {
                        "label": "Person",
                        "distance": round(float(dist1), 1),
                        "x": x1_min, "y": y1_min, "w": (x1_max - x1_min), "h": (y1_max - y1_min),
                        "center_x": round(float(cx1), 2), "center_y": round(float(cy1), 2)
                    },
                    {
                        "label": "Vehicle",
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
                # Camera online but no YOLO: Run Canny
                direction, obstacles, obstacle_count = detect_obstacles_canny(frame)

        # Draw boxes and direction on the frame
        vis = frame.copy()
        
        # Resize visual copy to 320x320 if not already
        if vis.shape[0] != 320 or vis.shape[1] != 320:
            vis = cv2.resize(vis, (320, 320))

        # Draw Left, Center, Right zone dividers on HUD
        cv2.line(vis, (int(320 * 0.33), 0), (int(320 * 0.33), 320), (50, 50, 100), 1)
        cv2.line(vis, (int(320 * 0.67), 0), (int(320 * 0.67), 320), (50, 50, 100), 1)

        for obs in obstacles:
            x, y, w, h = obs['x'], obs['y'], obs['w'], obs['h']
            lbl = obs['label']
            dst = obs['distance']
            if dst < 2.0:
                color = (0, 0, 255)  # Red
            elif dst < 3.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{lbl} {dst:.1f}m", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw navigation direction at top left
        cv2.putText(vis, f"DIR: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Calculate FPS
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / np.mean(frame_times) if frame_times else 0.0

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', vis)
        jpeg_bytes = buffer.tobytes()

        # Update globals safely
        with frame_lock:
            latest_jpeg_frame = jpeg_bytes

        with telemetry_lock:
            latest_telemetry.update({
                "direction": direction,
                "fps": round(fps, 1),
                "obstacle_count": obstacle_count,
                "obstacles": obstacles[:5],
                "status": {
                    "camera": "Connected" if camera_active else "Simulated",
                    "model": "YOLOv8 QAT" if (model_active and not is_simulated) else ("Canny Edge" if not is_simulated else "Simulated"),
                    "server": "Active"
                }
            })

        # Voice alerting logic
        if voice is not None and len(obstacles) > 0 and obstacles[0]['distance'] < 3.5:
            alert_text = f"{obstacles[0]['label']}, {obstacles[0]['distance']} meters ahead"
            voice.speak(alert_text)

        # MQTT telemetry logging
        if HAS_MQTT:
            try:
                # Publish status (e.g. STOP, FORWARD)
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

        # Throttle loop to ~10 FPS (100ms) to prevent Pi CPU pinning
        sleep_time = max(0.01, 0.10 - (time.perf_counter() - t_start))
        time.sleep(sleep_time)

    # Release camera on exit
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
    """Serves static files (style.css, script.js)."""
    return send_from_directory(UI_FOLDER, path)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/video')
def video_feed():
    """Streams the latest JPEG frame from memory as MJPEG."""
    def generate():
        while True:
            with frame_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.06)  # Stream at ~16 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def telemetry():
    """Returns real-time telemetry JSON."""
    with telemetry_lock:
        return jsonify(latest_telemetry)

@app.route('/api/mqtt-config')
def mqtt_config():
    """Exposes MQTT broker configurations dynamically."""
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
    """Receives navigation text and triggers TTS playback."""
    text = request.args.get('text', '').strip()
    if not text:
        return "No text provided", 400
    print(f"[AUDIO LOG] System Spoke: {text}")
    if voice is not None:
        voice.speak(text)
    return "OK", 200

def run_health_loop():
    import psutil
    
    def get_cpu_temp():
        try:
            # Standard RPi path
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            return 45.0  # Safe test default
            
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
        time.sleep(3.0)  # Publish every 3 seconds

@atexit.register
def cleanup():
    shutdown_event.set()

if __name__ == '__main__':
    # Start the navigation background thread
    nav_thread = threading.Thread(target=run_navigation_loop, daemon=True)
    nav_thread.start()

    # Start the health reporting thread
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
