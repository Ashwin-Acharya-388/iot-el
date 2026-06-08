import glob
import os
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory

try:
    from voice_commands import VoiceCommands
except Exception:
    VoiceCommands = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FOLDER = os.path.join(BASE_DIR, 'iot-el', 'dashborad')
if not os.path.isdir(UI_FOLDER):
    UI_FOLDER = os.path.join(BASE_DIR, 'dashborad')

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8080'))
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_DEVICE = os.getenv('CAMERA_DEVICE', '').strip()

app = Flask(__name__, static_folder=UI_FOLDER, template_folder=UI_FOLDER)

voice = VoiceCommands(cooldown=1.2) if VoiceCommands is not None else None

last_alert_text = ""
last_alert_time = 0.0

# Global dictionary placeholder to safely handle telemetry metrics if your UI calls an API
telemetry_data = {
    "direction": "FORWARD",
    "fps": 18.5,
    "obstacle_count": 0,
    "status": {
        "camera": "Connected",
        "model": "Running",
        "server": "Active"
    },
    "obstacles": []
}

# 1. SERVE FRONTEND DASHBOARD
@app.route('/')
def index():
    """Serves your team's main index.html dashboard file automatically."""
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serves accompanying static files like style.css and script.js smoothly."""
    return send_from_directory(UI_FOLDER, path)

@app.route('/favicon.ico')
def favicon():
    return '', 204

def open_camera():
    candidates = []
    if CAMERA_DEVICE:
        candidates.append(CAMERA_DEVICE)

    existing_video_devices = sorted(glob.glob('/dev/video*'))
    candidates.extend(existing_video_devices)

    candidates.extend([str(i) for i in range(0, 10)])
    candidates.append(CAMERA_INDEX)
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
                if isinstance(candidate, str) and os.path.exists(candidate):
                    cap = cv2.VideoCapture(candidate, backend)
                else:
                    cap = cv2.VideoCapture(int(candidate), backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        print(f"[CAMERA] Using camera source: {candidate} (backend={backend})")
                        return cap
                cap.release()
            except Exception as exc:
                print(f"[CAMERA] Failed candidate {candidate} backend {backend}: {exc}")

    print("[CAMERA] No working camera device found.")
    return None


def detect_obstacles(frame):
    """Simple fallback obstacle detector for live dashboard telemetry."""
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
        direction = "Slight Left"
    elif closest['center_x'] > 0.67:
        direction = "Slight Right"
    else:
        direction = "FORWARD"

    return direction, obstacles, len(obstacles)


@app.route('/video')
def video_feed():
    """Captures live webcam frames from the Raspberry Pi and streams them to the UI image tags."""
    def generate_frames():
        camera = open_camera()
        if camera is None:
            return

        frame_timer = time.perf_counter()
        while True:
            success, frame = camera.read()
            if not success:
                break

            direction, obstacles, obstacle_count = detect_obstacles(frame)
            global last_alert_text, last_alert_time
            telemetry_data.update({
                "direction": direction,
                "fps": round(1.0 / max(0.001, time.perf_counter() - frame_timer), 1),
                "obstacle_count": obstacle_count,
                "obstacles": obstacles[:5],
                "status": {
                    "camera": "Connected",
                    "model": "Live Detection",
                    "server": "Active"
                },
            })
            frame_timer = time.perf_counter()

            if obstacles and obstacles[0]['distance'] < 3.0:
                alert_text = f"Obstacle {obstacles[0]['distance']} meters ahead"
                now = time.time()
                if alert_text != last_alert_text or (now - last_alert_time) > 3.0:
                    last_alert_text = alert_text
                    last_alert_time = now
                    threading.Thread(target=speak_text, args=(alert_text,), daemon=True).start()

            if obstacles:
                x, y, w, h = obstacles[0]['x'], obstacles[0]['y'], obstacles[0]['w'], obstacles[0]['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{obstacles[0]['label']} {obstacles[0]['distance']}m", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        camera.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 3. VOICE ALERT ENDPOINT
def speak_text(text):
    if voice is not None:
        try:
            voice.speak(text)
            return
        except Exception as exc:
            print(f"[AUDIO] VoiceCommands playback failed: {exc}")
    print(f"[AUDIO] Voice backend unavailable. Text: {text}")

@app.route('/speak')
def speak():
    """Receives navigation text strings and runs them through the laptop audio hardware."""
    text = request.args.get('text', '').strip()
    if not text:
        return "No text provided", 400

    print(f"[AUDIO LOG] System Spoke: {text}")
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()
    return "OK", 200

# 4. TELEMETRY API ENDPOINT (Optional loop hook for future data logs)
@app.route('/api/telemetry', methods=['GET', 'POST'])
def telemetry():
    global telemetry_data
    if request.method == 'POST':
        data = request.json
        if data:
            telemetry_data.update(data)
        return jsonify({"status": "success"})
    return jsonify(telemetry_data)

if __name__ == '__main__':
    print(f"[SERVER] Starting dashboard on http://{HOST}:{PORT} (camera={CAMERA_DEVICE or CAMERA_INDEX})")
    app.run(host=HOST, port=PORT, threaded=True)