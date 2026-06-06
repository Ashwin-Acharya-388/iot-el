import os
import threading
import cv2
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

app = Flask(__name__, static_folder=UI_FOLDER, template_folder=UI_FOLDER)

voice = VoiceCommands(cooldown=1.2) if VoiceCommands is not None else None

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

# 2. LIVE CAMERA STREAM ENDPOINT
@app.route('/video')
def video_feed():
    """Captures live webcam frames from the laptop and streams them to the UI image tags."""
    def generate_frames():
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            print(f"[CAMERA] Camera index {CAMERA_INDEX} unavailable; trying fallback 0")
            camera = cv2.VideoCapture(0)

        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Compress the image matrix into a standard JPEG byte block
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
    print(f"[SERVER] Starting dashboard on http://{HOST}:{PORT} with camera index {CAMERA_INDEX}")
    app.run(host=HOST, port=PORT, threaded=True)