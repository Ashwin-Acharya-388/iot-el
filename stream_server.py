import os
import cv2
import pyttsx3
from flask import Flask, Response, request, jsonify, send_from_directory

# Automatically locate your UI folder based on your directory layout
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FOLDER = os.path.join(BASE_DIR, 'dashborad') # Matches your exact directory name

app = Flask(__name__, static_folder=UI_FOLDER, template_folder=UI_FOLDER)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Global dictionary placeholder to safely handle telemetry metrics if your UI calls an API
telemetry_data = {
    "direction": "FORWARD",
    "fps": 18.5,
    "obstacle_count": 0,
    "status": "System Online"
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
        camera = cv2.VideoCapture(0)  # Opens integrated laptop webcam
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
@app.route('/speak')
def speak():
    """Receives navigation text strings and runs them through your laptop audio hardware."""
    text = request.args.get('text', '')
    if text:
        print(f"[AUDIO LOG] System Spoke: {text}")
        engine.say(text)
        engine.runAndWait()
        return "OK", 200
    return "No text provided", 400

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
    # Binds globally across all network adapters over your active mobile hotspot connection
    app.run(host='0.0.0.0', port=8080, threaded=True)