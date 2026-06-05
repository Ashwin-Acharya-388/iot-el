import os
import sys
import time
import base64
import threading
import yaml
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.resolve()))

from navigation_system_rpi import (
    NavigationSystem,
    ONNXInference,
    SimpleTracker,
    TemporalSmoother,
    DirectionVoter,
    find_safe_direction,
    estimate_distance,
    CLASS_NAMES,
)

# Initialize Flask app
# Serve dashboard static files from 'dashborad' directory
app = Flask(__name__, static_folder='dashborad', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load settings
CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    HOST = config["dashboard"]["host"]
    PORT = config["dashboard"]["port"]
except Exception as e:
    print(f"[WARN] Failed to load settings.yaml: {e}. Using defaults.")
    HOST = "127.0.0.1"
    PORT = 5500

@app.route('/')
def index():
    return app.send_static_file('index.html')

# State variables
camera_active = False
model_active = False
navigation_thread = None
shutdown_event = threading.Event()

def run_navigation_loop():
    global camera_active, model_active
    
    print("\n[BACKEND] Starting Navigation and Video Stream Loop...")
    
    # 1. Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        print("[BACKEND] ✓ Camera opened successfully.")
        camera_active = True
    else:
        print("[BACKEND] ⚠ Camera index 0 could not be opened. Running in SIMULATED Camera mode.")
        camera_active = False
        cap.release()
        
    # 2. Initialize ONNX Inference Engine
    model_path = Path(__file__).parent / "models" / "yolov8n_qat.onnx"
    engine = None
    if model_path.exists():
        try:
            engine = ONNXInference(model_path)
            model_active = True
            print("[BACKEND] ✓ ONNX Model loaded successfully.")
        except Exception as e:
            print(f"[BACKEND] ⚠ Failed to load ONNX model: {e}. Running in SIMULATED Model mode.")
            model_active = False
    else:
        print(f"[BACKEND] ⚠ Model not found at {model_path}. Running in SIMULATED Model mode.")
        model_active = False

    # 3. Setup tracking, smoothing, and voting
    tracker = SimpleTracker(max_age=5, min_hits=2)
    smoother = TemporalSmoother(window=3)
    voter = DirectionVoter(window=5)
    
    # Simulation variables
    sim_t = 0.0
    frame_times = []
    
    while not shutdown_event.is_set():
        t_start = time.perf_counter()
        
        # ── Capture Frame ──
        frame = None
        if camera_active:
            ret, frame = cap.read()
            if not ret:
                print("[BACKEND] ⚠ Real camera read failed; switching to simulated frame.")
                camera_active = False
                cap.release()
                
        if frame is None:
            # Generate premium HUD synthetic frame
            frame = np.zeros((320, 320, 3), dtype=np.uint8)
            # Add futuristic grid
            for x in range(0, 320, 40):
                cv2.line(frame, (x, 0), (x, 320), (20, 20, 40), 1)
            for y in range(0, 320, 40):
                cv2.line(frame, (0, y), (320, y), (20, 20, 40), 1)
            # Draw ground perspective guidelines
            cv2.line(frame, (160, 160), (40, 320), (0, 100, 255), 1)
            cv2.line(frame, (160, 160), (280, 320), (0, 100, 255), 1)
            
        # ── Get Detections ──
        dets = []
        if camera_active and model_active and engine is not None:
            # Real camera + Real model
            tensor = engine.preprocess(frame)
            output = engine.infer(tensor)
            dets = engine.postprocess(output, conf_thresh=0.45)
        else:
            # Simulated mode: Move simulated obstacles on the frame
            sim_t += 0.05
            
            # Obstacle 1: A Person moving back and forth in the center-left
            # Position cy goes from 0.4 (far) to 0.9 (close)
            cy1 = 0.65 + 0.25 * np.sin(sim_t)
            cx1 = 0.25 + 0.05 * np.cos(sim_t)
            w1, h1 = 0.12, 0.28
            x1_min = int((cx1 - w1/2) * 320)
            x1_max = int((cx1 + w1/2) * 320)
            y1_min = int((cy1 - h1/2) * 320)
            y1_max = int((cy1 + h1/2) * 320)
            conf1 = 0.88 + 0.05 * np.sin(sim_t)
            cls_id1 = 0  # Person
            
            # Obstacle 2: A Car coming closer on the right
            cy2 = 0.70 + 0.20 * np.cos(sim_t * 0.7)
            cx2 = 0.75 + 0.03 * np.sin(sim_t * 0.7)
            w2, h2 = 0.22, 0.18
            x2_min = int((cx2 - w2/2) * 320)
            x2_max = int((cx2 + w2/2) * 320)
            y2_min = int((cy2 - h2/2) * 320)
            y2_max = int((cy2 + h2/2) * 320)
            conf2 = 0.92
            cls_id2 = 2  # Car
            
            # Append simulated detections
            dets = [
                (x1_min, y1_min, x1_max, y1_max, conf1, cls_id1),
                (x2_min, y2_min, x2_max, y2_max, conf2, cls_id2),
            ]
            
        # ── Tracking & Smoothing ──
        tracked = tracker.update(dets)
        smoothed = smoother.update(tracked)
        
        # ── Path Finding ──
        raw_direction, closest_obs = find_safe_direction(smoothed)
        voted_direction = voter.vote(raw_direction)
        # Estimate distances for all obstacles
        obstacles_list = []
        for det in smoothed:
            cls_id = int(det[5])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            dist = estimate_distance(det)
            obstacles_list.append({
                'label': class_name.capitalize(),
                'distance': round(float(dist), 1)
            })
            
        # Draw bounding boxes and HUD text on the frame
        vis = frame.copy()
        
        # Draw horizontal zone dividers (Left, Center, Right)
        # ZONE_LEFT_MAX = 0.33, ZONE_RIGHT_MIN = 0.67
        cv2.line(vis, (int(320 * 0.33), 0), (int(320 * 0.33), 320), (50, 50, 100), 1)
        cv2.line(vis, (int(320 * 0.67), 0), (int(320 * 0.67), 320), (50, 50, 100), 1)
        
        # Draw Bounding Boxes
        for det in smoothed:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            conf = det[4]
            cls = int(det[5])
            dist = estimate_distance(det)
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "unknown"
            
            # Color: Red for high-danger, Green for others
            color = (0, 0, 255) if cls in {0, 1, 2, 3, 4, 5, 6, 7} else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label.capitalize()} {dist:.1f}m", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        
        # Encode visual frame to base64 jpeg
        _, buffer = cv2.imencode('.jpg', vis)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate FPS
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / np.mean(frame_times) if frame_times else 0
        
        # Determine confidence
        confidence = 0.95
        if closest_obs:
            # Lower confidence if there are obstacles close
            confidence = max(0.5, 1.0 - (1.0 / (closest_obs[1] + 0.1)))
            
        # Emit real-time telemetry over socket
        socketio.emit('navigation_data', {
            'frame': 'data:image/jpeg;base64,' + frame_base64,
            'direction': voted_direction.upper(),
            'confidence': int(confidence * 100),
            'obstacle_count': len(smoothed),
            'obstacles': obstacles_list,
            'status': {
                'camera': 'Connected' if camera_active else 'Simulated',
                'model': 'Running' if model_active else 'Simulated',
                'server': 'Active'
            },
            'fps': round(fps, 1)
        })
        
        # Enforce rate limit (approx. 6 FPS)
        sleep_time = max(0.01, 0.16 - (time.perf_counter() - t_start))
        time.sleep(sleep_time)
        
    if camera_active:
        cap.release()
    print("[BACKEND] Navigation loop shutdown completed.")

@socketio.on('connect')
def handle_connect():
    print("[BACKEND] Web client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    print("[BACKEND] Web client disconnected.")

if __name__ == '__main__':
    # Start background navigation loop thread
    navigation_thread = threading.Thread(target=run_navigation_loop, daemon=True)
    navigation_thread.start()
    
    print(f"\n[BACKEND] Running AI Navigation Server on http://{HOST}:{PORT}")
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        if navigation_thread:
            navigation_thread.join(timeout=2.0)
        print("[BACKEND] Server stopped.")
