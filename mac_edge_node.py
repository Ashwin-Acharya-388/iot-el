import cv2
import onnxruntime as ort
import numpy as np
import paho.mqtt.client as mqtt
import json
import time

# ── MQTT config (connected to the public cloud broker matching your settings) ──
MQTT_BROKER   = "broker.emqx.io"
MQTT_PORT     = 1883
MQTT_TOPIC    = "iot/navigation/status"

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/freespace_int8.onnx"
print(f"[EDGE] Loading ONNX model from {MODEL_PATH}...")
freespace_session = ort.InferenceSession(MODEL_PATH)
freespace_input_name = freespace_session.get_inputs()[0].name
input_shape = freespace_session.get_inputs()[0].shape

# input_shape is [1, 3, 320, 320]
MODEL_H = input_shape[2] if input_shape[2] != -1 else 320
MODEL_W = input_shape[3] if input_shape[3] != -1 else 320
print(f"[EDGE] Model loaded. Expected input shape: {input_shape} (using {MODEL_W}x{MODEL_H})")

# ImageNet normalization parameters (to match kaggle training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── MQTT client ────────────────────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
print(f"[EDGE] Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}...")
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("[EDGE] MQTT connected successfully!")
except Exception as e:
    print(f"[EDGE] Failed to connect to MQTT broker: {e}")
    client = None

# ── Camera ─────────────────────────────────────────────────────────────────
# 0 = FaceTime camera. We will try 0, and fallback to 1 if it fails.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[EDGE] FaceTime Camera (index 0) could not be opened, trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open Mac camera. Please verify camera permissions in System Settings > Privacy & Security > Camera for your terminal or editor.")

print("\n[EDGE] Mac edge node running. Press 'q' in the preview window to stop.\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[EDGE] Failed to read frame — retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1

        # Run inference every 3rd frame to keep CPU load low
        if frame_count % 3 == 0:
            # Preprocess frame to match training input
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (MODEL_W, MODEL_H))
            
            # Normalize and transpose (HWC -> CHW)
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - MEAN) / STD
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0) # [1, 3, 320, 320]

            # Run freespace inference
            try:
                outputs = freespace_session.run(None, {freespace_input_name: tensor})
                # outputs[0] shape is [1, 2, 320, 320] (logits)
                # Take argmax along the class channel dimension (axis 1)
                pred_mask = np.argmax(outputs[0], axis=1)[0] # shape [320, 320]
                
                # Class 1 = walkable, Class 0 = obstacle
                free_ratio = float(np.mean(pred_mask == 1))
            except Exception as e:
                print(f"[EDGE] Inference error: {e}")
                free_ratio = 1.0
                pred_mask = None

            # Classify freespace status
            if free_ratio < 0.10:
                freespace_status = "BLOCKED"
            elif free_ratio < 0.30:
                freespace_status = "LIMITED"
            else:
                freespace_status = "CLEAR"

            # Simulate navigation status for testing (stuck on blocked)
            nav_status = "STOP" if freespace_status == "BLOCKED" else "GO"

            # Publish payload to MQTT status topic
            payload = {
                "navigation_status": nav_status,
                "freespace_status": freespace_status,
                "free_ratio": round(free_ratio, 3),
                "timestamp": time.time()
            }
            
            if client:
                client.publish(MQTT_TOPIC, json.dumps(payload))
                print(f"Published → nav: {nav_status} | freespace: {freespace_status} ({free_ratio:.1%} free)")

            # Create live preview overlay
            preview_frame = frame.copy()
            
            # Overlay freespace mask (colored green overlay) on resized preview
            if pred_mask is not None:
                # Resize pred_mask back to visual frame size for HUD overlay
                vis_mask = cv2.resize(pred_mask.astype(np.uint8), (preview_frame.shape[1], preview_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                green_overlay = np.zeros_like(preview_frame)
                green_overlay[vis_mask == 1] = [0, 255, 0] # Green for walkable
                
                # Blend overlay onto original frame
                preview_frame = cv2.addWeighted(preview_frame, 0.7, green_overlay, 0.3, 0)

            # Draw HUD labels
            label = f"{freespace_status}  {free_ratio:.1%} free"
            color = (0, 255, 0) if freespace_status == "CLEAR" else \
                    (0, 165, 255) if freespace_status == "LIMITED" else (0, 0, 255)
            
            cv2.putText(preview_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Mac Edge Node — Freespace Preview", preview_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if client:
        client.loop_stop()
        client.disconnect()
    print("[EDGE] Edge node stopped.")
