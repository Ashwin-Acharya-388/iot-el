Antigravity Agent Task: Integrate freespace_int8.onnx Model
Context
The project already has a working MQTT-based safety alert system:

Edge Node (Raspberry Pi) detects "STOP" commands and publishes to iot/navigation/status
Cloud Logic subscribes, detects consecutive "STOP" states, and publishes "DANGER" to iot/navigation/alerts
Caretaker Dashboard subscribes to iot/navigation/alerts via WebSocket and shows visual warnings

A new model file freespace_int8.onnx has been uploaded to the models/ folder.

⚠️ Testing Mode: For now, use the Mac laptop's built-in camera instead of the Raspberry Pi as the edge node. Everything else (MQTT, cloud logic, dashboard) remains the same. Once testing is complete the Mac camera script can be swapped back to the Pi with minimal changes.


🎯 Objective
Integrate models/freespace_int8.onnx into the full system pipeline using the Mac camera as the temporary edge node:

Capture frames from the Mac built-in camera using OpenCV
Run freespace inference on each frame
Publish freespace + navigation data to MQTT
Use freespace output in the cloud's danger detection logic
Display freespace results on the caretaker dashboard


📋 Step-by-Step Tasks

STEP 1 — Verify the Model File
- Confirm `models/freespace_int8.onnx` exists in the project
- Inspect the model's input/output shape using onnxruntime:

    import onnxruntime as ort
    session = ort.InferenceSession("models/freespace_int8.onnx")
    for inp in session.get_inputs():
        print("Input:", inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        print("Output:", out.name, out.shape, out.type)

- Record the exact input name, shape (e.g. [1, 3, H, W]), and output name/shape
- This determines how to preprocess frames and interpret results

STEP 2 — Edge Node: Mac Laptop Camera Script (Testing Replacement for Raspberry Pi)
Create a new file: mac_edge_node.py
This replaces the Pi edge node for local testing. It uses the Mac built-in camera via OpenCV.

2a. Install dependencies if not already installed:

    pip install opencv-python onnxruntime paho-mqtt numpy

2b. Create mac_edge_node.py with the following structure:

    import cv2
    import onnxruntime as ort
    import numpy as np
    import paho.mqtt.client as mqtt
    import json
    import time

    # ── MQTT config (match your existing broker settings) ──────────────────────
    MQTT_BROKER   = "localhost"   # or your cloud broker IP / hostname
    MQTT_PORT     = 1883
    MQTT_TOPIC    = "iot/navigation/status"

    # ── Model ──────────────────────────────────────────────────────────────────
    freespace_session    = ort.InferenceSession("models/freespace_int8.onnx")
    freespace_input_name = freespace_session.get_inputs()[0].name
    input_shape          = freespace_session.get_inputs()[0].shape
    # input_shape is typically [1, 3, H, W] — extract H and W:
    MODEL_H = input_shape[2] if input_shape[2] != -1 else 256
    MODEL_W = input_shape[3] if input_shape[3] != -1 else 256

    # ── MQTT client ────────────────────────────────────────────────────────────
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # ── Camera ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)   # 0 = Mac built-in FaceTime camera
    if not cap.isOpened():
        raise RuntimeError("Could not open Mac camera. Check camera permissions in System Settings > Privacy > Camera.")

    print("Mac edge node running. Press Q in the preview window to stop.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame — retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # ── Run inference every 3rd frame to keep CPU load low ────────────
            if frame_count % 3 == 0:

                # Preprocess
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (MODEL_W, MODEL_H))
                tensor  = resized.astype(np.float32) / 255.0
                tensor  = np.transpose(tensor, (2, 0, 1))   # HWC → CHW
                tensor  = np.expand_dims(tensor, axis=0)    # add batch dim → [1,3,H,W]

                # Inference
                try:
                    outputs        = freespace_session.run(None, {freespace_input_name: tensor})
                    freespace_mask = outputs[0]              # [1, H, W] or [1, 1, H, W]
                    free_ratio     = float(np.mean(freespace_mask > 0.5))
                except Exception as e:
                    print(f"Inference error: {e}")
                    free_ratio = 1.0

                # Classify
                if free_ratio < 0.10:
                    freespace_status = "BLOCKED"
                elif free_ratio < 0.30:
                    freespace_status = "LIMITED"
                else:
                    freespace_status = "CLEAR"

                # Simulate navigation status for testing
                # Replace this with your real navigation output when available
                nav_status = "STOP" if freespace_status == "BLOCKED" else "GO"

                # Publish to MQTT
                payload = {
                    "navigation_status": nav_status,
                    "freespace_status":  freespace_status,
                    "free_ratio":        round(free_ratio, 3),
                    "timestamp":         time.time()
                }
                client.publish(MQTT_TOPIC, json.dumps(payload))
                print(f"Published → nav: {nav_status} | freespace: {freespace_status} ({free_ratio:.1%} free)")

                # ── Optional: show live preview with overlay ──────────────────
                label = f"{freespace_status}  {free_ratio:.1%} free"
                color = (0, 200, 0) if freespace_status == "CLEAR" else \
                        (0, 165, 255) if freespace_status == "LIMITED" else (0, 0, 255)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.imshow("Mac Edge Node — Freespace Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()
        print("Edge node stopped.")

2c. Mac camera permission:
    - macOS requires explicit camera permission for Python/OpenCV
    - If the camera does not open, go to:
      System Settings → Privacy & Security → Camera → enable Terminal (or your IDE)
    - Re-run the script after granting permission

2d. To run the Mac edge node:

    python mac_edge_node.py

    A preview window will open showing the camera feed with the freespace status overlaid.
    Press Q to stop.

STEP 3 — Cloud Logic: Update Danger Detection
File to edit: the cloud handler/subscriber (e.g. cloud_handler.py, alert_service.py, or lambda_function.py)

3a. Update the MQTT message parser to read the new fields:

    data = json.loads(message.payload)
    nav_status = data.get("navigation_status", "")
    freespace_status = data.get("freespace_status", "CLEAR")
    free_ratio = data.get("free_ratio", 1.0)

3b. Update the danger detection logic to factor in freespace:
    - Original condition: N consecutive "STOP" commands → DANGER
    - New compound condition:

        is_nav_stuck = (nav_status == "STOP")
        is_path_blocked = (freespace_status == "BLOCKED")

        # Add to your existing consecutive-stop counter as before
        if is_nav_stuck:
            stop_counter += 1
        else:
            stop_counter = 0

        # Trigger DANGER if stuck AND path is confirmed blocked
        if stop_counter >= CONSECUTIVE_STOP_THRESHOLD or (is_nav_stuck and is_path_blocked):
            danger_payload = {
                "alert_type": "DANGER",
                "reason": "User is stuck with no free path",
                "freespace_status": freespace_status,
                "free_ratio": free_ratio,
                "consecutive_stops": stop_counter,
                "timestamp": time.time()
            }
            mqtt_client.publish("iot/navigation/alerts", json.dumps(danger_payload))
            stop_counter = 0  # reset after alert

3c. Also publish a separate freespace topic so the dashboard can visualize it independently:

    freespace_update = {
        "freespace_status": freespace_status,
        "free_ratio": free_ratio,
        "timestamp": time.time()
    }
    mqtt_client.publish("iot/navigation/freespace", json.dumps(freespace_update))

STEP 4 — Dashboard Frontend: Display Freespace Data
File to edit: the dashboard frontend (e.g. dashboard.js, App.jsx, index.html, or the relevant React/Vue component)

4a. Subscribe to the new freespace topic via WebSocket (alongside the existing alerts subscription):

    mqttClient.subscribe("iot/navigation/freespace");
    mqttClient.subscribe("iot/navigation/alerts");   // already exists

4b. Handle incoming freespace messages:

    mqttClient.on("message", (topic, message) => {
        const data = JSON.parse(message.toString());

        if (topic === "iot/navigation/alerts") {
            // existing alert handling — keep as-is
            showDangerAlert(data);
        }

        if (topic === "iot/navigation/freespace") {
            updateFreespacePanel(data);
        }
    });

4c. Implement updateFreespacePanel to render on the UI:

    function updateFreespacePanel(data) {
        const statusEl = document.getElementById("freespace-status");
        const ratioEl  = document.getElementById("freespace-ratio");

        statusEl.textContent = data.freespace_status;
        ratioEl.textContent  = `${(data.free_ratio * 100).toFixed(1)}% free`;

        // Color code the status
        statusEl.className = "";
        if (data.freespace_status === "BLOCKED")  statusEl.classList.add("status-danger");
        if (data.freespace_status === "LIMITED")  statusEl.classList.add("status-warning");
        if (data.freespace_status === "CLEAR")    statusEl.classList.add("status-safe");
    }

4d. Add the freespace panel to the dashboard HTML (place it near the navigation status section):

    <div id="freespace-panel" class="dashboard-card">
        <h3>🛣️ Freespace Detection</h3>
        <p>Status: <span id="freespace-status">--</span></p>
        <p>Path: <span id="freespace-ratio">--</span></p>
    </div>

4e. Add CSS for the status colors (add to your existing stylesheet):

    .status-danger  { color: #e74c3c; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-safe    { color: #27ae60; font-weight: bold; }

4f. Update the existing DANGER alert display to also show freespace info:
    - When a DANGER alert fires, show both the alert reason AND the freespace status
    - Example: "⚠️ DANGER: User is stuck — Path is BLOCKED (2% free)"

STEP 5 — MQTT Topic Summary (Confirm All Topics Are Correct)
After integration, these are the three active MQTT topics:

| Topic                        | Publisher     | Subscriber          | Payload Fields                                      |
|------------------------------|---------------|---------------------|-----------------------------------------------------|
| iot/navigation/status        | Raspberry Pi  | Cloud Handler       | navigation_status, freespace_status, free_ratio, timestamp |
| iot/navigation/alerts        | Cloud Handler | Dashboard           | alert_type, reason, freespace_status, free_ratio, consecutive_stops, timestamp |
| iot/navigation/freespace     | Cloud Handler | Dashboard           | freespace_status, free_ratio, timestamp             |

Confirm all three topics are publishing and receiving correctly before finishing.

STEP 6 — Testing & Validation on Mac
6a. Start services in this order:
    1. MQTT broker:         mosquitto (or your cloud broker)
    2. Cloud handler:       python cloud_handler.py  (or your existing cloud script)
    3. Dashboard:           open in browser
    4. Mac edge node:       python mac_edge_node.py

6b. Verify the preview window opens showing the Mac camera feed with the freespace label overlay.

6c. Cover the camera with your hand (simulates BLOCKED path):
    - Terminal should print: nav: STOP | freespace: BLOCKED
    - Dashboard freespace panel should turn RED and show BLOCKED
    - After consecutive STOPs, a DANGER alert should appear on the dashboard

6d. Uncover the camera (simulates CLEAR path):
    - Terminal should print: nav: GO | freespace: CLEAR
    - Dashboard panel should turn GREEN

6e. Test LIMITED state:
    - Partially cover the camera so ~70–90% is blocked
    - Should print: freespace: LIMITED and show ORANGE on dashboard

6f. Confirm no crashes:
    - Let it run for 2+ minutes
    - Confirm MQTT messages are arriving at the cloud handler (check cloud handler logs)

6g. When ready to move to Raspberry Pi later:
    - Copy mac_edge_node.py to the Pi
    - Change cv2.VideoCapture(0) to the correct Pi camera index (usually still 0)
    - Update MQTT_BROKER to the broker IP if different
    - No other changes needed

STEP 7 — Final Summary
After completing all steps, provide a summary with:
- Which files were modified and what was changed in each
- The exact input/output shape of freespace_int8.onnx as discovered in Step 1
- Confirmation that all three MQTT topics are active
- Any assumptions made about preprocessing (normalization values, resize dimensions)
- Any issues encountered and how they were resolved

⚠️ Important Notes for the Agent

Mac camera index is 0 — this is the built-in FaceTime HD camera; if it fails, try index 1
macOS camera permission is required — if cap.isOpened() returns False, the user must grant Terminal camera access in System Settings → Privacy & Security → Camera
mac_edge_node.py is a NEW file — do not modify the existing Pi edge node script; keep it intact for later
Simulated nav_status — the Mac script simulates navigation_status based on freespace output for testing; when the real nav model is connected, replace that line with the actual nav output
Do NOT change the existing iot/navigation/alerts topic structure — the dashboard already consumes it
Do NOT remove the consecutive-STOP counter logic — keep it and extend it
Use ort.InferenceSession — do not switch to a different ONNX runtime
The model is INT8 quantized — input must still be float32; quantization is internal to the model
If inference is slow on Mac, increase the frame skip from every 3rd to every 5th frame: if frame_count % 5 == 0