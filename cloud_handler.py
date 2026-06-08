import json
import yaml
import time
from collections import deque
import paho.mqtt.client as mqtt
from datetime import datetime

# Load settings
try:
    with open("config/settings.yaml", "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"[CLOUD] Error loading config/settings.yaml: {e}. Using defaults.")
    config = {"mqtt": {}}

BROKER = config["mqtt"].get("broker", "broker.emqx.io")
PORT = config["mqtt"].get("port", 8083)
STATUS_TOPIC = config["mqtt"].get("status_topic", "iot/navigation/status")
ALERTS_TOPIC = config["mqtt"].get("alerts_topic", "iot/navigation/alerts")

print(f"[CLOUD] Connecting to MQTT broker {BROKER}:{PORT} over WebSockets...")

# Keep track of recent STOP events: list of timestamps
stop_timestamps = deque()
WINDOW_SEC = 5.0      # Short window to check for consecutive stops
CONSECUTIVE_LIMIT = 3 # Number of consecutive stops to trigger DANGER

# Last time danger alert was published (cooldown to avoid flooding)
last_alert_time = 0
ALERT_COOLDOWN_SEC = 10.0

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[CLOUD] Connected successfully! Subscribing to: {STATUS_TOPIC}")
        client.subscribe(STATUS_TOPIC)
    else:
        print(f"[CLOUD] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global last_alert_time
    try:
        payload = json.loads(msg.payload.decode())
        direction = payload.get("direction", "")
        print(f"[CLOUD] Received status: direction={direction}")

        current_time = time.time()
        
        # Clean up timestamps older than the window
        while stop_timestamps and (current_time - stop_timestamps[0] > WINDOW_SEC):
            stop_timestamps.popleft()

        if direction == "STOP":
            stop_timestamps.append(current_time)
            print(f"[CLOUD] Stop event registered. Active stops in window: {len(stop_timestamps)}/{CONSECUTIVE_LIMIT}")
            
            if len(stop_timestamps) >= CONSECUTIVE_LIMIT:
                # Check cooldown
                if current_time - last_alert_time > ALERT_COOLDOWN_SEC:
                    alert_payload = {
                        "timestamp": str(datetime.now()),
                        "status": "DANGER",
                        "reason": f"User is stuck! Received {len(stop_timestamps)} consecutive STOP commands in {WINDOW_SEC}s."
                    }
                    client.publish(ALERTS_TOPIC, json.dumps(alert_payload))
                    print(f"[CLOUD] !!! DANGER ALERT PUBLISHED !!! to {ALERTS_TOPIC}")
                    last_alert_time = current_time
                    # Clear the window to avoid re-triggering immediately
                    stop_timestamps.clear()
        else:
            # If we receive a non-STOP command, it means the user is moving/safe again,
            # so we clear the stop window.
            if stop_timestamps:
                print("[CLOUD] User is moving again. Clearing stop window.")
                stop_timestamps.clear()

    except Exception as e:
        print(f"[CLOUD] Error processing message: {e}")

# WebSocket MQTT client (matches port 8083)
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    transport="websockets"
)
client.ws_set_options(path="/mqtt")
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, PORT, 60)
    print(f"[CLOUD] Loop starting...")
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[CLOUD] Exiting gracefully.")
except Exception as e:
    print(f"[CLOUD] Connection error: {e}")
