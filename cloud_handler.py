import json
import os
import signal
import time
from collections import deque
from datetime import datetime, timezone

import firebase_admin
import paho.mqtt.client as mqtt
import yaml
from firebase_admin import credentials, db, firestore


CONFIG_PATH = "config/settings.yaml"
DEFAULT_MQTT = {
    "broker": "broker.emqx.io",
    "port": 8083,
    "status_topic": "iot/navigation/status",
    "alerts_topic": "iot/navigation/alerts",
}

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "firebase-service-account.json")
FIREBASE_BACKEND = os.getenv("FIREBASE_BACKEND", "firestore").strip().lower()
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
INCIDENT_COLLECTION = os.getenv("FIREBASE_INCIDENT_COLLECTION", "incident_logs")

WINDOW_SEC = 5.0
CONSECUTIVE_STOP_LIMIT = 3
DANGER_ALERT_COOLDOWN_SEC = 10.0

stop_timestamps = deque()
last_alert_time = 0.0


def load_mqtt_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return {**DEFAULT_MQTT, **(config.get("mqtt") or {})}
    except Exception as exc:
        print(f"[CLOUD] Could not load {CONFIG_PATH}: {exc}. Using MQTT defaults.")
        return DEFAULT_MQTT.copy()


MQTT_CONFIG = load_mqtt_config()
BROKER = MQTT_CONFIG.get("broker", DEFAULT_MQTT["broker"])
PORT = int(MQTT_CONFIG.get("port", DEFAULT_MQTT["port"]))
STATUS_TOPIC = MQTT_CONFIG.get("status_topic", DEFAULT_MQTT["status_topic"])
ALERTS_TOPIC = MQTT_CONFIG.get("alerts_topic", DEFAULT_MQTT["alerts_topic"])


def utc_iso_timestamp():
    return datetime.now(timezone.utc).isoformat()


def initialize_firebase():
    if not os.path.exists(FIREBASE_CREDENTIALS):
        print(f"[FIREBASE] WARNING: Service account file not found at {FIREBASE_CREDENTIALS!r}. "
              "Firebase logging will be disabled. Running in MQTT-only mode.")
        return None

    try:
        options = {}
        if FIREBASE_BACKEND == "rtdb":
            if not FIREBASE_DATABASE_URL:
                print("[FIREBASE] WARNING: FIREBASE_DATABASE_URL is required when FIREBASE_BACKEND=rtdb. Firebase disabled.")
                return None
            options["databaseURL"] = FIREBASE_DATABASE_URL

        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, options)

        if FIREBASE_BACKEND == "rtdb":
            print(f"[FIREBASE] Connected to Realtime Database node: {INCIDENT_COLLECTION}")
            return db.reference(INCIDENT_COLLECTION)

        if FIREBASE_BACKEND == "firestore":
            print(f"[FIREBASE] Connected to Firestore collection: {INCIDENT_COLLECTION}")
            return firestore.client().collection(INCIDENT_COLLECTION)
    except Exception as exc:
        print(f"[FIREBASE] WARNING: Failed to initialize Firebase ({exc}). Firebase logging disabled.")
        return None

    print("[FIREBASE] WARNING: Firebase backend not recognized. Firebase logging disabled.")
    return None


incident_store = initialize_firebase()


def write_incident(record):
    record = {
        "timestamp": utc_iso_timestamp(),
        **record,
    }

    if incident_store is not None:
        try:
            if FIREBASE_BACKEND == "rtdb":
                incident_store.push(record)
            else:
                incident_store.add(record)
            print(f"[FIREBASE] Incident logged to Firebase: {record}")
        except Exception as exc:
            print(f"[FIREBASE] Error logging incident to Firebase: {exc}")
    else:
        print(f"[CONSOLE LOG] (Firebase Offline) Incident recorded: {record}")


def normalize_status(payload):
    if isinstance(payload, str):
        return payload.strip().upper()

    if not isinstance(payload, dict):
        return ""

    raw_status = payload.get("direction") or payload.get("status") or payload.get("command") or ""
    return str(raw_status).strip().upper()


def decode_payload(message):
    raw_payload = message.payload.decode("utf-8", errors="replace").strip()
    if not raw_payload:
        return {}

    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        return {"direction": raw_payload}


def publish_danger_alert(client, stop_count):
    alert_payload = {
        "status": "DANGER",
        "event": "DANGER_STUCK",
        "timestamp": utc_iso_timestamp(),
        "reason": "User stuck",
        "details": f"Received {stop_count} STOP commands within {WINDOW_SEC:.0f} seconds.",
    }

    client.publish(ALERTS_TOPIC, json.dumps(alert_payload), qos=1)
    print(f"[MQTT] Published DANGER alert to {ALERTS_TOPIC}: {alert_payload}")
    write_incident(
        {
            "event": "DANGER_STUCK",
            "reason": "User stuck",
        }
    )


def handle_status_update(client, payload):
    global last_alert_time

    status = normalize_status(payload)
    if not status:
        print(f"[CLOUD] Ignoring status message without direction/status: {payload}")
        return

    print(f"[CLOUD] Status received: {status}")
    now = time.time()

    while stop_timestamps and now - stop_timestamps[0] > WINDOW_SEC:
        stop_timestamps.popleft()

    if status != "STOP":
        if stop_timestamps:
            print("[CLOUD] Movement resumed. Clearing STOP streak.")
            stop_timestamps.clear()
        return

    stop_timestamps.append(now)
    write_incident({"event": "STOP_ALERT"})
    print(f"[CLOUD] STOP streak: {len(stop_timestamps)}/{CONSECUTIVE_STOP_LIMIT}")

    if len(stop_timestamps) < CONSECUTIVE_STOP_LIMIT:
        return

    if now - last_alert_time < DANGER_ALERT_COOLDOWN_SEC:
        print("[CLOUD] DANGER condition met but alert is in cooldown.")
        return

    last_alert_time = now
    publish_danger_alert(client, len(stop_timestamps))
    stop_timestamps.clear()


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"[MQTT] Connected to {BROKER}:{PORT}. Subscribing to {STATUS_TOPIC}")
        client.subscribe(STATUS_TOPIC, qos=1)
    else:
        print(f"[MQTT] Connection failed: {reason_code}")


def on_message(client, userdata, message):
    try:
        payload = decode_payload(message)
        handle_status_update(client, payload)
    except Exception as exc:
        print(f"[CLOUD] Error processing MQTT message: {exc}")


def build_mqtt_client():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"cloud_handler_{int(time.time())}",
        transport="websockets",
    )
    client.ws_set_options(path="/mqtt")
    client.on_connect = on_connect
    client.on_message = on_message
    return client


def main():
    print(f"[MQTT] Connecting to {BROKER}:{PORT} over WebSockets")
    print(f"[MQTT] Status topic: {STATUS_TOPIC}")
    print(f"[MQTT] Alerts topic: {ALERTS_TOPIC}")

    client = build_mqtt_client()
    should_stop = {"value": False}

    def stop_handler(signum, frame):
        should_stop["value"] = True
        client.disconnect()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()

    while not should_stop["value"]:
        time.sleep(0.5)

    client.loop_stop()
    print("[CLOUD] Shutdown complete.")


if __name__ == "__main__":
    main()
