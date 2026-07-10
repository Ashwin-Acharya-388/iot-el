import json
import yaml
import time
# pyrefly: ignore [missing-import]
import paho.mqtt.client as mqtt
from datetime import datetime
import threading

print("Starting MQTT Client...")

# Load config
with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

BROKER = config["mqtt"]["broker"]
PORT = config["mqtt"]["port"]
TOPIC = config["mqtt"]["topic"]
STATUS_TOPIC = config["mqtt"].get("status_topic", "iot/navigation/status")
ALERTS_TOPIC = config["mqtt"].get("alerts_topic", "iot/navigation/alerts")
HEALTH_TOPIC = config["mqtt"].get("health_topic", "iot/navigation/health")

# Deduplication states
_last_nav_direction = None
_last_nav_obs_count = None
_last_nav_obs_labels = None
_last_nav_closest_dist = None
_last_nav_time = 0.0

_last_status_direction = None

print("Connecting to broker in background...")

# WebSocket MQTT client
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    transport="websockets"
)

# WebSocket path
client.ws_set_options(path="/mqtt")

is_connected = False

def _connect_bg():
    global is_connected
    try:
        client.connect(BROKER, PORT, 60)
        is_connected = True
        client.loop_start()
        print("Connected successfully in background!")
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")

# Start connecting in background
threading.Thread(target=_connect_bg, daemon=True).start()

def publish_navigation(direction, obstacles, fps):
    global _last_nav_direction, _last_nav_obs_count, _last_nav_obs_labels, _last_nav_closest_dist, _last_nav_time
    
    current_time = time.time()
    obs_count = len(obstacles)
    
    # Extract closest distance and labels
    closest_dist = None
    lbls = []
    for obs in obstacles:
        if isinstance(obs, dict):
            lbls.append(obs.get("label", "Obstacle"))
            if closest_dist is None and "distance" in obs:
                closest_dist = obs["distance"]
        else:
            lbls.append(str(obs))
    
    # Deduplication checks
    direction_changed = (direction != _last_nav_direction)
    obs_count_changed = (obs_count != _last_nav_obs_count)
    obs_list_changed = (lbls != _last_nav_obs_labels)
    
    dist_changed = False
    if closest_dist is not None and _last_nav_closest_dist is not None:
        dist_changed = (abs(closest_dist - _last_nav_closest_dist) > 0.25)
    elif closest_dist != _last_nav_closest_dist:
        dist_changed = True
        
    time_elapsed = (current_time - _last_nav_time >= 1.0)
    
    if not (direction_changed or obs_count_changed or obs_list_changed or dist_changed or time_elapsed):
        print("[MQTT] Publish skipped (unchanged)")
        return
        
    # Update states
    _last_nav_direction = direction
    _last_nav_obs_count = obs_count
    _last_nav_obs_labels = lbls
    _last_nav_closest_dist = closest_dist
    _last_nav_time = current_time

    data = {
        "timestamp": str(datetime.now()),
        "direction": direction,
        "obstacles": lbls,
        "obstacle_count": obs_count,
        "fps": fps
    }
    message = json.dumps(data)
    try:
        client.publish(TOPIC, message, qos=1)
        print("Published Navigation:", message)
    except Exception as e:
        print(f"[MQTT] Failed to publish navigation: {e}")

def publish_status(direction):
    global _last_status_direction
    
    if direction == _last_status_direction:
        print("[MQTT] Publish skipped (unchanged)")
        return
        
    _last_status_direction = direction

    data = {
        "timestamp": str(datetime.now()),
        "direction": direction
    }
    message = json.dumps(data)
    try:
        client.publish(STATUS_TOPIC, message, qos=1)
        print("Published Status:", message)
    except Exception as e:
        print(f"[MQTT] Failed to publish status: {e}")

def publish_health(temp, cpu, mem):
    data = {
        "timestamp": str(datetime.now()),
        "cpu_temp": temp,
        "cpu_usage": cpu,
        "memory_usage": mem
    }
    message = json.dumps(data)
    try:
        client.publish(HEALTH_TOPIC, message, qos=1)
        print("Published Health:", message)
    except Exception as e:
        print(f"[MQTT] Failed to publish health: {e}")
