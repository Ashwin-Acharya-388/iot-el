import json
import yaml
import paho.mqtt.client as mqtt
from datetime import datetime

print("Starting MQTT Client...")

# Load config
with open("config/settings.yaml", "r") as file:
    config = yaml.safe_load(file)

BROKER = config["mqtt"]["broker"]
PORT = config["mqtt"]["port"]
TOPIC = config["mqtt"]["topic"]

print("Connecting to broker...")

# WebSocket MQTT client
client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    transport="websockets"
)

# WebSocket path
client.ws_set_options(path="/mqtt")

# Connect
client.connect(BROKER, PORT, 60)

print("Connected successfully!")

def publish_navigation(direction, obstacles, fps):

    data = {
        "timestamp": str(datetime.now()),
        "direction": direction,
        "obstacles": obstacles,
        "obstacle_count": len(obstacles),
        "fps": fps
    }

    message = json.dumps(data)

    client.publish(TOPIC, message)

    print("Published:", message)

if __name__ == "__main__":

    publish_navigation(
        "LEFT",
        ["person", "chair"],
        28
    )