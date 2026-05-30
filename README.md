# Head-Mounted Navigation Assistant for the Visually Impaired

### Smart Assistive Technology using Computer Vision and IoT

<p align="center">
<img src="https://img.shields.io/badge/python-3.10+-blue"
     alt="Python 3.10+">
<img src="https://img.shields.io/badge/PyTorch-2.0+-red"
     alt="PyTorch 2.0+">
<img src="https://img.shields.io/badge/YOLO-v8-black"
     alt="YOLOv8">
<img src="https://img.shields.io/badge/MQTT-enabled-green"
     alt="MQTT">
</p>

This project implements an end-to-end AI-powered navigation system that assists visually impaired individuals by providing real-time obstacle detection and environmental awareness through audio feedback.

---

## 🚀 Features

- **Real-Time Obstacle Detection**
  - Detects obstacles from 20cm to 25m
  - Classifies 21 common urban objects:
    `person, backpack, handbag, neck bag, traffic light, fire hydrant, stop sign, parking meter, bench, bird, bus, car, motorcycle, bicycle, scooter, taxi, truck, tram, boat, traffic light (horizontal), unknown object`

- **Multi-Layer Depth Perception**
  - **3D Bounding Boxes**: Provides distance, width, and height in real-time
  - **Semantic Depth**: Color-coded heatmap based on obstacle distance
  - **Obstacle Confidence**: Multi-threshold ranging (50% to 90% confidence)
  - **Object Size Estimation**: Estimates height and width for better situational awareness
  - **Depth Clustering**: Groups nearby objects for cleaner audio guidance

- **Advanced Audio Navigation**
  - **Directional Guidance**: Speaks the distance and direction of the nearest obstacle
  - **Distance Warning Tiers**:
    - ⚠️ **2m - 5m**: "Obstacle ahead in 4.5 meters"
    - ⚠️ **5m - 15m**: "Be careful, object in 10.2 meters"
    - ⛔ **0m - 2m**: "Stop! Obstacle 1.8 meters, width 0.6 meters"
  - **Multiple Announcement Modes**:
    - `fast`: Continuous 1-second interval updates
    - `normal`: 3-second interval with smooth transition
    - `safe`: 5-second interval with maximum smoothness
  - **Natural Voice Synthesis**: High-quality Text-to-Speech (TTS)

- **Multi-Sensor Integration**
  - **Camera**: RGB image capture for object detection
  - **LIDAR**: Depth data fusion for precise 3D perception
  - **GPS + IMU**: Navigation and location awareness
  - **Microphone**: Voice commands for hands-free operation

- **IoT Connectivity**
  - **MQTT Client**: Real-time data publishing to cloud platforms
  - **Command Center**: Voice-activated control panel
  - **Telepresence Mode**: Remote monitoring and guidance

- **Smart Features**
  - **Automatic Calibration**: Self-calibrating sensor alignment
  - **Energy-Efficient Processing**: CPU-only operation (no GPU required)
  - **Command Center**: Voice interface for system control
  - **User Profiling**: Personalized settings and preferences

---

## 🛠️ Tech Stack

- **AI Framework**: PyTorch 2.0+
- **Computer Vision**: YOLOv8 (Ultralytics)
- **3D Processing**: NumPy, SciPy
- **Sensor Integration**: OpenCV, LIDAR drivers, GPS libraries
- **IoT**: MQTT, WebSockets
- **TTS**: pyttsx3 (local) or cloud-based alternatives
- **Platform**: Linux, Windows, macOS

---

## 📂 Project Structure

```
iot-navigation-system/
├── src/                     # Source code modules
│   ├── main.py              # Main application entry point
│   ├── obstacle_detector.py # YOLOv8 obstacle detection
│   ├── depth_estimator.py   # 3D depth calculations
│   ├── audio_navigator.py   # Audio guidance system
│   ├── sensor_manager.py    # Sensor fusion and calibration
│   ├── command_center.py  # Voice command interface
│   └── mqtt_client.py       # IoT connectivity
├── datasets/                # Dataset files
│   ├── yolo_cityscapes/     # Cityscapes dataset
│   └── models/              # Pre-trained YOLOv8 models
├── experiments/             # Training and evaluation logs
│   ├── training_logs/       # Training history
│   └── evaluation_metrics/  # Performance metrics
├── config/                  # Configuration files
│   ├── settings.yaml        # System settings
│   └── class_mapping.json   # Object class mappings
├── data/                    # Raw and processed data
│   ├── cityscapes/          # Raw Cityscapes data
│   └── annotations/         # Processed annotations
├── models/                  # Trained model weights
│   ├── yolo_cityscapes.pt   # YOLOv8 model
│   └── custom_models/       # Custom trained models
└── requirements.txt         # Python dependencies
```

---

## 🔌 Sensor Configuration

The system supports both simulated and real sensor inputs.

### Required Sensors

| Sensor | Type | Purpose |
|--------|------|---------|
| **Camera** | Webcam / USB | Object detection |
| **LIDAR** | RPLIDAR / ROS | Depth measurements |
| **GPS** | U-Blox / USB | Geographic positioning |
| **IMU** | MPU6050 / USB | Orientation sensing |
| **Microphone** | USB / Built-in | Voice commands |

### Automatic Calibration

The system uses LiDAR SLAM to automatically calibrate sensor positions and orientations:

```python
# Automatic calibration sequence
1. LiDAR scan to create 2D map
2. Camera image capture for object detection
3. GPS fix for world-frame alignment
4. IMU initialization for orientation
5. SLAM refinement for sensor fusion
```

### Mounting Guidelines

```
┌──────────────────────────────────┐
│  ┌──────────────┐                │
│  │   Camera     │◄───────────────┤  Obstacle: 20cm - 25m
│  │ (Front-facing) │                │
│  └──────────────┘                │
│                                  │
│  ┌──────────────┐                │
│  │   LIDAR      │◄───────────────┤  Scanning angle: 360°
│  │ (360° scanner) │                │
│  └──────────────┘                │
│                                  │
│  ┌──────────────┐                │
│  │   GPS + IMU  │◄───────────────┤  Position + orientation
│  │   (Helmet)   │                │
│  └──────────────┘                │
│                                  │
│  ┌──────────────┐                │
│  │ Microphone   │◄───────────────┤  Voice commands
│  │ (Near mouth) │                │
│  └──────────────┘                │
│                                  │
└──────────────────────────────────┘
```

---

## 💾 Dataset Preparation

### Required Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| **Cityscapes** | Object detection training | ~4.5 GB |
| **Semantic KITTI** | Depth perception | ~300 GB |
| **nuScenes** | Full autonomous driving | ~800 GB |

### Automatic Dataset Download

```bash
# Download Cityscapes dataset
python download_datasets.py --dataset cityscapes --email [EMAIL_ADDRESS] --password [PASSWORD]

# Download all datasets
python download_datasets.py --dataset all
```

### Cityscapes Dataset

**Download links** (may require registration):

1. **leftImg8bit_trainvaltest.zip** - Image data (~2.4 GB)
   - [Download link](https://www.cityscapes-dataset.com/downloads/)

2. **gtFine_trainvaltest.zip** - Ground truth annotations (~240 MB)
   - [Download link](https://www.cityscapes-dataset.com/downloads/)

**Installation:**

```bash
# Create data
