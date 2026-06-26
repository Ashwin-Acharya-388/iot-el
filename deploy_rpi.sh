#!/usr/bin/env bash
set -euo pipefail

RPI_HOST="${1:-jatayu@jatayu.local}"
TARGET_DIR="/home/jatayu/freespace_navigation"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_DEVICE="${CAMERA_DEVICE:-}"

echo "Deploying project to ${RPI_HOST}:${TARGET_DIR}"
rsync -av --delete \
  --exclude '.git' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'audio/*.wav' \
  . "${RPI_HOST}:${TARGET_DIR}/"

ssh "${RPI_HOST}" "cd ${TARGET_DIR} && ( ./venv/bin/python -c 'import sys' 2>/dev/null || ( echo 'Recreating broken virtual environment...' && rm -rf venv && python3 -m venv venv ) ) && ./venv/bin/pip install flask numpy opencv-python-headless onnxruntime pyttsx3 pygame gTTS psutil PyYAML paho-mqtt firebase-admin --extra-index-url https://www.piwheels.org/simple && nohup python3 -m http.server 8000 -d caretaker-portal > portal.log 2>&1 & CAMERA_INDEX='${CAMERA_INDEX}' CAMERA_DEVICE='${CAMERA_DEVICE}' ./venv/bin/python app.py"
