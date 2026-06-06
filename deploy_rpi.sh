#!/usr/bin/env bash
set -euo pipefail

RPI_HOST="${1:-pi@raspberrypi.local}"
TARGET_DIR="/home/pi/iot-el"

echo "Deploying project to ${RPI_HOST}:${TARGET_DIR}"
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'audio/*.wav' \
  . "${RPI_HOST}:${TARGET_DIR}/"

ssh "${RPI_HOST}" "cd ${TARGET_DIR} && python3 -m pip install --user -r requirements.txt && python3 stream_server.py"
