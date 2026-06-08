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

ssh "${RPI_HOST}" "cd ${TARGET_DIR} && ( ./venv/bin/python -c 'import sys' 2>/dev/null || ( echo 'Recreating broken virtual environment...' && rm -rf venv && python3 -m venv venv ) ) && ./venv/bin/pip install -r requirements.txt && CAMERA_INDEX='${CAMERA_INDEX}' CAMERA_DEVICE='${CAMERA_DEVICE}' ./venv/bin/python stream_server.py"
