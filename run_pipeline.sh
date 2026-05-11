#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_pipeline.sh
# ══════════════════════════════════════════════════════════════
# Runs the complete training and deployment pipeline for the
# Head-Mounted Navigation Assistant.
#
# Steps:
#   1. Download and prepare Cityscapes dataset
#   2. Train YOLOv8n baseline on Cityscapes
#   3. Quantize to INT8 + run QAT fine-tuning
#   4. Evaluate all three models
#   5. Transfer models to Raspberry Pi
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#   ./run_pipeline.sh --rpi-ip 192.168.1.42   # with auto-deploy to RPi
#   ./run_pipeline.sh --skip-train             # skip training, just quantize
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── CONFIG ───────────────────────────────────────────────────
RPI_USER="pi"
RPI_IP="${1:-}"       # Pass as first argument or set here
RPI_DIR="~/navigation"
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

# Parse named args
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
SKIP_QAT=false
SKIP_EVAL=false
SKIP_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rpi-ip)       RPI_IP="$2"; shift 2 ;;
        --skip-download)SKIP_DOWNLOAD=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --skip-qat)     SKIP_QAT=true; shift ;;
        --skip-eval)    SKIP_EVAL=true; shift ;;
        --skip-deploy)  SKIP_DEPLOY=true; shift ;;
        *) shift ;;
    esac
done

# ── COLORS ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
    echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════════${RESET}"
    echo -e "${CYAN}${BOLD}  $1${RESET}"
    echo -e "${CYAN}${BOLD}══════════════════════════════════════════════${RESET}\n"
}

step() {
    echo -e "${GREEN}▶ $1${RESET}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

error_exit() {
    echo -e "${RED}✗ ERROR: $1${RESET}"
    exit 1
}

success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

# ── CHECKS ───────────────────────────────────────────────────
banner "PRE-FLIGHT CHECKS"

command -v python3 &>/dev/null || error_exit "python3 not found"
python3 -c "import ultralytics" 2>/dev/null || {
    warn "ultralytics not installed. Installing requirements..."
    pip install -r requirements.txt
}

mkdir -p ./models ./data ./runs ./audio

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  GPU: Apple Silicon MPS')
else:
    print('  GPU: None (CPU only — training will be slow)')
" 2>/dev/null || warn "Could not detect GPU"

success "Pre-flight checks passed"

# ── STEP 1: DATASET ──────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    banner "STEP 1: DATASET DOWNLOAD & PREPARATION"
    step "Downloading and converting Cityscapes dataset..."
    python3 download_datasets.py 2>&1 | tee -a "$LOG_FILE"
    success "Dataset ready"
else
    warn "Skipping dataset download (--skip-download)"
fi

# ── STEP 2: BASELINE TRAINING ────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    banner "STEP 2: BASELINE TRAINING (YOLOv8n → Cityscapes)"
    step "Starting fine-tuning (~2-4 hours on GPU)..."
    step "Training log: $LOG_FILE"
    python3 train_baseline.py 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "./models/yolov8n_cityscapes.pt" ]; then
        error_exit "Training failed — yolov8n_cityscapes.pt not found"
    fi
    success "Baseline model saved: ./models/yolov8n_cityscapes.pt"
else
    warn "Skipping training (--skip-train)"
    if [ ! -f "./models/yolov8n_cityscapes.pt" ]; then
        error_exit "No baseline model found. Run without --skip-train first."
    fi
fi

# ── STEP 3: QUANTIZATION + QAT ───────────────────────────────
if [ "$SKIP_QAT" = false ]; then
    banner "STEP 3: INT8 QUANTIZATION + QAT FINE-TUNING"
    step "Quantizing baseline model and running QAT..."
    python3 quantize_and_qat.py 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "./models/yolov8n_qat.onnx" ]; then
        error_exit "QAT failed — yolov8n_qat.onnx not found"
    fi
    success "QAT model saved: ./models/yolov8n_qat.onnx"
else
    warn "Skipping QAT (--skip-qat)"
fi

# ── STEP 4: ACCURACY EVALUATION ──────────────────────────────
if [ "$SKIP_EVAL" = false ]; then
    banner "STEP 4: ACCURACY EVALUATION"
    step "Evaluating all models on Cityscapes validation set..."
    python3 test_accuracy_laptop.py 2>&1 | tee -a "$LOG_FILE"
    success "Evaluation complete. Report saved to ./models/accuracy_report.json"
else
    warn "Skipping evaluation (--skip-eval)"
fi

# ── MODEL SUMMARY ────────────────────────────────────────────
banner "MODEL SUMMARY"
echo "  Models in ./models/:"
for f in ./models/*.pt ./models/*.onnx; do
    [ -f "$f" ] || continue
    size=$(du -sh "$f" | cut -f1)
    echo "    $size  $f"
done

# ── STEP 5: DEPLOY TO RPi ────────────────────────────────────
if [ -n "$RPI_IP" ] && [ "$SKIP_DEPLOY" = false ]; then
    banner "STEP 5: DEPLOY TO RASPBERRY PI"
    step "Deploying to ${RPI_USER}@${RPI_IP}:${RPI_DIR}"

    echo "  Creating directory on RPi..."
    ssh "${RPI_USER}@${RPI_IP}" "mkdir -p ${RPI_DIR}/models ${RPI_DIR}/audio" || {
        warn "Could not create directory on RPi. Check SSH access."
    }

    echo "  Copying Python scripts..."
    rsync -av --progress \
        navigation_system_rpi.py \
        voice_commands.py \
        benchmark_rpi.py \
        "${RPI_USER}@${RPI_IP}:${RPI_DIR}/"

    echo "  Copying models (this may take a few minutes)..."
    rsync -av --progress \
        ./models/yolov8n_qat.onnx \
        ./models/yolov8n_int8.onnx \
        ./models/yolov8n_fp32.onnx \
        "${RPI_USER}@${RPI_IP}:${RPI_DIR}/models/"

    echo "  Copying requirements..."
    rsync -av requirements.txt "${RPI_USER}@${RPI_IP}:${RPI_DIR}/"

    echo ""
    echo "  RPi setup commands (run these on your RPi):"
    echo "  ─────────────────────────────────────────────────────"
    echo "  cd ${RPI_DIR}"
    echo "  pip install numpy opencv-python-headless onnxruntime \\"
    echo "              supervision pyttsx3 pygame gTTS psutil PyYAML \\"
    echo "              --extra-index-url https://www.piwheels.org/simple"
    echo "  sudo apt install -y espeak espeak-data"
    echo ""
    echo "  # Run benchmark:"
    echo "  python3 benchmark_rpi.py"
    echo ""
    echo "  # Run navigation system:"
    echo "  python3 navigation_system_rpi.py"
    echo ""
    echo "  # Run headless (no monitor):"
    echo "  python3 navigation_system_rpi.py --model ./models/yolov8n_qat.onnx"
    echo "  ─────────────────────────────────────────────────────"

    success "Files transferred to RPi"
elif [ -z "$RPI_IP" ] && [ "$SKIP_DEPLOY" = false ]; then
    banner "DEPLOY TO RPi (Manual)"
    echo "  To deploy to Raspberry Pi, run:"
    echo ""
    echo "  # Copy scripts:"
    echo "  rsync -av navigation_system_rpi.py voice_commands.py benchmark_rpi.py \\"
    echo "         pi@<RPI_IP>:~/navigation/"
    echo ""
    echo "  # Copy models:"
    echo "  rsync -av ./models/ pi@<RPI_IP>:~/navigation/models/"
    echo ""
    echo "  Or re-run with:"
    echo "  ./run_pipeline.sh --rpi-ip <RPI_IP>"
fi

# ── DONE ─────────────────────────────────────────────────────
banner "PIPELINE COMPLETE ✓"
echo -e "${GREEN}${BOLD}  All steps finished successfully!${RESET}"
echo ""
echo "  Log file: $LOG_FILE"
echo ""
echo "  Summary:"
echo "    • Dataset:    ./data/yolo_cityscapes/"
echo "    • Baseline:   ./models/yolov8n_cityscapes.pt  (FP32)"
echo "    • Quantized:  ./models/yolov8n_int8.onnx      (INT8, no QAT)"
echo "    • Final:      ./models/yolov8n_qat.onnx       (INT8 + QAT) ← use this on RPi"
echo "    • Report:     ./models/accuracy_report.json"
echo ""
echo "  Expected RPi 4B performance (QAT model @ 320×320):"
echo "    FPS:      5-6 FPS"
echo "    Latency:  150-170ms"
echo "    Commands: Left / Right / Slight Left / Slight Right / Forward / Stop"
echo ""
