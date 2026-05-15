#!/bin/bash
# run_tests.sh
# Comprehensive testing suite for voice enhancement system
# 
# Usage:
#   chmod +x run_tests.sh
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --quick      # Skip time-consuming tests
#   ./run_tests.sh --audio      # Test audio only
#   ./run_tests.sh --voice      # Test voice input only

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TEST_MODE="${1:-all}"
PASSED=0
FAILED=0
SKIPPED=0

# ──────────────────────────────────────────────
# TEST FRAMEWORK
# ──────────────────────────────────────────────

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}\n"
}

test_start() {
    echo -ne "${YELLOW}[TEST]${NC} $1 ... "
}

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAILED++))
}

test_skip() {
    echo -e "${YELLOW}⊘ SKIP${NC}: $1"
    ((SKIPPED++))
}

python_test() {
    local test_name="$1"
    local python_code="$2"
    
    test_start "$test_name"
    if python3 -c "$python_code" 2>/dev/null; then
        test_pass
    else
        test_fail "Python execution error"
    fi
}

# ──────────────────────────────────────────────
# PHASE 1: DEPENDENCY CHECKS
# ──────────────────────────────────────────────

test_dependencies() {
    print_header "PHASE 1: Checking Dependencies"
    
    # Core packages
    python_test "numpy installed" "import numpy; print('✓')"
    python_test "opencv available" "import cv2; print('✓')"
    
    # Voice packages
    python_test "pyttsx3 installed" "import pyttsx3; print('✓')"
    python_test "pygame installed" "import pygame; print('✓')"
    python_test "gTTS installed" "import gtts; print('✓')"
    python_test "SpeechRecognition installed" "import speech_recognition; print('✓')"
    
    # ONNX (if available)
    test_start "onnxruntime installed"
    if python3 -c "import onnxruntime" 2>/dev/null; then
        test_pass
    else
        test_skip "ONNX Runtime (not critical for testing)"
    fi
}

# ──────────────────────────────────────────────
# PHASE 2: IMPORT CHECKS
# ──────────────────────────────────────────────

test_imports() {
    print_header "PHASE 2: Checking Module Imports"
    
    python_test "voice_commands imports" "from voice_commands import VoiceCommands; print('✓')"
    python_test "voice_input imports" "from voice_input import VoiceListener, VoiceCommandHandler; print('✓')"
    python_test "navigation imports" "from navigation_system_rpi import estimate_distance, get_closest_obstacle, find_safe_direction; print('✓')"
}

# ──────────────────────────────────────────────
# PHASE 3: AUDIO BACKEND TEST
# ──────────────────────────────────────────────

test_audio_backends() {
    print_header "PHASE 3: Testing Audio Backends"
    
    if [[ "$TEST_MODE" == "quick" ]]; then
        test_skip "Audio backend (--quick mode)"
        return
    fi
    
    echo "Running: python test_audio_rpi.py"
    python3 test_audio_rpi.py
}

# ──────────────────────────────────────────────
# PHASE 4: VOICE COMMANDS FUNCTIONALITY
# ──────────────────────────────────────────────

test_voice_commands() {
    print_header "PHASE 4: Testing Voice Commands Class"
    
    test_start "VoiceCommands initialization"
    python3 << 'EOF'
from voice_commands import VoiceCommands
import time

vc = VoiceCommands(cooldown=0.5)
print("✓")
time.sleep(0.5)
vc.shutdown()
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "Could not initialize VoiceCommands"
    fi
    
    test_start "VoiceCommands.speak() non-blocking"
    python3 << 'EOF'
from voice_commands import VoiceCommands
import time

vc = VoiceCommands(cooldown=0.2)
start = time.time()
vc.speak("Forward")
elapsed = time.time() - start

# speak() should return immediately (< 100ms)
if elapsed < 0.1:
    print(f"✓ (returned in {elapsed*1000:.1f}ms)")
else:
    print(f"✗ took {elapsed*1000:.1f}ms (should be <100ms)")
    exit(1)

time.sleep(1.0)
vc.shutdown()
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "speak() not non-blocking"
    fi
    
    test_start "VoiceCommands.speak_with_distance()"
    python3 << 'EOF'
from voice_commands import VoiceCommands
import time

vc = VoiceCommands(cooldown=0.2)
vc.speak_with_distance("Right", "car", 2.5)
print("✓")
time.sleep(1.0)
vc.shutdown()
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "speak_with_distance() error"
    fi
}

# ──────────────────────────────────────────────
# PHASE 5: DISTANCE ESTIMATION
# ──────────────────────────────────────────────

test_distance_estimation() {
    print_header "PHASE 5: Testing Distance Estimation"
    
    test_start "estimate_distance() function"
    python3 << 'EOF'
from navigation_system_rpi import estimate_distance

# Detection tuple: (x1, y1, x2, y2, conf, class_id)
# Car class is typically class_id=2

# Large bounding box (close object)
det_close = (50, 100, 150, 200, 0.9, 2)
dist_close = estimate_distance(det_close)

# Small bounding box (far object)
det_far = (100, 150, 130, 170, 0.85, 2)
dist_far = estimate_distance(det_far)

# Verify distances are reasonable
if 0.3 <= dist_close <= 10.0 and 0.3 <= dist_far <= 20.0:
    if dist_close > dist_far:
        print(f"✓ (close={dist_close:.1f}m > far={dist_far:.1f}m)")
    else:
        print(f"✗ distances reversed (close={dist_close:.1f}m < far={dist_far:.1f}m)")
        exit(1)
else:
    print(f"✗ distances out of range (close={dist_close:.1f}m, far={dist_far:.1f}m)")
    exit(1)
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "Distance estimation error"
    fi
    
    test_start "get_closest_obstacle() function"
    python3 << 'EOF'
from navigation_system_rpi import get_closest_obstacle

# Multiple detections
detections = [
    (50, 100, 150, 200, 0.9, 2),    # Car
    (200, 150, 250, 220, 0.8, 0),   # Person (far)
]

closest = get_closest_obstacle(detections)

if closest is None:
    print("✗ returned None")
    exit(1)

class_name, distance = closest
if isinstance(class_name, str) and isinstance(distance, float):
    print(f"✓ ({class_name}, {distance:.1f}m)")
else:
    print(f"✗ unexpected types: {type(class_name)}, {type(distance)}")
    exit(1)
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "get_closest_obstacle() error"
    fi
    
    test_start "find_safe_direction() with distance"
    python3 << 'EOF'
from navigation_system_rpi import find_safe_direction

# Detections in center and left zones
detections = [
    (100, 150, 150, 200, 0.9, 2),   # Center
    (50, 100, 100, 150, 0.8, 0),    # Left
]

direction, closest_obs = find_safe_direction(detections)

if not isinstance(direction, str):
    print(f"✗ direction is not string: {type(direction)}")
    exit(1)

if direction not in ["Left", "Right", "Forward", "Slight Left", "Slight Right", "Stop"]:
    print(f"✗ invalid direction: {direction}")
    exit(1)

if closest_obs is not None:
    class_name, distance = closest_obs
    if not isinstance(class_name, str) or not isinstance(distance, float):
        print(f"✗ unexpected obstacle format")
        exit(1)
    print(f"✓ (direction={direction}, obstacle={class_name},{distance:.1f}m)")
else:
    print(f"✓ (direction={direction}, no obstacles)")
EOF
    if [ $? -eq 0 ]; then
        test_pass
    else
        test_fail "find_safe_direction() error"
    fi
}

# ──────────────────────────────────────────────
# PHASE 6: VOICE INPUT (Microphone)
# ──────────────────────────────────────────────

test_voice_input() {
    print_header "PHASE 6: Testing Voice Input (Microphone)"
    
    if [[ "$TEST_MODE" == "quick" ]]; then
        test_skip "Voice input (--quick mode)"
        return
    fi
    
    test_start "VoiceListener initialization"
    python3 << 'EOF'
from voice_input import VoiceListener

# Check if microphone is available
listener = VoiceListener()

if not listener._mic_available:
    print("⊘ (no microphone detected - this is OK on laptop without mic)")
    exit(2)  # Exit code 2 = expected skip
else:
    print("✓ (microphone available)")
EOF
    
    if [ $? -eq 0 ]; then
        test_pass
    elif [ $? -eq 2 ]; then
        test_skip "Microphone not available"
    else
        test_fail "VoiceListener error"
    fi
}

# ──────────────────────────────────────────────
# PHASE 7: AUDIO FILE GENERATION
# ──────────────────────────────────────────────

test_audio_generation() {
    print_header "PHASE 7: Testing Audio File Generation"
    
    if [[ "$TEST_MODE" == "quick" ]]; then
        test_skip "Audio generation (--quick mode)"
        return
    fi
    
    test_start "Checking audio directory"
    if [ -d "./audio" ]; then
        audio_count=$(ls -1 ./audio/*.mp3 2>/dev/null | wc -l)
        if [ $audio_count -gt 0 ]; then
            test_pass "($audio_count audio files found)"
        else
            echo -e "${YELLOW}[INFO]${NC} No pre-generated audio files. Run: python generate_audio_files.py"
            test_skip "Audio files not generated"
        fi
    else
        echo -e "${YELLOW}[INFO]${NC} audio/ directory not found. Run: python generate_audio_files.py"
        test_skip "Audio directory missing"
    fi
}

# ──────────────────────────────────────────────
# PHASE 8: CODE SYNTAX CHECK
# ──────────────────────────────────────────────

test_syntax() {
    print_header "PHASE 8: Checking Python Syntax"
    
    for file in voice_commands.py voice_input.py navigation_system_rpi.py generate_audio_files.py integration_example.py; do
        test_start "Syntax: $file"
        if python3 -m py_compile "$file" 2>/dev/null; then
            test_pass
        else
            test_fail "Syntax error in $file"
        fi
    done
}

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

main() {
    echo -e "${GREEN}"
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║         VOICE ENHANCEMENT SYSTEM - TEST SUITE                 ║
║                   May 2026                                     ║
╚════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Show test mode
    if [[ "$TEST_MODE" == "quick" ]]; then
        echo "Mode: QUICK (skipping time-consuming tests)"
    elif [[ "$TEST_MODE" == "audio" ]]; then
        echo "Mode: AUDIO ONLY"
    elif [[ "$TEST_MODE" == "voice" ]]; then
        echo "Mode: VOICE INPUT ONLY"
    else
        echo "Mode: FULL TEST SUITE"
    fi
    echo ""
    
    # Run tests based on mode
    if [[ "$TEST_MODE" == "audio" ]]; then
        test_audio_backends
    elif [[ "$TEST_MODE" == "voice" ]]; then
        test_voice_input
    else
        test_dependencies
        test_imports
        test_syntax
        test_voice_commands
        test_distance_estimation
        test_audio_backends
        test_voice_input
        test_audio_generation
    fi
    
    # Summary
    print_header "TEST SUMMARY"
    echo -e "  ${GREEN}PASSED:${NC}  $PASSED"
    echo -e "  ${RED}FAILED:${NC}  $FAILED"
    echo -e "  ${YELLOW}SKIPPED:${NC} $SKIPPED"
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED${NC}\n"
        return 0
    else
        echo -e "${RED}✗ SOME TESTS FAILED${NC}\n"
        return 1
    fi
}

main
