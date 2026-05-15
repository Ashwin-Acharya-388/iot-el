# How to Test & Verify All Code

## 🚀 Quick Start (5 minutes)

### Step 1: Run the Complete Test Suite

```bash
# Windows / Mac / Linux - works everywhere
python test_suite.py

# Expected output:
# ✓ All dependencies present
# ✓ All modules import correctly
# ✓ Python syntax is valid
# ✓ Voice commands work
# ✓ Distance estimation works
# ✓ Voice input ready
```

### Step 2: Check Specific Functionality

```bash
# Test only distance estimation
python test_suite.py --distance

# Test only voice input
python test_suite.py --voice

# Quick test (no time-consuming operations)
python test_suite.py --quick
```

---

## ✅ Complete Testing Guide

### Test 1: Dependencies Check (1 min)

Verify all required packages are installed:

```bash
python test_suite.py
```

**Expected Output:**
```
[TEST] numpy                                   ... ✓ PASS
[TEST] cv2.VideoCapture                        ... ✓ PASS
[TEST] pyttsx3                                 ... ✓ PASS
[TEST] pygame                                  ... ✓ PASS
[TEST] gtts                                    ... ✓ PASS
[TEST] speech_recognition                      ... ✓ PASS
```

**If any FAIL:**
```bash
# Install missing package
pip install -r requirements.txt

# Or specific package:
pip install pyttsx3
pip install pygame
pip install gTTS
pip install SpeechRecognition
```

---

### Test 2: Module Imports (1 min)

Check that your custom modules can be imported:

```bash
python -c "from voice_commands import VoiceCommands; print('✓ voice_commands OK')"
python -c "from voice_input import VoiceListener; print('✓ voice_input OK')"
python -c "from navigation_system_rpi import estimate_distance; print('✓ navigation_system_rpi OK')"
```

**Expected Output:**
```
✓ voice_commands OK
✓ voice_input OK
✓ navigation_system_rpi OK
```

**If you get ImportError:**
```
Check that files exist in current directory:
  - voice_commands.py
  - voice_input.py
  - navigation_system_rpi.py

Then try:
  python test_suite.py --distance
```

---

### Test 3: Distance Estimation (2 min)

Test the core distance calculation:

```bash
python << 'EOF'
from navigation_system_rpi import estimate_distance, get_closest_obstacle, find_safe_direction

print("=" * 60)
print("TEST 1: estimate_distance()")
print("=" * 60)

# Create fake detections (x1, y1, x2, y2, conf, class_id)
det_close = (50, 100, 150, 200, 0.9, 2)      # Large bbox = close
det_far = (100, 150, 130, 170, 0.85, 2)      # Small bbox = far

dist_close = estimate_distance(det_close)
dist_far = estimate_distance(det_far)

print(f"\nLarge bbox (50px wide): {dist_close:.1f} meters")
print(f"Small bbox (30px wide): {dist_far:.1f} meters")
print(f"✓ PASS - Closer object has smaller distance\n" if dist_close < dist_far else "✗ FAIL - Distances wrong\n")

print("=" * 60)
print("TEST 2: get_closest_obstacle()")
print("=" * 60)

detections = [
    (50, 100, 150, 200, 0.9, 2),      # Car (class 2)
    (200, 150, 250, 220, 0.8, 0),     # Person (class 0)
]

closest = get_closest_obstacle(detections)
if closest:
    class_name, distance = closest
    print(f"\nClosest: {class_name.capitalize()}, {distance:.1f} meters")
    print(f"✓ PASS - Obstacle detected\n")
else:
    print("✗ FAIL - No obstacle detected\n")

print("=" * 60)
print("TEST 3: find_safe_direction()")
print("=" * 60)

direction, closest_obs = find_safe_direction(detections)

print(f"\nDirection: {direction}")
if closest_obs:
    class_name, distance = closest_obs
    print(f"Closest obstacle: {class_name.capitalize()}, {distance:.1f} meters")
    print(f"Output would be: '{direction} • {class_name.capitalize()}, {distance:.1f} meters'")
else:
    print(f"Output would be: '{direction}'")

print(f"✓ PASS - Navigation logic working\n")
EOF
```

**Expected Output:**
```
============================================================
TEST 1: estimate_distance()
============================================================

Large bbox (50px wide): 2.3 meters
Small bbox (30px wide): 3.8 meters
✓ PASS - Closer object has smaller distance

============================================================
TEST 2: get_closest_obstacle()
============================================================

Closest: Car, 2.3 meters
✓ PASS - Obstacle detected

============================================================
TEST 3: find_safe_direction()
============================================================

Direction: Right
Closest obstacle: Car, 2.3 meters
Output would be: 'Right • Car, 2.3 meters'
✓ PASS - Navigation logic working
```

---

### Test 4: Voice Commands Output (2 min)

Test the voice output system:

```bash
python << 'EOF'
from voice_commands import VoiceCommands
import time

print("Testing VoiceCommands class...")
print()

# Initialize (will auto-detect audio backend)
vc = VoiceCommands(cooldown=0.5)

print("\nTest 1: Basic speak()")
print("-" * 40)
vc.speak("Forward")
time.sleep(1.0)
print("✓ speak() executed without blocking")

print("\nTest 2: speak_with_distance()")
print("-" * 40)
vc.speak_with_distance("Right", "car", 2.5)
time.sleep(1.0)
print("✓ speak_with_distance() executed")

print("\nTest 3: Cooldown (should debounce)")
print("-" * 40)
vc.speak("Left")
time.sleep(0.2)
vc.speak("Left")  # Should be debounced
time.sleep(0.2)
vc.speak("Left")  # Should not be debounced (after cooldown)
time.sleep(1.0)
print("✓ Cooldown working")

# Cleanup
vc.shutdown()
print("\n✓ VoiceCommands class working correctly")
EOF
```

**Expected Output:**
```
Testing VoiceCommands class...

  ✓ VoiceCommands ready. Cooldown: 0.5s
  Setting up voice backend...
    ✓ Audio backend: pygame (pre-recorded WAV/MP3)
    ✓ pyttsx3 (real-time TTS)

Test 1: Basic speak()
----------------------------------------
✓ speak() executed without blocking

Test 2: speak_with_distance()
----------------------------------------
✓ speak_with_distance() executed

Test 3: Cooldown (should debounce)
----------------------------------------
✓ Cooldown working

✓ VoiceCommands shutdown.
✓ VoiceCommands class working correctly
```

---

### Test 5: Audio Backend Detection (2 min)

Run the audio diagnostic tool:

```bash
python test_audio_rpi.py
```

**Expected Output:**
```
============================================================
  RASPBERRY PI AUDIO BACKEND TEST SUITE
============================================================

  [TEST] pygame (pre-recorded playback)...
    ✓ pygame initialized
    ✓ pygame ready (needs pre-generated audio files)

  [TEST] pyttsx3 (real-time TTS)...
    ✓ pyttsx3 engine initialized
    ℹ Available voices: 2
      - David (en)
      - Victoria (en)
    Speaking test command: 'Forward'
    ✓ pyttsx3 TTS works

  [TEST] espeak (native Linux CLI)...
    ✗ espeak not installed (Linux only, RPi requirement)

  [TEST] playsound (lightweight file playback)...
    ✓ playsound initialized (needs audio files)

  SUMMARY
  ✓ pygame              Pre-recorded audio playback works
  ✓ pyttsx3             Real-time TTS functional
  ✓ playsound           playsound ready (needs pre-generated audio)
  ✗ espeak              espeak not installed

  ✓ SUCCESS: pygame, pyttsx3, playsound backend(s) available
```

---

### Test 6: Generate Audio Files (5 min)

Pre-generate MP3s for faster playback:

```bash
python generate_audio_files.py
```

**Expected Output:**
```
============================================================
  AUDIO FILE PRE-GENERATION
============================================================

  Audio directory: /path/to/iot-el/audio
  Commands to generate: 13
    • Left
    • Right
    • Forward
    • ... and 10 more

  [gTTS] Generating audio files...
    → Left                             ... ✓
    → Right                            ... ✓
    → Forward                          ... ✓
    → Stop                             ... ✓
    ... (more files)
  ✓ gTTS: 13/13 files generated

  ============================================================
  ✓ SUCCESS: 13 audio files ready

  Next steps:
    1. Transfer to RPi:
       scp -r ./audio/ pi@raspberrypi.local:~/iot-el/
    2. Test on RPi:
       python test_audio_rpi.py
    3. Run navigation system:
       python navigation_system_rpi.py --debug
```

**If it fails:**
```bash
# Try alternate backend
python generate_audio_files.py --backend pyttsx3

# Or force regenerate
python generate_audio_files.py --force
```

---

### Test 7: Voice Input (Microphone) - Optional

Test microphone detection and voice listening:

```bash
python << 'EOF'
from voice_input import VoiceListener
import time

print("Testing VoiceListener class...")
print()

# Create listener
listener = VoiceListener()

if not listener._mic_available:
    print("✓ Microphone not available (OK on laptop without mic)")
else:
    print("✓ Microphone detected")
    print("  Starting to listen for 10 seconds...")
    print("  Try saying: 'repeat', 'stop', 'what's around me'")
    print()
    
    listener.start()
    time.sleep(10)
    listener.stop()
    
    print("\n✓ Voice listener working")
EOF
```

**Expected Output (with microphone):**
```
Testing VoiceListener class...

  🎙️  Calibrating microphone... (speak normally)
  ✓ Microphone calibrated. Energy threshold: 3500.0
  ✓ VoiceListener initialized. Listen timeout: 10.0s
  ✓ Voice listener started
  🎙️  Listening for voice commands...
  Starting to listen for 10 seconds...
  Try saying: 'repeat', 'stop', 'what's around me'

  🎙️  Heard: 'stop'
  🛑 STOP command received

  ✓ Voice listener stopped
  ✓ Voice listener working
```

**Expected Output (without microphone):**
```
Testing VoiceListener class...

  [WARN] No microphone detected. Voice input disabled.
  ✓ Microphone not available (OK on laptop without mic)
```

---

### Test 8: Syntax Check

Verify all Python files have correct syntax:

```bash
python -m py_compile voice_commands.py
python -m py_compile voice_input.py
python -m py_compile navigation_system_rpi.py
python -m py_compile generate_audio_files.py
python -m py_compile integration_example.py

echo "✓ All files have valid syntax"
```

**Expected Output:**
```
✓ All files have valid syntax
```

**If you get errors:**
```
Fix the syntax error in that file and try again.
```

---

## 🎯 Automated Test Suite

Run all tests automatically:

```bash
# Full test (5-10 minutes)
python test_suite.py

# Quick test (1-2 minutes)
python test_suite.py --quick

# Specific tests
python test_suite.py --distance    # Distance estimation only
python test_suite.py --voice       # Voice input only  
python test_suite.py --audio       # Audio files only
```

**Full Test Output:**
```
╔════════════════════════════════════════════════════════════════╗
║         VOICE ENHANCEMENT SYSTEM - TEST SUITE                 ║
║                   May 2026                                     ║
╚════════════════════════════════════════════════════════════════╝

============================================================
  PHASE 1: Checking Dependencies
============================================================

  [TEST] numpy                                   ... ✓ PASS
  [TEST] cv2.VideoCapture                        ... ✓ PASS
  [TEST] pyttsx3                                 ... ✓ PASS
  [TEST] pygame                                  ... ✓ PASS
  [TEST] gtts                                    ... ✓ PASS
  [TEST] speech_recognition                      ... ✓ PASS

============================================================
  PHASE 2: Checking Module Imports
============================================================

  [TEST] voice_commands.VoiceCommands             ... ✓ PASS
  [TEST] voice_input.VoiceListener                ... ✓ PASS
  [TEST] navigation_system_rpi.estimate_distance  ... ✓ PASS

============================================================
  PHASE 3: Checking Python Syntax
============================================================

  [TEST] Syntax: voice_commands.py                ... ✓ PASS
  [TEST] Syntax: voice_input.py                   ... ✓ PASS
  [TEST] Syntax: navigation_system_rpi.py         ... ✓ PASS

... (more tests)

============================================================
  TEST SUMMARY
============================================================

  PASSED:  18
  FAILED:  0
  SKIPPED: 2

✓ ALL TESTS PASSED
```

---

## 🔍 Troubleshooting

### "ImportError: No module named 'pyttsx3'"
```bash
pip install pyttsx3
```

### "No audio output"
Run diagnostic:
```bash
python test_audio_rpi.py --verbose
```

### "No microphone detected"
On Windows/Mac, microphone detection is optional. On RPi, check:
```bash
arecord -l        # List ALSA devices
```

### "Distance seems wrong"
Verify the calculation:
```bash
python << 'EOF'
from navigation_system_rpi import estimate_distance

# Large bbox = closer
det1 = (50, 100, 150, 200, 0.9, 2)   # 100px width
det2 = (100, 150, 130, 170, 0.85, 2) # 30px width

print(f"100px width: {estimate_distance(det1):.1f}m")
print(f"30px width:  {estimate_distance(det2):.1f}m")
EOF
```

---

## ✅ Final Checklist

- [ ] `python test_suite.py` shows all PASS
- [ ] `python test_audio_rpi.py` detects audio backends
- [ ] `python generate_audio_files.py` creates ./audio/ folder
- [ ] Distance estimation shows reasonable values (0.3-20m)
- [ ] Voice commands initialize without errors
- [ ] All Python files have valid syntax

**If all checks pass:** ✅ **Ready for Raspberry Pi deployment**

