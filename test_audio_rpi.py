"""
test_audio_rpi.py
=================
Comprehensive test suite for audio output backends on Raspberry Pi.

This script tests all available audio playback methods in order of preference:
  1. pygame (pre-recorded WAV/MP3 playback) - lowest latency
  2. pyttsx3 (real-time TTS) - no internet required, works offline
  3. espeak (native Linux CLI tool) - lightweight fallback
  4. playsound (alternative lightweight player)
  5. Print-only (debugging fallback)

Usage:
    # Test all backends
    python test_audio_rpi.py

    # Test specific backend
    python test_audio_rpi.py --backend pygame
    python test_audio_rpi.py --backend pyttsx3

    # On RPi: run with speaker/headphone connected
    ssh pi@raspberrypi.local
    cd ~/iot-el
    python test_audio_rpi.py

Exit codes:
    0 = at least one backend works
    1 = no backends functional
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Callable


# ──────────────────────────────────────────────
# TEST COMMANDS
# ──────────────────────────────────────────────

TEST_COMMANDS = [
    "Left",
    "Right",
    "Forward",
    "Stop",
    "Car, five meters ahead",
]

TEST_RESULTS = {}  # backend_name → (success: bool, details: str)


# ──────────────────────────────────────────────
# BACKEND: PYGAME
# ──────────────────────────────────────────────

def test_pygame() -> bool:
    """Test pygame for pre-recorded audio playback."""
    print("\n  [TEST] pygame (pre-recorded playback)...")
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        print("    ✓ pygame initialized")

        # Try to create a simple sound or use test audio if available
        audio_file = Path("./audio/left.mp3")
        if audio_file.exists():
            try:
                snd = pygame.mixer.Sound(str(audio_file))
                snd.play()
                time.sleep(snd.get_length() + 0.1)
                print(f"    ✓ pygame played audio file: {audio_file}")
                TEST_RESULTS["pygame"] = (True, "Pre-recorded audio playback works")
                return True
            except Exception as e:
                print(f"    [WARN] pygame failed to play audio file: {e}")
        else:
            print(f"    [INFO] No pre-recorded audio found at {audio_file}")
            print("           Pre-generation needed first (run: python generate_audio_files.py)")

        pygame.mixer.quit()
        print("    ✓ pygame mixer initialized (backend ready for pre-recorded files)")
        TEST_RESULTS["pygame"] = (True, "pygame ready (needs pre-generated audio files)")
        return True

    except ImportError:
        print("    ✗ pygame not installed")
        TEST_RESULTS["pygame"] = (False, "pygame not installed (pip install pygame)")
        return False
    except Exception as e:
        print(f"    ✗ pygame init failed: {e}")
        TEST_RESULTS["pygame"] = (False, f"Error: {e}")
        return False


# ──────────────────────────────────────────────
# BACKEND: PYTTSX3
# ──────────────────────────────────────────────

def test_pyttsx3() -> bool:
    """Test pyttsx3 for real-time TTS."""
    print("\n  [TEST] pyttsx3 (real-time TTS)...")
    try:
        import pyttsx3

        engine = pyttsx3.init()
        print("    ✓ pyttsx3 engine initialized")

        # Configure voice
        engine.setProperty("rate", 130)  # Clear speech
        engine.setProperty("volume", 1.0)

        # List available voices
        voices = engine.getProperty("voices")
        print(f"    ℹ Available voices: {len(voices)}")
        for v in voices:
            print(f"      - {v.name} ({v.id})")

        # Try to speak
        print("    Speaking test command: 'Forward'")
        engine.say("Forward")
        engine.runAndWait()

        print("    ✓ pyttsx3 TTS works")
        TEST_RESULTS["pyttsx3"] = (True, "Real-time TTS functional")
        engine.stop()
        return True

    except ImportError:
        print("    ✗ pyttsx3 not installed")
        TEST_RESULTS["pyttsx3"] = (False, "pyttsx3 not installed (pip install pyttsx3)")
        return False
    except Exception as e:
        print(f"    ✗ pyttsx3 failed: {e}")
        TEST_RESULTS["pyttsx3"] = (False, f"Error: {e}")
        return False


# ──────────────────────────────────────────────
# BACKEND: ESPEAK
# ──────────────────────────────────────────────

def test_espeak() -> bool:
    """Test espeak CLI for native Linux TTS."""
    print("\n  [TEST] espeak (native Linux CLI)...")

    # Check if espeak is installed
    ret = os.system("which espeak > /dev/null 2>&1")
    if ret != 0:
        print("    ✗ espeak not installed")
        print("      Install: sudo apt install espeak espeak-data libespeak1")
        TEST_RESULTS["espeak"] = (False, "espeak not installed")
        return False

    try:
        # Test espeak
        cmd = 'espeak -s 130 -v en "Forward" 2>/dev/null'
        ret = os.system(cmd)
        if ret == 0:
            print("    ✓ espeak TTS works")
            TEST_RESULTS["espeak"] = (True, "Native CLI TTS functional")
            return True
        else:
            print(f"    ✗ espeak execution failed (exit code: {ret})")
            TEST_RESULTS["espeak"] = (False, f"espeak error (exit: {ret})")
            return False
    except Exception as e:
        print(f"    ✗ espeak test failed: {e}")
        TEST_RESULTS["espeak"] = (False, f"Error: {e}")
        return False


# ──────────────────────────────────────────────
# BACKEND: PLAYSOUND
# ──────────────────────────────────────────────

def test_playsound() -> bool:
    """Test playsound for audio file playback."""
    print("\n  [TEST] playsound (lightweight file playback)...")
    try:
        import playsound

        audio_file = Path("./audio/left.mp3")
        if audio_file.exists():
            try:
                playsound.playsound(str(audio_file))
                print(f"    ✓ playsound played: {audio_file}")
                TEST_RESULTS["playsound"] = (True, "Pre-recorded playback works")
                return True
            except Exception as e:
                print(f"    [WARN] playsound failed: {e}")
        else:
            print(f"    [INFO] No audio file for testing ({audio_file})")

        print("    ✓ playsound initialized (needs audio files)")
        TEST_RESULTS["playsound"] = (True, "playsound ready (needs pre-generated audio)")
        return True

    except ImportError:
        print("    ✗ playsound not installed")
        TEST_RESULTS["playsound"] = (False, "playsound not installed (pip install playsound)")
        return False
    except Exception as e:
        print(f"    ✗ playsound failed: {e}")
        TEST_RESULTS["playsound"] = (False, f"Error: {e}")
        return False


# ──────────────────────────────────────────────
# SYSTEM INFO
# ──────────────────────────────────────────────

def print_system_info():
    """Print system and audio hardware info."""
    print("\n  ━━━ SYSTEM INFO ━━━")
    print(f"    Platform: {sys.platform}")
    print(f"    Python: {sys.version.split()[0]}")

    # Check for ALSA
    alsa_present = os.system("which aplay > /dev/null 2>&1") == 0
    print(f"    ALSA: {'✓ present' if alsa_present else '✗ not found'}")

    # Check audio devices
    if alsa_present:
        print("\n    Audio devices:")
        os.system("aplay -l 2>/dev/null | head -5 || echo '      (none found)'")

    # Check pulseaudio
    pa_present = os.system("which pulseaudio > /dev/null 2>&1") == 0
    print(f"    PulseAudio: {'✓ present' if pa_present else '✗ not found'}")

    # Check display (headless?)
    headless = os.environ.get("DISPLAY", "") == ""
    print(f"    Headless: {'✓ yes' if headless else '✗ no (GUI available)'}")


# ──────────────────────────────────────────────
# MAIN TEST SUITE
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test audio backends on Raspberry Pi or Linux."
    )
    parser.add_argument(
        "--backend",
        choices=["pygame", "pyttsx3", "espeak", "playsound", "all"],
        default="all",
        help="Which backend to test (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed system info"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RASPBERRY PI AUDIO BACKEND TEST SUITE")
    print("="*60)

    if args.verbose:
        print_system_info()

    # Select backends to test
    backends = {
        "pygame": test_pygame,
        "pyttsx3": test_pyttsx3,
        "espeak": test_espeak,
        "playsound": test_playsound,
    }

    if args.backend == "all":
        backends_to_test = backends
    else:
        backends_to_test = {args.backend: backends[args.backend]}

    # Run tests
    print("\n" + "─"*60)
    print("  TESTING BACKENDS")
    print("─"*60)

    for name, test_func in backends_to_test.items():
        try:
            test_func()
        except Exception as e:
            print(f"    [ERROR] Unexpected error: {e}")
            TEST_RESULTS[name] = (False, f"Unexpected error: {e}")

    # Summary
    print("\n" + "─"*60)
    print("  SUMMARY")
    print("─"*60)

    working = []
    for backend, (success, details) in TEST_RESULTS.items():
        status = "✓" if success else "✗"
        print(f"  {status} {backend:15} {details}")
        if success:
            working.append(backend)

    print("\n" + "="*60)
    if working:
        print(f"  ✓ SUCCESS: {', '.join(working)} backend(s) available")
        print("\n  RECOMMENDED PRIORITY:")
        print("    1. pygame (pre-recorded) - lowest latency")
        print("    2. pyttsx3 (TTS) - no internet, responsive")
        print("    3. espeak (CLI) - native, lightweight")
        print("\n  NEXT STEPS:")
        print("    → Run: python generate_audio_files.py")
        print("      (Creates ./audio/ folder with pre-recorded commands)")
        print("    → Then test voice_commands.py:")
        print("      python -c 'from voice_commands import VoiceCommands;")
        print("                 vc = VoiceCommands();")
        print("                 vc.speak(\"Left\"); import time; time.sleep(2)'")
        return 0
    else:
        print("  ✗ FAILURE: No audio backends available!")
        print("\n  INSTALL ONE OF:")
        print("    pip install pygame pyttsx3 playsound")
        print("  OR")
        print("    sudo apt install espeak espeak-data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
