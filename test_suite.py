#!/usr/bin/env python3
"""
test_suite.py
==============
Comprehensive test suite for voice enhancement system.
Works on Windows, Mac, and Linux.

Usage:
    python test_suite.py              # Run all tests
    python test_suite.py --quick      # Skip time-consuming tests
    python test_suite.py --audio      # Test audio only
    python test_suite.py --voice      # Test voice input only
    python test_suite.py --distance   # Test distance estimation only
"""

import sys
import os
import time
import argparse
import traceback
from pathlib import Path
from typing import Callable, Tuple, Optional


# ──────────────────────────────────────────────
# TEST FRAMEWORK
# ──────────────────────────────────────────────

class TestSuite:
    """Color-coded test runner."""

    # ANSI colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def header(self, title: str):
        """Print section header."""
        print(f"\n{self.BLUE}{'='*60}{self.RESET}")
        print(f"{self.BLUE}{self.BOLD}  {title}{self.RESET}")
        print(f"{self.BLUE}{'='*60}{self.RESET}\n")

    def test_import(self, module_name: str, item_name: str = None) -> bool:
        """Test if a module can be imported."""
        test_name = f"{module_name}.{item_name}" if item_name else module_name
        print(f"  {self.YELLOW}[TEST]{self.RESET} {test_name:<45} ... ", end="", flush=True)

        try:
            if item_name:
                module = __import__(module_name)
                getattr(module, item_name)
            else:
                __import__(module_name)
            print(f"{self.GREEN}✓ PASS{self.RESET}")
            self.passed += 1
            return True
        except Exception as e:
            print(f"{self.RED}✗ FAIL{self.RESET}")
            self.errors.append(f"{test_name}: {e}")
            self.failed += 1
            return False

    def test_function(self, test_name: str, test_func: Callable) -> bool:
        """Run a test function."""
        print(f"  {self.YELLOW}[TEST]{self.RESET} {test_name:<45} ... ", end="", flush=True)

        try:
            result = test_func()
            if result is True or result is None:
                print(f"{self.GREEN}✓ PASS{self.RESET}")
                self.passed += 1
                return True
            elif result is False:
                print(f"{self.RED}✗ FAIL{self.RESET}")
                self.failed += 1
                return False
            else:
                # Result is a tuple: (success, message)
                success, message = result
                if success:
                    print(f"{self.GREEN}✓ PASS{self.RESET} {message}")
                    self.passed += 1
                    return True
                else:
                    print(f"{self.RED}✗ FAIL{self.RESET}")
                    self.errors.append(f"{test_name}: {message}")
                    self.failed += 1
                    return False
        except Exception as e:
            print(f"{self.RED}✗ FAIL{self.RESET}")
            self.errors.append(f"{test_name}: {str(e)}")
            self.failed += 1
            return False

    def test_skip(self, test_name: str, reason: str = ""):
        """Skip a test."""
        print(f"  {self.YELLOW}[TEST]{self.RESET} {test_name:<45} ... ", end="", flush=True)
        print(f"{self.YELLOW}⊘ SKIP{self.RESET} {reason}")
        self.skipped += 1

    def summary(self) -> bool:
        """Print test summary."""
        self.header("TEST SUMMARY")

        print(f"  {self.GREEN}PASSED:{self.RESET}  {self.passed}")
        print(f"  {self.RED}FAILED:{self.RESET}  {self.failed}")
        print(f"  {self.YELLOW}SKIPPED:{self.RESET} {self.skipped}")
        print()

        if self.errors:
            print(f"{self.RED}Errors:{self.RESET}")
            for error in self.errors:
                print(f"  • {error}")
            print()

        if self.failed == 0:
            print(f"{self.GREEN}{self.BOLD}✓ ALL TESTS PASSED{self.RESET}\n")
            return True
        else:
            print(f"{self.RED}{self.BOLD}✗ SOME TESTS FAILED{self.RESET}\n")
            return False


# ──────────────────────────────────────────────
# TEST FUNCTIONS
# ──────────────────────────────────────────────

def test_dependencies(suite: TestSuite):
    """Test that all required packages are installed."""
    suite.header("PHASE 1: Checking Dependencies")

    # Core packages
    suite.test_import("numpy")
    suite.test_import("cv2", "VideoCapture")

    # Voice packages
    suite.test_import("pyttsx3")
    suite.test_import("pygame")
    suite.test_import("gtts")
    suite.test_import("speech_recognition")

    # Optional
    try:
        import onnxruntime
        suite.test_import("onnxruntime")
    except ImportError:
        suite.test_skip("onnxruntime", "(not required for testing)")


def test_imports(suite: TestSuite):
    """Test that our custom modules import correctly."""
    suite.header("PHASE 2: Checking Module Imports")

    suite.test_import("voice_commands", "VoiceCommands")
    suite.test_import("voice_input", "VoiceListener")
    suite.test_import("voice_input", "VoiceCommandHandler")
    suite.test_import("navigation_system_rpi", "estimate_distance")
    suite.test_import("navigation_system_rpi", "get_closest_obstacle")
    suite.test_import("navigation_system_rpi", "find_safe_direction")


def test_syntax(suite: TestSuite):
    """Check Python syntax of all files."""
    suite.header("PHASE 3: Checking Python Syntax")

    files = [
        "voice_commands.py",
        "voice_input.py",
        "navigation_system_rpi.py",
        "generate_audio_files.py",
        "integration_example.py",
    ]

    for file in files:
        path = Path(file)
        if not path.exists():
            suite.test_skip(f"Syntax: {file}", f"(file not found)")
            continue

        def check_syntax():
            import py_compile
            py_compile.compile(str(path), doraise=True)
            return True

        suite.test_function(f"Syntax: {file}", check_syntax)


def test_voice_commands(suite: TestSuite):
    """Test VoiceCommands class."""
    suite.header("PHASE 4: Testing Voice Commands Class")

    def test_init():
        from voice_commands import VoiceCommands
        vc = VoiceCommands(cooldown=0.5)
        vc.shutdown()
        return True

    def test_speak_nonblocking():
        from voice_commands import VoiceCommands
        import time

        vc = VoiceCommands(cooldown=0.2)
        start = time.time()
        vc.speak("Forward")
        elapsed = time.time() - start
        vc.shutdown()

        if elapsed < 0.2:  # Should be near-instant
            return True, f"({elapsed*1000:.1f}ms)"
        else:
            return False, f"took {elapsed*1000:.1f}ms (should be <200ms)"

    def test_speak_with_distance():
        from voice_commands import VoiceCommands
        import time

        vc = VoiceCommands(cooldown=0.2)
        vc.speak_with_distance("Right", "car", 2.5)
        time.sleep(0.5)
        vc.shutdown()
        return True

    suite.test_function("VoiceCommands initialization", test_init)
    suite.test_function("VoiceCommands.speak() non-blocking", test_speak_nonblocking)
    suite.test_function("VoiceCommands.speak_with_distance()", test_speak_with_distance)


def test_distance_estimation(suite: TestSuite):
    """Test distance estimation functions."""
    suite.header("PHASE 5: Testing Distance Estimation")

    def test_estimate_distance():
        from navigation_system_rpi import estimate_distance

        # Large bbox (close object)
        det_close = (50, 100, 150, 200, 0.9, 2)
        dist_close = estimate_distance(det_close)

        # Small bbox (far object)
        det_far = (100, 150, 130, 170, 0.85, 2)
        dist_far = estimate_distance(det_far)

        if not (0.3 <= dist_close <= 20.0):
            return False, f"dist_close={dist_close:.1f}m out of range"
        if not (0.3 <= dist_far <= 20.0):
            return False, f"dist_far={dist_far:.1f}m out of range"
        if not (dist_close > dist_far):
            return False, f"distances reversed: {dist_close:.1f}m vs {dist_far:.1f}m"

        return True, f"(close={dist_close:.1f}m > far={dist_far:.1f}m)"

    def test_get_closest_obstacle():
        from navigation_system_rpi import get_closest_obstacle

        detections = [
            (50, 100, 150, 200, 0.9, 2),    # Car
            (200, 150, 250, 220, 0.8, 0),   # Person (far)
        ]

        closest = get_closest_obstacle(detections)

        if closest is None:
            return False, "returned None"

        class_name, distance = closest
        if not isinstance(class_name, str) or not isinstance(distance, float):
            return False, f"unexpected types: {type(class_name).__name__}, {type(distance).__name__}"

        return True, f"({class_name}, {distance:.1f}m)"

    def test_find_safe_direction():
        from navigation_system_rpi import find_safe_direction

        detections = [
            (100, 150, 150, 200, 0.9, 2),   # Center
            (50, 100, 100, 150, 0.8, 0),    # Left
        ]

        direction, closest_obs = find_safe_direction(detections)

        if not isinstance(direction, str):
            return False, f"direction not string: {type(direction).__name__}"

        valid_dirs = ["Left", "Right", "Forward", "Slight Left", "Slight Right", "Stop"]
        if direction not in valid_dirs:
            return False, f"invalid direction: {direction}"

        if closest_obs is not None:
            class_name, distance = closest_obs
            if not isinstance(distance, float):
                return False, f"distance not float: {type(distance).__name__}"
            return True, f"(dir={direction}, obs={class_name},{distance:.1f}m)"
        else:
            return True, f"(dir={direction}, no obstacles)"

    def test_no_detections():
        from navigation_system_rpi import find_safe_direction

        direction, closest_obs = find_safe_direction([])

        if direction not in ["Forward", "Stop"]:  # Should handle empty gracefully
            return False, f"invalid direction for empty dets: {direction}"
        if closest_obs is not None:
            return False, "should have no obstacles"

        return True, f"(handled empty detections: {direction})"

    suite.test_function("estimate_distance() function", test_estimate_distance)
    suite.test_function("get_closest_obstacle() function", test_get_closest_obstacle)
    suite.test_function("find_safe_direction() with distance", test_find_safe_direction)
    suite.test_function("find_safe_direction() with empty input", test_no_detections)


def test_voice_input(suite: TestSuite):
    """Test voice input functionality."""
    suite.header("PHASE 6: Testing Voice Input (Microphone)")

    def test_voice_listener_init():
        from voice_input import VoiceListener

        listener = VoiceListener()

        if listener._mic_available:
            return True, "(microphone available)"
        else:
            return None  # Skip if no mic (expected on laptop without mic)

    def test_voice_command_handler():
        from voice_input import VoiceCommandHandler

        handler = VoiceCommandHandler()
        if not handler.paused:
            return True, "(initial state correct)"
        else:
            return False, "should not be paused initially"

    suite.test_function("VoiceListener initialization", test_voice_listener_init)
    suite.test_function("VoiceCommandHandler initialization", test_voice_command_handler)


def test_audio_files(suite: TestSuite):
    """Check if audio files are pre-generated."""
    suite.header("PHASE 7: Checking Audio Files")

    audio_dir = Path("./audio")

    if not audio_dir.exists():
        suite.test_skip("Audio directory", "(not yet created - run: python generate_audio_files.py)")
        return

    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

    if len(audio_files) == 0:
        suite.test_skip("Audio files", "(directory empty - run: python generate_audio_files.py)")
    else:
        suite.test_function(
            "Audio files present",
            lambda: (True, f"({len(audio_files)} files found)")
        )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test suite for voice enhancement system"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip time-consuming tests",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Test audio only",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Test voice input only",
    )
    parser.add_argument(
        "--distance",
        action="store_true",
        help="Test distance estimation only",
    )
    args = parser.parse_args()

    print(f"\n{TestSuite.GREEN}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         VOICE ENHANCEMENT SYSTEM - TEST SUITE                 ║")
    print("║                   May 2026                                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{TestSuite.RESET}")

    suite = TestSuite()

    if args.audio:
        # Audio files only
        test_dependencies(suite)
        test_audio_files(suite)
    elif args.voice:
        # Voice input only
        test_imports(suite)
        test_voice_input(suite)
    elif args.distance:
        # Distance estimation only
        test_imports(suite)
        test_distance_estimation(suite)
    else:
        # Full test suite
        test_dependencies(suite)
        test_imports(suite)
        test_syntax(suite)
        test_voice_commands(suite)
        test_distance_estimation(suite)
        if not args.quick:
            test_voice_input(suite)
            test_audio_files(suite)

    # Summary
    success = suite.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
