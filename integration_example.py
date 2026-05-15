"""
integration_example.py
======================
Example: How to integrate voice input and distance-aware output
with the navigation system.

This shows how to:
1. Listen for voice commands while running navigation
2. Handle commands (repeat, stop, resume)
3. Output distance-aware navigation messages
4. Manage state transitions
"""

import time
import threading
from pathlib import Path
from typing import Optional

# Import navigation components
from navigation_system_rpi import (
    NavigationSystem,
    find_safe_direction,
    estimate_distance,
)

# Import voice components
from voice_commands import VoiceCommands
from voice_input import VoiceListener, VoiceCommandHandler


# ──────────────────────────────────────────────
# ENHANCED NAVIGATION SYSTEM WITH VOICE
# ──────────────────────────────────────────────

class VoiceEnabledNavigation:
    """
    Navigation system integrated with voice input and distance awareness.
    """

    def __init__(self, model_path: str, camera_idx: int = 0, debug: bool = False):
        """
        Initialize navigation with voice capabilities.
        
        Args:
            model_path: Path to ONNX model
            camera_idx: Camera device index
            debug: Enable debug output
        """
        print("\n  ▶ Initializing Voice-Enabled Navigation System...")

        # Core navigation
        self.nav = NavigationSystem(model_path, camera_idx, debug=debug)

        # Voice output
        self.voice = VoiceCommands()

        # Voice input & command handler
        self.cmd_handler = VoiceCommandHandler()
        self.listener = VoiceListener(
            on_command=self._on_voice_command,
            on_error=self._on_voice_error,
        )

        # State tracking
        self.last_direction = None
        self.last_obstacle = None
        self.paused = False
        self.repeat_count = 0

        print("  ✓ Voice-enabled navigation ready")

    def _on_voice_command(self, command: str):
        """Handle voice commands from microphone."""
        print(f"\n  🎙️  Voice command: {command}")

        if command == "repeat":
            if self.last_direction:
                self._speak_with_obstacle(self.last_direction, self.last_obstacle)
                self.repeat_count += 1
            else:
                print("  [INFO] No previous direction to repeat")

        elif command == "stop":
            print("  🛑 STOP - Navigation paused by voice")
            self.paused = True

        elif command == "pause":
            print("  ⏸️  PAUSE - Navigation paused by voice")
            self.paused = True

        elif command == "resume" or command == "start":
            print("  ▶️  RESUME - Navigation resumed by voice")
            self.paused = False

        elif command == "what_around" or command == "describe":
            self._describe_current_scene()

        elif command == "help":
            self._print_voice_help()

        else:
            print(f"  [INFO] Unknown voice command: {command}")

    def _on_voice_error(self, error: str):
        """Handle voice system errors."""
        print(f"  [VOICE ERROR] {error}")

    def _speak_with_obstacle(self, direction: str, obstacle: Optional[tuple]):
        """
        Speak direction with obstacle information if available.
        
        Args:
            direction: Direction command ("Left", "Right", "Forward", "Stop")
            obstacle: (class_name, distance_meters) or None
        """
        if obstacle:
            class_name, distance = obstacle
            self.voice.speak_with_distance(direction, class_name, distance)
        else:
            self.voice.speak(direction)

    def _describe_current_scene(self):
        """Speak description of current obstacles."""
        print("  ℹ️  Scene description requested (would describe obstacles)")
        # In real implementation, you'd call: self.nav.get_current_detections()
        # and describe them
        self.voice.speak("Current scene description not yet implemented")

    def _print_voice_help(self):
        """Print available voice commands."""
        self.cmd_handler._print_help()

    def run(self):
        """Main navigation loop with voice integration."""
        # Start voice listener
        self.listener.start()
        print("  🎙️  Voice listener started")

        # Use the navigation system's run loop but with our voice handling
        self._run_with_voice()

    def _run_with_voice(self):
        """
        Main loop that integrates navigation with voice handling.
        This is a simplified version - adapt to your actual navigation_system_rpi.py
        """
        import cv2
        import collections
        import numpy as np

        print("\n  ▶ Navigation system running (voice enabled)\n")

        self.nav._running = True
        frame_times = collections.deque(maxlen=30)
        prev_command = None

        try:
            while self.nav._running:
                t_frame_start = time.perf_counter()

                # Check if paused by voice command
                if self.paused:
                    time.sleep(0.1)  # Sleep to avoid busy-waiting
                    continue

                # ── Capture ────────────────────────────
                ret, frame = self.nav.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # ── Inference ──────────────────────────
                tensor = self.nav.engine.preprocess(frame)
                t0 = time.perf_counter()
                output = self.nav.engine.infer(tensor)
                lat_ms = (time.perf_counter() - t0) * 1000
                dets = self.nav.engine.postprocess(output, self.nav.conf)

                # ── Tracking ───────────────────────────
                tracked = self.nav.tracker.update(dets)

                # ── Temporal smoothing ─────────────────
                smoothed = self.nav.smoother.update(tracked)

                # ── Path finding with DISTANCE ─────────
                direction, closest_obs = find_safe_direction(smoothed)

                # ── Majority voting ────────────────────
                voted_direction = self.nav.voter.vote(direction)

                # ── Voice output ───────────────────────
                if self.nav.voter.stable and voted_direction != prev_command:
                    self._speak_with_obstacle(voted_direction, closest_obs)
                    self.last_direction = voted_direction
                    self.last_obstacle = closest_obs
                    prev_command = voted_direction

                # ── FPS tracking ───────────────────────
                t_frame_end = time.perf_counter()
                frame_times.append(t_frame_end - t_frame_start)
                fps = 1.0 / np.mean(frame_times) if frame_times else 0

                # ── Debug output ───────────────────────
                if self.nav.debug:
                    n_obs = len(smoothed)
                    state = "PAUSED" if self.paused else "RUNNING"
                    print(
                        f"  [{state}] FPS: {fps:4.1f} | Lat: {lat_ms:5.1f}ms | "
                        f"Obs: {n_obs:2d} | Dir: {voted_direction:<12} | "
                        f"Repeat: {self.repeat_count}"
                    )

        except KeyboardInterrupt:
            print("\n  [STOPPED by user]")

        finally:
            self.nav._running = False
            self.listener.stop()
            self.nav.cap.release()
            self.voice.shutdown()
            print("  ✓ Navigation system shutdown")


# ──────────────────────────────────────────────
# EXAMPLE USAGE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice-enabled navigation system example"
    )
    parser.add_argument(
        "--model", default="./models/yolov8n_qat.onnx", help="ONNX model path"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        exit(1)

    # Create voice-enabled navigation
    nav = VoiceEnabledNavigation(
        model_path=args.model,
        camera_idx=args.camera,
        debug=args.debug,
    )

    # Run
    nav.run()

    print("\n  Example complete")
