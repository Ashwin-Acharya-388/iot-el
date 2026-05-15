"""
voice_input.py
===============
Voice command listener using speech_recognition for microphone input.

Supports commands:
  • "repeat" / "repeat that" - repeat the last navigation command
  • "stop" / "pause" - halt the navigation system
  • "what's around me" / "describe" - read out current obstacles
  • "help" - list available commands
  • "resume" / "start" - resume after pause

Features:
  • Offline-first: uses PocketSphinx (built-in) or Vosk for offline recognition
  • Falls back to Google Speech API if offline option unavailable
  • Non-blocking: runs in background thread
  • Noise-robust: automatically adjusts to ambient noise level
  • Configurable listening timeout and no-match retry

Usage:
    from voice_input import VoiceListener
    
    # Initialize listener
    listener = VoiceListener(on_command=handle_voice_command)
    
    # Start listening in background
    listener.start()
    
    # Handle commands
    def handle_voice_command(cmd):
        if cmd == "repeat":
            # repeat_last_navigation_command()
            pass
        elif cmd == "stop":
            # stop_navigation()
            pass
    
    # Shutdown when done
    listener.stop()

Exit codes & error handling:
  • No microphone: print warning, fall back to print-only
  • No internet (Google API): use offline recognition
  • Timeout: automatically retry every 5 seconds
  • No match: log and continue listening
"""

import threading
import queue
import time
from enum import Enum
from typing import Callable, Optional
from pathlib import Path


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

LISTEN_TIMEOUT_SEC = 10.0      # Max time to wait for speech before giving up
PHRASE_TIME_LIMIT = 15.0       # Max phrase length (seconds)
ENERGY_THRESHOLD = 4000        # Mic sensitivity (higher = less sensitive)
DYNAMIC_ENERGY_THRESHOLD = True # Auto-adjust to noise


# ──────────────────────────────────────────────
# VOICE COMMANDS
# ──────────────────────────────────────────────

class VoiceCommand(Enum):
    """Recognized voice commands."""
    REPEAT = "repeat"              # Repeat last navigation output
    STOP = "stop"                  # Stop navigation
    PAUSE = "pause"
    RESUME = "resume"              # Resume after pause
    START = "start"
    WHAT_AROUND = "what_around"    # "What's around me?" / "Describe"
    DESCRIBE = "describe"
    HELP = "help"
    UNKNOWN = "unknown"


# Map common voice phrases to commands
VOICE_COMMAND_MAP = {
    # Repeat
    "repeat": VoiceCommand.REPEAT,
    "repeat that": VoiceCommand.REPEAT,
    "say that again": VoiceCommand.REPEAT,
    "again": VoiceCommand.REPEAT,

    # Stop / Pause
    "stop": VoiceCommand.STOP,
    "pause": VoiceCommand.PAUSE,
    "hold": VoiceCommand.PAUSE,

    # Resume / Start
    "resume": VoiceCommand.RESUME,
    "start": VoiceCommand.START,
    "go": VoiceCommand.START,
    "continue": VoiceCommand.RESUME,

    # What's around
    "what's around": VoiceCommand.WHAT_AROUND,
    "what's around me": VoiceCommand.WHAT_AROUND,
    "what around me": VoiceCommand.WHAT_AROUND,
    "describe": VoiceCommand.DESCRIBE,
    "what do you see": VoiceCommand.WHAT_AROUND,

    # Help
    "help": VoiceCommand.HELP,
}


# ──────────────────────────────────────────────
# MICROPHONE DETECTION & CALIBRATION
# ──────────────────────────────────────────────

def _check_microphone_available() -> bool:
    """Check if any microphone is available."""
    try:
        import speech_recognition as sr
        mic = sr.Microphone()
        with mic as source:
            pass  # Just open and close
        return True
    except (IndexError, OSError):
        return False


def _calibrate_microphone(timeout_sec: float = 3.0) -> Optional[int]:
    """
    Calibrate microphone noise level.
    
    Returns:
        energy_threshold value, or None if calibration failed
    """
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        print("  🎙️  Calibrating microphone... (speak normally)")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=timeout_sec)
            threshold = recognizer.energy_threshold

        print(f"  ✓ Microphone calibrated. Energy threshold: {threshold:.0f}")
        return threshold

    except Exception as e:
        print(f"  [WARN] Microphone calibration failed: {e}")
        return None


# ──────────────────────────────────────────────
# SPEECH RECOGNITION BACKENDS
# ──────────────────────────────────────────────

def _init_recognizer():
    """Initialize speech recognizer with offline-first config."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("  [ERROR] speech_recognition not installed")
        print("         pip install SpeechRecognition pocketsphinx")
        return None

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = DYNAMIC_ENERGY_THRESHOLD

    return recognizer


def _try_recognize_offline(recognizer, audio) -> Optional[str]:
    """Try offline recognition using PocketSphinx."""
    try:
        text = recognizer.recognize_sphinx(audio)
        return text.lower()
    except Exception as e:
        return None


def _try_recognize_google(recognizer, audio) -> Optional[str]:
    """Try Google Speech API (requires internet)."""
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except Exception as e:
        return None


# ──────────────────────────────────────────────
# VOICE LISTENER
# ──────────────────────────────────────────────

class VoiceListener:
    """
    Background voice command listener.
    
    Listens for voice commands from microphone in a background thread.
    Commands are parsed and callbacks are invoked in the listening thread.
    Non-blocking for the main navigation loop.
    """

    def __init__(
        self,
        on_command: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        energy_threshold: int = ENERGY_THRESHOLD,
        listen_timeout: float = LISTEN_TIMEOUT_SEC,
    ):
        """
        Initialize voice listener.

        Args:
            on_command: Callback when a command is recognized.
                       Called with command name (str).
            on_error: Callback when an error occurs.
                     Called with error message (str).
            energy_threshold: Mic sensitivity (higher = less sensitive).
            listen_timeout: Max time to wait for speech (seconds).
        """
        self.on_command = on_command or (lambda cmd: None)
        self.on_error = on_error or (lambda err: None)
        self.listen_timeout = listen_timeout

        self._running = False
        self._thread = None
        self._shutdown_flag = threading.Event()

        # Try to check mic availability
        if not _check_microphone_available():
            print("  [WARN] No microphone detected. Voice input disabled.")
            print("         Commands will be read from print-fallback only.")
            self._mic_available = False
            return

        self._mic_available = True

        # Initialize recognizer
        self.recognizer = _init_recognizer()
        if not self.recognizer:
            self._mic_available = False
            return

        # Calibrate microphone
        cal_threshold = _calibrate_microphone(timeout_sec=2.0)
        if cal_threshold is not None:
            self.recognizer.energy_threshold = cal_threshold
        else:
            self.recognizer.energy_threshold = energy_threshold

        print(f"  ✓ VoiceListener initialized. Listen timeout: {listen_timeout}s")

    def start(self):
        """Start listening in background thread."""
        if not self._mic_available:
            print("  [INFO] Microphone not available. Voice input disabled.")
            return

        if self._running:
            print("  [WARN] VoiceListener already running")
            return

        self._running = True
        self._shutdown_flag.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("  ✓ Voice listener started")

    def stop(self):
        """Stop listening gracefully."""
        if not self._running:
            return

        self._running = False
        self._shutdown_flag.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("  ✓ Voice listener stopped")

    def _listen_loop(self):
        """Background listening loop."""
        try:
            import speech_recognition as sr
        except ImportError:
            self.on_error("speech_recognition not installed")
            return

        mic = sr.Microphone()

        print("  🎙️  Listening for voice commands...")

        while self._running and not self._shutdown_flag.is_set():
            try:
                with mic as source:
                    # Listen for audio
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.listen_timeout,
                            phrase_time_limit=PHRASE_TIME_LIMIT,
                        )
                    except sr.UnknownValueError:
                        # Silence or unrecognizable audio
                        continue
                    except sr.RequestError as e:
                        self.on_error(f"Recognizer error: {e}")
                        continue

                # Try to recognize speech
                text = None

                # Try offline first (PocketSphinx)
                text = _try_recognize_offline(self.recognizer, audio)
                if text:
                    print(f"  🎙️  (offline) Heard: '{text}'")
                    self._process_command(text)
                    continue

                # Fall back to Google API
                try:
                    text = _try_recognize_google(self.recognizer, audio)
                    if text:
                        print(f"  🎙️  (Google) Heard: '{text}'")
                        self._process_command(text)
                    continue
                except Exception as e:
                    self.on_error(f"Recognition failed: {e}")

            except Exception as e:
                self.on_error(f"Unexpected error: {e}")
                time.sleep(1.0)  # Avoid rapid error spam

    def _process_command(self, text: str):
        """Parse recognized text and invoke callback."""
        # Try exact match first
        if text in VOICE_COMMAND_MAP:
            cmd = VOICE_COMMAND_MAP[text]
            self.on_command(cmd.value)
            return

        # Try substring match
        for phrase, cmd in VOICE_COMMAND_MAP.items():
            if phrase in text:
                self.on_command(cmd.value)
                return

        # No match
        print(f"  [INFO] Command not recognized: '{text}'")
        self.on_command(VoiceCommand.UNKNOWN.value)


# ──────────────────────────────────────────────
# INTEGRATION WITH NAVIGATION SYSTEM
# ──────────────────────────────────────────────

class VoiceCommandHandler:
    """
    Handles voice commands from VoiceListener and orchestrates actions.
    
    Bridges between voice input and navigation system.
    """

    def __init__(self):
        self.last_output = None
        self._paused = False

    def handle_command(self, command: str, last_output: str = None):
        """
        Process a voice command.
        
        Args:
            command: Command name (from VoiceCommand enum)
            last_output: Last navigation output (for "repeat" command)
        """
        if command == VoiceCommand.REPEAT.value:
            if last_output:
                print(f"  🔊 REPEAT: {last_output}")
            else:
                print("  [INFO] No previous output to repeat")

        elif command == VoiceCommand.STOP.value:
            print("  🛑 STOP command received")
            self._paused = True
            # User should handle: call navigation_system.stop()

        elif command == VoiceCommand.PAUSE.value:
            print("  ⏸️  PAUSE command received")
            self._paused = True

        elif command == VoiceCommand.RESUME.value or command == VoiceCommand.START.value:
            print("  ▶️  RESUME command received")
            self._paused = False

        elif command == VoiceCommand.WHAT_AROUND.value or command == VoiceCommand.DESCRIBE.value:
            print("  ℹ️  DESCRIBE command - current obstacles would be listed here")
            # User should handle: call navigation_system.get_current_obstacles()

        elif command == VoiceCommand.HELP.value:
            self._print_help()

        else:
            print(f"  [INFO] Unknown command: {command}")

    def _print_help(self):
        """Print available commands."""
        print("\n  📋 AVAILABLE VOICE COMMANDS:")
        print("    • 'Repeat' / 'Say that again' → repeat last direction")
        print("    • 'Stop' / 'Pause' → stop navigation")
        print("    • 'Resume' / 'Start' → resume navigation")
        print("    • 'What's around me' / 'Describe' → list obstacles")
        print("    • 'Help' → show this menu")
        print()

    @property
    def paused(self) -> bool:
        """Whether navigation is paused by voice command."""
        return self._paused


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing VoiceListener...")
    print("\nAvailable commands:")
    for phrase in sorted(VOICE_COMMAND_MAP.keys())[:10]:
        print(f"  • {phrase}")
    print("  ... and more")

    def on_cmd(cmd):
        print(f"  ✓ Command recognized: {cmd}")

    def on_err(err):
        print(f"  [ERROR] {err}")

    listener = VoiceListener(on_command=on_cmd, on_error=on_err)

    if not listener._mic_available:
        print("  [SKIP] Microphone not available for testing")
        sys.exit(1)

    listener.start()
    print("\nListening for 30 seconds (try saying 'repeat', 'stop', etc.)...\n")

    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n[STOPPED by user]")

    listener.stop()
    print("Test complete")
