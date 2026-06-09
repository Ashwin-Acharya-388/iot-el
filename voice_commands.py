"""
voice_commands.py
=================
Handles voice command output for the navigation assistant.

Features:
  • Pre-loads all directional commands as audio files (fastest playback)
  • Distance-aware warnings: "Car, 2.3 meters ahead" instead of just "Left"
  • 1.5-second cooldown prevents repeating same command
  • Background thread so TTS never blocks the inference loop
  • Falls back gracefully: preloaded WAV → pyttsx3 → print-only

Commands: 
  - Directions: "Left", "Right", "Slight Left", "Slight Right", "Forward", "Stop"
  - Distance-aware: "Car, 2.3 meters ahead", "Person, 1.8 meters", etc.

Usage:
    from voice_commands import VoiceCommands
    vc = VoiceCommands()
    vc.speak("Left")                                      # Basic
    vc.speak("Car, 3 meters ahead")                      # Distance-aware
    vc.speak_with_distance("Right", "car", 2.5)         # Helper method
    vc.shutdown()
"""

import os
import sys
import time
import shutil
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional



# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

COOLDOWN_SECONDS = 1.5     # Min gap between SAME command being repeated
BASE_COMMANDS     = ["Left", "Right", "Slight Left", "Slight Right", "Forward", "Stop"]

# Distance-based command templates
DISTANCE_COMMANDS = [
    "Person 1 meter",
    "Person 2 meters",
    "Car 1 meter",
    "Car 2 meters",
    "Car 3 meters",
    "Truck ahead",
    "Obstacle very close",
]

ALL_COMMANDS = BASE_COMMANDS + DISTANCE_COMMANDS

AUDIO_DIR        = Path("./audio")
VOICE_RATE       = 150     # pyttsx3 words-per-minute (default ~200, lower = clearer)
VOICE_VOLUME     = 1.0


# ──────────────────────────────────────────────
# AUDIO BACKEND DETECTION
# ──────────────────────────────────────────────

def _try_import_pygame() -> bool:
    """Check if pygame is importable (does NOT initialize mixer)."""
    try:
        import pygame
        return True
    except Exception:
        return False


def _try_import_pyttsx3():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        return engine
    except Exception:
        return None


def _try_import_playsound() -> bool:
    try:
        import playsound
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# PRE-RECORDED AUDIO GENERATION
# ──────────────────────────────────────────────

def generate_audio_files(commands: list, audio_dir: Path, engine=None) -> dict:
    """
    Pre-generate WAV files for all commands using pyttsx3 or gTTS.
    
    WHY PRE-GENERATE:
    Real-time TTS synthesis takes 50-200ms on RPi. Pre-generating at startup
    means playback is instant (<5ms), keeping the system responsive.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # Try gTTS first (better voice quality)
    try:
        from gtts import gTTS
        for cmd in commands:
            out = audio_dir / f"{cmd.replace(' ', '_').lower()}.mp3"
            if not out.exists():
                tts = gTTS(text=cmd, lang="en", slow=False)
                tts.save(str(out))
            files[cmd] = out
        print(f"  ✓ Audio files generated (gTTS) → {audio_dir}")
        return files
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] gTTS failed: {e}")

    # Fallback: pyttsx3 → save to WAV (pyttsx3 doesn't easily save files on all platforms)
    # We'll use a subprocess call to 'espeak' on Linux/RPi if available
    if shutil.which("espeak"):
        for cmd in commands:
            out = audio_dir / f"{cmd.replace(' ', '_').lower()}.wav"
            if not out.exists():
                os.system(f'espeak -s 130 -v en "{cmd}" --stdout > "{out}" 2>/dev/null')
            if out.exists():
                files[cmd] = out
        if files:
            print(f"  ✓ Audio files generated (espeak) → {audio_dir}")
            return files

    print("  [INFO] No audio file pre-generation available; using real-time TTS.")
    return {}


# ──────────────────────────────────────────────
# VOICE COMMANDS CLASS
# ──────────────────────────────────────────────

class VoiceCommands:
    """
    Thread-safe voice command output with cooldown and debounce.
    
    The speak() method is non-blocking: commands are queued and
    played by a background thread, so inference never waits for audio.
    """

    def __init__(self, cooldown: float = COOLDOWN_SECONDS):
        self.cooldown       = cooldown
        self._last_cmd      = None
        self._last_time     = 0.0
        self._queue         = queue.Queue(maxsize=2)   # Never accumulate stale commands
        self._shutdown_flag = threading.Event()
        self._lock          = threading.Lock()

        # Detect backend
        self._audio_files   = {}
        self._pygame_ready  = False
        self._pyttsx3_available = False
        self._fallback_print= False

        self._setup_backend()

        # Start audio playback thread
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

        print(f"  ✓ VoiceCommands ready. Cooldown: {cooldown}s")

    def _setup_backend(self):
        """Detect and configure the best available audio output method."""
        print("  Setting up voice backend...")

        # 1. Pre-generate audio files
        self._audio_files = generate_audio_files(ALL_COMMANDS, AUDIO_DIR)

        # 2. Try pygame for pre-recorded playback (lowest latency)
        if self._audio_files and _try_import_pygame():
            import pygame
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            except Exception as e:
                print(f"  [WARN] pygame mixer init failed: {e}")
            else:
                self._pygame_ready = True
                print("  ✓ Audio backend: pygame (pre-recorded WAV/MP3)")

        # 3. Check pyttsx3 availability without holding main thread COM engine
        try:
            import pyttsx3
            test_engine = pyttsx3.init()
            del test_engine
            self._pyttsx3_available = True
            print("  ✓ Audio backend: pyttsx3 (real-time TTS) detected as fallback")
        except Exception as e:
            print(f"  [WARN] Failed to detect pyttsx3: {e}")

        # 4. Try espeak directly as subprocess if no pyttsx3
        if not self._pyttsx3_available and shutil.which("espeak"):
            print("  ✓ Audio backend: espeak (subprocess) initialized as fallback")
            self._espeak = True

        # 5. Print-only fallback
        if not self._pygame_ready and not self._pyttsx3_available and not hasattr(self, "_espeak"):
            print("  [WARN] No audio backend found. Install pyttsx3 or espeak.")
            print("         Commands will be printed to console only.")
            self._fallback_print = True

    def speak(self, command: str):
        """
        Queue a voice command (non-blocking).
        Respects cooldown: same command won't repeat within cooldown period.
        """
        now = time.monotonic()

        with self._lock:
            if command == self._last_cmd and (now - self._last_time) < self.cooldown:
                return  # Debounced
            self._last_cmd  = command
            self._last_time = now

        # Drop stale command if queue is full (prefer freshest info)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

        self._queue.put(command)

    def speak_with_distance(self, direction: str, obstacle_class: str, distance: float):
        """
        Queue a distance-aware navigation command.
        
        Example:
            vc.speak_with_distance("Right", "car", 2.5)
            → Speaks: "Right • Car, 2.5 meters"
        
        Args:
            direction: Direction name ("Left", "Right", "Forward", etc.)
            obstacle_class: Object type ("car", "person", "truck", etc.)
            distance: Distance in meters
        """
        # Format: "Direction • Obstacle, X meters"
        msg = f"{direction} • {obstacle_class.capitalize()}, {distance:.1f} meters"
        self.speak(msg)

    def _playback_loop(self):
        """Background thread: dequeue and play commands."""
        is_windows = sys.platform.startswith('win')
        if is_windows:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception as e:
                print(f"  [WARN] CoInitialize failed on playback thread: {e}")

        pyttsx3_engine = None
        if self._pyttsx3_available:
            try:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
                pyttsx3_engine.setProperty("rate",   VOICE_RATE)
                pyttsx3_engine.setProperty("volume", VOICE_VOLUME)
                # Try to select a clear voice
                voices = pyttsx3_engine.getProperty("voices")
                for v in voices:
                    if "english" in v.name.lower() or "en" in v.id.lower():
                        pyttsx3_engine.setProperty("voice", v.id)
                        break
            except Exception as e:
                print(f"  [WARN] Failed to initialize pyttsx3 on background thread: {e}")

        while not self._shutdown_flag.is_set():
            try:
                cmd = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._play(cmd, pyttsx3_engine)

        if pyttsx3_engine is not None:
            try:
                pyttsx3_engine.stop()
            except Exception:
                pass
            del pyttsx3_engine

        if is_windows:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def _play(self, command: str, pyttsx3_engine=None):
        """Play a single command using the best available backend."""
        if self._fallback_print:
            print(f"\n  🔊 DIRECTION: {command}")
            return

        # pygame: play pre-recorded file
        if self._pygame_ready and command in self._audio_files:
            try:
                import pygame
                snd = pygame.mixer.Sound(str(self._audio_files[command]))
                snd.play()
                # Wait for playback to finish (non-blocking for the inference thread)
                time.sleep(snd.get_length() + 0.05)
                return
            except Exception as e:
                print(f"  [WARN] pygame play failed: {e}")

        # pyttsx3: real-time TTS
        if pyttsx3_engine:
            try:
                pyttsx3_engine.say(command)
                pyttsx3_engine.runAndWait()
                return
            except Exception as e:
                print(f"  [WARN] pyttsx3 failed: {e}")

        # espeak: subprocess
        if hasattr(self, "_espeak"):
            try:
                os.system(f'espeak -s 130 -v en "{command}" 2>/dev/null')
                return
            except Exception as e:
                print(f"  [WARN] espeak failed: {e}")

        # Last resort
        print(f"\n  🔊 DIRECTION: {command}")

    def shutdown(self):
        """Stop the playback thread cleanly."""
        self._shutdown_flag.set()
        self._thread.join(timeout=2.0)
        if self._pygame_ready:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass
        print("  ✓ VoiceCommands shutdown.")


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing VoiceCommands...")
    vc = VoiceCommands(cooldown=0.5)
    time.sleep(0.5)

    for cmd in ALL_COMMANDS:
        print(f"  Speaking: {cmd}")
        vc.speak(cmd)
        time.sleep(1.0)

    # Test cooldown: same command twice rapidly
    print("\n  Testing cooldown (should only say 'Left' once)...")
    vc.speak("Left")
    time.sleep(0.3)
    vc.speak("Left")   # Should be debounced
    time.sleep(1.5)
    vc.speak("Left")   # Should play again (cooldown elapsed)
    time.sleep(1.5)

    vc.shutdown()
    print("  Done.")
