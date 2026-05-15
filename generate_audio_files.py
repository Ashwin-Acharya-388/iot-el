#!/usr/bin/env python3
"""
generate_audio_files.py
========================
Pre-generate all navigation command audio files for instant RPi playback.

Instead of synthesizing audio in real-time (50-200ms per command on RPi),
pre-generate all possible commands once and store them as MP3s.
Playback is then instant (<5ms), keeping the system responsive.

Supported methods:
  1. gTTS (Google TTS) - best quality, requires internet (one-time)
  2. pyttsx3 + ffmpeg - offline, good quality
  3. espeak - lightweight native CLI tool

Usage:
    # Generate all audio files
    python generate_audio_files.py

    # Force regenerate (overwrites existing)
    python generate_audio_files.py --force

    # Use specific backend
    python generate_audio_files.py --backend gtts
    python generate_audio_files.py --backend pyttsx3
    python generate_audio_files.py --backend espeak

    # On RPi after transfer
    scp pi@raspberrypi.local:~/iot-el/audio/* ./audio/

Output:
    ./audio/
    ├── left.mp3
    ├── right.mp3
    ├── forward.mp3
    ├── stop.mp3
    ├── slight_left.mp3
    ├── slight_right.mp3
    ├── person_1_meter.mp3
    ├── person_2_meters.mp3
    ├── car_1_meter.mp3
    ├── car_2_meters.mp3
    ├── car_3_meters.mp3
    └── ... (distance variants)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────
# COMMANDS TO PRE-GENERATE
# ──────────────────────────────────────────────

BASE_COMMANDS = [
    "Left",
    "Right",
    "Forward",
    "Stop",
    "Slight Left",
    "Slight Right",
]

DISTANCE_COMMANDS = [
    "Person, 1 meter ahead",
    "Person, 2 meters ahead",
    "Car, 1 meter ahead",
    "Car, 2 meters ahead",
    "Car, 3 meters ahead",
    "Truck, very close",
    "Obstacle ahead",
]

ALL_COMMANDS = BASE_COMMANDS + DISTANCE_COMMANDS

AUDIO_DIR = Path("./audio")


# ──────────────────────────────────────────────
# FILENAME MAPPING
# ──────────────────────────────────────────────

def command_to_filename(cmd: str) -> str:
    """Convert command text to safe filename."""
    # "Person, 1 meter ahead" → "person_1_meter_ahead.mp3"
    name = cmd.lower()
    name = name.replace(", ", "_")
    name = name.replace(" ", "_")
    name = name.replace(".", "")
    name = name.strip("_")
    return f"{name}.mp3"


# ──────────────────────────────────────────────
# BACKEND 1: GTTS (GOOGLE TEXT-TO-SPEECH)
# ──────────────────────────────────────────────

def generate_gtts(commands: List[str], output_dir: Path, force: bool = False) -> bool:
    """Generate audio using Google TTS (requires internet connection)."""
    print("\n  [gTTS] Generating audio files...")

    try:
        from gtts import gTTS
    except ImportError:
        print("  ✗ gTTS not installed (pip install gTTS)")
        return False

    generated = 0
    for cmd in commands:
        output_file = output_dir / command_to_filename(cmd)

        if output_file.exists() and not force:
            print(f"    ✓ {output_file.name} (already exists)")
            generated += 1
            continue

        try:
            print(f"    → {cmd:<30} ...", end="", flush=True)
            tts = gTTS(text=cmd, lang="en", slow=False)
            tts.save(str(output_file))
            print(f" ✓")
            generated += 1
        except Exception as e:
            print(f" ✗ ({e})")

    print(f"  ✓ gTTS: {generated}/{len(commands)} files generated")
    return generated > 0


# ──────────────────────────────────────────────
# BACKEND 2: PYTTSX3 + FFMPEG
# ──────────────────────────────────────────────

def generate_pyttsx3(commands: List[str], output_dir: Path, force: bool = False) -> bool:
    """Generate audio using pyttsx3 (offline, converts WAV to MP3)."""
    print("\n  [pyttsx3] Generating audio files...")

    try:
        import pyttsx3
    except ImportError:
        print("  ✗ pyttsx3 not installed (pip install pyttsx3)")
        return False

    # Check for ffmpeg (for WAV → MP3 conversion)
    has_ffmpeg = os.system("which ffmpeg > /dev/null 2>&1") == 0
    if not has_ffmpeg:
        print("  ✗ ffmpeg not found (required for MP3 encoding)")
        print("    Install: apt install ffmpeg  OR  brew install ffmpeg")
        return False

    engine = pyttsx3.init()
    engine.setProperty("rate", 130)
    engine.setProperty("volume", 1.0)

    generated = 0
    for cmd in commands:
        mp3_file = output_dir / command_to_filename(cmd)

        if mp3_file.exists() and not force:
            print(f"    ✓ {mp3_file.name} (already exists)")
            generated += 1
            continue

        try:
            print(f"    → {cmd:<30} ...", end="", flush=True)

            # Save as WAV first
            wav_file = output_dir / f"_temp_{command_to_filename(cmd)[:-4]}.wav"
            engine.save_to_file(cmd, str(wav_file))
            engine.runAndWait()

            # Convert WAV to MP3 using ffmpeg
            if wav_file.exists():
                os.system(f'ffmpeg -i "{wav_file}" -q:a 5 "{mp3_file}" 2>/dev/null')
                wav_file.unlink()  # Delete temp WAV

            print(f" ✓")
            generated += 1
        except Exception as e:
            print(f" ✗ ({e})")

    print(f"  ✓ pyttsx3: {generated}/{len(commands)} files generated")
    return generated > 0


# ──────────────────────────────────────────────
# BACKEND 3: ESPEAK (NATIVE LINUX CLI)
# ──────────────────────────────────────────────

def generate_espeak(commands: List[str], output_dir: Path, force: bool = False) -> bool:
    """Generate audio using espeak (lightweight native tool)."""
    print("\n  [espeak] Generating audio files...")

    # Check if espeak is installed
    if os.system("which espeak > /dev/null 2>&1") != 0:
        print("  ✗ espeak not installed")
        print("    Install: apt install espeak espeak-data")
        return False

    # Check for ffmpeg (for WAV → MP3 conversion)
    has_ffmpeg = os.system("which ffmpeg > /dev/null 2>&1") == 0
    if not has_ffmpeg:
        print("  [WARN] ffmpeg not found, generating WAV only (not MP3)")

    generated = 0
    for cmd in commands:
        filename = command_to_filename(cmd)
        wav_file = output_dir / filename.replace(".mp3", ".wav")
        mp3_file = output_dir / filename

        if mp3_file.exists() and not force:
            print(f"    ✓ {mp3_file.name} (already exists)")
            generated += 1
            continue

        try:
            print(f"    → {cmd:<30} ...", end="", flush=True)

            # Generate WAV with espeak
            os.system(f'espeak -s 130 -v en "{cmd}" -w "{wav_file}" 2>/dev/null')

            # Convert to MP3 if ffmpeg available
            if has_ffmpeg and wav_file.exists():
                os.system(f'ffmpeg -i "{wav_file}" -q:a 5 "{mp3_file}" 2>/dev/null')
                wav_file.unlink()  # Delete temp WAV
                print(f" ✓")
            elif wav_file.exists():
                print(f" ✓ (WAV only)")

            generated += 1
        except Exception as e:
            print(f" ✗ ({e})")

    print(f"  ✓ espeak: {generated}/{len(commands)} files generated")
    return generated > 0


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate audio files for instant RPi playback."
    )
    parser.add_argument(
        "--backend",
        choices=["gtts", "pyttsx3", "espeak", "auto"],
        default="auto",
        help="Which TTS backend to use (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all files (overwrite existing)",
    )
    parser.add_argument(
        "--commands",
        choices=["base", "distance", "all"],
        default="all",
        help="Which command set to generate",
    )
    args = parser.parse_args()

    # Create output directory
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Audio directory: {AUDIO_DIR.absolute()}")

    # Select commands
    if args.commands == "base":
        commands = BASE_COMMANDS
    elif args.commands == "distance":
        commands = DISTANCE_COMMANDS
    else:
        commands = ALL_COMMANDS

    print(f"  Commands to generate: {len(commands)}")
    for cmd in commands[:5]:
        print(f"    • {cmd}")
    if len(commands) > 5:
        print(f"    ... and {len(commands) - 5} more")

    # Select backend
    print("\n" + "="*60)
    print("  AUDIO FILE PRE-GENERATION")
    print("="*60)

    success = False

    if args.backend == "auto":
        # Try in order: gTTS → pyttsx3 → espeak
        print("\n  [AUTO] Trying backends in priority order...")
        success = (
            generate_gtts(commands, AUDIO_DIR, args.force)
            or generate_pyttsx3(commands, AUDIO_DIR, args.force)
            or generate_espeak(commands, AUDIO_DIR, args.force)
        )
    elif args.backend == "gtts":
        success = generate_gtts(commands, AUDIO_DIR, args.force)
    elif args.backend == "pyttsx3":
        success = generate_pyttsx3(commands, AUDIO_DIR, args.force)
    elif args.backend == "espeak":
        success = generate_espeak(commands, AUDIO_DIR, args.force)

    # Summary
    print("\n" + "="*60)
    if success:
        audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.wav"))
        print(f"  ✓ SUCCESS: {len(audio_files)} audio files ready")
        print(f"\n  Next steps:")
        print(f"    1. Transfer to RPi:")
        print(f"       scp -r ./audio/ pi@raspberrypi.local:~/iot-el/")
        print(f"    2. Test on RPi:")
        print(f"       python test_audio_rpi.py")
        print(f"    3. Run navigation system:")
        print(f"       python navigation_system_rpi.py --debug")
        return 0
    else:
        print("  ✗ FAILURE: No audio files generated")
        print("\n  Troubleshooting:")
        print("    • gTTS: pip install gTTS (requires internet)")
        print("    • pyttsx3: pip install pyttsx3 && apt install ffmpeg")
        print("    • espeak: apt install espeak espeak-data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
