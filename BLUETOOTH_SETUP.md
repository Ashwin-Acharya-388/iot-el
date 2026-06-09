# 🎧 Bluetooth Earphone Integration Guide — Raspberry Pi
## AI Navigation System · Voice Output via Bluetooth

---

## Overview

This guide covers:
1. Installing Bluetooth + audio stack on RPi
2. Pairing & connecting a Bluetooth earphone
3. Setting it as the **default audio sink** so TTS voice output plays through it
4. Making the connection **persistent across reboots**
5. Verifying audio from the navigation system

---

## Step 1 — Install Required Packages

```bash
sudo apt update && sudo apt upgrade -y

# Core Bluetooth stack
sudo apt install -y bluez bluez-tools

# PulseAudio Bluetooth support (needed for A2DP audio profile)
sudo apt install -y pulseaudio pulseaudio-module-bluetooth

# Audio utilities
sudo apt install -y alsa-utils pavucontrol

# Python TTS (if not already installed)
pip install pyttsx3 gTTS pygame
```

---

## Step 2 — Enable & Start Services

```bash
# Enable Bluetooth service
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

# Start PulseAudio for the current user (not as root)
pulseaudio --start

# Check that Bluetooth is unblocked
sudo rfkill unblock bluetooth

# Verify Bluetooth daemon is running
sudo systemctl status bluetooth
```

---

## Step 3 — Put Your Earphone in Pairing Mode

> **Put your Bluetooth earphone into pairing mode** (usually hold the power button for 5–7 seconds until the LED flashes rapidly or you hear "pairing mode").

---

## Step 4 — Pair the Earphone via `bluetoothctl`

Run the interactive Bluetooth controller:

```bash
bluetoothctl
```

Then inside the `bluetoothctl` prompt, run these commands **one by one**:

```
# Turn on the Bluetooth adapter
power on

# Make RPi discoverable (optional)
discoverable on

# Start scanning for nearby devices
scan on
```

Wait a few seconds — you will see a list of nearby Bluetooth devices:

```
[NEW] Device AA:BB:CC:DD:EE:FF YourEarphoneName
```

Copy the **MAC address** (e.g. `AA:BB:CC:DD:EE:FF`) of your earphone, then:

```
# Stop scanning
scan off

# Pair with the earphone (replace with YOUR MAC address)
pair AA:BB:CC:DD:EE:FF

# Trust the device so it auto-reconnects on boot
trust AA:BB:CC:DD:EE:FF

# Connect to it
connect AA:BB:CC:DD:EE:FF
```

You should see: `Connection successful`

Exit `bluetoothctl`:
```
quit
```

---

## Step 5 — Load A2DP Profile (High-Quality Audio)

After connecting, load the A2DP (stereo audio) profile:

```bash
# Check the connected sink name
pactl list sinks short
```

You will see something like:
```
1   bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink   ...
```

Set it as the **default audio output sink**:

```bash
# Replace with your actual sink name from the command above
pactl set-default-sink bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink
```

Test audio output:

```bash
# Play a test tone through the earphone
speaker-test -t wav -c 2 -D bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink

# Or play an MP3/WAV file
aplay -D bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink /usr/share/sounds/alsa/Front_Center.wav
```

---

## Step 6 — Make It Persistent (Auto-connect on Boot)

### 6a. Create a startup script

```bash
sudo nano /usr/local/bin/bt-audio-connect.sh
```

Paste this (replace `AA:BB:CC:DD:EE:FF` with your earphone's MAC):

```bash
#!/bin/bash
# Bluetooth earphone auto-connect for AI Navigation system
EARPHONE_MAC="AA:BB:CC:DD:EE:FF"
SINK_NAME="bluez_sink.${EARPHONE_MAC//:/_}.a2dp_sink"

sleep 8  # Wait for BT service to be ready

echo "Connecting to Bluetooth earphone: $EARPHONE_MAC"
bluetoothctl connect "$EARPHONE_MAC"

sleep 3

# Set as default audio sink
pactl set-default-sink "$SINK_NAME" 2>/dev/null || true

echo "Bluetooth audio setup complete."
```

Make it executable:

```bash
sudo chmod +x /usr/local/bin/bt-audio-connect.sh
```

### 6b. Create a systemd service

```bash
sudo nano /etc/systemd/system/bt-audio.service
```

Paste:

```ini
[Unit]
Description=Bluetooth Earphone Auto-Connect for AI Navigation
After=bluetooth.service pulseaudio.service
Wants=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/bt-audio-connect.sh
User=pi
Environment=PULSE_RUNTIME_PATH=/run/user/1000/pulse
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable bt-audio.service
sudo systemctl start bt-audio.service

# Check status
sudo systemctl status bt-audio.service
```

---

## Step 7 — Configure pyttsx3 / Navigation System to Use Bluetooth

If using **pyttsx3** (offline TTS), edit `voice_commands.py` to force ALSA output:

```python
import os
# Force pyttsx3 to use PulseAudio (which routes to BT earphone)
os.environ['AUDIODRIVER'] = 'pulse'
```

If using **pygame** for MP3 playback (gTTS files):

```python
import pygame
import subprocess

# Get default PulseAudio sink
sink = subprocess.check_output(
    ['pactl', 'get-default-sink'], text=True
).strip()

# Set pygame to use PulseAudio
os.environ['SDL_AUDIODRIVER'] = 'pulse'
pygame.mixer.init()
```

For **mpg123** (lightweight MP3 player), add this to `voice_commands.py`:

```bash
# Install
sudo apt install -y mpg123

# Play through default PulseAudio sink (auto-routed to BT earphone)
mpg123 /path/to/audio.mp3
```

---

## Step 8 — Verify End-to-End Audio Flow

```bash
# 1. Check BT device is connected
bluetoothctl info AA:BB:CC:DD:EE:FF | grep "Connected"

# 2. Check default audio sink
pactl get-default-sink

# 3. Test with a quick TTS command
python3 -c "
import pyttsx3
engine = pyttsx3.init()
engine.say('Obstacle detected ahead. Please stop.')
engine.runAndWait()
"

# 4. Check the navigation system's /api/bt-status endpoint
# (once app.py is running)
curl http://localhost:5500/api/bt-status
```

---

## Quick Reference — Most Used Commands

| Task | Command |
|------|---------|
| Start BT scan | `bluetoothctl scan on` |
| Pair device | `bluetoothctl pair AA:BB:CC:DD:EE:FF` |
| Connect device | `bluetoothctl connect AA:BB:CC:DD:EE:FF` |
| Trust device | `bluetoothctl trust AA:BB:CC:DD:EE:FF` |
| List audio sinks | `pactl list sinks short` |
| Set default sink | `pactl set-default-sink <sink_name>` |
| Test speaker | `speaker-test -t wav -c 2` |
| Check BT status | `bluetoothctl info` |
| Reconnect manually | `bluetoothctl connect AA:BB:CC:DD:EE:FF` |

---

## Troubleshooting

### ❌ "No default controller available"
```bash
sudo hciconfig hci0 up
```

### ❌ Audio plays through HDMI/3.5mm instead of BT earphone
```bash
# Force PulseAudio to use BT sink
pactl set-default-sink bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink
# Also move existing streams to BT
pactl move-sink-input $(pactl list sink-inputs short | awk '{print $1}') bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink
```

### ❌ Earphone connects but no A2DP (only HFP/HSP profile — low quality)
```bash
# Load A2DP module manually
pactl load-module module-bluetooth-policy
pactl load-module module-bluetooth-discover

# Restart PulseAudio
pulseaudio -k && pulseaudio --start
```

### ❌ PulseAudio crashes / earphone disconnects
```bash
# Check PulseAudio logs
journalctl --user-unit=pulseaudio -f

# Restart PulseAudio
systemctl --user restart pulseaudio
```

### ❌ "org.bluez.Error.AuthenticationFailed"
- Re-put earphone in **pairing mode**
- Run `bluetoothctl remove AA:BB:CC:DD:EE:FF` then pair again

---

## Full One-Shot Setup Script

Save as `setup_bluetooth_audio.sh` and run once on the RPi:

```bash
#!/bin/bash
# ============================================================
# AI Navigation System — Bluetooth Earphone Setup
# Run: bash setup_bluetooth_audio.sh <EARPHONE_MAC>
# Example: bash setup_bluetooth_audio.sh AA:BB:CC:DD:EE:FF
# ============================================================

set -e
MAC="${1:?Usage: $0 <EARPHONE_MAC_ADDRESS>}"

echo "=== Installing packages ==="
sudo apt update -qq
sudo apt install -y bluez bluez-tools pulseaudio pulseaudio-module-bluetooth alsa-utils mpg123

echo "=== Enabling services ==="
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
sudo rfkill unblock bluetooth
pulseaudio --start 2>/dev/null || true

echo "=== Pairing with $MAC ==="
bluetoothctl << EOF
power on
discoverable on
pair $MAC
trust $MAC
connect $MAC
quit
EOF

echo "=== Setting A2DP sink ==="
sleep 3
SINK="bluez_sink.${MAC//:/_}.a2dp_sink"
pactl set-default-sink "$SINK" 2>/dev/null && echo "Sink set: $SINK" || echo "WARNING: Sink not found yet — run manually after connecting"

echo "=== Creating auto-connect service ==="
sudo tee /usr/local/bin/bt-audio-connect.sh > /dev/null << SCRIPT
#!/bin/bash
sleep 8
bluetoothctl connect "$MAC"
sleep 3
pactl set-default-sink "$SINK" 2>/dev/null || true
SCRIPT

sudo chmod +x /usr/local/bin/bt-audio-connect.sh

sudo tee /etc/systemd/system/bt-audio.service > /dev/null << UNIT
[Unit]
Description=Bluetooth Earphone Auto-Connect
After=bluetooth.service
[Service]
Type=oneshot
ExecStart=/usr/local/bin/bt-audio-connect.sh
User=pi
Environment=PULSE_RUNTIME_PATH=/run/user/1000/pulse
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable bt-audio.service

echo ""
echo "✅  Bluetooth earphone setup complete!"
echo "    Device : $MAC"
echo "    Sink   : $SINK"
echo ""
echo "Test with:"
echo "  python3 -c \"import pyttsx3; e=pyttsx3.init(); e.say('Obstacle detected'); e.runAndWait()\""
```
