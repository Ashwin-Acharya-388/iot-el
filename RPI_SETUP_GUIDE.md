# Raspberry Pi MQTT & Bluetooth Setup Guide

This guide provides step-by-step instructions to configure a local MQTT broker with WebSockets on your Raspberry Pi and route navigation voice commands/telemetry to a Bluetooth earphone.

---

## Part 1: MQTT & Safety Alert System Activation

Since you configured the broker as `127.0.0.1` in `settings.yaml`, you must install and run a local MQTT broker (Mosquitto) on the Raspberry Pi and configure it to support WebSockets for the browser dashboard.

### Step 1: Install Mosquitto on the RPi
Run the following commands in the SSH terminal of your Raspberry Pi:
```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
```

### Step 2: Configure WebSockets Support
By default, Mosquitto only listens on port 1883 (TCP). The caretaker dashboard requires WebSockets on port 8083.
1. Create a configuration file on the RPi:
   ```bash
   sudo nano /etc/mosquitto/conf.d/websockets.conf
   ```
2. Paste the following configuration lines:
   ```ini
   # Standard TCP listener
   listener 1883
   allow_anonymous true

   # WebSockets listener for Caretaker Dashboard
   listener 8083
   protocol websockets
   allow_anonymous true
   ```
3. Save and exit (Press `Ctrl+O`, `Enter`, then `Ctrl+X`).
4. Restart the Mosquitto service:
   ```bash
   sudo systemctl restart mosquitto
   sudo systemctl enable mosquitto
   ```

### Step 3: Run the Services
Start the services in the following order in separate terminals/tmux windows on the Raspberry Pi:
1. **Cloud Danger Handler**:
   ```bash
   cd ~/freespace_navigation
   ./venv/bin/python cloud_handler.py
   ```
2. **Navigation Dashboard Server**:
   ```bash
   cd ~/freespace_navigation
   # Starts the server on port 5500
   ./venv/bin/python app.py
   ```
3. Open `http://<RPI_IP>:5500` in your MacBook browser. Login with `admin` / `blind2024`.
   *(The dashboard will dynamically connect to the RPi's WebSocket broker via port 8083).*

---

## Part 2: Bluetooth Setup for Earphone Voice Commands

To route the audio output (navigation voice) and the microphone input (voice commands) to a Bluetooth earphone, you need to pair the device and configure the audio server (PipeWire/WirePlumber or PulseAudio).

### Step 1: Pair and Connect the Earphone
1. Open the Bluetooth control utility on the RPi:
   ```bash
   sudo bluetoothctl
   ```
2. Run these commands inside the `bluetoothctl` prompt:
   ```text
   power on
   agent on
   default-agent
   scan on
   ```
3. Turn on pairing mode on your Bluetooth earphone. Look for its MAC address (e.g. `XX:XX:XX:XX:XX:XX`) in the scan output.
4. Pair and trust the earphone (replace with your earphone's MAC address):
   ```text
   pair XX:XX:XX:XX:XX:XX
   trust XX:XX:XX:XX:XX:XX
   connect XX:XX:XX:XX:XX:XX
   ```
5. Exit the prompt:
   ```text
   exit
   ```

### Step 2: Configure Audio Routing

#### Option A: Raspberry Pi OS Bookworm (PipeWire)
Modern RPi OS uses **PipeWire** by default.
1. Install `wireplumber` utilities:
   ```bash
   sudo apt install -y wireplumber
   ```
2. List your audio sources (microphones) and sinks (speakers):
   ```bash
   wpctl status
   ```
3. Note the ID of your Bluetooth earphone sink and source. Set them as defaults:
   ```bash
   # Set default output (playback)
   wpctl set-default <sink_id>
   
   # Set default input (microphone)
   wpctl set-default <source_id>
   ```

#### Option B: Raspberry Pi OS Bullseye/Legacy (PulseAudio)
Older RPi OS versions use **PulseAudio**.
1. Install PulseAudio Bluetooth modules:
   ```bash
   sudo apt install -y pulseaudio-module-bluetooth
   sudo systemctl restart bluetooth
   ```
2. Start PulseAudio if it isn't running:
   ```bash
   pulseaudio --start
   ```
3. Set default sink and source:
   ```bash
   # List sinks and search for your Bluetooth headset
   pactl list sinks short
   pactl set-default-sink <bluetooth_sink_name>

   # List sources and search for your Bluetooth headset microphone
   pactl list sources short
   pactl set-default-source <bluetooth_source_name>
   ```

### Step 3: Verify Audio and Microphone Input
1. **Verify Speaker Playback**:
   ```bash
   # Play a test wav file to the earphone
   aplay -D default /usr/share/sounds/alsa/Front_Center.wav
   ```
2. **Verify Microphone Recording**:
   ```bash
   # Record audio from your earphone mic for 5 seconds
   arecord -D default -f S16_LE -c1 -r16000 -d 5 test_mic.wav
   
   # Play it back to listen to the quality
   aplay test_mic.wav
   ```

If you can hear the playback, your Bluetooth earphone is fully integrated! The voice commands listener (`voice_input.py`) and playback engine (`voice_commands.py`) will now automatically use the Bluetooth device.
