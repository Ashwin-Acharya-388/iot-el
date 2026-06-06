# Raspberry Pi deployment

Run this from your laptop:

```bash
cd C:/Users/amrut/iot-el
bash deploy_rpi.sh pi@raspberrypi.local
```

If your Pi uses a different hostname or IP, replace `pi@raspberrypi.local`.

On the Pi, the script will install requirements and start the dashboard.

Manual fallback:

```bash
ssh pi@raspberrypi.local
cd /home/pi/iot-el
python3 -m pip install --user -r requirements.txt
CAMERA_INDEX=0 python3 stream_server.py
```

If your external webcam is not camera 0, set `CAMERA_INDEX=1` or the correct index.
