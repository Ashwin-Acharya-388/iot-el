# Raspberry Pi deployment

Run this from your laptop:

```bash
cd C:/Users/amrut/iot-el
bash deploy_rpi.sh jatayu@jatayu.local
```

If your Pi uses a different hostname or IP, replace `jatayu@jatayu.local`.

On the Pi, the script will install requirements and start the dashboard.

Manual fallback:

```bash
ssh jatayu@jatayu.local
cd /home/jatayu/freespace_navigation
./venv/bin/python -m pip install --user -r requirements.txt
CAMERA_INDEX=0 ./venv/bin/python stream_server.py
```

If your external webcam is not camera 0, set `CAMERA_INDEX=1` or the correct index. If the Pi sees the webcam as `/dev/video1`, run:

```bash
ssh jatayu@jatayu.local
cd /home/jatayu/freespace_navigation
CAMERA_DEVICE=/dev/video1 ./venv/bin/python stream_server.py
```
