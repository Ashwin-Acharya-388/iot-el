param (
    [string]$RpiHost = "jatayu@jatayu.local",
    [string]$TargetDir = "/home/jatayu/freespace_navigation",
    [string]$CameraIndex = "0",
    [string]$CameraDevice = ""
)

Write-Host "Creating archive, excluding venv and large files..."
tar.exe -czf deploy.tar.gz --exclude=".git" --exclude="venv" --exclude="__pycache__" --exclude=".pytest_cache" --exclude="audio/*.wav" .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create archive."
    exit 1
}

Write-Host "Transferring archive to $RpiHost..."
scp deploy.tar.gz "${RpiHost}:/tmp/deploy.tar.gz"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to transfer archive via SCP."
    exit 1
}

Write-Host "Extracting on Raspberry Pi and starting servers..."
$remoteCmd = @"
mkdir -p $TargetDir && cd $TargetDir && \
tar -xzf /tmp/deploy.tar.gz && \
( ./venv/bin/python -c 'import sys' 2>/dev/null || ( echo 'Recreating broken virtual environment...' && rm -rf venv && python3 -m venv venv ) ) && \
./venv/bin/pip install flask numpy opencv-python-headless onnxruntime pyttsx3 pygame gTTS psutil PyYAML paho-mqtt firebase-admin --extra-index-url https://www.piwheels.org/simple && \
nohup python3 -m http.server 8000 -d caretaker-portal > portal.log 2>&1 & \
nohup env CAMERA_INDEX='$CameraIndex' CAMERA_DEVICE='$CameraDevice' ./venv/bin/python app.py > app.log 2>&1 &
"@

ssh $RpiHost $remoteCmd
