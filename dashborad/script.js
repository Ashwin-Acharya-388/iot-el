function updateClock() {
  const now = new Date();
  let hours = now.getHours();
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';

  hours = hours % 12;
  hours = hours ? hours : 12;

  document.getElementById('time').innerHTML = `${hours}:${minutes}:${seconds} ${ampm}`;
  document.getElementById('date').innerHTML = now.toDateString();
}

let alertTracker = new Set();
let lastSpoken = '';

function setStatus(isActive) {
  const serverStatus = document.getElementById('status-server');
  if (serverStatus) {
    serverStatus.innerText = isActive ? 'Active' : 'Offline';
    serverStatus.style.color = isActive ? '#35ff4f' : '#ffaa35';
  }

  const circle = document.getElementById('status-circle');
  if (circle) {
    circle.style.backgroundColor = isActive ? '#35ff4f' : '#ffaa35';
    circle.style.boxShadow = isActive ? '0 0 15px #35ff4f' : '0 0 15px #ffaa35';
  }
}

function formatCommand(data) {
  let currentCommand = data.direction || 'FORWARD';
  if (data.obstacles && data.obstacles.length > 0) {
    const closest = data.obstacles.reduce((prev, curr) => (prev.distance < curr.distance ? prev : curr));
    currentCommand = `${data.direction || 'FORWARD'} • ${closest.label || 'Obstacle'} ${closest.distance}m`;
  }
  return currentCommand;
}

function updateDashboard(data) {
  const dirElement = document.getElementById('val-direction');
  if (dirElement) dirElement.innerText = data.direction || 'FORWARD';

  const fpsElement = document.getElementById('val-fps');
  if (fpsElement) fpsElement.innerText = data.fps || '18.5';

  const countElement = document.getElementById('val-obstacle-count');
  if (countElement) countElement.innerText = data.obstacle_count || 0;

  const camStatus = document.getElementById('status-camera');
  if (camStatus) {
    camStatus.innerText = data.status?.camera || 'Connected';
    camStatus.style.color = data.status?.camera === 'Connected' ? '#35ff4f' : '#ffaa35';
  }

  const modelStatus = document.getElementById('status-model');
  if (modelStatus) {
    modelStatus.innerText = data.status?.model || 'Running';
    modelStatus.style.color = data.status?.model === 'Running' ? '#35ff4f' : '#ffaa35';
  }

  setStatus(true);

  const voiceMain = document.getElementById('voice-main');
  const voiceHistory = document.getElementById('voice-history');

  if (voiceMain && voiceHistory) {
    const currentCommand = formatCommand(data);
    if (voiceMain.innerText !== currentCommand) {
      voiceMain.innerText = currentCommand;

      const p = document.createElement('p');
      p.innerText = currentCommand;
      if ((data.direction || 'FORWARD') === 'STOP') p.className = 'danger';
      voiceHistory.insertBefore(p, voiceHistory.firstChild);
      while (voiceHistory.children.length > 3) {
        voiceHistory.removeChild(voiceHistory.lastChild);
      }
    }

    const shouldSpeak = currentCommand &&
      currentCommand !== lastSpoken &&
      (data.obstacles?.length > 0 || (data.direction || 'FORWARD') !== 'FORWARD');

    if (shouldSpeak) {
      fetch(`/speak?text=${encodeURIComponent(currentCommand)}`)
        .catch(() => console.warn('[FRONTEND] Voice request failed.'));
      lastSpoken = currentCommand;
    }
  }

  const alertsBody = document.getElementById('alerts-body');
  if (alertsBody && data.obstacles) {
    data.obstacles.forEach((obs) => {
      if (obs.distance < 4.5) {
        const alertKey = `${obs.label}_${obs.distance.toFixed(0)}`;
        if (!alertTracker.has(alertKey)) {
          alertTracker.add(alertKey);
          if (alertTracker.size > 20) alertTracker.clear();

          const now = new Date();
          const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          const tr = document.createElement('tr');
          const tdTime = document.createElement('td');
          tdTime.innerText = timeStr;
          tr.appendChild(tdTime);

          const tdLabel = document.createElement('td');
          tdLabel.innerText = obs.label;
          if (obs.distance < 2.0) {
            tdLabel.className = 'yellow';
            tdLabel.style.color = '#ff3131';
          } else if (obs.distance < 3.5) {
            tdLabel.className = 'cyan';
          } else {
            tdLabel.className = 'green';
          }
          tr.appendChild(tdLabel);

          const tdDist = document.createElement('td');
          tdDist.innerText = `${obs.distance}m`;
          if (obs.distance < 2.0) {
            tdDist.style.color = '#ff3131';
            tdDist.style.fontWeight = 'bold';
          }
          tr.appendChild(tdDist);

          alertsBody.insertBefore(tr, alertsBody.firstChild);
          while (alertsBody.children.length > 5) {
            alertsBody.removeChild(alertsBody.lastChild);
          }
        }
      }
    });
  }
}

async function loadTelemetry() {
  try {
    const response = await fetch('/api/telemetry');
    if (!response.ok) return;
    const data = await response.json();
    updateDashboard(data);
  } catch (error) {
    console.warn('[FRONTEND] Telemetry fetch failed:', error);
    setStatus(false);
  }
}

updateClock();
setInterval(updateClock, 1000);
loadTelemetry();
setInterval(loadTelemetry, 1000);

// Initialize camera stream once on page load
window.addEventListener('DOMContentLoaded', () => {
  const cameraImg = document.getElementById('camera-feed');
  if (cameraImg) {
    cameraImg.src = '/video';
  }
});