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

setInterval(updateClock, 1000);
updateClock();

// Connect to Socket.IO server
const socket = io();

socket.on('connect', () => {
  console.log('[FRONTEND] Connected to AI Navigation Server');
  const serverStatus = document.getElementById('status-server');
  if (serverStatus) {
    serverStatus.innerText = 'Active';
    serverStatus.style.color = '#35ff4f';
  }
});

socket.on('disconnect', () => {
  console.log('[FRONTEND] Disconnected from AI Navigation Server');
  const serverStatus = document.getElementById('status-server');
  if (serverStatus) {
    serverStatus.innerText = 'Offline';
    serverStatus.style.color = '#ff3535';
  }
});

let alertTracker = new Set();

socket.on('navigation_data', (data) => {
  // 1. Update Camera Stream
  const cameraImg = document.getElementById('camera-feed');
  if (cameraImg && data.frame) {
    cameraImg.src = data.frame;
  }

  // 2. Update HUD metrics
  const dirElement = document.getElementById('val-direction');
  if (dirElement) dirElement.innerText = data.direction;

  const fpsElement = document.getElementById('val-fps');
  if (fpsElement) fpsElement.innerText = data.fps;

  const countElement = document.getElementById('val-obstacle-count');
  if (countElement) countElement.innerText = data.obstacle_count;

  // 3. Update Detailed system status
  const camStatus = document.getElementById('status-camera');
  if (camStatus) {
    camStatus.innerText = data.status.camera;
    camStatus.style.color = data.status.camera === 'Connected' ? '#35ff4f' : '#ffaa35';
  }

  const modelStatus = document.getElementById('status-model');
  if (modelStatus) {
    modelStatus.innerText = data.status.model;
    modelStatus.style.color = data.status.model === 'Running' ? '#35ff4f' : '#ffaa35';
  }

  const serverStatus = document.getElementById('status-server');
  if (serverStatus) {
    serverStatus.innerText = data.status.server;
    serverStatus.style.color = data.status.server === 'Active' ? '#35ff4f' : '#ffaa35';
  }

  // Status indicator LED color matching
  const circle = document.getElementById('status-circle');
  if (circle) {
    const active = data.status.camera === 'Connected' && data.status.model === 'Running';
    circle.style.backgroundColor = active ? '#35ff4f' : '#ffaa35';
    circle.style.boxShadow = active ? '0 0 15px #35ff4f' : '0 0 15px #ffaa35';
  }

  // 4. Update Voice output log
  const voiceMain = document.getElementById('voice-main');
  const voiceHistory = document.getElementById('voice-history');
  
  if (voiceMain && voiceHistory) {
    let currentCommand = data.direction;
    if (data.obstacles && data.obstacles.length > 0) {
      // Find the closest obstacle to list in the directional announcement
      const closest = data.obstacles.reduce((prev, curr) => prev.distance < curr.distance ? prev : curr);
      currentCommand = `${data.direction} • ${closest.label} ${closest.distance}m`;
    }

    if (voiceMain.innerText !== currentCommand) {
      voiceMain.innerText = currentCommand;

      const p = document.createElement('p');
      p.innerText = currentCommand;
      if (data.direction === 'STOP') {
        p.className = 'danger';
      }
      
      voiceHistory.insertBefore(p, voiceHistory.firstChild);

      // Clean old logs
      while (voiceHistory.children.length > 3) {
        voiceHistory.removeChild(voiceHistory.lastChild);
      }
    }
  }

  // 5. Update Recent Alerts list dynamically
  const alertsBody = document.getElementById('alerts-body');
  if (alertsBody && data.obstacles) {
    data.obstacles.forEach((obs) => {
      // Only trigger alerts for close items (< 4.5m)
      if (obs.distance < 4.5) {
        const alertKey = `${obs.label}_${obs.distance.toFixed(0)}`;
        if (!alertTracker.has(alertKey)) {
          alertTracker.add(alertKey);
          
          // Clean up track set occasionally
          if (alertTracker.size > 20) alertTracker.clear();

          const now = new Date();
          const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          
          const tr = document.createElement('tr');
          
          const tdTime = document.createElement('td');
          tdTime.innerText = timeStr;
          tr.appendChild(tdTime);

          const tdLabel = document.createElement('td');
          tdLabel.innerText = obs.label;
          
          // Color based on safety threshold
          if (obs.distance < 2.0) {
            tdLabel.className = 'yellow';
            tdLabel.style.color = '#ff3131'; // Warning alert
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

          // Keep table size clean
          while (alertsBody.children.length > 5) {
            alertsBody.removeChild(alertsBody.lastChild);
          }
        }
      }
    });
  }
});