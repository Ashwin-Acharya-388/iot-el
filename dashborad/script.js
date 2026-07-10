/* ================================================================
   AI Navigation Command Center – Dashboard Script
   ================================================================ */

// ── Clock ─────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  let hours = now.getHours();
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12;
  document.getElementById('time').innerHTML = `${hours}:${minutes}:${seconds} ${ampm}`;
  document.getElementById('date').innerHTML = now.toDateString();
}

// ── State ─────────────────────────────────────────────────────────
let alertTracker = new Set();
let lastSpoken = '';
let activeLogFilter = 'all';
let logViewCleared = false;
let knownLogTimestamps = new Set();
let voiceEnabled = false;   // User must click once to enable (browser policy)

// ── Web Speech API (Laptop Speaker) ──────────────────────────────
const synth = window.speechSynthesis;

function speakOnLaptop(text) {
  if (!voiceEnabled || !synth) return;
  synth.cancel();                          // Stop any current speech
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate   = 1.1;                      // Slightly faster than default
  utter.pitch  = 1.0;
  utter.volume = 1.0;
  // Prefer a clear English voice if available
  const voices = synth.getVoices();
  const preferred = voices.find(v =>
    v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.localService)
  );
  if (preferred) utter.voice = preferred;
  synth.speak(utter);
}

function enableVoice() {
  voiceEnabled = true;
  const btn = document.getElementById('voice-enable-btn');
  if (btn) {
    btn.innerText = '🔊 Voice ON';
    btn.style.background = 'linear-gradient(135deg, #00e5ff33, #35ff4f33)';
    btn.style.borderColor = '#35ff4f';
    btn.style.color = '#35ff4f';
  }
  // Trigger a test utterance so browser unlocks audio
  speakOnLaptop('Voice enabled');
}

// ── Status helpers ────────────────────────────────────────────────
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

// ── Voice command formatting ──────────────────────────────────────
const DIR_PHRASES = {
  'FORWARD':      'Forward',
  'LEFT':         'Turn left',
  'RIGHT':        'Turn right',
  'SLIGHT LEFT':  'Slight left',
  'SLIGHT RIGHT': 'Slight right',
  'STOP':         'Stop',
};

function formatCommand(data) {
  const dir = (data.direction || 'FORWARD').toUpperCase().trim();
  const dirPhrase = DIR_PHRASES[dir] || data.direction || 'Forward';
  const hasObstacle = (data.obstacle_count || 0) > 0;
  if (hasObstacle) {
    return `${dirPhrase}. Obstacle ahead.`;
  }
  return `${dirPhrase}.`;
}

// ── Dashboard update ──────────────────────────────────────────────
function updateDashboard(data) {
  const dirElement = document.getElementById('val-direction');
  if (dirElement) dirElement.innerText = data.direction || 'FORWARD';

  const fpsElement = document.getElementById('val-fps');
  if (fpsElement) fpsElement.innerText = data.fps || '0';

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
    modelStatus.style.color = '#35ff4f';
  }

  setStatus(true);

  // Voice text panel
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
      speakOnLaptop(currentCommand);   // Plays through laptop speaker
      lastSpoken = currentCommand;
    }
  }

  // Alerts table
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
          // Always display generic "Obstacle" label
          tdLabel.innerText = 'Obstacle';
          if (obs.distance < 2.0) {
            tdLabel.style.color = '#ff3131';
            tdLabel.style.fontWeight = 'bold';
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

// ── Telemetry polling ─────────────────────────────────────────────
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

// ── Activity Log ──────────────────────────────────────────────────
function setLogFilter(btn, level) {
  document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  activeLogFilter = level;
  renderLogEntries(window._lastLogEntries || []);
}

function clearLogView() {
  logViewCleared = true;
  knownLogTimestamps.clear();
  const tbody = document.getElementById('log-body');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="4" style="color:rgba(0,229,255,0.25); text-align:center; padding:16px; font-size:12px;">Log view cleared. New entries will appear below.</td></tr>';
  }
}

function renderLogEntries(entries) {
  window._lastLogEntries = entries;
  const tbody = document.getElementById('log-body');
  if (!tbody) return;

  const filtered = activeLogFilter === 'all'
    ? entries
    : entries.filter(e => e.level === activeLogFilter);

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="color:rgba(0,229,255,0.3); text-align:center; padding:16px; font-size:12px;">No entries for this filter.</td></tr>';
    return;
  }

  // Only prepend rows that are genuinely new
  let addedCount = 0;
  const fragment = document.createDocumentFragment();

  for (const entry of filtered) {
    const key = `${entry.timestamp}|${entry.message}`;
    if (knownLogTimestamps.has(key)) continue;
    if (logViewCleared) {
      // After a clear, only show entries added after the clear (approximate)
      knownLogTimestamps.add(key);
      continue; // Skip pre-clear entries silently
    }
    knownLogTimestamps.add(key);

    const tr = document.createElement('tr');
    if (entry.level === 'warn')   tr.className = 'log-row-warn';
    if (entry.level === 'danger') tr.className = 'log-row-danger';

    const tdTs = document.createElement('td');
    tdTs.innerText = entry.timestamp;
    tr.appendChild(tdTs);

    const tdType = document.createElement('td');
    tdType.innerText = entry.type;
    tr.appendChild(tdType);

    const tdLevel = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `log-badge ${entry.level}`;
    badge.innerText = entry.level.toUpperCase();
    tdLevel.appendChild(badge);
    tr.appendChild(tdLevel);

    const tdMsg = document.createElement('td');
    tdMsg.innerText = entry.message;
    tr.appendChild(tdMsg);

    fragment.appendChild(tr);
    addedCount++;
  }

  if (addedCount > 0) {
    // Clear placeholder if present
    if (tbody.querySelector('td[colspan]') && tbody.children.length === 1) {
      tbody.innerHTML = '';
    }
    tbody.insertBefore(fragment, tbody.firstChild);

    // Limit displayed rows to 80
    while (tbody.children.length > 80) {
      tbody.removeChild(tbody.lastChild);
    }
  }

  // After a clear, reset the flag so future entries show
  if (logViewCleared) logViewCleared = false;
}

async function loadActivityLog() {
  try {
    const res = await fetch('/api/logs?limit=50');
    if (!res.ok) return;
    const entries = await res.json();
    renderLogEntries(entries);
  } catch (e) {
    // Silently ignore
  }
}

// ── Camera stream ─────────────────────────────────────────────────
const cameraImg = document.getElementById('camera-feed');
if (cameraImg) {
  cameraImg.src = '/video';
}

// ── Bluetooth status (via server-side /api/bt-status) ────────────
async function checkBluetooth() {
  const dot = document.getElementById('bt-dot');
  const statusText = document.getElementById('bt-status-text');
  const deviceName = document.getElementById('bt-device-name');
  const audioStatus = document.getElementById('status-audio');

  try {
    const res = await fetch('/api/bt-status');
    if (!res.ok) throw new Error('no endpoint');
    const data = await res.json();

    if (data.connected) {
      if (dot) { dot.classList.add('connected'); }
      if (statusText) statusText.innerText = 'Connected';
      if (deviceName) deviceName.innerText = data.device || 'Bluetooth Device';
      if (audioStatus) { audioStatus.innerText = 'BT Connected'; audioStatus.style.color = '#35ff4f'; }
    } else {
      if (dot) dot.classList.remove('connected');
      if (statusText) statusText.innerText = data.message || 'Not Connected';
      if (deviceName) deviceName.innerText = '—';
      if (audioStatus) { audioStatus.innerText = 'BT Offline'; audioStatus.style.color = '#ffaa35'; }
    }
  } catch (_) {
    // BT endpoint not available (Windows dev), show graceful fallback
    if (dot) dot.classList.remove('connected');
    if (statusText) statusText.innerText = 'N/A (RPi only)';
    if (deviceName) deviceName.innerText = '—';
    if (audioStatus) { audioStatus.innerText = 'N/A'; audioStatus.style.color = 'rgba(0,229,255,0.4)'; }
  }
}

// ── MQTT Safety Alert System ──────────────────────────────────────
function showDangerToast(reason) {
  const existing = document.getElementById('danger-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'danger-toast';
  toast.innerHTML = `
    <div class="danger-toast-content">
      <span class="warning-icon">⚠️</span>
      <div class="danger-toast-text">
        <strong>SAFETY CRITICAL ALERT:</strong>
        <p>${reason}</p>
      </div>
      <button onclick="document.getElementById('danger-toast').remove()">DISMISS</button>
    </div>
  `;
  document.body.appendChild(toast);
}

function triggerDangerAlert(payload) {
  const container = document.querySelector('.container');
  if (container) {
    container.classList.add('flash-danger');
    setTimeout(() => container.classList.remove('flash-danger'), 3000);
  }

  showDangerToast(payload.reason);

  const alertsBody = document.getElementById('alerts-body');
  if (alertsBody) {
    const tr = document.createElement('tr');
    tr.style.backgroundColor = 'rgba(255, 49, 49, 0.15)';
    tr.style.border = '1px solid #ff3131';

    const tdTime = document.createElement('td');
    const now = new Date(payload.timestamp || new Date());
    tdTime.innerText = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    tr.appendChild(tdTime);

    const tdLabel = document.createElement('td');
    tdLabel.innerHTML = '🚨 <span style="color:#ff3131; font-weight:bold; text-shadow:0 0 10px #ff3131;">DANGER</span>';
    tr.appendChild(tdLabel);

    const tdDist = document.createElement('td');
    tdDist.innerText = 'STUCK';
    tdDist.style.color = '#ff3131';
    tdDist.style.fontWeight = 'bold';
    tr.appendChild(tdDist);

    alertsBody.insertBefore(tr, alertsBody.firstChild);
    while (alertsBody.children.length > 5) {
      alertsBody.removeChild(alertsBody.lastChild);
    }
  }
}

async function initMQTT() {
  try {
    const res = await fetch('/api/mqtt-config');
    if (!res.ok) return;
    const mqttConfig = await res.json();

    console.log('[MQTT] Initializing connection to:', mqttConfig.broker, mqttConfig.port);
    const clientId = 'caretaker_dash_' + Math.random().toString(16).substr(2, 8);
    const client = new Paho.MQTT.Client(mqttConfig.broker, Number(mqttConfig.port), '/mqtt', clientId);

    client.onConnectionLost = (responseObject) => {
      console.warn('[MQTT] Connection lost:', responseObject.errorMessage);
      setTimeout(initMQTT, 5000);
    };

    client.onMessageArrived = (message) => {
      try {
        const payload = JSON.parse(message.payloadString);
        if (payload.status === 'DANGER') {
          console.log('[MQTT] Danger alert received:', payload);
          triggerDangerAlert(payload);
        }

        if (message.destinationName === (mqttConfig.health_topic || "iot/navigation/health")) {
          const tempEl = document.getElementById('health-temp');
          const cpuEl = document.getElementById('health-cpu');
          const memEl = document.getElementById('health-mem');
          
          if (tempEl) {
            tempEl.innerText = payload.cpu_temp + "°C";
            tempEl.style.color = payload.cpu_temp > 75 ? "#ff3131" : (payload.cpu_temp > 60 ? "#ffaa35" : "#35ff4f");
          }
          if (cpuEl) {
            cpuEl.innerText = payload.cpu_usage + "%";
            cpuEl.style.color = payload.cpu_usage > 85 ? "#ff3131" : (payload.cpu_usage > 60 ? "#ffaa35" : "#35ff4f");
          }
          if (memEl) {
            memEl.innerText = payload.memory_usage + "%";
            memEl.style.color = payload.memory_usage > 85 ? "#ff3131" : (payload.memory_usage > 65 ? "#ffaa35" : "#35ff4f");
          }
        }
      } catch (e) {
        console.error('[MQTT] Error parsing message:', e);
      }
    };

    const isSecurePort = Number(mqttConfig.port) === 8084;
    client.connect({
      onSuccess: () => {
        console.log('[MQTT] Connected! Subscribing to:', mqttConfig.alerts_topic);
        client.subscribe(mqttConfig.alerts_topic);
        client.subscribe(healthTopic);
      },
      onFailure: (err) => {
        console.error('[MQTT] Connection failed:', err);
        setTimeout(initMQTT, 5000);
      },
      useSSL: window.location.protocol === 'https:' || isSecurePort
    });
  } catch (err) {
    console.warn('[MQTT] Setup failed:', err);
  }
}

// ── Init ──────────────────────────────────────────────────────────
updateClock();
setInterval(updateClock, 1000);

loadTelemetry();
setInterval(loadTelemetry, 1000);

loadActivityLog();
setInterval(loadActivityLog, 3000);

checkBluetooth();
setInterval(checkBluetooth, 10000);

initMQTT();

// ── Voice Enable Button (browser requires user gesture) ───────────
// Inject a floating enable button if not already in HTML
if (!document.getElementById('voice-enable-btn')) {
  const btn = document.createElement('button');
  btn.id = 'voice-enable-btn';
  btn.innerText = '🔇 Click to Enable Voice';
  btn.title = 'Enable laptop speaker voice commands';
  Object.assign(btn.style, {
    position:     'fixed',
    bottom:       '24px',
    right:        '24px',
    zIndex:       '9999',
    padding:      '10px 18px',
    borderRadius: '30px',
    border:       '1px solid #00e5ff',
    background:   'rgba(0,0,0,0.7)',
    color:        '#00e5ff',
    fontFamily:   'inherit',
    fontSize:     '13px',
    cursor:       'pointer',
    backdropFilter: 'blur(8px)',
    transition:   'all 0.3s ease',
  });
  btn.onclick = enableVoice;
  document.body.appendChild(btn);
}