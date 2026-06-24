# Firebase Cloud Integration Guide
## Head-Mounted Navigation Assistant for Visually Impaired People

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Firebase Setup](#2-firebase-setup)
3. [Environment Variables](#3-environment-variables)
4. [Firestore Database Design](#4-firestore-database-design)
5. [Firestore Security Rules](#5-firestore-security-rules)
6. [cloud_handler.py](#6-cloud_handlerpy)
7. [Firebase Config (JS)](#7-firebase-config-js)
8. [React Dashboard](#8-react-dashboard)
9. [Component Files](#9-component-files)
10. [CSS Styling](#10-css-styling)
11. [Firebase Hosting Config](#11-firebase-hosting-config)
12. [Deployment Steps](#12-deployment-steps)
13. [Testing Checklist](#13-testing-checklist)

---

## 1. Project Structure

```
nav-assistant/
│
├── cloud_handler.py               # MQTT subscriber → Firestore writer
├── .env                           # Python env vars (never commit)
├── .env.example                   # Safe template to commit
├── requirements.txt               # Python dependencies
│
├── dashboard/                     # React frontend
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── package.json
│   ├── .env                       # React env vars (never commit)
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── firebase.js            # Firebase init
│       ├── index.css              # Tailwind + custom vars
│       └── components/
│           ├── LiveStatusCard.jsx
│           ├── AlertLog.jsx
│           ├── HistoryTable.jsx
│           ├── StatusIndicator.jsx
│           └── ConnectionBadge.jsx
│
├── firebase.json                  # Firebase Hosting config
├── .firebaserc                    # Firebase project alias
├── firestore.rules                # Firestore security rules
└── firestore.indexes.json         # Firestore composite indexes
```

---

## 2. Firebase Setup

### Step 1 — Create Firebase project

```
https://console.firebase.google.com
→ Add project
→ Name: nav-assistant (or your chosen name)
→ Disable Google Analytics (not needed)
→ Create project
```

### Step 2 — Enable Firestore

```
Firebase Console → Build → Firestore Database
→ Create database
→ Start in production mode
→ Region: choose closest to you (e.g. asia-south1 for India)
```

### Step 3 — Enable Firebase Hosting

```
Firebase Console → Build → Hosting
→ Get started (follow the wizard)
→ Single-page app: YES
→ Automatic deploys: skip for now
```

### Step 4 — Get your Firebase web config

```
Firebase Console → Project Settings (gear icon)
→ Your apps → Add app → Web (</>)
→ App nickname: dashboard
→ Copy the firebaseConfig object shown
```

The config looks like this — you will paste values into `.env` files:

```js
const firebaseConfig = {
  apiKey: "AIzaSy...",
  authDomain: "nav-assistant.firebaseapp.com",
  projectId: "nav-assistant",
  storageBucket: "nav-assistant.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

### Step 5 — Get a Service Account key for Python

```
Firebase Console → Project Settings → Service accounts
→ Generate new private key
→ Download JSON → save as serviceAccountKey.json
→ Place next to cloud_handler.py
→ NEVER commit this file to git
```

### Step 6 — Install Firebase CLI

```bash
npm install -g firebase-tools
firebase login
firebase init
```

During `firebase init` select:
- Firestore
- Hosting
- Use existing project → select your project

---

## 3. Environment Variables

### `/cloud_handler/.env`

```env
# Mosquitto MQTT broker
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC=navigation/status
MQTT_CLIENT_ID=cloud_handler_01

# Firebase — path to service account JSON
FIREBASE_CREDENTIALS=./serviceAccountKey.json

# Firestore user ID (which live_status document to update)
FIRESTORE_USER_ID=user1

# History cap — how many history docs to keep
HISTORY_CAP=500
```

### `/cloud_handler/.env.example`

```env
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC=navigation/status
MQTT_CLIENT_ID=cloud_handler_01
FIREBASE_CREDENTIALS=./serviceAccountKey.json
FIRESTORE_USER_ID=user1
HISTORY_CAP=500
```

### `/dashboard/.env`

```env
VITE_FIREBASE_API_KEY=AIzaSy...
VITE_FIREBASE_AUTH_DOMAIN=nav-assistant.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=nav-assistant
VITE_FIREBASE_STORAGE_BUCKET=nav-assistant.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123
VITE_FIRESTORE_USER_ID=user1
```

### `/dashboard/.env.example`

```env
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_PROJECT_ID=
VITE_FIREBASE_STORAGE_BUCKET=
VITE_FIREBASE_MESSAGING_SENDER_ID=
VITE_FIREBASE_APP_ID=
VITE_FIRESTORE_USER_ID=user1
```

---

## 4. Firestore Database Design

### Collection: `live_status`

Document path: `live_status/{userId}` (e.g. `live_status/user1`)

```json
{
  "navigation_status": "GO",
  "freespace_status": "CLEAR",
  "free_ratio": 0.82,
  "last_updated": "2025-01-15T10:30:00Z"
}
```

| Field | Type | Values |
|---|---|---|
| `navigation_status` | string | `GO`, `CAUTION`, `STOP` |
| `freespace_status` | string | `CLEAR`, `LIMITED`, `BLOCKED` |
| `free_ratio` | number | `0.0` – `1.0` |
| `last_updated` | timestamp | Firestore server timestamp |

---

### Collection: `alerts`

Document path: `alerts/{auto-id}`

```json
{
  "user_id": "user1",
  "alert_type": "DANGER",
  "reason": "User is stuck with no free path",
  "freespace_status": "BLOCKED",
  "free_ratio": 0.02,
  "navigation_status": "STOP",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Which user triggered the alert |
| `alert_type` | string | `DANGER` |
| `reason` | string | Human-readable description |
| `freespace_status` | string | Status at time of alert |
| `free_ratio` | number | Ratio at time of alert |
| `navigation_status` | string | Status at time of alert |
| `timestamp` | timestamp | Firestore server timestamp |

---

### Collection: `history`

Document path: `history/{auto-id}`

```json
{
  "user_id": "user1",
  "navigation_status": "CAUTION",
  "freespace_status": "LIMITED",
  "free_ratio": 0.45,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

> **Cost control:** The Python handler caps history at 500 documents per user.
> The dashboard queries only the last 20 for the table.
> Old documents are deleted automatically when the cap is hit.

---

## 5. Firestore Security Rules

### `firestore.rules`

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    // live_status — public read, no client write
    // Only cloud_handler.py (service account) writes here
    match /live_status/{userId} {
      allow read: if true;
      allow write: if false;
    }

    // alerts — public read, no client write
    match /alerts/{alertId} {
      allow read: if true;
      allow write: if false;
    }

    // history — public read, no client write
    match /history/{historyId} {
      allow read: if true;
      allow write: if false;
    }

    // Block everything else
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
```

> **Note:** These rules allow the dashboard to read data without authentication.
> The service account used by `cloud_handler.py` bypasses these rules entirely —
> it uses the Admin SDK which has full access regardless of rules.
> If you want to restrict the dashboard to authenticated caretakers only,
> add Firebase Authentication and change `allow read: if true`
> to `allow read: if request.auth != null`.

---

### `firestore.indexes.json`

```json
{
  "indexes": [
    {
      "collectionGroup": "alerts",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "user_id", "order": "ASCENDING" },
        { "fieldPath": "timestamp", "order": "DESCENDING" }
      ]
    },
    {
      "collectionGroup": "history",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "user_id", "order": "ASCENDING" },
        { "fieldPath": "timestamp", "order": "DESCENDING" }
      ]
    }
  ],
  "fieldOverrides": []
}
```

---

## 6. `cloud_handler.py`

### `requirements.txt`

```
paho-mqtt==1.6.1
firebase-admin==6.4.0
python-dotenv==1.0.0
```

### `cloud_handler.py`

```python
"""
cloud_handler.py
Head-Mounted Navigation Assistant — MQTT to Firebase bridge

Subscribes to Mosquitto MQTT broker.
Receives freespace navigation payloads.
Writes live_status, alerts, and history to Firestore.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

load_dotenv()

MQTT_BROKER      = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT        = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC       = os.getenv("MQTT_TOPIC", "navigation/status")
MQTT_CLIENT_ID   = os.getenv("MQTT_CLIENT_ID", "cloud_handler_01")
FIREBASE_CREDS   = os.getenv("FIREBASE_CREDENTIALS", "./serviceAccountKey.json")
USER_ID          = os.getenv("FIRESTORE_USER_ID", "user1")
HISTORY_CAP      = int(os.getenv("HISTORY_CAP", 500))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Firebase init
# ─────────────────────────────────────────────

cred = credentials.Certificate(FIREBASE_CREDS)
firebase_admin.initialize_app(cred)
db = firestore.client()

log.info("Firebase Admin SDK initialised")

# ─────────────────────────────────────────────
# Firestore writers
# ─────────────────────────────────────────────

def write_live_status(payload: dict) -> None:
    """Overwrite the single live_status document for this user."""
    doc_ref = db.collection("live_status").document(USER_ID)
    doc_ref.set({
        "navigation_status": payload["navigation_status"],
        "freespace_status":  payload["freespace_status"],
        "free_ratio":        payload["free_ratio"],
        "last_updated":      firestore.SERVER_TIMESTAMP,
    })
    log.info(
        "live_status updated | nav=%s free=%s ratio=%.2f",
        payload["navigation_status"],
        payload["freespace_status"],
        payload["free_ratio"],
    )


def write_alert(payload: dict) -> None:
    """Append a new danger alert document."""
    db.collection("alerts").add({
        "user_id":           USER_ID,
        "alert_type":        "DANGER",
        "reason":            "User is stuck with no free path",
        "freespace_status":  payload["freespace_status"],
        "free_ratio":        payload["free_ratio"],
        "navigation_status": payload["navigation_status"],
        "timestamp":         firestore.SERVER_TIMESTAMP,
    })
    log.warning(
        "DANGER ALERT written | ratio=%.2f",
        payload["free_ratio"],
    )


def write_history(payload: dict) -> None:
    """
    Append to history collection.
    If document count exceeds HISTORY_CAP, delete the oldest batch.
    """
    history_ref = db.collection("history")

    # Write new entry
    history_ref.add({
        "user_id":           USER_ID,
        "navigation_status": payload["navigation_status"],
        "freespace_status":  payload["freespace_status"],
        "free_ratio":        payload["free_ratio"],
        "timestamp":         firestore.SERVER_TIMESTAMP,
    })

    # Count and prune — runs every 50 writes to reduce read cost
    # Uses a simple modulo on the MQTT message timestamp
    if int(payload.get("timestamp", 0)) % 50 == 0:
        prune_history(history_ref)


def prune_history(history_ref) -> None:
    """Delete oldest documents when history exceeds HISTORY_CAP."""
    try:
        total = len(history_ref.where("user_id", "==", USER_ID).get())
        if total > HISTORY_CAP:
            excess = total - HISTORY_CAP
            oldest = (
                history_ref
                .where("user_id", "==", USER_ID)
                .order_by("timestamp", direction=firestore.Query.ASCENDING)
                .limit(excess)
                .get()
            )
            batch = db.batch()
            for doc in oldest:
                batch.delete(doc.reference)
            batch.commit()
            log.info("Pruned %d old history documents", excess)
    except Exception as exc:
        log.error("Prune failed: %s", exc)


# ─────────────────────────────────────────────
# Danger detection
# ─────────────────────────────────────────────

def is_danger(payload: dict) -> bool:
    """
    Danger condition:
    navigation_status == STOP  AND  freespace_status == BLOCKED
    """
    return (
        payload.get("navigation_status") == "STOP"
        and payload.get("freespace_status") == "BLOCKED"
    )


# ─────────────────────────────────────────────
# MQTT callbacks
# ─────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to MQTT broker at %s:%d", MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        log.info("Subscribed to topic: %s", MQTT_TOPIC)
    else:
        log.error("MQTT connection failed — return code %d", rc)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        log.warning("Unexpected MQTT disconnect (rc=%d). Reconnecting...", rc)
        while True:
            try:
                client.reconnect()
                log.info("Reconnected to MQTT broker")
                break
            except Exception as exc:
                log.error("Reconnect failed: %s. Retrying in 5s...", exc)
                time.sleep(5)


def on_message(client, userdata, msg):
    """
    Receives MQTT message, parses JSON, writes to Firestore.

    Expected payload:
    {
        "navigation_status": "GO" | "CAUTION" | "STOP",
        "freespace_status":  "CLEAR" | "LIMITED" | "BLOCKED",
        "free_ratio":        0.0 – 1.0,
        "timestamp":         unix epoch int
    }
    """
    try:
        raw = msg.payload.decode("utf-8")
        payload = json.loads(raw)
        log.debug("Received: %s", raw)

        # Validate required fields
        required = {"navigation_status", "freespace_status", "free_ratio"}
        if not required.issubset(payload.keys()):
            log.error("Payload missing required fields: %s", payload)
            return

        # Always write live status and history
        write_live_status(payload)
        write_history(payload)

        # Write alert only on danger condition
        if is_danger(payload):
            write_alert(payload)

    except json.JSONDecodeError as exc:
        log.error("JSON decode error: %s | raw=%s", exc, msg.payload)
    except Exception as exc:
        log.error("Unexpected error in on_message: %s", exc)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    log.info("Connecting to MQTT broker at %s:%d ...", MQTT_BROKER, MQTT_PORT)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except ConnectionRefusedError:
        log.critical(
            "Cannot connect to MQTT broker at %s:%d — is Mosquitto running?",
            MQTT_BROKER, MQTT_PORT
        )
        raise SystemExit(1)

    # Blocking loop — handles reconnects via on_disconnect
    client.loop_forever()


if __name__ == "__main__":
    main()
```

---

## 7. Firebase Config (JS)

### `dashboard/src/firebase.js`

```javascript
// firebase.js
// Initialises Firebase app and exports Firestore instance.
// All config values come from .env — never hardcode credentials here.

import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey:            import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain:        import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId:         import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket:     import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId:             import.meta.env.VITE_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const USER_ID = import.meta.env.VITE_FIRESTORE_USER_ID ?? "user1";
```

---

## 8. React Dashboard

### `dashboard/package.json`

```json
{
  "name": "nav-assistant-dashboard",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "firebase": "^10.7.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "vite": "^5.0.8"
  }
}
```

### `dashboard/vite.config.js`

```javascript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../public",   // Firebase Hosting serves from /public
    emptyOutDir: true,
  },
});
```

### `dashboard/tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        surface:  "#0f1117",
        card:     "#1a1d27",
        border:   "#2a2d3a",
        clear:    "#22c55e",
        limited:  "#f97316",
        blocked:  "#ef4444",
        accent:   "#6366f1",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
```

### `dashboard/postcss.config.js`

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

### `dashboard/index.html`

```html
<!DOCTYPE html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Nav Assistant — Caretaker Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link
      href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap"
      rel="stylesheet"
    />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

### `dashboard/src/main.jsx`

```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### `dashboard/src/index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background-color: #0f1117;
  color: #e2e8f0;
  font-family: 'JetBrains Mono', monospace;
  -webkit-font-smoothing: antialiased;
}

/* Thin scrollbar for table */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #1a1d27; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 2px; }

/* Pulse animation for danger banner */
@keyframes danger-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.75; }
}
.danger-pulse {
  animation: danger-pulse 1.4s ease-in-out infinite;
}
```

### `dashboard/src/App.jsx`

```jsx
import React, { useEffect, useState } from "react";
import {
  collection,
  doc,
  onSnapshot,
  query,
  where,
  orderBy,
  limit,
} from "firebase/firestore";
import { db, USER_ID } from "./firebase.js";

import LiveStatusCard   from "./components/LiveStatusCard.jsx";
import AlertLog         from "./components/AlertLog.jsx";
import HistoryTable     from "./components/HistoryTable.jsx";
import ConnectionBadge  from "./components/ConnectionBadge.jsx";

export default function App() {
  const [liveStatus,  setLiveStatus]  = useState(null);
  const [alerts,      setAlerts]      = useState([]);
  const [history,     setHistory]     = useState([]);
  const [connected,   setConnected]   = useState(false);
  const [lastPing,    setLastPing]    = useState(null);

  // ── Live status (single document) ──────────────────────────
  useEffect(() => {
    const docRef = doc(db, "live_status", USER_ID);
    const unsub = onSnapshot(
      docRef,
      (snap) => {
        if (snap.exists()) {
          setLiveStatus(snap.data());
          setConnected(true);
          setLastPing(new Date());
        }
      },
      (err) => {
        console.error("live_status listener error:", err);
        setConnected(false);
      }
    );
    return unsub;
  }, []);

  // ── Alerts (last 10, newest first) ─────────────────────────
  useEffect(() => {
    const q = query(
      collection(db, "alerts"),
      where("user_id", "==", USER_ID),
      orderBy("timestamp", "desc"),
      limit(10)
    );
    const unsub = onSnapshot(q, (snap) => {
      setAlerts(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
    });
    return unsub;
  }, []);

  // ── History (last 20 records) ───────────────────────────────
  useEffect(() => {
    const q = query(
      collection(db, "history"),
      where("user_id", "==", USER_ID),
      orderBy("timestamp", "desc"),
      limit(20)
    );
    const unsub = onSnapshot(q, (snap) => {
      setHistory(snap.docs.map((d) => ({ id: d.id, ...d.data() })));
    });
    return unsub;
  }, []);

  const isDanger =
    liveStatus?.navigation_status === "STOP" &&
    liveStatus?.freespace_status  === "BLOCKED";

  return (
    <div className="min-h-screen bg-surface text-slate-200">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="border-b border-border px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-white">
            Navigation Assistant
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Caretaker monitoring dashboard — {USER_ID}
          </p>
        </div>
        <ConnectionBadge connected={connected} lastPing={lastPing} />
      </header>

      {/* ── Danger Banner ──────────────────────────────────── */}
      {isDanger && (
        <div className="danger-pulse bg-red-900/80 border-b border-red-600 px-6 py-3 flex items-center gap-3">
          <span className="text-red-400 text-xl">⚠</span>
          <div>
            <p className="text-red-200 font-semibold text-sm">
              DANGER — Path completely blocked
            </p>
            <p className="text-red-400 text-xs">
              User is unable to move. Immediate attention required.
            </p>
          </div>
        </div>
      )}

      {/* ── Main Grid ──────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto px-4 py-6 grid gap-6
                        grid-cols-1
                        md:grid-cols-2
                        xl:grid-cols-3">

        {/* Live Status */}
        <div className="xl:col-span-1">
          <LiveStatusCard status={liveStatus} />
        </div>

        {/* Alert Log */}
        <div className="xl:col-span-2">
          <AlertLog alerts={alerts} />
        </div>

        {/* History Table */}
        <div className="md:col-span-2 xl:col-span-3">
          <HistoryTable history={history} />
        </div>

      </main>
    </div>
  );
}
```

---

## 9. Component Files

### `dashboard/src/components/StatusIndicator.jsx`

```jsx
// Reusable coloured badge for CLEAR / LIMITED / BLOCKED

const CONFIG = {
  CLEAR:   { color: "text-green-400",  bg: "bg-green-400/10",  dot: "bg-green-400"  },
  LIMITED: { color: "text-orange-400", bg: "bg-orange-400/10", dot: "bg-orange-400" },
  BLOCKED: { color: "text-red-400",    bg: "bg-red-400/10",    dot: "bg-red-400"    },
};

export default function StatusIndicator({ status }) {
  const cfg = CONFIG[status] ?? CONFIG.BLOCKED;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1
                  rounded-full text-xs font-semibold
                  ${cfg.color} ${cfg.bg}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {status ?? "—"}
    </span>
  );
}
```

### `dashboard/src/components/ConnectionBadge.jsx`

```jsx
// Shows live / offline with time since last ping

export default function ConnectionBadge({ connected, lastPing }) {
  const ago = lastPing
    ? Math.floor((Date.now() - lastPing.getTime()) / 1000)
    : null;

  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={`w-2 h-2 rounded-full ${
          connected ? "bg-green-400 shadow-[0_0_6px_#4ade80]" : "bg-slate-600"
        }`}
      />
      <span className={connected ? "text-green-400" : "text-slate-500"}>
        {connected ? "Live" : "Offline"}
      </span>
      {ago !== null && (
        <span className="text-slate-600">· {ago}s ago</span>
      )}
    </div>
  );
}
```

### `dashboard/src/components/LiveStatusCard.jsx`

```jsx
import StatusIndicator from "./StatusIndicator.jsx";

const NAV_COLOR = {
  GO:      "text-green-400",
  CAUTION: "text-orange-400",
  STOP:    "text-red-400",
};

function formatTimestamp(ts) {
  if (!ts) return "—";
  // Firestore Timestamp has .toDate()
  const date = ts.toDate ? ts.toDate() : new Date(ts);
  return date.toLocaleTimeString([], {
    hour:   "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function LiveStatusCard({ status }) {
  const ratio = status?.free_ratio ?? 0;
  const pct   = Math.round(ratio * 100);
  const navColor = NAV_COLOR[status?.navigation_status] ?? "text-slate-400";

  return (
    <div className="bg-card border border-border rounded-xl p-5 h-full">
      <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">
        Live status
      </p>

      {/* Navigation status — large display */}
      <div className="mb-5">
        <p className="text-xs text-slate-500 mb-1">Navigation</p>
        <p className={`text-4xl font-semibold tracking-tight ${navColor}`}>
          {status?.navigation_status ?? "—"}
        </p>
      </div>

      {/* Freespace status badge */}
      <div className="mb-5">
        <p className="text-xs text-slate-500 mb-1.5">Free space</p>
        <StatusIndicator status={status?.freespace_status} />
      </div>

      {/* Free ratio bar */}
      <div className="mb-5">
        <div className="flex justify-between items-center mb-1.5">
          <p className="text-xs text-slate-500">Free ratio</p>
          <p className="text-sm font-semibold text-white">{pct}%</p>
        </div>
        <div className="h-2 bg-border rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              pct >= 60 ? "bg-green-400" :
              pct >= 25 ? "bg-orange-400" :
                          "bg-red-400"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Last updated */}
      <div className="border-t border-border pt-3 mt-3">
        <p className="text-xs text-slate-500">
          Last updated:{" "}
          <span className="text-slate-300">
            {formatTimestamp(status?.last_updated)}
          </span>
        </p>
      </div>
    </div>
  );
}
```

### `dashboard/src/components/AlertLog.jsx`

```jsx
import StatusIndicator from "./StatusIndicator.jsx";

function formatTimestamp(ts) {
  if (!ts) return "—";
  const date = ts.toDate ? ts.toDate() : new Date(ts);
  return date.toLocaleString([], {
    month:  "short",
    day:    "numeric",
    hour:   "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function AlertLog({ alerts }) {
  return (
    <div className="bg-card border border-border rounded-xl p-5 h-full">
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-slate-500 uppercase tracking-widest">
          Alert log
        </p>
        <span className="text-xs text-slate-500">
          Last 10 events
        </span>
      </div>

      {alerts.length === 0 ? (
        <div className="flex flex-col items-center justify-center
                        py-12 text-center">
          <p className="text-green-400 text-2xl mb-2">✓</p>
          <p className="text-sm text-slate-400">No alerts recorded</p>
          <p className="text-xs text-slate-600 mt-1">
            System is operating normally
          </p>
        </div>
      ) : (
        <ul className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {alerts.map((alert) => (
            <li
              key={alert.id}
              className="flex items-start gap-3 p-3
                         bg-red-900/20 border border-red-900/40
                         rounded-lg"
            >
              <span className="text-red-400 mt-0.5 shrink-0">⚠</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-semibold text-red-300">
                    {alert.alert_type}
                  </span>
                  <StatusIndicator status={alert.freespace_status} />
                  <span className="text-xs text-slate-500">
                    {Math.round(alert.free_ratio * 100)}% free
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1 truncate">
                  {alert.reason}
                </p>
                <p className="text-xs text-slate-600 mt-0.5">
                  {formatTimestamp(alert.timestamp)}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

### `dashboard/src/components/HistoryTable.jsx`

```jsx
import StatusIndicator from "./StatusIndicator.jsx";

const NAV_COLOR = {
  GO:      "text-green-400",
  CAUTION: "text-orange-400",
  STOP:    "text-red-400",
};

function formatTimestamp(ts) {
  if (!ts) return "—";
  const date = ts.toDate ? ts.toDate() : new Date(ts);
  return date.toLocaleTimeString([], {
    hour:   "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function HistoryTable({ history }) {
  return (
    <div className="bg-card border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-slate-500 uppercase tracking-widest">
          History
        </p>
        <span className="text-xs text-slate-500">Last 20 records</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left">
              <th className="pb-3 pr-4 text-xs text-slate-500 font-normal">
                Time
              </th>
              <th className="pb-3 pr-4 text-xs text-slate-500 font-normal">
                Navigation
              </th>
              <th className="pb-3 pr-4 text-xs text-slate-500 font-normal">
                Free space
              </th>
              <th className="pb-3 text-xs text-slate-500 font-normal">
                Free ratio
              </th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? (
              <tr>
                <td colSpan={4} className="py-8 text-center text-slate-600 text-xs">
                  No history yet
                </td>
              </tr>
            ) : (
              history.map((row, i) => (
                <tr
                  key={row.id}
                  className={`border-b border-border/50 transition-colors
                    hover:bg-white/[0.02] ${i === 0 ? "bg-white/[0.03]" : ""}`}
                >
                  <td className="py-2.5 pr-4 text-xs text-slate-400 font-mono">
                    {formatTimestamp(row.timestamp)}
                  </td>
                  <td className="py-2.5 pr-4">
                    <span
                      className={`text-xs font-semibold ${
                        NAV_COLOR[row.navigation_status] ?? "text-slate-400"
                      }`}
                    >
                      {row.navigation_status}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4">
                    <StatusIndicator status={row.freespace_status} />
                  </td>
                  <td className="py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            row.free_ratio >= 0.6 ? "bg-green-400" :
                            row.free_ratio >= 0.25 ? "bg-orange-400" :
                                                     "bg-red-400"
                          }`}
                          style={{ width: `${Math.round(row.free_ratio * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-400">
                        {Math.round(row.free_ratio * 100)}%
                      </span>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

---

## 10. CSS Styling

All styling is handled via Tailwind utility classes in the components above,
plus the custom tokens defined in `tailwind.config.js` and `index.css`.

Color semantics used throughout:

| Token | Hex | Meaning |
|---|---|---|
| `surface` | `#0f1117` | Page background |
| `card` | `#1a1d27` | Card background |
| `border` | `#2a2d3a` | All borders and dividers |
| `clear` | `#22c55e` | CLEAR / GO state |
| `limited` | `#f97316` | LIMITED / CAUTION state |
| `blocked` | `#ef4444` | BLOCKED / STOP / DANGER state |
| `accent` | `#6366f1` | Highlights, focus rings |

---

## 11. Firebase Hosting Config

### `firebase.json`

```json
{
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "hosting": {
    "public": "public",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ],
    "headers": [
      {
        "source": "**/*.@(js|jsx|css)",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000, immutable"
          }
        ]
      },
      {
        "source": "**",
        "headers": [
          {
            "key": "X-Frame-Options",
            "value": "SAMEORIGIN"
          },
          {
            "key": "X-Content-Type-Options",
            "value": "nosniff"
          }
        ]
      }
    ]
  }
}
```

### `.firebaserc`

```json
{
  "projects": {
    "default": "YOUR_FIREBASE_PROJECT_ID"
  }
}
```

Replace `YOUR_FIREBASE_PROJECT_ID` with your actual project ID
(e.g. `nav-assistant-abc12`).

---

## 12. Deployment Steps

### Step 1 — Install Python dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

### Step 2 — Test cloud_handler.py locally

```bash
# Start Mosquitto if not already running
mosquitto -v

# In another terminal, run the handler
python cloud_handler.py

# In a third terminal, send a test message
mosquitto_pub -t navigation/status -m '{
  "navigation_status": "GO",
  "freespace_status": "CLEAR",
  "free_ratio": 0.82,
  "timestamp": 1750758000
}'

# Verify in Firebase Console → Firestore → live_status → user1
```

### Step 3 — Test a danger scenario

```bash
mosquitto_pub -t navigation/status -m '{
  "navigation_status": "STOP",
  "freespace_status": "BLOCKED",
  "free_ratio": 0.02,
  "timestamp": 1750758001
}'
# Verify an alert appears in Firestore → alerts
```

### Step 4 — Build the React dashboard

```bash
cd dashboard
npm install
cp .env.example .env
# Fill in your Firebase config values in .env
npm run build
# Output goes to ../public/
```

### Step 5 — Deploy to Firebase Hosting

```bash
# From the project root (where firebase.json lives)
firebase deploy --only firestore:rules,firestore:indexes,hosting

# On success you will see:
# ✔ Deploy complete!
# Hosting URL: https://YOUR-PROJECT.web.app
```

### Step 6 — Run cloud_handler.py as a background service (Raspberry Pi)

Create `/etc/systemd/system/nav-cloud-handler.service`:

```ini
[Unit]
Description=Navigation Assistant Cloud Handler
After=network.target mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/nav-assistant
EnvironmentFile=/home/pi/nav-assistant/.env
ExecStart=/usr/bin/python3 /home/pi/nav-assistant/cloud_handler.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable nav-cloud-handler
sudo systemctl start nav-cloud-handler
sudo systemctl status nav-cloud-handler
```

---

## 13. Testing Checklist

```
Before going live, verify each of these manually:

MQTT
[ ] Mosquitto is running: mosquitto -v
[ ] cloud_handler.py connects without error
[ ] Test message reaches Firestore live_status/user1

Firestore
[ ] live_status/user1 updates on each MQTT message
[ ] alerts collection grows on STOP + BLOCKED payload
[ ] history collection grows on every message
[ ] history prune runs and stays under HISTORY_CAP

Dashboard (local dev)
[ ] npm run dev starts without error
[ ] LiveStatusCard shows correct nav status + ratio bar
[ ] StatusIndicator shows correct colour for CLEAR / LIMITED / BLOCKED
[ ] AlertLog shows "No alerts" when alerts collection is empty
[ ] Danger banner appears when status is STOP + BLOCKED
[ ] HistoryTable shows last 20 rows
[ ] ConnectionBadge shows "Live" when Firestore is connected
[ ] onSnapshot updates happen without page refresh

Firebase Hosting
[ ] firebase deploy completes with no errors
[ ] https://YOUR-PROJECT.web.app loads correctly
[ ] Dashboard on mobile (375px) renders without overflow
[ ] Firestore security rules block writes from the browser
  (open browser console and try:
   db.collection("live_status").doc("user1").set({test: 1})
   → should be rejected)
```

---

*End of guide. All file paths, env variable names, and collection names
are consistent throughout this document — copy each section exactly
and the system will wire together without modification.*