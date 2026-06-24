# 🔥 Firebase Setup Guide

Complete step-by-step guide to connect your IoT Navigation Assistant to Firebase Cloud
so a caretaker can monitor from **anywhere in the world**.

---

## Overview

After completing this guide:
- ✅ The Raspberry Pi pushes **live telemetry** (direction, obstacles, FPS) to Firestore every ~5s
- ✅ **Danger alerts** (obstacle too close) are pushed instantly to Firestore
- ✅ **Warn/Danger logs** are mirrored to Firestore
- ✅ The caretaker opens a **public hosted URL** (e.g. `https://your-project.web.app`) and sees everything in real time — no login needed

---

## PART 1 — Create a Firebase Project (5 minutes)

### Step 1: Create the project

1. Go to **https://console.firebase.google.com**
2. Click **"Add project"**
3. Enter a project name, e.g. `iot-navigation-guardian`
4. Disable Google Analytics (not needed) → **Create project**
5. Wait ~30 seconds for the project to be created → **Continue**

---

### Step 2: Enable Firestore Database

1. In the left sidebar → **Build → Firestore Database**
2. Click **"Create database"**
3. Choose **"Start in test mode"** (we'll tighten rules later)
4. Select a region close to you (e.g. `asia-south1` for India)
5. Click **"Enable"**

> ⚠️ **Important**: Test mode allows all reads/writes for 30 days. After testing,
> tighten the rules using the template at the bottom of this guide.

---

### Step 3: Get the Server Service Account (for Raspberry Pi)

1. In the left sidebar → **Project Settings** (gear icon ⚙️)
2. Click the **"Service accounts"** tab
3. Click **"Generate new private key"** → **"Generate key"**
4. A JSON file downloads — **this is your `firebase-service-account.json`**
5. **Copy it** into your `iot-el/` project folder:

```bash
cp ~/Downloads/iot-navigation-guardian-*.json /path/to/iot-el/firebase-service-account.json
```

> 🔒 **Security**: Never commit this file to Git. It's already in `.gitignore`.
> Keep it only on the Raspberry Pi.

---

### Step 4: Get the Web App Config (for Caretaker Portal)

1. Still in **Project Settings** → click the **"General"** tab
2. Scroll down to **"Your apps"** → click the **`</>`** (Web) icon
3. Enter app nickname: `caretaker-portal` → **Register app**
4. You'll see a `firebaseConfig` object like this:

```js
const firebaseConfig = {
  apiKey: "AIzaSy...",
  authDomain: "iot-navigation-guardian.firebaseapp.com",
  projectId: "iot-navigation-guardian",
  storageBucket: "iot-navigation-guardian.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

5. Open `caretaker-portal/firebase-config.js` and paste these values:

```js
window.FIREBASE_CONFIG = {
  apiKey:            "AIzaSy...",          // ← paste yours
  authDomain:        "iot-navigation-guardian.firebaseapp.com",
  projectId:         "iot-navigation-guardian",
  storageBucket:     "iot-navigation-guardian.appspot.com",
  messagingSenderId: "123456789",
  appId:             "1:123456789:web:abc123"
};
```

---

## PART 2 — Configure the Raspberry Pi (2 minutes)

### Step 5: Verify the service account file is in place

```bash
ls -la /path/to/iot-el/firebase-service-account.json
```

The file should exist. If not, redo Step 3.

### Step 6: Install firebase-admin (if not already installed)

```bash
cd /path/to/iot-el
pip install firebase-admin>=6.5.0
# or if using the venv:
source venv/bin/activate
pip install firebase-admin>=6.5.0
```

### Step 7: Run the app and verify Firebase is connecting

```bash
python app.py
```

Look for these log lines at startup:
```
[FIREBASE] ✓ Connected to Firestore. Caretaker portal sync active.
```

And every ~5 seconds during navigation:
```
[FIREBASE] Telemetry pushed: FORWARD | 0 obstacles
```

If you see:
```
[FIREBASE] ⚠  Service account not found at 'firebase-service-account.json'
```
→ Check the file path in `config/settings.yaml` → `firebase.credentials`

---

## PART 3 — Deploy the Caretaker Portal (3 minutes)

### Step 8: Install Firebase CLI

```bash
npm install -g firebase-tools
# or use npx (no install needed):
npx firebase-tools --version
```

### Step 9: Login and deploy

```bash
cd /path/to/iot-el/caretaker-portal

# Login to Firebase
npx firebase-tools login

# Set your project
npx firebase-tools use YOUR_PROJECT_ID

# Deploy to Firebase Hosting
npx firebase-tools deploy --only hosting
```

After deploy completes, you'll see:
```
✔  Deploy complete!
Hosting URL: https://your-project-id.web.app
```

### Step 10: Share the URL with the caretaker

Send the URL (e.g. `https://iot-navigation-guardian.web.app`) to the caretaker.
They can open it on **any device, anywhere** — phone, tablet, laptop.

---

## PART 4 — Tighten Firestore Security Rules (Recommended)

After testing, replace the default rules with these in
**Firebase Console → Firestore → Rules**:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    // Telemetry: RPi writes (server-side), anyone can read
    match /telemetry/{document} {
      allow read: if true;
      allow write: if false;  // Only server (Admin SDK) can write
    }

    // Alerts: RPi writes, anyone can read
    match /alerts/{document} {
      allow read: if true;
      allow write: if false;
    }

    // Activity logs: RPi writes, anyone can read
    match /activity_logs/{document} {
      allow read: if true;
      allow write: if false;
    }
  }
}
```

> Note: `allow write: if false` means only the **Firebase Admin SDK** (your RPi)
> can write. The caretaker browser can only read.

---

## PART 5 — What the Caretaker Sees

The portal at `https://your-project.web.app` shows:

| Feature | Details |
|---------|---------|
| **Navigation Direction** | FORWARD / STOP / SLIGHT LEFT / SLIGHT RIGHT — color coded |
| **Obstacle Count** | How many obstacles are detected |
| **Closest Obstacle** | Distance in meters |
| **System Status** | Camera / Model / Server status |
| **FPS** | Processing speed |
| **Live Alerts Feed** | Danger alerts with timestamps |
| **Activity Log** | All warn/danger events from the device |
| **Last Updated** | "Updated 3s ago" — so caretaker knows if device is offline |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Service account not found` | Check the JSON file is in `iot-el/` and matches `config/settings.yaml → firebase.credentials` |
| `Permission denied` on Firestore | Firestore rules are too strict — use test mode rules during development |
| Portal shows "Connecting…" forever | Check your `firebase-config.js` values are correct (no typos in projectId) |
| Portal shows "Demo Mode" | `firebase-config.js` still has `YOUR_API_KEY` — paste real values |
| No data in Firestore | Make sure `app.py` is running on RPi and you see `[FIREBASE] ✓ Connected` |

---

## File Summary

```
iot-el/
├── firebase_cloud.py              ← RPi Firebase push module (NEW)
├── firebase-service-account.json ← Your secret key (NEVER commit to Git)
├── config/settings.yaml          ← Firebase config section added
└── caretaker-portal/
    ├── index.html                 ← Caretaker web portal
    ├── style.css                  ← Premium dark UI
    ├── firebase-config.js         ← YOUR WEB CONFIG GOES HERE
    ├── firebase.json              ← Firebase Hosting config
    └── .firebaserc                ← Firebase project alias
```
