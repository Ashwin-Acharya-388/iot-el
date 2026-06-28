"""
firebase_cloud.py
=================
Firebase cloud integration for the IoT Navigation Assistant.
Pushes real-time telemetry, danger alerts, and activity logs to Firestore
so a caretaker can monitor from anywhere via the hosted caretaker portal.

Firestore Schema (matches manual schema):
    live_status / user1          ← live telemetry (overwritten every ~5s)
    alerts      / <auto-id>      ← append-only danger/warning events
    history     / <auto-id>      ← activity log (warn/danger only)

Usage in app.py:
    import firebase_cloud
    firebase_cloud.initialize()
    firebase_cloud.push_telemetry({...})
    firebase_cloud.push_alert("DANGER", "Path Blocked", 0.03)
    firebase_cloud.push_log({"type": "NAV", "message": "...", "level": "danger"})
"""

import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = os.getenv(
    "FIREBASE_CREDENTIALS",
    str(BASE_DIR / "firebase-service-account.json")
)

# Firestore collection / document names (match your manual schema)
COLLECTION_LIVE    = "telemetry"       # Collection ID
DOCUMENT_USER      = "live"            # Document ID inside live_status
COLLECTION_ALERTS  = "alerts"          # alerts collection
COLLECTION_HISTORY = "activity_logs"   # history / activity log collection

import yaml

# Read settings to get telemetry_interval_sec
try:
    with open(BASE_DIR / "config" / "settings.yaml", "r") as f:
        _cfg = yaml.safe_load(f)
        TELEMETRY_MIN_INTERVAL = float(_cfg.get("firebase", {}).get("telemetry_interval_sec", 5.0))
        FRAME_MIN_INTERVAL = float(_cfg.get("firebase", {}).get("frame_interval_sec", 0.2))
except Exception:
    TELEMETRY_MIN_INTERVAL = 5.0
    FRAME_MIN_INTERVAL = 0.2

# Separate Firestore document for fast camera frame streaming
DOCUMENT_FRAME = "camera_frame"

# ── Internal state ──────────────────────────────────────────────────────────
_db              = None
_initialized     = False
_last_telem_push = 0.0
_last_frame_push = 0.0
_push_lock       = threading.Lock()
_frame_lock      = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize() -> bool:
    """
    Initialize Firebase Admin SDK.
    Returns True if successful, False if credentials are missing/invalid.
    Call once at app startup.
    """
    global _db, _initialized

    if _initialized:
        return _db is not None
    _initialized = True

    if not os.path.exists(CREDENTIALS_FILE):
        print(
            f"[FIREBASE] ⚠  Service account not found at '{CREDENTIALS_FILE}'.\n"
            "           Firebase cloud sync disabled.\n"
            "           Follow FIREBASE_SETUP_GUIDE.md to enable."
        )
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore as fs

        if not firebase_admin._apps:
            cred = credentials.Certificate(CREDENTIALS_FILE)
            firebase_admin.initialize_app(cred)

        _db = fs.client()
        print("[FIREBASE] ✓ Connected to Firestore — caretaker portal sync active.")
        return True

    except Exception as exc:
        print(f"[FIREBASE] ✗ Initialization failed: {exc}")
        _db = None
        return False


# ── Public API ───────────────────────────────────────────────────────────────

def push_telemetry(telemetry: dict) -> None:
    """
    Overwrite live_status/user1 with the latest navigation state.
    Throttled to at most once every TELEMETRY_MIN_INTERVAL seconds.

    Maps internal app.py telemetry keys → Firestore field names:
        direction       → navigation_status   (e.g. "FORWARD", "STOP")
        status.camera   → freespace_status    (e.g. "CLEAR", "BLOCKED")
        zone_free_ratio → free_ratio          (0.0 – 1.0)
        fps             → fps
        timestamp       → last_updated
    """
    global _last_telem_push

    if _db is None:
        return

    now = time.time()
    with _push_lock:
        if now - _last_telem_push < TELEMETRY_MIN_INTERVAL:
            return
        _last_telem_push = now

    direction = telemetry.get("direction", "UNKNOWN").upper()

    # Derive freespace_status from camera status and obstacle data
    obs_count = telemetry.get("obstacle_count", 0)
    closest   = None
    if telemetry.get("obstacles"):
        closest = telemetry["obstacles"][0].get("distance")

    if direction == "STOP" or (closest is not None and closest < 1.5):
        freespace_status = "BLOCKED"
    elif obs_count > 0:
        freespace_status = "CAUTION"
    else:
        freespace_status = "CLEAR"

    # free_ratio: percentage of walkable area (0.0 to 1.0)
    # Estimated from obstacle count; real value comes from freespace model
    free_ratio = telemetry.get("free_ratio", None)
    if free_ratio is None:
        if freespace_status == "BLOCKED":
            free_ratio = 0.03
        elif freespace_status == "CAUTION":
            free_ratio = 0.45
        else:
            free_ratio = 0.85

    payload = {
        # ── Fields matching your Firestore schema ──
        "navigation_status": direction,
        "freespace_status":  freespace_status,
        "free_ratio":        round(float(free_ratio), 3),
        "fps":               telemetry.get("fps", 0.0),
        "last_updated":      _utc_now(),
        # ── Extra context fields ──
        "obstacle_count":    obs_count,
        "camera_status":     telemetry.get("status", {}).get("camera", "Unknown"),
        "model_status":      telemetry.get("status", {}).get("model", "Unknown"),
        "closest_obstacle_m": closest,
        "server_status":     telemetry.get("status", {}).get("server", "Active"),
    }

    threading.Thread(target=_write_live, args=(payload,), daemon=True).start()


def push_alert(alert_type: str, reason: str, free_ratio: float = 0.0) -> None:
    """
    Append a danger/warning event to the 'alerts' collection.

    Args:
        alert_type:  e.g. "DANGER", "WARNING"
        reason:      e.g. "Path Blocked", "Obstacle at 0.8m"
        free_ratio:  walkable area ratio at time of alert (0.0–1.0)
    """
    if _db is None:
        return

    payload = {
        # ── Fields matching your Firestore schema ──
        "alert_type":  alert_type.upper(),
        "reason":      reason,
        "free_ratio":  round(float(free_ratio), 3),
        # ── Extra context ──
        "timestamp":   _utc_now(),
    }
    threading.Thread(target=_write_alert, args=(payload,), daemon=True).start()


def push_log(entry: dict) -> None:
    """
    Mirror a warn/danger activity log entry to the 'history' collection.
    Info-level entries are skipped to avoid Firestore write quota exhaustion.

    Args:
        entry: dict with keys: timestamp, type, message, level
    """
    if _db is None:
        return

    level = entry.get("level", "info")
    if level == "info":
        return  # Only mirror warn/danger to Firestore

    # Map to navigation_status / freespace_status for consistency
    message = entry.get("message", "")
    nav_status = "STOP" if "STOP" in message.upper() else entry.get("type", "SYSTEM")
    fs_status  = "BLOCKED" if level == "danger" else "CAUTION"

    payload = {
        # ── Fields matching your Firestore schema ──
        "navigation_status": nav_status,
        "freespace_status":  fs_status,
        "free_ratio":        0.03 if level == "danger" else 0.35,
        # ── Extra context ──
        "event_type":        entry.get("type", "SYSTEM"),
        "message":           message,
        "level":             level,
        "timestamp":         entry.get("timestamp", _utc_now()),
    }
    threading.Thread(target=_write_history, args=(payload,), daemon=True).start()


def push_frame(base64_frame: str, fps: float = 0.0) -> None:
    """
    Push a camera frame to a separate Firestore document for fast streaming.
    Throttled independently from telemetry at FRAME_MIN_INTERVAL.
    """
    global _last_frame_push

    if _db is None or not base64_frame:
        return

    now = time.time()
    with _frame_lock:
        if now - _last_frame_push < FRAME_MIN_INTERVAL:
            return
        _last_frame_push = now

    payload = {
        "frame": base64_frame,
        "fps": round(float(fps), 1),
        "timestamp": _utc_now(),
    }
    threading.Thread(target=_write_frame, args=(payload,), daemon=True).start()


# ── Internal write helpers ───────────────────────────────────────────────────

def _write_frame(payload: dict) -> None:
    try:
        _db.collection(COLLECTION_LIVE).document(DOCUMENT_FRAME).set(payload)
    except Exception as exc:
        print(f"[FIREBASE] Frame push error: {exc}")


def _write_live(payload: dict) -> None:
    try:
        _db.collection(COLLECTION_LIVE).document(DOCUMENT_USER).set(payload)
        print(f"[FIREBASE] ↑ Telemetry: {payload['navigation_status']} | "
              f"free={payload['free_ratio']:.2f} | fps={payload['fps']}")
    except Exception as exc:
        print(f"[FIREBASE] Telemetry push error: {exc}")


def _write_alert(payload: dict) -> None:
    try:
        _db.collection(COLLECTION_ALERTS).add(payload)
        print(f"[FIREBASE] ⚠ Alert pushed: {payload['alert_type']} — {payload['reason']}")
    except Exception as exc:
        print(f"[FIREBASE] Alert push error: {exc}")


def _write_history(payload: dict) -> None:
    try:
        _db.collection(COLLECTION_HISTORY).add(payload)
    except Exception as exc:
        print(f"[FIREBASE] History push error: {exc}")
