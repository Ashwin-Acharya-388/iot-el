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

# ── Write rate limits (tuned for Firebase Spark free tier and user requirements) ──
TELEMETRY_MIN_INTERVAL = 1.0   # Push telemetry once every 1 second max
FRAME_MIN_INTERVAL     = 5.0   # Upload camera frames once every 5 seconds max
HEALTH_MIN_INTERVAL    = 5.0   # Push health data once every 5 seconds max

# ── Circuit breaker: disable writes after repeated 429s ──────────────────────
_CIRCUIT_BREAKER_THRESHOLD = 3       # trips after this many consecutive 429s
_CIRCUIT_BREAKER_COOLDOWN  = 300.0   # 5 minutes of silence before re-enabling
_circuit_broken_until      = 0.0     # epoch time when circuit resets
_consecutive_429s          = 0

# ── Internal state ──────────────────────────────────────────────────────────
_db              = None
_initialized     = False
_last_telem_push = 0.0
_last_frame_push = 0.0
_last_health_push = 0.0
_push_lock       = threading.Lock()
_frame_lock      = threading.Lock()

# Queue with hard cap (maxsize=20) — drops oldest if full, prevents memory bloat
import queue
_write_queue = queue.Queue(maxsize=20)
_worker_thread = None
_frame_upload_in_progress = False
_frame_upload_lock = threading.Lock()
_last_pushed_telem = {}
_last_history_nav_status = None

# Exception classes for retry logic
try:
    from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, ServiceUnavailable
except ImportError:
    ResourceExhausted = Exception
    DeadlineExceeded = Exception
    ServiceUnavailable = Exception


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Background Worker ────────────────────────────────────────────────────────

def _worker_loop() -> None:
    """
    Background worker thread loop that pulls write requests from the queue,
    gathers all pending items to conflate/batch-process them, and writes to Firestore.
    Checks the circuit breaker before each batch to pause during 429 storms.
    """
    while True:
        try:
            item = _write_queue.get()
            if item is None:
                # Shutdown signal
                break

            # Check circuit breaker: if tripped, drop and wait
            now = time.time()
            if now < _circuit_broken_until:
                remaining = _circuit_broken_until - now
                print(f"[FIREBASE] ⚡ Circuit breaker active — skipping writes for {remaining:.0f}s more.")
                # Drain remaining items in queue to prevent backlog
                while not _write_queue.empty():
                    try:
                        _write_queue.get_nowait()
                    except queue.Empty:
                        break
                time.sleep(min(remaining, 30.0))
                continue

            # Gather all currently available items in the queue to conflate
            items = [item]
            while not _write_queue.empty():
                try:
                    items.append(_write_queue.get_nowait())
                except queue.Empty:
                    break

            _process_batch(items)
        except Exception as err:
            print(f"[FIREBASE] Error in background worker: {err}")
        finally:
            try:
                _write_queue.task_done()
            except ValueError:
                pass


def _process_batch(items: list) -> None:
    """
    Groups and conflates queued writes, then commits them in a single Firestore batch.
    - Telemetry ('live'), frame ('frame'), and health ('health') updates are conflated (only the latest is written).
    - Alerts ('alert') and history logs ('history') are batched (all are appended).
    """
    global _db
    if _db is None:
        return

    latest_live = None
    latest_frame = None
    latest_health = None
    alerts = []
    histories = []

    for item in items:
        itype = item.get("type")
        payload = item.get("payload")
        if itype == "live":
            latest_live = payload
        elif itype == "frame":
            latest_frame = payload
        elif itype == "health":
            latest_health = payload
        elif itype == "alert":
            alerts.append(payload)
        elif itype == "history":
            histories.append(payload)

    batch = _db.batch()
    has_ops = False

    # 1. Telemetry live document (merged with set to preserve health stats)
    if latest_live is not None:
        doc_ref = _db.collection(COLLECTION_LIVE).document(DOCUMENT_USER)
        batch.set(doc_ref, latest_live, merge=True)
        has_ops = True

    # 2. System health document (merged into live status document to prevent stale data)
    if latest_health is not None:
        doc_ref = _db.collection(COLLECTION_LIVE).document(DOCUMENT_USER)
        batch.set(doc_ref, latest_health, merge=True)
        has_ops = True

    # 3. Camera frame document
    if latest_frame is not None:
        doc_ref = _db.collection(COLLECTION_LIVE).document(DOCUMENT_FRAME)
        batch.set(doc_ref, latest_frame)
        has_ops = True

    # 4. Danger alerts collection (append-only)
    for alert in alerts:
        doc_ref = _db.collection(COLLECTION_ALERTS).document()
        batch.set(doc_ref, alert)
        has_ops = True

    # 5. History / activity logs collection (append-only)
    for history in histories:
        doc_ref = _db.collection(COLLECTION_HISTORY).document()
        batch.set(doc_ref, history)
        has_ops = True

    if not has_ops:
        return

    has_frame = (latest_frame is not None)
    try:
        _commit_batch_with_retry(batch)
    finally:
        # Always clear the frame in progress flag when done
        if has_frame:
            global _frame_upload_in_progress
            with _frame_upload_lock:
                _frame_upload_in_progress = False


def _commit_batch_with_retry(batch) -> None:
    """
    Commits a batch with exponential backoff on HTTP 429 (ResourceExhausted) or timeouts.
    Trips the circuit breaker after _CIRCUIT_BREAKER_THRESHOLD consecutive failures.
    """
    global _consecutive_429s, _circuit_broken_until
    import random
    max_retries = 3  # Reduced from 5 — don't hammer a quota-exceeded endpoint
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            batch.commit()
            _consecutive_429s = 0  # Reset on success
            return
        except (ResourceExhausted, DeadlineExceeded, ServiceUnavailable) as exc:
            _consecutive_429s += 1
            if _consecutive_429s >= _CIRCUIT_BREAKER_THRESHOLD:
                _circuit_broken_until = time.time() + _CIRCUIT_BREAKER_COOLDOWN
                print(f"[FIREBASE] ⚡ Circuit breaker TRIPPED — pausing all writes for {_CIRCUIT_BREAKER_COOLDOWN:.0f}s after {_consecutive_429s} consecutive quota errors.")
                return
            if attempt == max_retries - 1:
                print(f"[FIREBASE] ✗ Batch write failed after {max_retries} attempts: {exc}")
                return
            delay = (base_delay * (2 ** attempt)) + random.uniform(0.0, 1.0)
            print(f"[FIREBASE] ⚠ Quota/Timeout: retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("429", "quota", "timeout", "exhausted", "deadline")):
                _consecutive_429s += 1
                if _consecutive_429s >= _CIRCUIT_BREAKER_THRESHOLD:
                    _circuit_broken_until = time.time() + _CIRCUIT_BREAKER_COOLDOWN
                    print(f"[FIREBASE] ⚡ Circuit breaker TRIPPED — pausing all writes for {_CIRCUIT_BREAKER_COOLDOWN:.0f}s.")
                    return
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0.0, 1.0)
                    print(f"[FIREBASE] ⚠ Quota exception: retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
            print(f"[FIREBASE] ✗ Non-retryable write error: {exc}")
            break


# ── Initialization ──────────────────────────────────────────────────────────

def initialize() -> bool:
    """
    Initialize Firebase Admin SDK.
    Returns True if successful, False if credentials are missing/invalid.
    Call once at app startup.
    """
    global _db, _initialized, _worker_thread

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

        # Start the background writer thread
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        print("[FIREBASE] Background write worker started.")
        return True

    except Exception as exc:
        print(f"[FIREBASE] ✗ Initialization failed: {exc}")
        _db = None
        return False


# ── Public API ───────────────────────────────────────────────────────────────

def push_telemetry(telemetry: dict) -> None:
    """
    Queue live status telemetry to the background queue.
    - Throttled to once every 1.0 seconds.
    - Skips if values have not changed significantly.
    """
    global _last_telem_push, _last_pushed_telem

    if _db is None:
        return

    now = time.time()
    if now - _last_telem_push < TELEMETRY_MIN_INTERVAL:
        print("[FIREBASE] Upload skipped (throttled)")
        return

    direction = telemetry.get("direction", "UNKNOWN").upper()
    obs_count = telemetry.get("obstacle_count", 0)
    closest = None
    if telemetry.get("obstacles"):
        closest = telemetry["obstacles"][0].get("distance")

    if direction == "STOP" or (closest is not None and closest < 1.5):
        freespace_status = "BLOCKED"
    elif obs_count > 0:
        freespace_status = "CAUTION"
    else:
        freespace_status = "CLEAR"

    free_ratio = telemetry.get("free_ratio", None)
    if free_ratio is None:
        if freespace_status == "BLOCKED":
            free_ratio = 0.03
        elif freespace_status == "CAUTION":
            free_ratio = 0.45
        else:
            free_ratio = 0.85

    # Significant change check
    significant = False
    if not _last_pushed_telem:
        significant = True
    else:
        if direction != _last_pushed_telem.get("direction"):
            significant = True
        elif freespace_status != _last_pushed_telem.get("freespace_status"):
            significant = True
        elif obs_count != _last_pushed_telem.get("obs_count"):
            significant = True
        elif closest != _last_pushed_telem.get("closest"):
            prev_closest = _last_pushed_telem.get("closest")
            if prev_closest is None or closest is None:
                significant = True
            elif abs(closest - prev_closest) >= 0.3:
                significant = True
        elif abs(float(free_ratio) - float(_last_pushed_telem.get("free_ratio", 0.0))) >= 0.1:
            significant = True

    if not significant:
        print("[FIREBASE] Upload skipped (unchanged)")
        return

    _last_pushed_telem = {
        "direction": direction,
        "freespace_status": freespace_status,
        "obs_count": obs_count,
        "closest": closest,
        "free_ratio": float(free_ratio)
    }
    _last_telem_push = now

    payload = {
        "navigation_status": direction,
        "freespace_status":  freespace_status,
        "free_ratio":        round(float(free_ratio), 3),
        "fps":               telemetry.get("fps", 0.0),
        "last_updated":      _utc_now(),
        "obstacle_count":    obs_count,
        "camera_status":     telemetry.get("status", {}).get("camera", "Unknown"),
        "model_status":      telemetry.get("status", {}).get("model", "Unknown"),
        "closest_obstacle_m": closest,
        "server_status":     telemetry.get("status", {}).get("server", "Active"),
    }

    try:
        _write_queue.put_nowait({"type": "live", "payload": payload})
    except queue.Full:
        pass  # Queue full — drop this telemetry update


def push_alert(alert_type: str, reason: str, free_ratio: float = 0.0) -> None:
    """
    Enqueues a danger/warning event to the alerts collection.
    """
    if _db is None:
        return

    payload = {
        "alert_type":  alert_type.upper(),
        "reason":      reason,
        "free_ratio":  round(float(free_ratio), 3),
        "timestamp":   _utc_now(),
    }
    try:
        _write_queue.put_nowait({"type": "alert", "payload": payload})
    except queue.Full:
        print(f"[FIREBASE] ⚠ Queue full — alert dropped: {alert_type}")


def push_log(entry: dict) -> None:
    """
    Enqueues warn/danger logs to the Firestore history/activity log collection.
    - Only written when the navigation direction changes or a danger event occurs.
    - Does not store duplicate sequential direction entries.
    """
    global _last_history_nav_status

    if _db is None:
        return

    level = entry.get("level", "info")
    if level == "info":
        return  # Only log warn/danger

    message = entry.get("message", "")

    nav_status = None
    if "DIR=" in message:
        for part in message.split("|"):
            if "DIR=" in part:
                nav_status = part.split("=")[1].strip().upper()
                break
    if not nav_status:
        nav_status = "STOP" if "STOP" in message.upper() else entry.get("type", "SYSTEM")

    is_danger = (level == "danger")

    should_write = False
    if is_danger:
        should_write = True
    elif nav_status != _last_history_nav_status:
        should_write = True

    if not should_write:
        print("[FIREBASE] Upload skipped (unchanged)")
        return

    _last_history_nav_status = nav_status
    fs_status = "BLOCKED" if level == "danger" else "CAUTION"

    payload = {
        "navigation_status": nav_status,
        "freespace_status":  fs_status,
        "free_ratio":        0.03 if level == "danger" else 0.35,
        "event_type":        entry.get("type", "SYSTEM"),
        "message":           message,
        "level":             level,
        "timestamp":         entry.get("timestamp", _utc_now()),
    }
    try:
        _write_queue.put_nowait({"type": "history", "payload": payload})
    except queue.Full:
        pass  # Drop history entry if queue is full


def push_frame(base64_frame: str, fps: float = 0.0) -> None:
    """
    Enqueues a camera frame for fast streaming to Firestore.
    - Throttled to once every 4.0 seconds.
    - Skipped if the previous frame upload is still in progress.
    """
    global _last_frame_push, _frame_upload_in_progress

    if _db is None or not base64_frame:
        return

    now = time.time()
    if now - _last_frame_push < FRAME_MIN_INTERVAL:
        print("[FIREBASE] Upload skipped (throttled)")
        return

    with _frame_upload_lock:
        if _frame_upload_in_progress:
            print("[FIREBASE] Upload skipped (throttled)")
            return
        _frame_upload_in_progress = True

    _last_frame_push = now

    payload = {
        "frame": base64_frame,
        "fps": round(float(fps), 1),
        "timestamp": _utc_now(),
    }
    try:
        _write_queue.put_nowait({"type": "frame", "payload": payload})
    except queue.Full:
        with _frame_upload_lock:
            _frame_upload_in_progress = False
        print("[FIREBASE] Upload skipped (throttled)")


def push_health(cpu: float, memory: float, temp: float) -> None:
    """
    Enqueues CPU, memory, and temperature health data to the live telemetry document.
    Throttled to once every 30 seconds.
    """
    global _last_health_push
    if _db is None:
        return

    now = time.time()
    if now - _last_health_push < HEALTH_MIN_INTERVAL:
        print("[FIREBASE] Upload skipped (throttled)")
        return
    _last_health_push = now

    payload = {
        "cpu": round(float(cpu), 1),
        "memory": round(float(memory), 1),
        "temperature": round(float(temp), 1),
        "health_last_updated": _utc_now(),
    }
    try:
        _write_queue.put_nowait({"type": "health", "payload": payload})
    except queue.Full:
        pass

