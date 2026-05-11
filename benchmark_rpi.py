"""
benchmark_rpi.py
================
Benchmarks inference performance of FP32, INT8, and QAT models on
the Raspberry Pi 4B. Reports FPS, latency, CPU temperature, and
estimated battery drain.

Results are logged to benchmark_results.csv.

Usage:
    python benchmark_rpi.py
    python benchmark_rpi.py --frames 200 --model-dir ./models
    python benchmark_rpi.py --model ./models/yolov8n_qat.onnx --frames 100
"""

import os
import sys
import csv
import time
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_DIR    = Path("./models")
RESULTS_CSV  = Path("./benchmark_results.csv")
N_WARMUP     = 10      # Warmup frames (discarded from stats)
N_BENCHMARK  = 100     # Benchmark frames

MODELS_TO_BENCHMARK = {
    "FP32 Baseline":   "yolov8n_fp32.onnx",
    "INT8 Quantized":  "yolov8n_int8.onnx",
    "QAT Fine-Tuned":  "yolov8n_qat.onnx",
}

# RPi 4B USB-C draws ~15W under heavy CPU load
# Battery capacity: typical 10,000mAh @ 5V = 50Wh
RPI_IDLE_WATTS      = 3.0
RPI_INFERENCE_WATTS = 6.5   # Approximate CPU + camera active load


# ──────────────────────────────────────────────
# SYSTEM INFO
# ──────────────────────────────────────────────

def get_cpu_temperature() -> Optional[float]:
    """Read CPU temperature on Raspberry Pi."""
    try:
        # RPi thermal zone
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        pass
    try:
        # Fallback: vcgencmd
        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                       stderr=subprocess.DEVNULL).decode()
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


# Add Optional to imports at the top (forgot to include)
from typing import Optional


def get_cpu_freq_mhz() -> Optional[float]:
    """Get current CPU frequency in MHz."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return None


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def get_system_info() -> dict:
    """Collect system metadata for benchmark report."""
    info = {
        "timestamp":    datetime.now().isoformat(),
        "platform":     platform.platform(),
        "machine":      platform.machine(),
        "processor":    platform.processor(),
        "python":       platform.python_version(),
        "cpu_temp_c":   get_cpu_temperature(),
        "cpu_freq_mhz": get_cpu_freq_mhz(),
        "mem_mb":       get_memory_usage_mb(),
    }

    # onnxruntime version
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except Exception:
        info["onnxruntime"] = "N/A"

    return info


# ──────────────────────────────────────────────
# BENCHMARK RUNNER
# ──────────────────────────────────────────────

def create_dummy_frame(size=(320, 320)) -> np.ndarray:
    """Create a random BGR frame for benchmarking (no camera needed)."""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Match preprocessing used in navigation_system_rpi.py."""
    import cv2
    img  = cv2.resize(frame, (320, 320))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = img.astype(np.float32) / 255.0
    img  = np.transpose(img, (2, 0, 1))
    img  = np.expand_dims(img, 0)
    return np.ascontiguousarray(img)


def benchmark_model(label: str, model_path: Path, n_frames: int = N_BENCHMARK) -> dict:
    """
    Benchmark a single ONNX model.
    Returns statistics dict.
    """
    import onnxruntime as ort

    print(f"\n  Benchmarking: {label} ({model_path.name})")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load {model_path.name}: {e}")
        return {"label": label, "error": str(e)}

    input_name = sess.get_inputs()[0].name

    # Warm up (not measured)
    print(f"    Warming up ({N_WARMUP} frames)...")
    for _ in range(N_WARMUP):
        frame  = create_dummy_frame()
        tensor = preprocess_frame(frame)
        sess.run(None, {input_name: tensor})

    # Benchmark
    print(f"    Running benchmark ({n_frames} frames)...")
    latencies    = []
    temps_before = []
    temps_after  = []

    for i in range(n_frames):
        frame  = create_dummy_frame()
        tensor = preprocess_frame(frame)

        t_before = get_cpu_temperature()
        t0       = time.perf_counter()
        sess.run(None, {input_name: tensor})
        t1       = time.perf_counter()
        t_after  = get_cpu_temperature()

        latencies.append((t1 - t0) * 1000)
        if t_before: temps_before.append(t_before)
        if t_after:  temps_after.append(t_after)

        if (i+1) % 25 == 0:
            print(f"    {i+1}/{n_frames}  avg lat: {np.mean(latencies):.1f}ms  "
                  f"fps: {1000/np.mean(latencies):.2f}")

    lat_arr = np.array(latencies)
    avg_fps = 1000 / lat_arr.mean()

    # Battery estimate
    # Inference runs continuously; estimate runtime from battery
    battery_capacity_wh = 50.0   # 10,000mAh @ 5V
    runtime_hours       = battery_capacity_wh / RPI_INFERENCE_WATTS

    result = {
        "label":           label,
        "model":           model_path.name,
        "n_frames":        n_frames,
        "avg_fps":         round(avg_fps, 2),
        "avg_lat_ms":      round(lat_arr.mean(), 2),
        "min_lat_ms":      round(lat_arr.min(), 2),
        "max_lat_ms":      round(lat_arr.max(), 2),
        "p50_lat_ms":      round(np.percentile(lat_arr, 50), 2),
        "p95_lat_ms":      round(np.percentile(lat_arr, 95), 2),
        "p99_lat_ms":      round(np.percentile(lat_arr, 99), 2),
        "std_lat_ms":      round(lat_arr.std(), 2),
        "cpu_temp_start_c":round(np.mean(temps_before), 1) if temps_before else None,
        "cpu_temp_end_c":  round(np.mean(temps_after), 1)  if temps_after  else None,
        "est_draw_watts":  RPI_INFERENCE_WATTS,
        "est_runtime_h":   round(runtime_hours, 1),
        "model_size_mb":   round(model_path.stat().st_size / 1024**2, 2),
        "error":           None,
    }

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  {label:<34}│")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Avg FPS:       {result['avg_fps']:>8.2f}            │")
    print(f"  │  Avg Latency:   {result['avg_lat_ms']:>8.2f} ms          │")
    print(f"  │  P95 Latency:   {result['p95_lat_ms']:>8.2f} ms          │")
    print(f"  │  Model Size:    {result['model_size_mb']:>8.2f} MB          │")
    print(f"  │  Est. Runtime:  {result['est_runtime_h']:>8.1f} h (10Ah batt)  │")
    if result['cpu_temp_end_c']:
        print(f"  │  CPU Temp:      {result['cpu_temp_end_c']:>8.1f} °C          │")
    print(f"  └─────────────────────────────────────┘")

    return result


# ──────────────────────────────────────────────
# COMPARISON TABLE & REPORTING
# ──────────────────────────────────────────────

def print_comparison_table(results: list) -> None:
    """Print side-by-side comparison of all benchmarked models."""
    print("\n")
    print("=" * 90)
    print("  BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    header = (f"  {'Model':<22} {'FPS':>6} {'Avg(ms)':>8} {'P95(ms)':>8} "
              f"{'Size(MB)':>9} {'Temp(°C)':>9} {'Runtime(h)':>11}")
    print(header)
    print(f"  {'-'*86}")

    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<22}  ERROR: {r['error']}")
            continue
        temp_str = f"{r['cpu_temp_end_c']:.1f}" if r.get("cpu_temp_end_c") else "N/A"
        print(
            f"  {r['label']:<22} "
            f"{r['avg_fps']:>6.2f} "
            f"{r['avg_lat_ms']:>8.2f} "
            f"{r['p95_lat_ms']:>8.2f} "
            f"{r['model_size_mb']:>9.2f} "
            f"{temp_str:>9} "
            f"{r['est_runtime_h']:>11.1f}"
        )

    print()
    # Speedup column
    valid = [r for r in results if not r.get("error") and "avg_lat_ms" in r]
    if len(valid) >= 2:
        baseline_lat = valid[0]["avg_lat_ms"]
        print(f"  Speedup vs FP32 baseline:")
        for r in valid[1:]:
            speedup = baseline_lat / r["avg_lat_ms"]
            print(f"    {r['label']}: {speedup:.2f}×")


def save_csv(results: list, sysinfo: dict, path: Path) -> None:
    """Write benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp", "label", "model", "n_frames",
        "avg_fps", "avg_lat_ms", "min_lat_ms", "max_lat_ms",
        "p50_lat_ms", "p95_lat_ms", "p99_lat_ms", "std_lat_ms",
        "cpu_temp_start_c", "cpu_temp_end_c",
        "est_draw_watts", "est_runtime_h", "model_size_mb",
        "platform", "onnxruntime",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {**r, **sysinfo}
            writer.writerow(row)

    print(f"\n  ✓ Results saved → {path.resolve()}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark navigation models on RPi 4B.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--model",     default=None, help="Benchmark a single model instead")
    parser.add_argument("--frames",    type=int, default=N_BENCHMARK)
    parser.add_argument("--output",    default=str(RESULTS_CSV))
    args = parser.parse_args()

    try:
        import onnxruntime
        import cv2
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install onnxruntime opencv-python-headless")
        sys.exit(1)

    print("\n" + "="*60)
    print("  HEAD-MOUNTED NAVIGATION — RPi BENCHMARK")
    print("="*60)

    sysinfo = get_system_info()
    print(f"\n  Platform:      {sysinfo['platform']}")
    print(f"  Machine:       {sysinfo['machine']}")
    print(f"  ONNX Runtime:  {sysinfo['onnxruntime']}")
    if sysinfo["cpu_temp_c"]:
        print(f"  CPU Temp now:  {sysinfo['cpu_temp_c']}°C")
    if sysinfo["cpu_freq_mhz"]:
        print(f"  CPU Freq now:  {sysinfo['cpu_freq_mhz']} MHz")

    model_dir = Path(args.model_dir)
    results   = []

    if args.model:
        # Single model benchmark
        p = Path(args.model)
        if not p.exists():
            print(f"[ERROR] Model not found: {p}")
            sys.exit(1)
        results.append(benchmark_model(p.stem, p, args.frames))
    else:
        # Benchmark all available models
        for label, filename in MODELS_TO_BENCHMARK.items():
            p = model_dir / filename
            if p.exists():
                results.append(benchmark_model(label, p, args.frames))
            else:
                print(f"\n  [SKIP] {label}: {filename} not found in {model_dir}")

    if not results:
        print("\n[ERROR] No models found to benchmark.")
        print(f"Expected models in: {model_dir.resolve()}")
        sys.exit(1)

    print_comparison_table(results)
    save_csv(results, sysinfo, Path(args.output))

    print("\n  PERFORMANCE TARGETS (RPi 4B):")
    print("    Target FPS:     5-6 FPS (QAT model)")
    print("    Target latency: 150-170ms (QAT model)")
    for r in results:
        if "qat" in r.get("model", "").lower() and not r.get("error"):
            meets_fps = r["avg_fps"] >= 5.0
            meets_lat = r["avg_lat_ms"] <= 170.0
            print(f"\n  QAT Model ({r['model']}):")
            print(f"    FPS:     {r['avg_fps']:.2f} {'✓' if meets_fps else '✗ (below target)'}")
            print(f"    Latency: {r['avg_lat_ms']:.2f}ms {'✓' if meets_lat else '✗ (above target)'}")


if __name__ == "__main__":
    main()
