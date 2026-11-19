import time
import json
import os
import numpy as np
import requests

# ===================== CONFIG =====================

WINDOW = 20
PAUSE_THRESHOLD = 2.0

BASELINE_FILE = "baseline.json"
BACKEND_URL = "http://127.0.0.1:5000/software-data"

MAX_MOUSE_JITTER = 500.0
MAX_PAUSES = 5
MAX_BACKSPACES = 10
MAX_TASK_SWITCHES = 6
EPS = 1e-5

# ===================== BASELINE HANDLING =====================

def load_baseline():
    if os.path.exists(BASELINE_FILE):
        try:
            with open(BASELINE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Baseline corrupted. Resetting baseline.")

    return {
        "typing_variance": [],
        "mouse_jitter": [],
        "pause_count": [],
        "backspace_count": [],
        "task_switches": []
    }

def update_baseline(baseline, features):
    for key in baseline:
        baseline[key].append(features[key])
        baseline[key] = baseline[key][-50:]

    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f)

# ===================== ABNORMALITY =====================

def z_score(current, history):
    if len(history) < 5:
        return 0.0

    mean = np.mean(history)
    std = np.std(history)

    if std < EPS:
        return 0.0

    return (current - mean) / std

# ===================== ENGINE =====================

def start_engine():
    from pynput.keyboard import Listener as KeyboardListener, Key
    from pynput.mouse import Listener as MouseListener

    key_times = []
    mouse_moves = []
    backspace_count = 0
    task_switches = 0

    baseline = load_baseline()

    # -------- Keyboard Handler --------
    def on_key_press(key):
        nonlocal backspace_count, task_switches
        key_times.append(time.time())

        if key == Key.backspace:
            backspace_count += 1
        if key in (Key.alt_l, Key.alt_r):
            task_switches += 1

    # -------- Mouse Handler --------
    def on_mouse_move(x, y):
        mouse_moves.append((time.time(), x, y))

    kb = KeyboardListener(on_press=on_key_press)
    ms = MouseListener(on_move=on_mouse_move)
    kb.start()
    ms.start()

    print("🧠 Cognitive Load Engine running & sending data to backend...\n")

    start_time = time.time()

    while True:
        time.sleep(1)

        if time.time() - start_time >= WINDOW:

            # ---------- FEATURES ----------
            if len(key_times) > 1:
                delays = np.diff(key_times)
                typing_variance = np.var(delays)
                pause_count = int(np.sum(delays > PAUSE_THRESHOLD))
                typing_speed = len(key_times)
            else:
                typing_variance = 0.0
                pause_count = 0
                typing_speed = 0

            if len(mouse_moves) > 1:
                positions = np.array([(x, y) for _, x, y in mouse_moves])
                diffs = np.diff(positions, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
                mouse_jitter = np.var(distances)
            else:
                mouse_jitter = 0.0

            # ---------- CLI ----------
            norm_typing = min(
                typing_variance / (np.mean(baseline["typing_variance"]) + EPS),
                1.0
            )
            norm_mouse = min(
                np.log1p(mouse_jitter) / np.log1p(MAX_MOUSE_JITTER),
                1.0
            )
            norm_pause = min(pause_count / MAX_PAUSES, 1.0)
            norm_back = min(backspace_count / MAX_BACKSPACES, 1.0)
            norm_task = min(task_switches / MAX_TASK_SWITCHES, 1.0)

            cli = (
                0.30 * norm_typing +
                0.20 * norm_mouse +
                0.20 * norm_pause +
                0.15 * norm_back +
                0.15 * norm_task
            ) * 100

            cli = round(min(cli, 100), 2)

            if cli < 30:
                load_state = "Low"
            elif cli < 55:
                load_state = "Medium"
            elif cli < 75:
                load_state = "High"
            else:
                load_state = "Overload"

            # ---------- ABNORMALITY ----------
            abnormality = np.mean([
                abs(z_score(typing_variance, baseline["typing_variance"])),
                abs(z_score(mouse_jitter, baseline["mouse_jitter"])),
                abs(z_score(pause_count, baseline["pause_count"])),
                abs(z_score(backspace_count, baseline["backspace_count"])),
                abs(z_score(task_switches, baseline["task_switches"]))
            ])

            if abnormality > 2.5 and cli > 60:
                stress_state = "Cognitive Stress"
            elif abnormality > 2.5:
                stress_state = "Behavioral Anomaly"
            else:
                stress_state = "Normal"

            # ---------- SEND TO BACKEND ----------
            payload = {
                "timestamp": time.time(),
                "cli": cli,
                "load_state": load_state,
                "abnormality": round(abnormality, 2),
                "stress_state": stress_state,
                "features": {
                    "typing_variance": typing_variance,
                    "mouse_jitter": mouse_jitter,
                    "pause_count": pause_count,
                    "backspace_count": backspace_count,
                    "task_switches": task_switches
                }
            }

            try:
                requests.post(BACKEND_URL, json=payload, timeout=1)
            except requests.exceptions.RequestException:
                print("⚠️ Backend not reachable")

            # ---------- LOCAL OUTPUT ----------
            print("===================================")
            print(f"CLI: {cli} | Load: {load_state}")
            print(f"Stress State     : {stress_state}")
            print(f"Abnormality      : {abnormality:.2f}")
            print("===================================\n")

            update_baseline(baseline, payload["features"])

            key_times.clear()
            mouse_moves.clear()
            backspace_count = 0
            task_switches = 0
            start_time = time.time()

# ===================== ENTRY =====================

if __name__ == "__main__":
    start_engine()
