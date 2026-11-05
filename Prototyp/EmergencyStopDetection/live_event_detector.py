#!/usr/bin/env python3
"""
live_event_detector.py — Streaming-Event-Erkennung für den Lift.
=================================================================
TranAD-only Pipeline mit JSONL-Output für LLM (Qwen).

Hardware:
  LEDs:         GPIO 5=gestoppt, 6=anfahren, 12=fahren, 13=abbremsen
  Taster:       GPIO 23 = Pause/Resume

Output:
  --events-jsonl  (Default events.jsonl)  : eine JSON-Zeile pro Event
  --log-raw       (optional, nur --debug) : binäres Rolling-Ring

Steuerung Tastatur:
  p = Pause/Resume (auch zum Fortsetzen nach Anomalie!)
  r = Detektor-Reset (kurz pausieren)
  i = Info
"""
import argparse
import json
import os
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from gpiozero import LED, Button
    _HAS_GPIO = True
except ImportError:
    LED = None
    Button = None
    _HAS_GPIO = False

try:
    from live_anomaly import LiveAnomalyScorer, resolve_run_dir, MODEL_NAME
    _HAS_ANOMALY = True
except ImportError:
    LiveAnomalyScorer = None
    resolve_run_dir = None
    MODEL_NAME = "tranad"
    _HAS_ANOMALY = False

CFG = {
    "fs":                    100,
    "env_s":                 0.4,
    "lowpass_s":             0.3,
    "calibration_s":         5.0,
    "peak_thresh_factor":    3.0,
    "enter_factor":          1.5,
    "exit_factor":           0.7,
    "stop_thresh_default":   0.02,
    "min_event_duration_s":  0.8,
    "min_stop_duration_s":   2.0,
    "confirm_anfahren_s":    0.5,
    "confirm_fahren_s":      0.3,
    "confirm_abbremsen_s":   0.3,
    "max_duration_s": {
        "anfahren":  15.0,
        "fahren":    180.0,
        "abbremsen": 15.0,
    },
    "gestoppt_led_off_after_s": 10.0,
}

BUTTON_PINS = {"pause": 23}

COLORS = {
    "gestoppt":  "\033[90m",
    "anfahren":  "\033[94m",
    "fahren":    "\033[92m",
    "abbremsen": "\033[91m",
    "watchdog":  "\033[93m",
    "abort":     "\033[95m",
    "trigger":   "\033[96m",
    "alarm":     "\033[91m\033[1m",
    "ok":        "\033[92m",
    "dim":       "\033[2m",
    "reset":     "\033[0m",
}


# ======================================================================
# LEDs
# ======================================================================
class StateLEDs:
    PINS = {"gestoppt": 5, "anfahren": 6, "fahren": 12, "abbremsen": 13}

    def __init__(self, enabled=True, gestoppt_off_after_s=0.0):
        self.enabled = enabled and _HAS_GPIO
        self.gestoppt_off_after_s = float(gestoppt_off_after_s or 0.0)
        self.leds = {}
        self.current_state = None
        self.state_entered_t = None
        if self.enabled:
            try:
                for state, pin in self.PINS.items():
                    self.leds[state] = LED(pin)
                self._all_off()
            except Exception as e:
                print(f"[!] LED-Init fehlgeschlagen: {e}")
                self.enabled = False
                self.leds = {}

    def _all_off(self):
        for led in self.leds.values():
            try: led.off()
            except Exception: pass

    def set_state(self, state, now=None):
        if not self.enabled: return
        self.current_state = state
        self.state_entered_t = now if now is not None else time.time()
        for s, led in self.leds.items():
            try: led.on() if s == state else led.off()
            except Exception: pass

    def update(self, now=None):
        if not self.enabled or self.current_state != "gestoppt": return
        if self.gestoppt_off_after_s <= 0 or self.state_entered_t is None: return
        now = now if now is not None else time.time()
        if (now - self.state_entered_t) >= self.gestoppt_off_after_s:
            try: self.leds["gestoppt"].off()
            except Exception: pass

    def blink(self, state, times=3, on_s=0.15, off_s=0.1):
        if not self.enabled or state not in self.leds: return
        led = self.leds[state]
        try:
            for _ in range(times):
                led.on(); time.sleep(on_s); led.off(); time.sleep(off_s)
        except Exception: pass

    def flash(self, state, duration_s=0.8):
        """Nicht-blockierendes Aufblitzen der Phase-LED (Anomalie-Anzeige)."""
        if not self.enabled or state not in self.leds: return
        led = self.leds[state]
        def _run():
            try:
                led.on(); time.sleep(duration_s)
                if self.current_state != state: led.off()
            except Exception: pass
        threading.Thread(target=_run, daemon=True).start()

    def close(self):
        if not self.enabled: return
        self._all_off()
        for led in self.leds.values():
            try: led.close()
            except Exception: pass


# ======================================================================
# Steuerung — nur Pause/Reset
# ======================================================================
class ControlPanel:
    def __init__(self, enabled=True, on_pause=None, on_reset=None,
                 use_hardware_buttons=False):
        self.enabled = enabled and self._can_read_keyboard()
        self._stop = threading.Event()
        self._thread = None
        self.on_pause = on_pause
        self.on_reset = on_reset
        self.buttons = {}

        if use_hardware_buttons and _HAS_GPIO and Button is not None:
            try:
                btn = Button(BUTTON_PINS["pause"],
                             pull_up=False, bounce_time=0.2)
                btn.when_pressed = self._hw_pause
                self.buttons["pause"] = btn
                print(f"[i] HW-Taster: GPIO {BUTTON_PINS['pause']} = Pause/Resume")
            except Exception as e:
                print(f"[!] HW-Taster-Init: {e}")

        if self.enabled:
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()

    def _hw_pause(self):
        c, r = COLORS.get("trigger", ""), COLORS.get("reset", "")
        if self.on_pause: self.on_pause()
        print(f"\n  {c}[HW] Pause/Resume{r}\n")

    @staticmethod
    def _can_read_keyboard():
        try:
            import msvcrt  # noqa
            return True
        except ImportError:
            pass
        try:
            import termios  # noqa
            return sys.stdin.isatty()
        except ImportError:
            return False

    def _listen(self):
        try:
            import msvcrt
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode(errors="ignore").lower()
                    self._handle_key(ch)
                time.sleep(0.05)
            return
        except ImportError:
            pass
        try:
            import select, termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop.is_set():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1).lower()
                        self._handle_key(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception as e:
            print(f"[!] Key-Listener beendet: {e}")

    def _handle_key(self, ch):
        c, r = COLORS.get("trigger", ""), COLORS.get("reset", "")
        if ch == "p":
            if self.on_pause: self.on_pause()
        elif ch == "r":
            if self.on_reset: self.on_reset()
        elif ch == "i":
            print(f"\n  {c}[INFO] 'p'=Pause/Resume  'r'=Reset{r}\n")

    def close(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)
        for btn in self.buttons.values():
            try: btn.close()
            except Exception: pass


# ======================================================================
# JSONL Event-Logger (append-only, LLM-freundlich)
# ======================================================================
class EventLogger:
    def __init__(self, path: Path, session_id=None, resume=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.seq_idx = 0
        self.ride_id = 0
        if resume and self.path.exists() and self.path.stat().st_size > 0:
            self._resume()
        self._f = open(self.path, "a", encoding="utf-8")

    def _resume(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try: ev = json.loads(line)
                    except json.JSONDecodeError: continue
                    if "seq_idx" in ev:
                        self.seq_idx = max(self.seq_idx, int(ev["seq_idx"]))
                    if "ride_id" in ev:
                        self.ride_id = max(self.ride_id, int(ev["ride_id"]))
            print(f"[i] Event-Log fortgesetzt: seq={self.seq_idx}, "
                  f"ride={self.ride_id}")
        except Exception as e:
            print(f"[!] Resume-Fehler: {e}")

    def next_seq(self):
        s = self.seq_idx; self.seq_idx += 1
        return s

    def log(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        if self._f and not self._f.closed:
            self._f.flush(); self._f.close()


# ======================================================================
# Optionaler binärer Raw-Logger (nur mit --debug)
# ======================================================================
class BinaryRawLogger:
    """8 Byte/Sample: <float32 z> <float32 t_rel>. Kein Header."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "wb")

    def log(self, z, t_rel):
        self._f.write(struct.pack("<ff", float(z), float(t_rel)))

    def close(self):
        self._f.close()


# ======================================================================
# Detektor
# ======================================================================
class LiveEventDetector:
    def __init__(self, cfg=None, leds=None, control=None):
        self.cfg = cfg or CFG
        fs = self.cfg["fs"]
        self.fs = fs

        buf_len = max(int(fs * 10.0), 200)
        self.buf_x = deque(maxlen=buf_len)
        self.buf_y = deque(maxlen=buf_len)
        self.buf_z = deque(maxlen=buf_len)
        self.buf_t = deque(maxlen=buf_len)

        self.env_window = max(int(fs * self.cfg["env_s"]), 5)
        self.lowpass_window = max(int(fs * self.cfg["lowpass_s"]), 3)

        self.state = "gestoppt"
        self.state_start_t = None
        self.current_events = []
        self.last_stop_t = None
        self.ride_id = 0
        self.axis = None

        self.offset = None
        self._cal_offset_samples = []

        self.pending_state = None
        self.pending_since_t = None
        self.env_buffer = []

        self.stop_thresh = self.cfg["stop_thresh_default"]
        self.peak_thresh = self.stop_thresh * self.cfg["peak_thresh_factor"]
        self.enter_thresh = self.peak_thresh * self.cfg["enter_factor"]
        self.exit_thresh = self.peak_thresh * self.cfg["exit_factor"]

        self.cal_samples = []
        self.calibrated = False
        self.calibration_failed = False
        self.calibration_start_t = None

        self.print_events = True
        self.print_debug = False

        self.on_event = None
        self.on_abort_marker = None
        self.on_ride_end = None

        self.paused = False
        self.anomaly_scorer = None
        self._ride_end_logged = False

        self.leds = leds
        self.control = control

        self.n_rides_ok = 0
        self.n_rides_aborted = 0
        self.n_rides_anomal = 0

    def _set_leds(self, state):
        if self.leds: self.leds.set_state(state, now=time.time())

    def set_paused(self, paused: bool):
        self.paused = paused
        self._clear_buffers()
        self.state = "gestoppt"
        now = time.time()
        self.state_start_t = now
        self.last_stop_t = now
        self._ride_end_logged = True
        if self.leds:
            if paused: self.leds._all_off()
            else: self.leds.set_state("gestoppt", now=now)
        c, r = COLORS.get("watchdog" if paused else "ok", ""), COLORS.get("reset", "")
        print(f"\n  {c}{'⏸  PAUSE AKTIV' if paused else '▶  RESUME'}{r}\n")

    def _clear_buffers(self):
        self.buf_x.clear(); self.buf_y.clear()
        self.buf_z.clear(); self.buf_t.clear()
        self.env_buffer.clear()
        self.current_events = []
        self.pending_state = None
        self.pending_since_t = None

    # ------------------------------------------------------------------
    # Sensor-Feed
    # ------------------------------------------------------------------
    def feed(self, x, y, z, timestamp):
        if self.paused: return
        self.buf_x.append(x); self.buf_y.append(y)
        self.buf_z.append(z); self.buf_t.append(timestamp)

        if len(self.buf_x) < self.env_window + self.lowpass_window + 5:
            return

        if self.axis is None:
            stds = {"x": float(np.std(self.buf_x)),
                    "y": float(np.std(self.buf_y)),
                    "z": float(np.std(self.buf_z))}
            self.axis = max(stds, key=stds.get)
            if self.print_events:
                print(f"[i] Hauptachse (auto): {self.axis.upper()}")

        raw = np.array(self.buf_x if self.axis == "x"
                       else self.buf_y if self.axis == "y"
                       else self.buf_z)

        if self.offset is None:
            self._cal_offset_samples.append(float(np.mean(raw)))
            if not self.calibrated and not self.calibration_failed:
                if self.calibration_start_t is None:
                    self.calibration_start_t = timestamp
                ac = raw - np.mean(raw)
                if len(ac) >= self.lowpass_window:
                    k = np.ones(self.lowpass_window) / self.lowpass_window
                    low = np.convolve(ac, k, mode="valid")
                else: low = ac
                self.cal_samples.append(
                    float(np.mean(np.abs(low[-self.env_window:])))
                )
                if (timestamp - self.calibration_start_t) >= self.cfg["calibration_s"]:
                    self._finish_calibration(timestamp)
            return

        raw_ac = raw - self.offset
        if len(raw_ac) < self.lowpass_window: return
        k = np.ones(self.lowpass_window) / self.lowpass_window
        low_series = np.convolve(raw_ac, k, mode="valid")
        low_env = float(np.mean(np.abs(low_series[-self.env_window:])))

        if self.print_debug:
            print(f"  low_env={low_env:.4f} state={self.state}")

        if self.calibrated or self.calibration_failed:
            self.env_buffer.append(low_env)

        if self._check_watchdog(timestamp): return
        self._update_state(low_env, timestamp)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    def _check_watchdog(self, timestamp):
        if self.state == "gestoppt" or self.state_start_t is None:
            return False
        max_dur = self.cfg["max_duration_s"].get(self.state)
        if max_dur is None: return False
        elapsed = timestamp - self.state_start_t
        if elapsed <= max_dur: return False
        self._abort_ride(
            reason=f"{self.state} dauerte {elapsed:.1f}s (max {max_dur:.0f}s)",
            timestamp=timestamp)
        return True

    def _emit_event(self, ev, ride_id):
        if self.paused: return
        if self.on_event: self.on_event(ev, ride_id)

    def _write_active_event_on_abort(self, timestamp, ride_id):
        if self.state == "gestoppt" or self.state_start_t is None: return
        ev_dur = timestamp - self.state_start_t
        env_mean, env_max, env_std, n_samples, peak_pos_rel = self._env_stats()
        ev = {
            "type": self.state, "start_t": self.state_start_t,
            "end_t": timestamp, "duration_s": ev_dur,
            "env_mean": env_mean, "env_max": env_max,
            "env_std": env_std, "n_samples": n_samples,
            "peak_pos_rel": peak_pos_rel, "aborted": 1,
        }
        self._emit_event(ev, ride_id)
        if self.print_events:
            print(f"  {COLORS.get(self.state, '')}"
                  f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                  f"{self.state:<10} ({ev_dur:5.2f}s) [ABORT]{COLORS['reset']}")

    def _write_abort_marker(self, ride_id, reason):
        ev = {"type": "ABORT", "start_t": time.time(), "end_t": time.time(),
              "duration_s": 0.0, "env_mean": 0.0, "env_max": 0.0,
              "env_std": 0.0, "n_samples": 0, "peak_pos_rel": 0.0,
              "aborted": 1, "reason": reason}
        if self.on_abort_marker: self.on_abort_marker(ev, ride_id, reason)

    def _report_ride_end(self, aborted: bool, summary=None):
        if self.on_ride_end:
            self.on_ride_end(self.ride_id, summary, aborted)

    # ------------------------------------------------------------------
    # Abbruch
    # ------------------------------------------------------------------
    def _abort_ride(self, reason, timestamp=None):
        if timestamp is None: timestamp = time.time()
        if self.print_events:
            print(f"\n  {COLORS['watchdog']}"
                  f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                  f"⚠ WATCHDOG — Fahrt #{self.ride_id} abgebrochen{COLORS['reset']}")
            print(f"  {COLORS['watchdog']}         Grund: {reason}{COLORS['reset']}")
        if self.leds: self.leds.blink("gestoppt", times=3)

        self._write_active_event_on_abort(timestamp, self.ride_id)
        self._write_abort_marker(self.ride_id, reason)

        summary = None
        if self.anomaly_scorer is not None:
            summary = self.anomaly_scorer.discard_current_ride()

        self._report_ride_end(aborted=True, summary=summary)

        self.current_events = []
        self.env_buffer.clear()
        self.state = "gestoppt"
        self.state_start_t = timestamp
        self.last_stop_t = timestamp
        self.pending_state = None
        self.pending_since_t = None
        self.n_rides_aborted += 1
        self._set_leds("gestoppt")
        self._ride_end_logged = True

    def _env_stats(self):
        buf = self.env_buffer
        if buf:
            arr = np.asarray(buf, dtype=float)
            return (float(arr.mean()), float(arr.max()), float(arr.std()),
                    int(len(arr)), float(np.argmax(arr) / max(len(arr) - 1, 1)))
        return 0.0, 0.0, 0.0, 0, 0.0

    # ------------------------------------------------------------------
    # Kalibrierung
    # ------------------------------------------------------------------
    def _finish_calibration(self, timestamp):
        self.offset = float(np.mean(self._cal_offset_samples)) if self._cal_offset_samples else 0.0
        samples = np.array(self.cal_samples)
        if len(samples) < 10:
            self.calibration_failed = True
        else:
            p90 = float(np.percentile(samples, 90))
            if p90 > 0.15:
                self.calibration_failed = True
            else:
                self.stop_thresh = max(p90 * 1.5, 0.01)
                self.peak_thresh = self.stop_thresh * self.cfg["peak_thresh_factor"]
                self.enter_thresh = self.peak_thresh * self.cfg["enter_factor"]
                self.exit_thresh = self.peak_thresh * self.cfg["exit_factor"]
                self.calibrated = True

        if self.print_events:
            print(f"[i] Kalibrierung {'OK' if self.calibrated else 'FEHLGESCHLAGEN'}")
            print(f"    offset={self.offset:.4f}")
            print(f"    stop={self.stop_thresh:.4f}  peak={self.peak_thresh:.4f}")
            print(f"    enter={self.enter_thresh:.4f}  exit={self.exit_thresh:.4f}")

        self.state_start_t = timestamp
        self.last_stop_t = timestamp
        self._set_leds(self.state)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _update_state(self, low_env, timestamp):
        prev = self.state
        candidate = None; confirm_s = 0.0

        if prev == "gestoppt" and low_env > self.enter_thresh:
            candidate, confirm_s = "anfahren", self.cfg["confirm_anfahren_s"]
        elif prev == "anfahren" and low_env < self.exit_thresh:
            candidate, confirm_s = "fahren", self.cfg["confirm_fahren_s"]
        elif prev == "fahren" and low_env > self.enter_thresh:
            candidate, confirm_s = "abbremsen", self.cfg["confirm_abbremsen_s"]
        elif prev == "abbremsen" and low_env < self.exit_thresh:
            candidate, confirm_s = "gestoppt", self.cfg["confirm_abbremsen_s"]

        if candidate is None or candidate == prev:
            self.pending_state = None; self.pending_since_t = None
            return
        if candidate != self.pending_state:
            self.pending_state = candidate
            self.pending_since_t = timestamp
            return
        if (timestamp - self.pending_since_t) >= confirm_s:
            self._transition(prev, candidate, timestamp, low_env)
            self.pending_state = None; self.pending_since_t = None

    def _transition(self, prev, new, timestamp, low_env):
        if prev == "gestoppt" and new != "gestoppt":
            if self.state_start_t is not None:
                ev_dur = timestamp - self.state_start_t
                if self.print_events:
                    print(f"  {COLORS.get(prev, '')}"
                          f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                          f"{prev:<10} ({ev_dur:5.2f}s){COLORS['reset']}")
            self.current_events = []
            self.ride_id += 1
            self._ride_end_logged = False
            if self.print_events:
                print(f"\n  [=== Fahrt #{self.ride_id} startet ===]\n")
            self.state = new
            self.state_start_t = timestamp
            self.env_buffer.clear()
            self._set_leds(new)
            return

        if prev is not None and self.state_start_t is not None:
            ev_dur = timestamp - self.state_start_t
            if ev_dur < self.cfg["min_event_duration_s"] and prev != "gestoppt":
                self.state = new
                self.state_start_t = timestamp
                self.env_buffer.clear()
                self._set_leds(new)
                if new == "gestoppt": self.last_stop_t = timestamp
                return
            env_mean, env_max, env_std, n_samples, peak_pos_rel = self._env_stats()
            ev = {"type": prev, "start_t": self.state_start_t,
                  "end_t": timestamp, "duration_s": ev_dur,
                  "env_mean": env_mean, "env_max": env_max,
                  "env_std": env_std, "n_samples": n_samples,
                  "peak_pos_rel": peak_pos_rel, "aborted": 0}
            self._emit_event(ev, self.ride_id)
            if self.print_events:
                print(f"  {COLORS.get(prev, '')}"
                      f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                      f"{prev:<10} ({ev['duration_s']:5.2f}s) mean={env_mean:.4f}"
                      f"{COLORS['reset']}")

        self.state = new
        self.state_start_t = timestamp
        self.env_buffer.clear()
        self._set_leds(new)
        if new == "gestoppt": self.last_stop_t = timestamp

    def check_ride_end(self, timestamp):
        if self.paused: return
        if not (self.calibrated or self.calibration_failed): return
        if self.state != "gestoppt" or self.last_stop_t is None: return
        if self.ride_id == 0: return
        if self._ride_end_logged: return
        if (timestamp - self.last_stop_t) < self.cfg["min_stop_duration_s"]: return

        env_mean, env_max, env_std, n_samples, peak_pos_rel = self._env_stats()
        ev = {"type": "gestoppt", "start_t": self.state_start_t,
              "end_t": timestamp,
              "duration_s": timestamp - self.state_start_t,
              "env_mean": env_mean, "env_max": env_max,
              "env_std": env_std, "n_samples": n_samples,
              "peak_pos_rel": peak_pos_rel, "aborted": 0}
        self._emit_event(ev, self.ride_id)

        if self.print_events:
            print(f"  {COLORS['gestoppt']}"
                  f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                  f"{'gestoppt':<10} ({ev['duration_s']:5.2f}s){COLORS['reset']}")

        summary = None
        if self.anomaly_scorer is not None:
            summary = self.anomaly_scorer.finalize_current_ride()

        self.n_rides_ok += 1
        if summary and summary.get("alarmed"):
            self.n_rides_anomal += 1

        self._report_ride_end(aborted=False, summary=summary)

        self.state_start_t = timestamp
        self.last_stop_t = timestamp
        self.env_buffer.clear()
        self._ride_end_logged = True


# ======================================================================
# Sensor
# ======================================================================
def load_sensor(mock=False):
    if mock:
        import math, random
        class MockSensor:
            def __init__(self): self.t0 = time.time()
            def read_accel_ms2(self):
                t = time.time() - self.t0
                if t < 7.0:
                    return (random.gauss(0, 0.005), random.gauss(0, 0.005),
                            9.81 + random.gauss(0, 0.005))
                cycle = (t - 7.0) % 25
                z = 9.81
                if 3 <= cycle < 6:
                    z += 0.25 * math.sin((cycle - 3) / 3 * math.pi)
                elif 6 <= cycle < 14:
                    z += 0.02 * math.sin(t * 8)
                elif 14 <= cycle < 17:
                    z -= 0.25 * math.sin((cycle - 14) / 3 * math.pi)
                return (random.gauss(0, 0.005), random.gauss(0, 0.005), z)
            def close(self): pass
        return MockSensor()
    try:
        from record_sensor import LIS2DW12
    except ImportError as e:
        raise SystemExit(f"[!] Sensor-Treiber fehlt: {e}")
    return LIS2DW12()


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    BASE_DIR = Path(__file__).parent.resolve()
    
    # Logs landen standardmäßig in Desktop/live_data/raw (so wie bei dir)
    DEFAULT_DATA_DIR = BASE_DIR.parent / "live_data" / "raw"

    ap.add_argument("--test", action="store_true")
    ap.add_argument("--axis", choices=["x", "y", "z", "auto"], default="z")
    ap.add_argument("--events-jsonl", type=Path,
                    default=DEFAULT_DATA_DIR / "events.jsonl")
    ap.add_argument("--log-raw", type=Path, default=None,
                    help="Binäres Rolling-Ring (nur mit --debug sinnvoll)")
    ap.add_argument("--context-file", type=Path, default=None,
                    help="JSON mit externem Kontext (Tür, Waage, ...)")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-leds", action="store_true")
    ap.add_argument("--no-buttons", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--anomaly-run-dir", type=Path, default=None,
                    help="Run-Dir ODER Parent (z.B. Desktop/models). "
                         "Default: finde automatisch mittel_w4")
    ap.add_argument("--anomaly-model-name", type=str, default=None,
                    help="Bevorzugter Subdir-Name (Default: mittel_w4)")
    ap.add_argument("--threshold-factor", type=float, default=1.5)
    ap.add_argument("--confirmation-events", type=int, default=1)
    ap.add_argument("--accept-below-ratio", type=float, default=0.75,
                    help="max_score < threshold × ratio → Fahrt in Pool")

    args = ap.parse_args()

    print("=" * 62)
    print("  Live Event Detector — TranAD-only")
    print("=" * 62)
    print(f"  Modus:    {'MOCK' if args.test else 'LIVE'}")
    print(f"  Sampling: {CFG['fs']} Hz")
    print(f"  Achse:    {args.axis.upper() if args.axis != 'auto' else 'AUTO'}")
    print(f"  Beenden mit Strg+C\n")

    # --- LEDs ---
    leds = None if args.no_leds else StateLEDs(
        enabled=True, gestoppt_off_after_s=CFG["gestoppt_led_off_after_s"])
    if leds and leds.enabled:
        print("  LEDs: an  (5=gestoppt, 6=anfahren, 12=fahren, 13=abbremsen)")
    else:
        print("  LEDs: aus")

    # --- Sensor ---
    try:
        sensor = load_sensor(mock=args.test)
    except SystemExit as e:
        if leds: leds.close()
        print(e); return

    # --- Event-Logger ---
    evt_logger = EventLogger(args.events_jsonl, resume=True)
    print(f"[i] Event-Log: {evt_logger.path}")

    # --- Detektor ---
    det = LiveEventDetector(dict(CFG), leds=leds)
    det.print_events = True
    det.print_debug = args.debug
    det.ride_id = evt_logger.ride_id

    if args.axis != "auto":
        det.axis = args.axis
        print(f"[i] Hauptachse (fix): {args.axis.upper()}")
    if leds and leds.enabled:
        leds.set_state("gestoppt")

    # --- Anomaly-Scorer ---
    anomaly_scorer = None
    if _HAS_ANOMALY:
        try:
            run_dir = resolve_run_dir(
                args.anomaly_run_dir,
                preferred_name=args.anomaly_model_name,
            )
            print(f"[i] Modell gefunden: {run_dir}")
            anomaly_scorer = LiveAnomalyScorer(
                run_dir=run_dir,
                warmup_csv=args.events_jsonl if args.events_jsonl.exists() else None,
                verbose=True,
                threshold_factor=args.threshold_factor,
                confirmation_events=args.confirmation_events,
                accept_below_ratio=args.accept_below_ratio,
            )
            det.anomaly_scorer = anomaly_scorer
            print(f"[i] Live-Scoring: AN (TranAD)")
        except SystemExit as e:
            print(e)
            print("[!] Scoring deaktiviert — Events werden trotzdem geloggt.")
    else:
        print("[!] live_anomaly.py nicht gefunden — kein Scoring")

    # --- Context-Provider ---
    def _get_context():
        ctx = {}
        if args.context_file and args.context_file.exists():
            try:
                ctx = json.loads(args.context_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return ctx

    # --- Event-Handler: JSONL + Score + CMD + PAUSE BEI ANOMALIE ---
    def _on_event(ev, ride_id):
        seq = evt_logger.next_seq()
        ctx = _get_context()

        score_info = None
        if anomaly_scorer is not None:
            res = anomaly_scorer.score_event(ev, ride_id)
            score_info = {
                "score": res["score"],
                "threshold": res["threshold"],
                "alarm": res["alarm"],
                "warming_up": res["warming_up"],
            }
            
            if res["alarm"]:
                c, r = COLORS.get("alarm", ""), COLORS.get("reset", "")
                print(f"\n  {c}[!] ANOMALIE ride=#{ride_id}  "
                      f"score={res['score']:.4f} "
                      f"thr={res['threshold']:.4f}{r}")
                if leds:
                    leds.flash("fahren", 0.8) # Kurzes Aufblitzen

                # 1. Scorer anweisen, die kaputten Daten NICHT in den Normal-Pool zu übernehmen
                aborted_summary = anomaly_scorer.discard_current_ride()
                
                # 2. Detector pausieren (ignoriert ab jetzt alle Sensordaten)
                det.set_paused(True)
                
                # 3. Das Fahrt-Ende (mit der Anomalie) für die LLM loggen
                det._report_ride_end(aborted=True, summary=aborted_summary)
                
        record = {
            "kind": "event",
            "seq_idx": seq,
            "session_id": evt_logger.session_id,
            "timestamp_unix": ev.get("end_t", time.time()),
            "timestamp_iso": datetime.fromtimestamp(
                ev.get("end_t", time.time())).isoformat(timespec="milliseconds"),
            "ride_id": ride_id,
            "event_type": ev.get("type"),
            "duration_s": ev.get("duration_s"),
            "env_mean": ev.get("env_mean"),
            "env_max": ev.get("env_max"),
            "env_std": ev.get("env_std"),
            "peak_pos_rel": ev.get("peak_pos_rel"),
            "n_samples": ev.get("n_samples"),
            "aborted": ev.get("aborted", 0),
            "context": ctx,
            "anomaly": score_info,
        }
        evt_logger.log(record)
    det.on_event = _on_event

    def _on_abort_marker(ev, ride_id, reason):
        seq = evt_logger.next_seq()
        evt_logger.log({
            "kind": "abort_marker",
            "seq_idx": seq,
            "session_id": evt_logger.session_id,
            "timestamp_unix": ev.get("end_t", time.time()),
            "timestamp_iso": datetime.fromtimestamp(
                ev.get("end_t", time.time())).isoformat(timespec="milliseconds"),
            "ride_id": ride_id,
            "reason": reason,
            "context": _get_context(),
        })
    det.on_abort_marker = _on_abort_marker

    # --- Ride-End: JSONL + CMD-Feedback + PAUSE BEI ANOMALIE ---
    def _on_ride_end(ride_id, summary, aborted):
        seq = evt_logger.next_seq()
        tag = "ABORT" if aborted else "ENDE"

        print("\n" + "=" * 62)
        print(f"  FAHRT #{ride_id} — {tag}")
        print("-" * 62)
        if summary is None:
            print("  (keine gescorten Events)")
        else:
            alarmed = summary.get("alarmed", False)
            verdict = "ANOMAL" if alarmed else "normal"
            col = COLORS.get("alarm" if alarmed else "ok", "")
            print(f"  TranAD      : {col}{verdict}{COLORS['reset']}")
            print(f"  max score   : {summary.get('max_score', 0.0):.4f}")
            thr = anomaly_scorer.threshold if anomaly_scorer else 0.0
            print(f"  Schwelle    : {thr:.4f}")
            print(f"  Events      : {summary.get('n_events', 0)} "
                  f"({summary.get('n_scored', 0)} gescort)")
            pool_tag = ("→ in Normal-Pool"
                        if summary.get("accepted_into_pool")
                        else "→ verworfen (nicht in Pool)")
            print(f"  Normal-Pool : {pool_tag}")
            
            # Falls die Anomalie erst am Ende der Fahrt erkannt wurde
            # (WICHTIG: det.paused check verhindert doppeltes Pausieren!)
            if alarmed and not det.paused:
                det.set_paused(True)
        print("=" * 62 + "\n")

        evt_logger.log({
            "kind": "ride_end",
            "seq_idx": seq,
            "session_id": evt_logger.session_id,
            "timestamp_unix": time.time(),
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "ride_id": ride_id,
            "aborted": aborted,
            "summary": summary,
            "context": _get_context(),
        })
    det.on_ride_end = _on_ride_end

    # --- Control ---
    def _toggle_pause():
        det.set_paused(not det.paused)

    def _reset_detector():
        if det.paused:
            det.set_paused(False)
        else:
            det.set_paused(True); time.sleep(0.05); det.set_paused(False)

    control = None
    if not args.no_control:
        control = ControlPanel(
            enabled=True, on_pause=_toggle_pause, on_reset=_reset_detector,
            use_hardware_buttons=(not args.no_buttons),
        )
        det.control = control

    # --- Raw-Logger (nur wenn explizit) ---
    raw_logger = BinaryRawLogger(args.log_raw) if args.log_raw else None

    # --- Mainloop ---
    t_start = time.time()
    sample_period = 1.0 / CFG["fs"]
    next_t = t_start
    last_ride_check = t_start
    print(f"\n[i] Starte Streaming...\n")

    try:
        while True:
            now = time.time()
            x, y, z = sensor.read_accel_ms2()
            det.feed(x, y, z, now)
            if now - last_ride_check >= 0.2:
                det.check_ride_end(now)
                last_ride_check = now
            if leds: leds.update(now)
            if raw_logger: raw_logger.log(z, now - t_start)
            next_t += sample_period
            sleep = next_t - time.time()
            if sleep > 0: time.sleep(sleep)
            else: next_t = time.time()
    except KeyboardInterrupt:
        print("\n[i] Strg+C — beende.")
    finally:
        if det.state != "gestoppt" and det.state_start_t is not None:
            det._write_active_event_on_abort(time.time(), det.ride_id)
            det._write_abort_marker(det.ride_id, reason="shutdown")

        print()
        print("=" * 62)
        print("  Statistik")
        print("=" * 62)
        print(f"  Fahrten abgeschlossen:   {det.n_rides_ok}")
        print(f"  davon als ANOMAL:        {det.n_rides_anomal}")
        print(f"  Fahrten abgebrochen:     {det.n_rides_aborted}")

        if raw_logger: raw_logger.close()
        if anomaly_scorer: anomaly_scorer.close()
        evt_logger.close()
        if control: control.close()
        if leds: leds.close()
        sensor.close()
        print("\n[✓] Beendet.")


if __name__ == "__main__":
    main()