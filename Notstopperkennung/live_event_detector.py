#!/usr/bin/env python3
"""
live_event_detector.py — Streaming-Event-Erkennung für den Lift.
=================================================================
Mit:
- Fahrt-Abbruch (Watchdog)
- Status-LEDs auf 4 GPIO-Pins (5, 6, 12, 13)
- Software-Trigger: Tasten 'n', 'x', 'i', 'p', 'r', 'a'
- Hardware-Taster (aktiv HIGH, externe 10k-Pulldowns):
    GPIO 22 (Pin 15) = Nothalt markieren
    GPIO 23 (Pin 16) = Pause/Resume
- Live-Anomalie-Scoring (Sliding Window)
- Fahrt-Report mit Ground Truth vs. Erkennung
- Pause/Resume: Ignoriert unsaubere Phasen nach Nothalt
- NEU: Setzt ride_id und seq_idx beim Neustart fort, ohne Marker zu löschen.
"""

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from stream_logger import StreamLogger
    _HAS_STREAM_LOGGER = True
except ImportError:
    StreamLogger = None
    _HAS_STREAM_LOGGER = False

try:
    from gpiozero import LED, Button
    _HAS_GPIO = True
except ImportError:
    LED = None
    Button = None
    _HAS_GPIO = False

try:
    from live_anomaly import LiveAnomalyScorer
    _HAS_ANOMALY = True
except ImportError:
    LiveAnomalyScorer = None
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

# Hardware-Taster (aktiv HIGH, externe 10k-Pulldowns gegen GND)
BUTTON_PINS = {
    "nothalt": 22,   # GPIO 22 (Pin 15)
    "pause":   23,   # GPIO 23 (Pin 16)
}

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
    "reset":     "\033[0m",
}


# ======================================================================
# LED-Statusanzeige
# ======================================================================
class StateLEDs:
    PINS = {
        "gestoppt":  5,
        "anfahren":  6,
        "fahren":    12,
        "abbremsen": 13,
    }
    ANOMALY_LED = {
        "gru_ae": "anfahren",
        "tranad": "fahren",
    }

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
                print(f"[!] LED-Init fehlgeschlagen: {e} — LEDs deaktiviert")
                self.enabled = False
                self.leds = {}

    def _all_off(self):
        for led in self.leds.values():
            try:
                led.off()
            except Exception:
                pass

    def set_state(self, state, now=None):
        if not self.enabled:
            return
        self.current_state = state
        self.state_entered_t = now if now is not None else time.time()
        for s, led in self.leds.items():
            try:
                led.on() if s == state else led.off()
            except Exception:
                pass

    def update(self, now=None):
        if not self.enabled or self.current_state != "gestoppt":
            return
        if self.gestoppt_off_after_s <= 0 or self.state_entered_t is None:
            return
        now = now if now is not None else time.time()
        if (now - self.state_entered_t) >= self.gestoppt_off_after_s:
            try:
                self.leds["gestoppt"].off()
            except Exception:
                pass

    def blink(self, state, times=3, on_s=0.15, off_s=0.1):
        if not self.enabled or state not in self.leds:
            return
        led = self.leds[state]
        try:
            for _ in range(times):
                led.on(); time.sleep(on_s); led.off(); time.sleep(off_s)
        except Exception:
            pass

    def set_anomaly_status(self, model_name, is_anomaly):
        if not self.enabled:
            return
        led_name = self.ANOMALY_LED.get(model_name)
        if not led_name or led_name not in self.leds:
            return
        try:
            self.leds[led_name].on() if is_anomaly else self.leds[led_name].off()
        except Exception:
            pass

    def clear_anomaly_status(self):
        if not self.enabled:
            return
        for led_name in set(self.ANOMALY_LED.values()):
            if led_name in self.leds:
                try:
                    self.leds[led_name].off()
                except Exception:
                    pass

    def close(self):
        if not self.enabled:
            return
        self._all_off()
        for led in self.leds.values():
            try:
                led.close()
            except Exception:
                pass


# ======================================================================
# Software-Trigger + Hardware-Taster
# ======================================================================
class TestTrigger:
    def __init__(self, marker_path: Path, session_id=None, enabled=True,
                 on_force_alarm=None, on_pause=None, on_reset=None,
                 use_hardware_buttons=False):
        self.marker_path = Path(marker_path)
        self.session_id = session_id
        self.enabled = enabled and self._can_read_keyboard()
        self.next_ride_is_test = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.marked_rides = []

        self.on_force_alarm = on_force_alarm
        self.on_pause = on_pause
        self.on_reset = on_reset

        # --- Hardware-Taster (aktiv HIGH, externe 10k-Pulldowns) ---
        self.buttons = {}
        if use_hardware_buttons and _HAS_GPIO and Button is not None:
            try:
                btn_nothalt = Button(
                    BUTTON_PINS["nothalt"],
                    pull_up=False,   # externer 10k-Pulldown
                    bounce_time=0.2,
                )
                btn_nothalt.when_pressed = self._hw_nothalt_pressed
                self.buttons["nothalt"] = btn_nothalt

                btn_pause = Button(
                    BUTTON_PINS["pause"],
                    pull_up=False,
                    bounce_time=0.2,
                )
                btn_pause.when_pressed = self._hw_pause_pressed
                self.buttons["pause"] = btn_pause

                print(f"[i] Hardware-Taster: AN (aktiv HIGH, externe Pulldowns)")
                print(f"    GPIO {BUTTON_PINS['nothalt']} = Nothalt markieren")
                print(f"    GPIO {BUTTON_PINS['pause']}   = Pause/Resume")
            except Exception as e:
                print(f"[!] Hardware-Taster-Init fehlgeschlagen: {e}")
                self.buttons = {}

        if self.enabled:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            # FIX: Wir überschreiben die Datei nicht mehr, sondern legen sie nur an, falls sie fehlt
            if not self.marker_path.exists():
                self.marker_path.touch()
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()

    def _hw_nothalt_pressed(self):
        c = COLORS.get("trigger", "")
        r = COLORS.get("reset", "")
        with self._lock:
            self.next_ride_is_test = True
        print(f"\n  {c}[HW-TASTER] Nächste Fahrt wird als NOTHALT markiert{r}\n")

    def _hw_pause_pressed(self):
        c = COLORS.get("trigger", "")
        r = COLORS.get("reset", "")
        if self.on_pause is not None:
            self.on_pause()
        print(f"\n  {c}[HW-TASTER] Pause/Resume{r}\n")

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
            print(f"[!] Trigger-Listener beendet: {e}")

    def _handle_key(self, ch):
        c, r = COLORS.get("trigger", ""), COLORS.get("reset", "")
        if ch == "n":
            with self._lock:
                already = self.next_ride_is_test
                self.next_ride_is_test = True
            print(f"\n  {c}[TRIGGER] Nächste Fahrt NOTHALT{r}\n" if not already
                  else f"\n  {c}[TRIGGER] Flag bereits gesetzt{r}\n")
        elif ch == "x":
            with self._lock:
                was = self.next_ride_is_test
                self.next_ride_is_test = False
            print(f"\n  {c}[TRIGGER] Markierung "
                  f"{'aufgehoben' if was else 'nicht aktiv'}{r}\n")
        elif ch == "i":
            with self._lock:
                flag = self.next_ride_is_test
            print(f"\n  {c}[TRIGGER] next_ride_is_test={flag} | "
                  f"markiert: {len(self.marked_rides)}{r}\n")
        elif ch == "a":
            if self.on_force_alarm:
                self.on_force_alarm()
                print(f"\n  {c}[TRIGGER] Manuelle Anomalie (Test){r}\n")
        elif ch == "p":
            if self.on_pause is not None:
                self.on_pause()
        elif ch == "r":
            if self.on_reset is not None:
                self.on_reset()

    def consume_flag(self):
        with self._lock:
            flag = self.next_ride_is_test
            self.next_ride_is_test = False
        return flag

    def mark_ride(self, ride_id, ride_start_unix):
        if not self.enabled:
            return
        record = {
            "ride_id": int(ride_id),
            "type": "nothalt",
            "session_id": self.session_id,
            "start_unix": float(ride_start_unix),
            "marked_unix": float(time.time()),
        }
        self.marked_rides.append(int(ride_id))
        with open(self.marker_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for btn in self.buttons.values():
            try:
                btn.close()
            except Exception:
                pass


# ======================================================================
# Detektor
# ======================================================================
class LiveEventDetector:
    def __init__(self, cfg=None, leds=None, test_trigger=None):
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
        self.high_buf = deque(maxlen=self.env_window)

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

        self.anomaly_models = []
        self.ride_alarm_status = {}
        self.ride_ground_truth = {}
        self._last_reported_ride = None

        self.stats = {}

        self.n_rides_ok = 0
        self.n_rides_aborted = 0
        
        self._ride_end_logged = False

        self.leds = leds
        self.test_trigger = test_trigger

    def _set_leds(self, state):
        if self.leds is not None:
            self.leds.set_state(state, now=time.time())

    def report_alarm(self, model_name):
        if model_name in self.ride_alarm_status:
            self.ride_alarm_status[model_name] = True

    def set_paused(self, paused: bool):
        self.paused = paused
        if paused:
            self.buf_x.clear(); self.buf_y.clear()
            self.buf_z.clear(); self.buf_t.clear()
            self.env_buffer.clear()
            self.current_events = []
            self.state = "gestoppt"
            self.state_start_t = time.time()
            self.last_stop_t = time.time()
            self.pending_state = None
            self.pending_since_t = None
            self._ride_end_logged = True
            if self.leds is not None:
                self.leds._all_off()
            c, r = COLORS.get("watchdog", ""), COLORS.get("reset", "")
            print(f"\n  {c}⏸  PAUSE AKTIV — Events werden ignoriert{r}\n")
        else:
            self.buf_x.clear(); self.buf_y.clear()
            self.buf_z.clear(); self.buf_t.clear()
            self.env_buffer.clear()
            self.current_events = []
            self.state = "gestoppt"
            now = time.time()
            self.state_start_t = now
            self.last_stop_t = now
            self.pending_state = None
            self.pending_since_t = None
            self._ride_end_logged = True
            if self.leds is not None:
                self.leds.set_state("gestoppt", now=now)
                self.leds.clear_anomaly_status()
            c, r = COLORS.get("ok", ""), COLORS.get("reset", "")
            print(f"\n  {c}▶  RESUME — Detektor zurückgesetzt{r}\n")

    def feed(self, x, y, z, timestamp):
        if self.paused:
            return
        self.buf_x.append(x); self.buf_y.append(y)
        self.buf_z.append(z); self.buf_t.append(timestamp)

        n = len(self.buf_x)
        if n < self.env_window + self.lowpass_window + 5:
            return

        if self.axis is None:
            stds = {
                "x": float(np.std(self.buf_x)),
                "y": float(np.std(self.buf_y)),
                "z": float(np.std(self.buf_z)),
            }
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
                    kernel = np.ones(self.lowpass_window) / self.lowpass_window
                    low = np.convolve(ac, kernel, mode="valid")
                else:
                    low = ac
                low_env = float(np.mean(np.abs(low[-self.env_window:])))
                self.cal_samples.append(low_env)
                if (timestamp - self.calibration_start_t) >= self.cfg["calibration_s"]:
                    self._finish_calibration(timestamp)
            return

        raw_ac = raw - self.offset
        if len(raw_ac) < self.lowpass_window:
            return

        kernel = np.ones(self.lowpass_window) / self.lowpass_window
        low_series = np.convolve(raw_ac, kernel, mode="valid")
        low_env = float(np.mean(np.abs(low_series[-self.env_window:])))

        if self.print_debug:
            print(f"  low_env={low_env:.4f} state={self.state}")

        if self.calibrated or self.calibration_failed:
            self.env_buffer.append(low_env)

        if self._check_watchdog(timestamp):
            return

        self._update_state(low_env, timestamp)

    def _check_watchdog(self, timestamp):
        if self.state == "gestoppt" or self.state_start_t is None:
            return False
        max_dur = self.cfg["max_duration_s"].get(self.state, None)
        if max_dur is None:
            return False
        elapsed = timestamp - self.state_start_t
        if elapsed <= max_dur:
            return False
        self._abort_ride(
            reason=f"{self.state} dauerte {elapsed:.1f}s (max {max_dur:.0f}s)",
            timestamp=timestamp,
        )
        return True

    def _emit_event(self, ev, ride_id):
        if self.paused:
            return
        if self.on_event:
            self.on_event(ev, ride_id)

    def _write_active_event_on_abort(self, timestamp, ride_id):
        if self.state == "gestoppt" or self.state_start_t is None:
            return
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
                  f"{self.state:<10} ({ev_dur:5.2f}s) [ABORT]"
                  f"{COLORS['reset']}")

    def _write_abort_marker(self, ride_id, reason):
        ev = {
            "type": "ABORT", "start_t": time.time(), "end_t": time.time(),
            "duration_s": 0.0, "env_mean": 0.0, "env_max": 0.0,
            "env_std": 0.0, "n_samples": 0, "peak_pos_rel": 0.0,
            "aborted": 1, "reason": reason,
        }
        if self.on_abort_marker:
            self.on_abort_marker(ev, ride_id, reason)

    def _report_ride_end(self, aborted: bool):
        if self.ride_id == 0 or self.ride_id == self._last_reported_ride:
            return
        self._last_reported_ride = self.ride_id
        if self.on_ride_end:
            self.on_ride_end(self.ride_id, dict(self.ride_alarm_status), aborted)

    def _abort_ride(self, reason, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if self.print_events:
            print(f"\n  {COLORS['watchdog']}"
                  f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                  f"⚠ WATCHDOG — Fahrt #{self.ride_id} abgebrochen"
                  f"{COLORS['reset']}")
            print(f"  {COLORS['watchdog']}         Grund: {reason}"
                  f"{COLORS['reset']}")
        if self.leds is not None:
            self.leds.blink("gestoppt", times=3)
        self._write_active_event_on_abort(timestamp, self.ride_id)
        self._write_abort_marker(self.ride_id, reason)
        self._report_ride_end(aborted=True)
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
            return (
                float(arr.mean()), float(arr.max()),
                float(arr.std()), int(len(arr)),
                float(np.argmax(arr) / max(len(arr) - 1, 1)),
            )
        return 0.0, 0.0, 0.0, 0, 0.0

    def _finish_calibration(self, timestamp):
        if self._cal_offset_samples:
            self.offset = float(np.mean(self._cal_offset_samples))
        else:
            self.offset = 0.0

        samples = np.array(self.cal_samples)
        if len(samples) < 10:
            self.calibration_failed = True
            if self.print_events:
                print("[!] Kalibrierung fehlgeschlagen — Default-Thresholds")
                print(f"    offset={self.offset:.4f}")
                print(f"    enter={self.enter_thresh:.4f}  exit={self.exit_thresh:.4f}")
            self.state_start_t = timestamp
            self.last_stop_t = timestamp
            self._set_leds(self.state)
            return

        p90 = float(np.percentile(samples, 90))

        if p90 > 0.15:
            self.calibration_failed = True
            if self.print_events:
                print(f"[!] Kalibrierung ungültig (p90={p90:.4f} > 0.15)")
                print(f"    offset={self.offset:.4f}")
                print(f"    enter={self.enter_thresh:.4f}  exit={self.exit_thresh:.4f}")
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
            print(f"    Watchdog: anfahren≤{self.cfg['max_duration_s']['anfahren']:.0f}s  "
                  f"fahren≤{self.cfg['max_duration_s']['fahren']:.0f}s  "
                  f"abbremsen≤{self.cfg['max_duration_s']['abbremsen']:.0f}s")

        self.state_start_t = timestamp
        self.last_stop_t = timestamp
        self._set_leds(self.state)

    def _update_state(self, low_env, timestamp):
        prev = self.state
        candidate = None
        confirm_s = 0.0

        if prev == "gestoppt" and low_env > self.enter_thresh:
            candidate, confirm_s = "anfahren", self.cfg["confirm_anfahren_s"]
        elif prev == "anfahren" and low_env < self.exit_thresh:
            candidate, confirm_s = "fahren", self.cfg["confirm_fahren_s"]
        elif prev == "fahren" and low_env > self.enter_thresh:
            candidate, confirm_s = "abbremsen", self.cfg["confirm_abbremsen_s"]
        elif prev == "abbremsen" and low_env < self.exit_thresh:
            candidate, confirm_s = "gestoppt", self.cfg["confirm_abbremsen_s"]

        if candidate is None or candidate == prev:
            self.pending_state = None
            self.pending_since_t = None
            return

        if candidate != self.pending_state:
            self.pending_state = candidate
            self.pending_since_t = timestamp
            return

        if (timestamp - self.pending_since_t) >= confirm_s:
            self._transition(prev, candidate, timestamp, low_env)
            self.pending_state = None
            self.pending_since_t = None

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

            is_test_ride = False
            if self.test_trigger is not None and self.test_trigger.consume_flag():
                self.test_trigger.mark_ride(self.ride_id, timestamp)
                is_test_ride = True
            self.ride_ground_truth[self.ride_id] = (
                "nothalt" if is_test_ride else "normal"
            )

            self.ride_alarm_status = {
                name: False for name in self.anomaly_models
            }
            self._last_reported_ride = None

            if self.leds is not None:
                self.leds.clear_anomaly_status()

            if self.print_events:
                if is_test_ride:
                    c, r = COLORS.get("trigger", ""), COLORS.get("reset", "")
                    print(f"\n  {c}!!!!!!!! TESTFAHRT #{self.ride_id} (Nothalt) !!!!!!!!{r}")
                    print(f"  {c}[=== Fahrt #{self.ride_id} startet ===]{r}\n")
                else:
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
                if new == "gestoppt":
                    self.last_stop_t = timestamp
                return

            env_mean, env_max, env_std, n_samples, peak_pos_rel = self._env_stats()
            ev = {
                "type": prev, "start_t": self.state_start_t,
                "end_t": timestamp, "duration_s": ev_dur,
                "env_mean": env_mean, "env_max": env_max,
                "env_std": env_std, "n_samples": n_samples,
                "peak_pos_rel": peak_pos_rel, "aborted": 0,
            }
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
        if new == "gestoppt":
            self.last_stop_t = timestamp

    def check_ride_end(self, timestamp):
        if self.paused:
            return
        if not (self.calibrated or self.calibration_failed):
            return
        if self.state != "gestoppt" or self.last_stop_t is None:
            return
        if self.ride_id == 0:
            return
        if self._ride_end_logged:
            return
        if (timestamp - self.last_stop_t) < self.cfg["min_stop_duration_s"]:
            return

        env_mean, env_max, env_std, n_samples, peak_pos_rel = self._env_stats()
        ev = {
            "type": "gestoppt", "start_t": self.state_start_t,
            "end_t": timestamp,
            "duration_s": timestamp - self.state_start_t,
            "env_mean": env_mean, "env_max": env_max,
            "env_std": env_std, "n_samples": n_samples,
            "peak_pos_rel": peak_pos_rel, "aborted": 0,
        }
        self._emit_event(ev, self.ride_id)

        if self.print_events:
            print(f"  {COLORS['gestoppt']}"
                  f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}] "
                  f"{'gestoppt':<10} ({ev['duration_s']:5.2f}s)"
                  f"{COLORS['reset']}")
            print(f"  [=== Fahrt #{self.ride_id} beendet ===]\n")

        self.n_rides_ok += 1
        self._report_ride_end(aborted=False)

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


class RawLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp", "t_rel", "x", "y", "z"])
    def log(self, ts, tr, x, y, z):
        self.w.writerow([f"{ts:.6f}", f"{tr:.6f}",
                         f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
    def close(self): self.f.close()

# ======================================================================
# Helfer zum Lesen der letzten IDs
# ======================================================================
def get_last_ids(events_path: Path):
    last_ride = 0
    last_seq = 0
    if events_path.exists():
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("ride_id")
                    if rid and str(rid).strip().isdigit():
                        last_ride = int(rid)
                    seq = row.get("seq_idx")
                    if seq and str(seq).strip().isdigit():
                        last_seq = int(seq)
        except Exception as e:
            print(f"[!] Fehler beim Lesen der letzten IDs: {e}")
    return last_seq, last_ride


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    BASE_DIR = Path(__file__).parent.resolve()
    DEFAULT_DATA_DIR = BASE_DIR.parent / "live_data" / "raw"

    ap.add_argument("--test", action="store_true")
    ap.add_argument("--axis", choices=["x", "y", "z", "auto"], default="z")
    ap.add_argument("--log-raw", type=Path,
                    default=DEFAULT_DATA_DIR / "stream_raw.csv")
    ap.add_argument("--log-events", type=Path,
                    default=DEFAULT_DATA_DIR / "stream_events.csv")
    ap.add_argument("--test-markers", type=Path, default=None)
    ap.add_argument("--no-trigger", action="store_true")
    ap.add_argument("--no-stream-logger", action="store_true")
    ap.add_argument("--no-leds", action="store_true")
    ap.add_argument("--no-buttons", action="store_true",
                    help="Hardware-Taster (GPIO 22/23) deaktivieren")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--anomaly-run-dir", type=Path, default=None)
    ap.add_argument("--anomaly-models", type=str, default="gru_ae,tranad")
    ap.add_argument("--score-log", type=Path, default=None)

    args = ap.parse_args()

    if args.test_markers is None:
        args.test_markers = Path(args.log_events).parent / "test_markers.jsonl"
    if args.score_log is None:
        args.score_log = Path(args.log_events).parent / "live_scores.csv"

    print("=" * 62)
    print("  Live Event Detector")
    print("=" * 62)
    print(f"  Modus: {'MOCK' if args.test else 'LIVE'}")
    print(f"  Sampling: {CFG['fs']} Hz")
    print(f"  Achse: {args.axis.upper() if args.axis != 'auto' else 'AUTO'}")
    print(f"  Beenden mit Strg+C\n")

    leds = None if args.no_leds else StateLEDs(
        enabled=True,
        gestoppt_off_after_s=CFG["gestoppt_led_off_after_s"],
    )
    if leds and leds.enabled:
        print(f"  LEDs: an  (gestoppt=GPIO5, anfahren=GPIO6, "
              f"fahren=GPIO12, abbremsen=GPIO13)")
        print(f"        Nach Fahrt: GPIO6=GRU-AE, GPIO12=TranAD")
    else:
        print("  LEDs: aus")

    try:
        sensor = load_sensor(mock=args.test)
    except SystemExit as e:
        if leds: leds.close()
        print(e); return

    cfg = dict(CFG)
    det = LiveEventDetector(cfg, leds=leds)
    det.print_events = True
    det.print_debug = args.debug

    # --- NEU: Letzte IDs auslesen und fortsetzen ---
    last_seq, last_ride = get_last_ids(args.log_events)
    det.ride_id = last_ride
    print(f"[i] Setze ride_id fort bei: {last_ride + 1}")

    if args.axis != "auto":
        det.axis = args.axis
        print(f"[i] Hauptachse (fix): {args.axis.upper()}")

    if leds and leds.enabled:
        leds.set_state("gestoppt")

    raw_logger = RawLogger(args.log_raw) if args.log_raw else None

    stream_logger = None
    if (not args.no_stream_logger) and _HAS_STREAM_LOGGER:
        try:
            stream_logger = StreamLogger(
                path=args.log_events, rotate_daily=False, verbose=True,
            )
            # --- NEU: seq_idx fortsetzen ---
            if hasattr(stream_logger, "seq_idx"):
                stream_logger.seq_idx = last_seq + 1
                print(f"[i] Setze seq_idx fort bei: {last_seq + 1}")
        except Exception as e:
            print(f"[!] StreamLogger-Fehler: {e}")

    model_list = [
        s.strip() for s in args.anomaly_models.split(",") if s.strip()
    ]
    det.anomaly_models = model_list
    det.stats = {name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                 for name in model_list}

    anomaly_scorer = None
    if args.anomaly_run_dir is not None:
        if not _HAS_ANOMALY:
            print("[!] live_anomaly.py nicht gefunden — Scoring deaktiviert")
        else:
            try:
                anomaly_scorer = LiveAnomalyScorer(
                    run_dir=args.anomaly_run_dir,
                    models=model_list,
                    score_log_path=args.score_log,
                    warmup_csv=args.log_events,
                    verbose=True,
                )
                print(f"[i] Live-Scoring: AN (Modelle: {model_list})")
                print(f"    Score-Log: {args.score_log}")
            except SystemExit as e:
                print(e)
                anomaly_scorer = None

    if stream_logger is not None:
        def _on_event(ev, ride_id):
            seq = stream_logger.seq_idx
            stream_logger.log_event(ev, ride_id=ride_id)
            if anomaly_scorer is not None:
                res = anomaly_scorer.score_event(ev, ride_id, seq_idx=seq)
                if res is not None:
                    alerts = [n for n, a in res["alarms"].items() if a]
                    if alerts:
                        for n in alerts:
                            det.report_alarm(n)
                        c, r = COLORS.get("alarm", ""), COLORS.get("reset", "")
                        parts = "  ".join(
                            f"{n}={res['scores'][n]:.4f}" for n in alerts
                        )
                        print(f"  {c}[!] ANOMALIE ride=#{ride_id}  {parts}{r}")
        det.on_event = _on_event

        def _on_abort_marker(ev, ride_id, reason):
            stream_logger.log_event(ev, ride_id=ride_id)
        det.on_abort_marker = _on_abort_marker

    def _on_ride_end(ride_id, alarm_status, aborted):
        tag = "ABORT" if aborted else "ENDE"
        gt = det.ride_ground_truth.get(ride_id, "normal")

        print("\n" + "=" * 62)
        print(f"  FAHRT #{ride_id} — {tag}")
        print("-" * 62)
        if gt == "nothalt":
            print(f"  Ground Truth : {COLORS.get('alarm','')}"
                  f"⚠ NOTHALT (markiert){COLORS['reset']}")
        else:
            print(f"  Ground Truth : ✓ normal")

        for name in model_list:
            alarm = alarm_status.get(name, False)
            if not aborted:
                if gt == "nothalt" and alarm:
                    det.stats[name]["tp"] += 1
                    verdict = "True Positive"
                elif gt == "nothalt" and not alarm:
                    det.stats[name]["fn"] += 1
                    verdict = "False Negative"
                elif gt == "normal" and alarm:
                    det.stats[name]["fp"] += 1
                    verdict = "False Positive"
                else:
                    det.stats[name]["tn"] += 1
                    verdict = "True Negative"
            else:
                verdict = "(ABORT — keine Wertung)"

            sym = "⚠ ANOMALIE" if alarm else "✓ normal"
            col = COLORS.get("alarm", "") if alarm else COLORS.get("ok", "")
            print(f"  {name:<10} : {col}{sym}{COLORS['reset']:<20} → {verdict}")
        print("=" * 62 + "\n")

        if leds:
            for name in model_list:
                leds.set_anomaly_status(name, alarm_status.get(name, False))

    det.on_ride_end = _on_ride_end

    def _force_alarm():
        for name in model_list:
            det.report_alarm(name)
            if leds:
                leds.set_anomaly_status(name, True)

    def _toggle_pause():
        det.set_paused(not det.paused)

    def _reset_detector():
        if det.paused:
            det.set_paused(False)
        else:
            det.set_paused(True)
            time.sleep(0.05)
            det.set_paused(False)

    test_trigger = None
    if not args.no_trigger:
        session_id = (getattr(stream_logger, "session_id", None)
                      if stream_logger else None)
        test_trigger = TestTrigger(
            marker_path=args.test_markers,
            session_id=session_id,
            enabled=True,
            on_force_alarm=_force_alarm,
            on_pause=_toggle_pause,
            on_reset=_reset_detector,
            use_hardware_buttons=(not args.no_buttons),
        )
        det.test_trigger = test_trigger
        if test_trigger.enabled:
            print(f"[i] Software-Trigger: AN")
            print(f"    Taste 'n' = nächste Fahrt NOTHALT")
            print(f"    Taste 'p' = Pause an/aus")
            print(f"    Taste 'r' = Reset")
            print(f"    Taste 'a' = manuelle Anomalie (Test)")

    t_start = time.time()
    sample_period = 1.0 / cfg["fs"]
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

            if leds is not None:
                leds.update(now)
            if raw_logger:
                raw_logger.log(now, now - t_start, x, y, z)

            next_t += sample_period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()

    except KeyboardInterrupt:
        print("\n[i] Strg+C — beende Detector.")

    finally:
        now = time.time()
        if det.state != "gestoppt" and det.state_start_t is not None:
            det._write_active_event_on_abort(now, det.ride_id)
            det._write_abort_marker(det.ride_id, reason="shutdown")

        print()
        print("=" * 62)
        print("  Statistik")
        print("=" * 62)
        print(f"  Fahrten erfolgreich abgeschlossen:  {det.n_rides_ok}")
        print(f"  Fahrten abgebrochen (ABORT):        {det.n_rides_aborted}")
        if test_trigger is not None:
            print(f"  Markierte Testfahrten (Nothalt):    "
                  f"{len(test_trigger.marked_rides)}")

        print()
        print("=" * 62)
        print("  Klassifikations-Statistik (nicht-abortierte Fahrten)")
        print("=" * 62)
        print(f"  {'Modell':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  "
              f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("  " + "-" * 60)
        for name, s in det.stats.items():
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec / max(prec + rec, 1e-8)
                  if (prec + rec) > 0 else 0.0)
            print(f"  {name:<10} {tp:>4} {fp:>4} {fn:>4} {tn:>4}  "
                  f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

        if raw_logger: raw_logger.close()
        if stream_logger: stream_logger.close()
        if anomaly_scorer: anomaly_scorer.close()
        if test_trigger: test_trigger.close()
        if leds: leds.close()
        sensor.close()
        print("\n[✓] Beendet.")


if __name__ == "__main__":
    main()