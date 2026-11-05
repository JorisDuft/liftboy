#!/usr/bin/env python3
"""
live_anomaly.py — Event-Level Scoring (Sliding Window) mit
rolling Normal-Pool, Force-Normal-Rides, Pool-Persistenz
und Anti-Fehlalarm-Logik.
"""
import json
import csv
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn

EVENT_TYPES = ["gestoppt", "anfahren", "fahren", "abbremsen"]
FEATURE_LEN = 9
CONTINUOUS_IDX = [4, 5, 6, 7, 8]

from retrain_from_stream import GRUAE, TranAD


class LiveAnomalyScorer:
    def __init__(self, run_dir, models=("gru_ae",), score_log_path=None,
                 warmup_csv=None, verbose=False,
                 threshold_factor=1.0,
                 confirmation_events=1,
                 force_normal_rides=(2,3),
                 pool_path=None):
        self.run_dir = Path(run_dir)
        self.verbose = verbose
        self.threshold_factor = float(threshold_factor)
        self.confirmation_events = max(1, int(confirmation_events))
        self.force_normal_rides = set(int(r) for r in force_normal_rides)
        self.pool_path = (Path(pool_path) if pool_path
                          else self.run_dir / "normal_pool.npy")

        cfg_path = self.run_dir / "config.json"
        sc_path = self.run_dir / "scaler.npz"
        if not cfg_path.exists() or not sc_path.exists():
            raise SystemExit(
                f"[!] {self.run_dir} enthält kein config.json/scaler.npz"
            )

        cfg = json.loads(cfg_path.read_text())
        sc = np.load(str(sc_path))
        self.window = int(cfg["window"])
        self.mean = sc["mean"]
        self.std = sc["std"]

        self.models = {}
        self.thresholds = {}
        self.anomaly_detected = False

        for name in models:
            ckpt_path = self.run_dir / f"{name}.pt"
            if not ckpt_path.exists():
                print(f"[!] {name}.pt fehlt in {self.run_dir} — übersprungen")
                continue

            ckpt = torch.load(str(ckpt_path), map_location="cpu",
                              weights_only=False)
            m_args = ckpt.get("args", {})

            if name == "gru_ae":
                m = GRUAE(
                    ckpt.get("n_features", FEATURE_LEN),
                    hidden=m_args.get("hidden", 32),
                    latent=m_args.get("latent", 8),
                    dropout=m_args.get("dropout", 0.0),
                    max_len=self.window + 50,
                    pos_emb_scale=m_args.get("pos_emb_scale", 1.0),
                )
            elif name == "tranad":
                m = TranAD(
                    ckpt.get("n_features", FEATURE_LEN),
                    d_model=m_args.get("d_model", 64),
                    nhead=m_args.get("nhead", 4),
                    nlayers=m_args.get("nlayers", 2),
                    dropout=m_args.get("dropout", 0.0),
                )
            else:
                print(f"[!] Unbekanntes Modell: {name}")
                continue

            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.models[name] = m

            thr_p = self.run_dir / f"thresholds_{name}.json"
            if thr_p.exists():
                raw_thr = float(json.loads(thr_p.read_text())["threshold"])
                self.thresholds[name] = raw_thr * self.threshold_factor
                if self.verbose and self.threshold_factor != 1.0:
                    print(f"[i] {name}: threshold {raw_thr:.4f} → "
                          f"{self.thresholds[name]:.4f} "
                          f"(× {self.threshold_factor})")
            else:
                self.thresholds[name] = float("inf")

        self.buf = deque(maxlen=self.window)
        self._seq_counter = 0  # Wird gleich aus der CSV überschrieben (falls vorhanden)
        self.streak = {n: 0 for n in self.models}

        # Rolling-Pool aus Events sauberen Fahrten
        self.normal_pool = deque(maxlen=max(self.window * 2, 16))

        # Fahrt-Tracking
        self.current_ride_id = None
        self.current_ride_events = []
        self.current_ride_dirty = False
        self.last_normal_ride_id = None
        self.last_normal_ride_len = 0

        # 1) Persistierten Pool laden (falls vorhanden)
        loaded_from_disk = False
        if self.pool_path and self.pool_path.exists():
            try:
                self.load_pool(self.pool_path)
                loaded_from_disk = True
            except Exception as e:
                print(f"[!] Pool-Load fehlgeschlagen: {e}")

        # 2) Sonst aus warmup_csv seeden
        if not loaded_from_disk and warmup_csv and Path(warmup_csv).exists():
            print(f"[AnomalyScorer] Seed-Normaldaten aus {warmup_csv} ...")
            try:
                with open(warmup_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ev = {
                            "type": row["event_type"],
                            "duration_s": float(row["duration_s"]),
                            "env_mean": float(row["env_mean"]),
                            "env_max": float(row["env_max"]),
                            "env_std": float(row["env_std"]),
                            "peak_pos_rel": float(row["peak_pos_rel"]),
                        }
                        self.normal_pool.append(self._feature_vector(ev))
                for x in list(self.normal_pool)[-self.window:]:
                    self.buf.append(x)
                print(f"[AnomalyScorer] Warmup: {len(self.normal_pool)} "
                      f"Events im Pool, {len(self.buf)} im Puffer.")
            except Exception as e:
                print(f"[!] Fehler beim Warmup: {e}")

        if not self.normal_pool:
            print("[i] Normal-Pool leer — wird erst durch saubere Fahrten "
                  "gefüllt. Force-Normal-Rides helfen beim Bootstrap.")

        # ------------------------------------------------------------------
        # HIER IST DIE ÄNDERUNG FÜR DIE FORTSETZUNG DES SEQ_IDX
        # ------------------------------------------------------------------
        self.score_log_path = Path(score_log_path) if score_log_path else None
        self._f, self._w = None, None
        if self.score_log_path:
            self.score_log_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = ((not self.score_log_path.exists())
                      or (self.score_log_path.stat().st_size == 0))
            
            # Wenn die Datei schon existiert: Lese den letzten seq_idx aus der CSV
            if not is_new:
                try:
                    with open(self.score_log_path, "r", encoding="utf-8") as f_read:
                        lines = f_read.readlines()
                        for line in reversed(lines):
                            line = line.strip()
                            if line:
                                parts = line.split(",")
                                if parts and parts[0].isdigit():
                                    self._seq_counter = int(parts[0]) + 1
                                    print(f"[i] Score-Log existiert bereits. Setze seq_idx fort bei: {self._seq_counter}")
                                    break
                except Exception as e:
                    print(f"[!] Konnte letzten seq_idx nicht lesen: {e}")

            self._f = open(self.score_log_path, "a", newline="")
            self._w = csv.writer(self._f)
            if is_new:
                header = ["seq_idx", "timestamp_unix", "event_type",
                          "ride_id", "forced"]
                for name in self.models:
                    header += [f"score_{name}", f"thr_{name}",
                               f"alarm_{name}"]
                self._w.writerow(header)
                self._f.flush()

    # ------------------------------------------------------------------
    # Feature-Vektor
    # ------------------------------------------------------------------
    def _feature_vector(self, ev):
        x = np.zeros((FEATURE_LEN,), dtype=np.float32)
        et = ev.get("type", "")
        if et in EVENT_TYPES:
            x[EVENT_TYPES.index(et)] = 1.0
        x[4] = float(ev.get("duration_s", 0.0))
        x[5] = float(ev.get("env_mean", 0.0))
        x[6] = float(ev.get("env_max", 0.0))
        x[7] = float(ev.get("env_std", 0.0))
        x[8] = float(ev.get("peak_pos_rel", 0.0))
        x[CONTINUOUS_IDX] = (x[CONTINUOUS_IDX] - self.mean) / self.std
        return x

    # ------------------------------------------------------------------
    # Ride-Finalisierung + Pool-Aufnahme
    # ------------------------------------------------------------------
    def _finalize_previous_ride(self, new_ride_id):
        if (self.current_ride_id is None
                or self.current_ride_id == new_ride_id):
            return
        forced = self.current_ride_id in self.force_normal_rides
        clean = (not self.current_ride_dirty) or forced
        n_events = len(self.current_ride_events)

        if clean and n_events > 0:
            for x in self.current_ride_events:
                self.normal_pool.append(x)
            self.last_normal_ride_id = self.current_ride_id
            self.last_normal_ride_len = n_events

        if forced and self.current_ride_dirty:
            tag = "FORCED-NORMAL"
        elif self.current_ride_dirty:
            tag = "ANOMAL"
        else:
            tag = "SAUBER"

        print(f"[Ride] {self.current_ride_id}: events={n_events} "
              f"dirty={self.current_ride_dirty} forced={forced} "
              f"→ {tag} "
              f"{'AKZEPTIERT' if clean else 'VERWORFEN'} "
              f"(Pool: {len(self.normal_pool)})")

    def _refill_buffer_from_normals(self):
        self.buf.clear()
        if not self.normal_pool:
            if self.verbose:
                print("  [Refill] Kein Normal-Pool — Puffer bleibt leer.")
            return False
        for x in list(self.normal_pool)[-self.window:]:
            self.buf.append(x)
        if self.verbose:
            src = (f"letzte Normalfahrt {self.last_normal_ride_id}"
                   if self.last_normal_ride_id is not None else "Pool")
            print(f"  [Refill] {len(self.buf)}/{self.window} Events "
                  f"aus {src} in Puffer.")
        return True

    def _build_window(self, x):
        if not self.current_ride_dirty:
            self.buf.append(x)
            return list(self.buf)
        else:
            combined = list(self.buf) + [x]
            return combined[-self.window:]

    # ------------------------------------------------------------------
    # Event-Scoring
    # ------------------------------------------------------------------
    def score_event(self, ev, ride_id, seq_idx=None):
        ride_id = int(ride_id)
        forced = ride_id in self.force_normal_rides

        # 1) Neue Fahrt → alte finalisieren
        if self.current_ride_id is None or ride_id != self.current_ride_id:
            self._finalize_previous_ride(ride_id)
            self.current_ride_id = ride_id
            self.current_ride_events = []
            self.current_ride_dirty = False

        # 2) Nach Alarm: Puffer aus Normal-Pool auffüllen
        if self.anomaly_detected:
            self._refill_buffer_from_normals()
            self.anomaly_detected = False
            for k in self.streak:
                self.streak[k] = 0

        # 3) Feature-Vektor, immer fürs Ride-Tracking
        x = self._feature_vector(ev)
        self.current_ride_events.append(x)

        # 4) Scoring-Fenster
        window_arr = self._build_window(x)

        if seq_idx is None:
            seq_idx = self._seq_counter
        self._seq_counter = seq_idx + 1

        if len(window_arr) < self.window:
            return None

        W = np.stack(window_arr).astype(np.float32)[None, :, :]
        t = torch.from_numpy(W)

        result = {
            "seq_idx": seq_idx,
            "ride_id": ride_id,
            "forced": forced,
            "scores": {},
            "alarms": {},
        }

        with torch.no_grad():
            for name, m in self.models.items():
                r = m(t)
                s = float(((r - t) ** 2).mean().item())
                result["scores"][name] = s

                if forced:
                    result["alarms"][name] = False
                    self.streak[name] = 0
                else:
                    if s > self.thresholds[name]:
                        self.streak[name] += 1
                    else:
                        self.streak[name] = 0
                    result["alarms"][name] = (
                        self.streak[name] >= self.confirmation_events
                    )

        # 5) State-Update (nicht für force-normal-Rides)
        if not forced and any(result["alarms"].values()):
            self.current_ride_dirty = True
            self.anomaly_detected = True

        # 6) Log
        if self._w is not None:
            row = [seq_idx, f"{ev.get('start_t', time.time()):.6f}",
                   ev.get("type", ""), ride_id, int(forced)]
            for name in self.models:
                row.append(f"{result['scores'].get(name, 0.0):.6f}")
                row.append(f"{self.thresholds.get(name, 0.0):.6f}")
                row.append(int(result["alarms"].get(name, False)))
            self._w.writerow(row)
            self._f.flush()

        return result

    # ------------------------------------------------------------------
    # Pool-Persistenz
    # ------------------------------------------------------------------
    def save_pool(self, path=None):
        path = Path(path) if path else self.pool_path
        if not self.normal_pool:
            print("[i] Pool leer — nichts zu speichern.")
            return
        arr = np.stack(list(self.normal_pool)).astype(np.float32)
        np.save(str(path), arr)
        print(f"[Pool] {len(arr)} Events gespeichert → {path}")

    def load_pool(self, path=None):
        path = Path(path) if path else self.pool_path
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] != FEATURE_LEN:
            raise ValueError(f"Pool-Form {arr.shape} passt nicht zu "
                             f"({FEATURE_LEN} features)")
        for x in arr[-self.normal_pool.maxlen:]:
            self.normal_pool.append(x.astype(np.float32))
        if self.normal_pool:
            for x in list(self.normal_pool)[-self.window:]:
                self.buf.append(x)
        print(f"[Pool] {len(self.normal_pool)} Events geladen aus {path} "
              f"(Puffer: {len(self.buf)}/{self.window}).")

    def close(self):
        if self.pool_path and self.normal_pool:
            try:
                self.save_pool(self.pool_path)
            except Exception as e:
                print(f"[!] Pool-Save fehlgeschlagen: {e}")
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()