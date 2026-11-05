#!/usr/bin/env python3
import os
import json
import csv
import math
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn

EVENT_TYPES = ["gestoppt", "anfahren", "fahren", "abbremsen"]
FEATURE_LEN = 9
CONTINUOUS_IDX = [4, 5, 6, 7, 8]

MODEL_NAME = "tranad"

# ======================================================================
# Modell-Definition (exakt kopiert aus retrain_from_stream.py)
# ======================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TranAD(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4,
                 nlayers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=nlayers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        memory = self.encoder(h)
        out = self.decoder(h, memory)
        return self.output_proj(out)

# ======================================================================
# Pfad-Auflösung (findet Desktop/models/mittel_w4 automatisch)
# ======================================================================
_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR =  _BASE_DIR.parent / "models"

DEFAULT_MODEL_ROOT = Path(os.path.expanduser(
    os.environ.get("LIFT_MODEL_ROOT", str(_DEFAULT_MODELS_DIR))
))
PREFERRED_MODEL_NAME = os.environ.get("LIFT_MODEL_NAME", "EmergencyStopAnomaly_medium_w4")


def _is_run_dir(p: Path) -> bool:
    return (p.is_dir()
            and (p / "config.json").exists()
            and (p / "scaler.npz").exists()
            and (p / f"{MODEL_NAME}.pt").exists()
            and (p / f"thresholds_{MODEL_NAME}.json").exists())


def resolve_run_dir(path=None, preferred_name=None):
    preferred_name = preferred_name or PREFERRED_MODEL_NAME
    if path is not None:
        p = Path(os.path.expanduser(str(path)))
    else:
        p = DEFAULT_MODEL_ROOT

    if _is_run_dir(p):
        return p

    if not p.is_dir():
        raise SystemExit(f"[!] Modell-Pfad existiert nicht: {p}")

    candidates = [d for d in p.iterdir() if _is_run_dir(d)]
    if not candidates:
        raise SystemExit(
            f"[!] Kein gültiges Run-Dir unter {p}\n"
            f"    Erwartet pro Subdir: config.json, scaler.npz, "
            f"{MODEL_NAME}.pt, thresholds_{MODEL_NAME}.json"
        )

    for c in candidates:
        if c.name == preferred_name:
            return c
    for c in candidates:
        if (c / "selected.json").exists():
            return c
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]

# ======================================================================
# Scorer
# ======================================================================
class LiveAnomalyScorer:
    def __init__(self, run_dir, warmup_csv=None, verbose=False,
                 threshold_factor=1.5,
                 confirmation_events=1,
                 pool_path=None,
                 accept_below_ratio=0.75):
        self.run_dir = Path(run_dir)
        self.verbose = verbose
        self.threshold_factor = float(threshold_factor)
        self.confirmation_events = max(1, int(confirmation_events))
        self.accept_below_ratio = float(accept_below_ratio)

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

        # --- Modell laden (TranAD) ---
        ckpt_path = self.run_dir / f"{MODEL_NAME}.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"[!] {MODEL_NAME}.pt fehlt in {self.run_dir}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
        m_args = ckpt.get("args", {})
        
        m = TranAD(
            ckpt.get("n_features", FEATURE_LEN),
            d_model=m_args.get("d_model", 32),
            nhead=m_args.get("nhead", 2),
            nlayers=m_args.get("nlayers", 1),
            dropout=m_args.get("dropout", 0.0),
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        self.model = m

        # --- Threshold ---
        thr_p = self.run_dir / f"thresholds_{MODEL_NAME}.json"
        if not thr_p.exists():
            raise SystemExit(f"[!] thresholds_{MODEL_NAME}.json fehlt")
        raw_thr = float(json.loads(thr_p.read_text())["threshold"])
        self.threshold = raw_thr * self.threshold_factor
        self.accept_threshold = self.threshold * self.accept_below_ratio
        
        if self.verbose:
            print(f"[i] {MODEL_NAME}: base={raw_thr:.4f} → "
                  f"alarm>{self.threshold:.4f} "
                  f"(×{self.threshold_factor}), "
                  f"pool-accept<{self.accept_threshold:.4f}")

        # --- Buffer + Pool ---
        self.buf = deque(maxlen=self.window)
        self.streak = 0
        self.anomaly_detected = False
        self.normal_pool = deque(maxlen=max(self.window * 2, 16))
        self.pool_path = (Path(pool_path) if pool_path
                          else self.run_dir / "normal_pool.npy")

        # --- Ride-Tracking ---
        self.current_ride_id = None
        self.current_ride_events = []
        self.current_ride_max_score = 0.0
        self.current_ride_alarmed = False
        self.current_ride_n_scored = 0

        # 1) Persistierten Pool laden
        loaded = False
        if self.pool_path.exists():
            try:
                self.load_pool(self.pool_path)
                loaded = True
            except Exception as e:
                print(f"[!] Pool-Load fehlgeschlagen: {e}")

        # 2) Sonst aus (kuratierter) warmup_csv seeden
        if not loaded and warmup_csv and Path(warmup_csv).exists():
            self._seed_from_csv(Path(warmup_csv))

        if not self.normal_pool:
            print("[i] Normal-Pool leer — bootstrap aus ersten sauberen "
                  "Fahrten. Bei Deadlock: --warmup-csv mit kuratiertem "
                  "Normal-CSV verwenden.")

    def _seed_from_csv(self, path):
        print(f"[AnomalyScorer] Seed-Normaldaten aus {path} ...")
        try:
            with open(path, "r", encoding="utf-8") as f:
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
            print(f"[AnomalyScorer] Warmup: {len(self.normal_pool)} im Pool.")
        except Exception as e:
            print(f"[!] Warmup-Fehler: {e}")

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

    def _finalize_current_ride(self):
        if self.current_ride_id is None or not self.current_ride_events:
            return None
        n = len(self.current_ride_events)
        pool_empty = len(self.normal_pool) == 0
        clean = (not self.current_ride_alarmed
                 and self.current_ride_max_score < self.accept_threshold
                 and (pool_empty or self.current_ride_n_scored > 0))
        if clean:
            for x in self.current_ride_events:
                self.normal_pool.append(x)
        summary = {
            "ride_id": self.current_ride_id,
            "n_events": n,
            "n_scored": self.current_ride_n_scored,
            "max_score": self.current_ride_max_score,
            "alarmed": self.current_ride_alarmed,
            "accepted_into_pool": clean,
        }
        self.current_ride_id = None
        self.current_ride_events = []
        self.current_ride_max_score = 0.0
        self.current_ride_alarmed = False
        self.current_ride_n_scored = 0
        return summary

    def finalize_current_ride(self):
        """Explizit am Fahrt-Ende aufrufen (check_ride_end)."""
        summary = self._finalize_current_ride()
        if self.anomaly_detected:
            self._refill_buffer()
            self.anomaly_detected = False
            self.streak = 0
        return summary

    def discard_current_ride(self):
        """Bei Abbruch — kein Pool-Update."""
        summary = None
        if self.current_ride_id is not None and self.current_ride_events:
            summary = {
                "ride_id": self.current_ride_id,
                "n_events": len(self.current_ride_events),
                "n_scored": self.current_ride_n_scored,
                "max_score": self.current_ride_max_score,
                "alarmed": self.current_ride_alarmed,
                "accepted_into_pool": False,
                "aborted": True,
            }
        self.current_ride_id = None
        self.current_ride_events = []
        self.current_ride_max_score = 0.0
        self.current_ride_alarmed = False
        self.current_ride_n_scored = 0
        if self.anomaly_detected:
            self._refill_buffer()
            self.anomaly_detected = False
            self.streak = 0
        return summary

    def _refill_buffer(self):
        self.buf.clear()
        if not self.normal_pool:
            return False
        for x in list(self.normal_pool)[-self.window:]:
            self.buf.append(x)
        return True

    def _build_window(self, x):
        if not self.current_ride_alarmed:
            self.buf.append(x)
            return list(self.buf)
        return (list(self.buf) + [x])[-self.window:]

    def score_event(self, ev, ride_id):
        ride_id = int(ride_id)
        ride_summary = None

        # 1) Neue Fahrt?
        if self.current_ride_id is None or ride_id != self.current_ride_id:
            ride_summary = self._finalize_current_ride()
            if self.anomaly_detected:
                self._refill_buffer()
                self.anomaly_detected = False
                self.streak = 0
            self.current_ride_id = ride_id
            self.current_ride_events = []
            self.current_ride_max_score = 0.0
            self.current_ride_alarmed = False
            self.current_ride_n_scored = 0

        x = self._feature_vector(ev)
        self.current_ride_events.append(x)
        window_arr = self._build_window(x)

        if len(window_arr) < self.window:
            return {
                "score": None,
                "alarm": False,
                "threshold": self.threshold,
                "warming_up": True,
                "ride_summary": ride_summary,
            }

        W = np.stack(window_arr).astype(np.float32)[None, :, :]
        t = torch.from_numpy(W)
        with torch.no_grad():
            r = self.model(t)
            s = float(((r - t) ** 2).mean().item())

        self.current_ride_n_scored += 1
        if s > self.current_ride_max_score:
            self.current_ride_max_score = s

        self.streak = self.streak + 1 if s > self.threshold else 0
        alarm = self.streak >= self.confirmation_events
        if alarm:
            self.current_ride_alarmed = True
            self.anomaly_detected = True

        return {
            "score": s,
            "alarm": alarm,
            "threshold": self.threshold,
            "warming_up": False,
            "ride_summary": ride_summary,
        }

    def save_pool(self, path=None):
        path = Path(path) if path else self.pool_path
        if not self.normal_pool:
            return
        arr = np.stack(list(self.normal_pool)).astype(np.float32)
        np.save(str(path), arr)
        if self.verbose:
            print(f"[Pool] {len(arr)} Events → {path}")

    def load_pool(self, path=None):
        path = Path(path) if path else self.pool_path
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] != FEATURE_LEN:
            raise ValueError(f"Pool-Form {arr.shape} ungültig")
        for x in arr[-self.normal_pool.maxlen:]:
            self.normal_pool.append(x.astype(np.float32))
        if self.normal_pool:
            for x in list(self.normal_pool)[-self.window:]:
                self.buf.append(x)
        if self.verbose:
            print(f"[Pool] {len(self.normal_pool)} Events ← {path}")

    def close(self):
        if self.pool_path and self.normal_pool:
            try:
                self.save_pool(self.pool_path)
            except Exception as e:
                print(f"[!] Pool-Save fehlgeschlagen: {e}")