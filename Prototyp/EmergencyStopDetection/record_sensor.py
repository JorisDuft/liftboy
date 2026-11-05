#!/usr/bin/env python3
"""
record_sensor.py — LIS2DW12 (SEN0405) Aufnahme im SensorLogger-Format.
======================================================================

Speichert:
  <name>.zip
    ├── Accelerometer.csv   (time, seconds_elapsed, z, y, x)
    └── Metadata.csv        (Info zur Aufnahme)

"""

import argparse
import json
import struct
import sys
import time
import zipfile
from csv import writer
from datetime import datetime
from io import StringIO
from pathlib import Path

# Sensor-Konfiguration
I2C_BUS = 1
I2C_ADDR_CANDIDATES = [0x19, 0x18]   # SA0=1 → 0x19, SA0=0 → 0x18

# LIS2DW12 Register
REG_WHO_AM_I  = 0x0F
REG_CTRL1     = 0x20
REG_CTRL2     = 0x21
REG_CTRL6     = 0x25
REG_OUT_X_L   = 0x28

# Erwarteter WHO_AM_I Wert
WHO_AM_I_LIS2DW12 = 0x44

# Sensitivität bei ±2g (14-bit)
SENSITIVITY_MG = 0.244        # mg/LSB
G_TO_MS2       = 9.80665

SAMPLE_RATE_HZ = 100
SAMPLE_PERIOD  = 1.0 / SAMPLE_RATE_HZ


# ==================================================================
# Sensor-Treiber (raw I2C, keine zusätzliche Bibliothek nötig)
# ==================================================================
class LIS2DW12:
    def __init__(self, bus_id=I2C_BUS):
        try:
            from smbus2 import SMBus
        except ImportError:
            raise SystemExit(
                "[!] smbus2 fehlt. Installiere mit:\n"
                "    pip install smbus2"
            )

        self.bus = SMBus(bus_id)
        self.addr = None

        for a in I2C_ADDR_CANDIDATES:
            try:
                who = self.bus.read_byte_data(a, REG_WHO_AM_I)
                if who == WHO_AM_I_LIS2DW12:
                    self.addr = a
                    break
            except OSError:
                continue

        if self.addr is None:
            raise SystemExit(
                "[!] LIS2DW12 nicht gefunden. Prüfe:\n"
                "    - Verdrahtung (VCC, GND, SDA, SCL)\n"
                "    - I2C aktiviert: sudo raspi-config\n"
                "    - i2cdetect -y 1 (sollte 0x19 oder 0x18 zeigen)"
            )

        self._configure()

    def _configure(self):
        # CTRL2: Soft-Reset + BDU
        self.bus.write_byte_data(self.addr, REG_CTRL2, 0x40)
        time.sleep(0.05)

        # CTRL6: FS = ±2g (00)
        self.bus.write_byte_data(self.addr, REG_CTRL6, 0x00)

        # CTRL1: ODR = 100 Hz (0101), Mode = High-Performance (01)
        # Bits: [ODR3:0][MODE1:0][LP_MODE][0]
        #   100 Hz = 0101
        #   Mode High-Performance = 01
        #   LP_MODE = 0
        #   → 0b0101_01_0_0 = 0x54
        self.bus.write_byte_data(self.addr, REG_CTRL1, 0x54)

        time.sleep(0.05)

    def read_accel_ms2(self):
        """Liest x,y,z in m/s²."""
        # Auto-Increment-Bit setzen für Mehrbyte-Read
        raw = self.bus.read_i2c_block_data(self.addr, REG_OUT_X_L | 0x80, 6)
        x = struct.unpack('<h', bytes(raw[0:2]))[0]
        y = struct.unpack('<h', bytes(raw[2:4]))[0]
        z = struct.unpack('<h', bytes(raw[4:6]))[0]

        # 14-bit signed → nach unten schieben (2 Bits) --> kleinsten zwei bits verwerfen da nur noise
        x >>= 2; y >>= 2; z >>= 2

        scale = SENSITIVITY_MG / 1000.0 * G_TO_MS2   # m/s² pro LSB
        return x * scale, y * scale, z * scale

    def close(self):
        self.bus.close()


# ==================================================================
# Mock-Sensor (zum Pipeline-Testen ohne Hardware)
# ==================================================================
class MockSensor:

    def __init__(self):
        import math
        self._t0 = time.time()
        self._math = math

    def read_accel_ms2(self):
        import math, random
        t = time.time() - self._t0

        # Profil: 3 s gestoppt, 3 s anfahren, 8 s fahren, 3 s abbremsen, 3 s gestoppt
        z = 0.0
        if 3.0 <= t < 6.0:
            # anfahren: positiver Puls (nach oben) oder negativ (nach unten)
            z = 0.25 * math.sin((t - 3.0) / 3.0 * math.pi)
        elif 6.0 <= t < 14.0:
            z = 0.02 * math.sin(t * 8.0)   # leichte Vibration
        elif 14.0 <= t < 17.0:
            z = -0.25 * math.sin((t - 14.0) / 3.0 * math.pi)

        z += random.gauss(0, 0.005)
        x = random.gauss(0, 0.005)
        y = random.gauss(0, 0.005)
        return x, y, z

    def close(self):
        pass


# ==================================================================
# Aufnahme + Speichern
# ==================================================================
def record(sensor, name, duration_s, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datenpuffer
    rows = []           # (time_ns, seconds_elapsed, z, y, x)
    t_start = time.time()
    t_start_ns = int(t_start * 1e9)

    print(f"[i] Aufnahme: {name}  ({'bis Strg+C' if duration_s == 0 else f'{duration_s} s'})")
    print(f"[i] Sample-Rate: {SAMPLE_RATE_HZ} Hz")
    print(f"[i] Strg+C zum vorzeitigen Beenden\n")

    next_t = t_start
    try:
        while True:
            now = time.time()
            if duration_s > 0 and (now - t_start) >= duration_s:
                break

            x, y, z = sensor.read_accel_ms2()
            t_ns = int(now * 1e9)
            rows.append((t_ns, now - t_start, z, y, x))

            # Fortschritt alle 1 s
            elapsed = now - t_start
            if int(elapsed) != int(elapsed - SAMPLE_PERIOD):
                print(f"    {elapsed:5.1f} s  |  z = {z:+.4f} m/s²", end="\r")

            # Sleep bis zum nächsten Sample
            next_t += SAMPLE_PERIOD
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[i] Manuell beendet.")

    elapsed_total = time.time() - t_start
    print(f"\n[i] Aufnahme beendet nach {elapsed_total:.1f} s "
          f"({len(rows)} Samples, Ø {len(rows)/elapsed_total:.1f} Hz)")

    if len(rows) < 10:
        print("[!] Zu wenige Samples — Datei wird nicht gespeichert.")
        return None

    zip_path = out_dir / f"{name}.zip"

    # Accelerometer.csv
    acc_buf = StringIO()
    w = writer(acc_buf)
    w.writerow(["time", "seconds_elapsed", "z", "y", "x"])
    for t_ns, sec, z, y, x in rows:
        w.writerow([t_ns, f"{sec:.6f}", f"{z:.6f}", f"{y:.6f}", f"{x:.6f}"])
    acc_csv = acc_buf.getvalue()

    # Metadata.csv
    now_dt = datetime.fromtimestamp(t_start)
    meta_buf = StringIO()
    mw = writer(meta_buf)
    mw.writerow([
        "version", "device name", "recording epoch time", "recording time",
        "recording timezone", "platform", "appVersion", "device id",
        "sensors", "sampleRateMs", "standardisation", "platform version"
    ])
    mw.writerow([
        3, "RPi-SEN0405", int(t_start * 1000),
        now_dt.strftime("%Y-%m-%d_%H-%M-%S"),
        "Europe/Zurich", "raspberrypi", "1.0",
        "local", "Accelerometer",
        f"{int(1000/SAMPLE_RATE_HZ)}", "false", "1.0"
    ])
    meta_csv = meta_buf.getvalue()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Accelerometer.csv", acc_csv)
        zf.writestr("Metadata.csv", meta_csv)

    print(f"[✓] Gespeichert: {zip_path}  ({zip_path.stat().st_size} Bytes)")
    print(f"    Kopiere es nach Train/raw/ oder Test/raw/.")
    return zip_path


# ==================================================================
# Diagnose-Modus
# ==================================================================
def check_sensor():
    print("[i] Prüfe Sensor...")
    try:
        sensor = LIS2DW12()
    except SystemExit as e:
        print(e)
        return

    print(f"[✓] LIS2DW12 gefunden bei Adresse 0x{sensor.addr:02X}")
    print("[i] Lese 5 s lang Werte (Strg+C zum Beenden)...\n")
    try:
        for _ in range(500):
            x, y, z = sensor.read_accel_ms2()
            print(f"    x={x:+.4f}  y={y:+.4f}  z={z:+.4f}  m/s²", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
    print("\n[✓] Sensor-Check beendet.")


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default=None,
                    help="Name der Aufnahme (ohne .zip). Default: auto aus Zeitstempel.")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="Dauer in Sekunden. 0 = bis Strg+C.")
    ap.add_argument("--out", type=Path, default=Path("."),
                    help="Zielordner (Default: aktueller Ordner).")
    ap.add_argument("--mock", action="store_true",
                    help="Mock-Sensor nutzen (kein Hardware nötig).")
    ap.add_argument("--check", action="store_true",
                    help="Nur Sensor prüfen, nichts aufnehmen.")
    args = ap.parse_args()

    if args.check:
        check_sensor()
        return

    if args.name is None:
        args.name = "rec_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"[i] Modus: {'MOCK' if args.mock else 'LIVE'}")

    if args.mock:
        sensor = MockSensor()
    else:
        sensor = LIS2DW12()

    try:
        record(sensor, args.name, args.duration, args.out)
    finally:
        sensor.close()


if __name__ == "__main__":
    main()