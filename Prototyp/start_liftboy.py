#!/usr/bin/env python3
"""
start_liftboy.py — Master-Skript zum Starten des Lift-Systems.
Prüft die Umgebung und startet den live_event_detector im Unterordner.
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SUBDIR = BASE_DIR / "EmergencyStopDetection"
DETECTOR_SCRIPT = SUBDIR / "live_event_detector.py"

# WICHTIG: Fügt den Unterordner zum Suchpfad hinzu, damit der Import klappt
sys.path.insert(0, str(SUBDIR))

def check_environment():
    print("=" * 62)
    print("  Lift-System Check")
    print("=" * 62)
    
    # 1. Prüfen ob der Detector existiert
    if not DETECTOR_SCRIPT.exists():
        print(f"[!] Fehler: {DETECTOR_SCRIPT.name} nicht gefunden!")
        sys.exit(1)
    print(f"[✓] Detector gefunden: {DETECTOR_SCRIPT.name}")
    
    # 2. Prüfen ob live_anomaly.py importierbar ist
    try:
        import live_anomaly
        print("[✓] live_anomaly.py importierbar.")
    except Exception as e:
        print(f"[!] Warnung: live_anomaly.py konnte nicht geladen werden: {e}")
        print("    Das System läuft, aber ohne Anomalie-Erkennung.")
    
    # 3. Prüfen ob torch installiert ist
    try:
        import torch
        print(f"[✓] PyTorch installiert (Version {torch.__version__})")
    except ImportError:
        print("[!] PyTorch fehlt! Bitte installieren: pip install torch")
        sys.exit(1)
        
    print("=" * 62)
    print("  Starte System...")
    print("=" * 62 + "\n")

if __name__ == "__main__":
    check_environment()
    
    # Starte den Detector als Subprocess
    try:
        subprocess.run([sys.executable, str(DETECTOR_SCRIPT)], check=True)
    except KeyboardInterrupt:
        print("\n[i] System durch Benutzer beendet.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Detector mit Fehler beendet: {e}")