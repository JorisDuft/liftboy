#!/usr/bin/env python3
"""
start_liftboy.py — Master-Skript zum Starten des Lift-Systems.
Startet den Detector und den LLM-Daemon parallel.
"""
import subprocess
import sys
import time
from pathlib import Path

# ======================================================================
# KONFIGURATION
# ======================================================================
BASE_DIR = Path(__file__).resolve().parent
SUBDIR = BASE_DIR / "EmergencyStopDetection"

DETECTOR_SCRIPT = SUBDIR / "live_event_detector.py"
LLM_SCRIPT = BASE_DIR / "llm_daemon.py"  # Muss im selben Ordner wie dieses Skript liegen

# PFAD ZU DEINEM QWEN MODELL (Bitte anpassen!)
MODEL_PATH = "~/Desktop/models/Qwen2-0.5B-Instruct" 
# ======================================================================

# Fügt den Unterordner zum Suchpfad hinzu
sys.path.insert(0, str(SUBDIR))

def check_environment():
    print("=" * 62)
    print("  Lift-System Check")
    print("=" * 62)
    
    if not DETECTOR_SCRIPT.exists():
        print(f"[!] Fehler: {DETECTOR_SCRIPT.name} nicht gefunden!")
        sys.exit(1)
    print(f"[✓] Detector gefunden: {DETECTOR_SCRIPT.name}")

    if not LLM_SCRIPT.exists():
        print(f"[!] Fehler: {LLM_SCRIPT.name} nicht gefunden im Hauptordner!")
        sys.exit(1)
    print(f"[✓] LLM-Daemon gefunden: {LLM_SCRIPT.name}")
    
    try:
        import live_anomaly
        print("[✓] live_anomaly.py importierbar.")
    except Exception as e:
        print(f"[!] Warnung: live_anomaly.py nicht ladbar: {e}")
    
    try:
        import torch
        print(f"[✓] PyTorch installiert (Version {torch.__version__})")
    except ImportError:
        print("[!] PyTorch fehlt! Bitte installieren: pip install torch")
        sys.exit(1)
        
    print("=" * 62)
    print("  Starte System (Detector + LLM)...")
    print("  Beenden mit Strg+C")
    print("=" * 62 + "\n")

if __name__ == "__main__":
    check_environment()
    
    # Starte den Detector
    detector_proc = subprocess.Popen([sys.executable, str(DETECTOR_SCRIPT)])

    # Kurze Pause, damit der Detector die JSONL anlegen kann
    time.sleep(2.0)

    # Starte den LLM-Daemon
    print("\n[i] Starte LLM-Daemon... (Modell wird geladen)")
    llm_proc = subprocess.Popen([
        sys.executable, str(LLM_SCRIPT),
        "--model-path", str(Path(MODEL_PATH).expanduser()),
        "--events-jsonl", str(BASE_DIR.parent / "live_data" / "raw" / "events.jsonl")
    ])

    try:
        # Warte, bis einer der beiden Prozesse beendet wird
        detector_proc.wait()
        llm_proc.wait()
    except KeyboardInterrupt:
        print("\n[i] Strg+C empfangen. Beende alle Prozesse...")
        detector_proc.terminate()
        llm_proc.terminate()
        detector_proc.wait()
        llm_proc.wait()
        print("[✓] System heruntergefahren.")