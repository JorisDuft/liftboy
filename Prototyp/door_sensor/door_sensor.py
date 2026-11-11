#!/usr/bin/env python3
"""
door_sensor.py — Liest den VL53L1X ToF-Sensor auf Software-I2C Bus 3 aus,
ermittelt den Türstatus (offen/geschlossen) per Schwellenwert und
übergibt das Ergebnis an die LLM (Qwen).
"""
import time
import json
import board
import busio
import adafruit_vl53l1x

# ======================================================================
# KONFIGURATION
# ======================================================================
# Schwellenwert in mm (oder cm, je nach Sensorausgabe).
# Wenn die gemessene Distanz DIESEN WERT UNTERSCHREITET, gilt die Tür als GESCHLOSSEN.
# Wenn die Distanz DIESEN WERT ÜBERSCHREITET, gilt die Tür als OFFEN.
# Bitte an die Einbausituation anpassen!
DOOR_CLOSED_THRESHOLD_MM = 50  # Beispiel: 5 cm
# Adresse des Sensors (Standard: 0x29)
SENSOR_I2C_ADDRESS = 0x29
# ======================================================================

def main():
    print("[i] Initialisiere Software-I2C Bus 3 (GPIO23=SDA, GPIO24=SCL)...")
    # WICHTIG: busio.I2C benötigt die Pins SCL und SDA.
    # Auf dem Raspberry Pi mit Blinka werden die GPIO-Nummern verwendet.
    # GPIO24 = SCL, GPIO23 = SDA
    i2c = busio.I2C(board.SCL, board.SDA)  # Dies nutzt den konfigurierten Software-I2C-Bus

    # Hinweis: Für Software-I2C auf spezifischen Pins kann je nach Blinka-Version
    # auch die direkte Angabe der Pins nötig sein:
    # i2c = busio.I2C(scl=board.GPIO24, sda=board.GPIO23)

    print(f"[i] Initialisiere VL53L1X (Adresse 0x{SENSOR_I2C_ADDRESS:02X})...")
    try:
        vl53 = adafruit_vl53l1x.VL53L1X(i2c, address=SENSOR_I2C_ADDRESS)
    except Exception as e:
        print(f"[!] Fehler beim Initialisieren des Sensors: {e}")
        print("    Prüfe die Verkabelung und ob der Software-I2C-Bus aktiviert ist.")
        return

    # Konfiguration: Kurze Distanz, 100ms Timing Budget
    vl53.distance_mode = 1  # SHORT
    vl53.timing_budget = 100

    print("[i] Starte Distanzmessung...")
    vl53.start_ranging()

    try:
        while True:
            if vl53.data_ready:
                distance_mm = vl53.distance  # Distanz in mm
                vl53.clear_interrupt()

                if distance_mm is None:
                    print("[!] Keine gültige Messung (Sensor blockiert?)")
                    time.sleep(1.0)
                    continue

                # Türstatus anhand des Schwellenwerts bestimmen
                is_open = distance_mm > DOOR_CLOSED_THRESHOLD_MM
                door_status = "offen" if is_open else "geschlossen"

                # CMD-Ausgabe
                print(f"[{time.strftime('%H:%M:%S')}] Distanz: {distance_mm:4d} mm  ->  Tür: {door_status.upper()}")

                # --- Übergabe an die LLM ---
                # Hier wird ein Event-Dict im gleichen Format wie in der events.jsonl erzeugt.
                # Dieses Event kann dann von einem LLM-Daemon (z.B. llm_daemon.py) verarbeitet werden.
                llm_event = {
                    "kind": "event",
                    "sensor_type": "door",
                    "timestamp_unix": time.time(),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "door_status": door_status,
                    "distance_mm": distance_mm,
                    "threshold_mm": DOOR_CLOSED_THRESHOLD_MM,
                }
                # In der Praxis würdest du dieses Event an die Pipeline übergeben.
                # Zum Testen wird es hier als JSON ausgegeben.
                print(f"    -> LLM-Event: {json.dumps(llm_event)}")

            time.sleep(1.0)  # 1 Hz Abtastrate

    except KeyboardInterrupt:
        print("\n[i] Beende Türsensor-Überwachung.")
    finally:
        vl53.stop_ranging()
        i2c.deinit()

if __name__ == "__main__":
    main()