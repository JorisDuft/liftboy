#!/usr/bin/env python3
"""
llm_daemon.py — Zentraler LLM-Daemon für den Lift.
Kann durch verschiedene Ereignisse getriggert werden (Anomalien, Zeitpläne, Audio).
"""
import argparse
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_ride_context(current_ride_events, ride_summary, ride_id):
    """Baut den Text-Kontext aus den Event-Daten der aktuellen Fahrt."""
    if not current_ride_events:
        return "Keine Event-Details verfügbar."
    
    text = f"--- FAHRT #{ride_id} DATEN ---\n"
    for ev in current_ride_events:
        ev_type = ev.get("event_type")
        dur = ev.get("duration_s", 0.0)
        e_max = ev.get("env_max", 0.0)
        e_std = ev.get("env_std", 0.0)
        
        anomaly = ev.get("anomaly")
        anom_str = ""
        if anomaly and anomaly.get("alarm"):
            anom_str = f" [ANOMALIE! Score={anomaly['score']:.4f}]"
                
        text += f"- {ev_type} ({dur:.2f}s): max={e_max:.4f}, std={e_std:.4f}{anom_str}\n"
    
    if ride_summary:
        status = "ANOMAL" if ride_summary.get("alarmed") else "normal"
        text += f"Fazit TranAD: {status} (Score: {ride_summary.get('max_score', 0.0):.4f})\n"
    return text

def analyze_with_llm(model, tokenizer, context_text, trigger_reason, system_prompt, user_prompt):
    """Zentrale Funktion, die die LLM aufruft."""
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔔 TRIGGER: {trigger_reason}")
    
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nEreignis: {trigger_reason}\n\n"
        f"Hier sind die aktuellen Daten:\n{context_text}\n\n"
        f"Aufgabe: {user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids, max_new_tokens=256, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print("\n" + "*"*50)
    print("  ANTWORT VON QWEN:")
    print("*"*50)
    print(response.strip())
    print("*"*50 + "\n")

def watch_silence(model, tokenizer, args, last_ride_time_ref):
    """
    Überwacht, ob der Lift zu lange still steht (Ausfall-Erkennung).
    (Platzhalter für deine zukünftige Kalender-Logik)
    """
    while True:
        time.sleep(60) # Jede Minute prüfen
        now = time.time()
        # Beispiel: Wenn es zwischen 8:00 und 18:00 Uhr ist (High Traffic)
        # und seit 2 Stunden keine Fahrt mehr stattfand -> Alarm!
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 18:
            if (now - last_ride_time_ref[0]) > 7200: # 7200 Sekunden = 2 Stunden
                context = "Es wurden seit über 2 Stunden keine Fahrten mehr aufgezeichnet, obwohl Hochbetrieb ist."
                analyze_with_llm(model, tokenizer, context, "Möglicher Lift-Ausfall (Stille)", 
                                 args.system_prompt, "Was könnte die Ursache sein? Sollte ein Techniker gerufen werden?")
                last_ride_time_ref[0] = now # Timer zurücksetzen, um Spam zu vermeiden

def main():
    ap = argparse.ArgumentParser(description="Fortlaufender Qwen LLM Analyzer")
    ap.add_argument("--model-path", type=str, required=True, help="Pfad zum Qwen-Modell")
    ap.add_argument("--events-jsonl", type=Path, default=Path("../live_data/raw/events.jsonl"))
    ap.add_argument("--analyze_all", action="store_true", help="Jede Fahrt analysieren (nicht nur Anomalien)")
    ap.add_argument("--system-prompt", type=str, 
                    default="Du bist der zentrale Sicherheits-Assistent des Aufzugs. "
                            "Analysiere die Sensordaten, Zeitpläne oder Hilferufe und fasse dich kurz.")
    ap.add_argument("--user-prompt", type=str, 
                    default="Bewerte die Situation und nenne die wahrscheinlichste Ursache.")
    
    args = ap.parse_args()

    print(f"[i] Lade Modell von {args.model_path}... (Bitte warten)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="cpu", torch_dtype=torch.float32, trust_remote_code=True
    )
    print("[✓] Modell geladen. Lausche auf Trigger...\n")

    # Referenz auf die letzte Fahrt (für den Silence-Watcher)
    last_ride_time_ref = [time.time()]

    # Starte den Hintergrund-Thread für Stille-Erkennung (Ausfall)
    silence_thread = threading.Thread(target=watch_silence, args=(model, tokenizer, args, last_ride_time_ref), daemon=True)
    silence_thread.start()

    while not args.events_jsonl.exists():
        time.sleep(1)

    current_ride_events = []
    current_ride_id = None

    with open(args.events_jsonl, "r", encoding="utf-8") as f:
        f.seek(0, 2) # Ans Ende springen
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            
            line = line.strip()
            if not line: continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Event einsammeln
            if rec.get("kind") == "event":
                rid = rec.get("ride_id")
                if rid != current_ride_id:
                    current_ride_id = rid
                    current_ride_events = []
                current_ride_events.append(rec)

            # Fahrt beendet
            elif rec.get("kind") == "ride_end":
                rid = rec.get("ride_id")
                summary = rec.get("summary", {})
                alarmed = summary.get("alarmed", False)
                
                # Update der letzten Fahrt-Zeit
                last_ride_time_ref[0] = time.time()

                # TRIGGER-LOGIK
                # 1. Wenn Anomalie -> Immer auswerten
                # 2. Wenn --analyze_all gesetzt ist -> Jede Fahrt auswerten
                if alarmed or args.analyze_all:
                    reason = "ANOMALIE ERKANNT" if alarmed else "Reguläre Fahrtbeobachtung"
                    context = build_ride_context(current_ride_events, summary, rid)
                    analyze_with_llm(model, tokenizer, context, reason, args.system_prompt, args.user_prompt)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Fahrt #{rid} war normal. (Ignoriert)")

                current_ride_events = []
                current_ride_id = None

if __name__ == "__main__":
    main()