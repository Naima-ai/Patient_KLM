"""
seed_p003.py
Loads P-003 (patient + pathology triples) into the existing patient_klm.db.

Run AFTER the pipeline steps have generated the JSON files:
    python demo_3/generate_ehr_p003.py
    python demo_3/generate_genomics_p003.py
    python demo_3/extract_triples_p003.py
    python demo_3/generate_pathology_p003.py

Then run this to load everything into the DB:
    python demo_3/seed_p003.py

P-003 will immediately be available via:
    GET /patient/P-003
    GET /patient/P-003/timeline
    GET /patient/P-003/genomics
"""

import json
import hashlib
import os
import re
import sqlite3
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "patient":   os.path.join(BASE_DIR, "data", "p003_patient_triples.json"),
    "pathology": os.path.join(BASE_DIR, "data", "p003_pathology_triples.json"),
}

DB_PATH = os.environ.get(
    "PATIENT_KLM_DB_PATH",
    os.path.join(BASE_DIR, "data", "patient_klm.db")
)


def make_id(head, relation, tail, timestamp):
    content = f"{head}{relation}{tail}{timestamp}"
    return "RT" + hashlib.md5(content.encode()).hexdigest()[:6].upper()


def strip_comments(text):
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def load_json_file(path, label):
    if not os.path.exists(path):
        print(f"  [WARN] {label}: file not found at {path} — skipping")
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = strip_comments(f.read())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] {label}: JSON parse failed — {e}")
        return []
    if isinstance(data, dict):
        triples = data.get("triples", [])
    elif isinstance(data, list):
        triples = data
    else:
        print(f"  [WARN] {label}: unexpected structure — skipping")
        return []
    print(f"  {label}: found {len(triples)} triples")
    return triples


def normalise(t, domain):
    head      = str(t.get("head", "")).strip()
    relation  = str(t.get("relation", "")).strip()
    tail      = str(t.get("tail", "")).strip()
    if not head or not relation or not tail:
        return None
    timestamp      = t.get("timestamp", datetime.now().strftime("%Y-%m-%d"))
    confidence     = float(t.get("confidence", 0.85))
    evidence_level = t.get("evidence_level", "II")
    source         = t.get("source", domain.upper())
    klm_source     = t.get("klm_source", f"patient_klm_{domain}")
    triple_id      = t.get("triple_id") or make_id(head, relation, tail, timestamp)
    return (triple_id, head, relation, tail,
            confidence, evidence_level, source, timestamp, klm_source)


def store(conn, tuples):
    inserted = skipped = 0
    for row in tuples:
        if row is None:
            skipped += 1
            continue
        try:
            conn.execute("""
                INSERT OR REPLACE INTO triples
                (triple_id, head, relation, tail, confidence,
                 evidence_level, source, timestamp, klm_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
            inserted += 1
        except Exception as e:
            print(f"    [SKIP] {row[0] if row else '?'}: {e}")
            skipped += 1
    conn.commit()
    return inserted, skipped


def main():
    print("=" * 60)
    print("  P-003 KLM Seed")
    print(f"  Database: {DB_PATH}")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"\n[ERROR] Database not found at {DB_PATH}")
        print("  Run run_pipeline.py first to initialise the DB.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    total_inserted = total_skipped = 0

    for domain, path in PATHS.items():
        print(f"\nLoading {domain} ...")
        raw = load_json_file(path, domain)
        tuples = [normalise(t, domain) for t in raw]
        ins, skp = store(conn, tuples)
        print(f"  → stored {ins}, skipped {skp}")
        total_inserted += ins
        total_skipped  += skp

    conn.close()

    print("\n" + "=" * 60)
    print(f"  Done!")
    print(f"  Total triples stored : {total_inserted}")
    print(f"  Total skipped        : {total_skipped}")
    print(f"\n  P-003 is now available via:")
    print(f"    GET /patient/P-003")
    print(f"    GET /patient/P-003/timeline")
    print(f"    GET /patient/P-003/genomics")
    print("=" * 60)


if __name__ == "__main__":
    main()
