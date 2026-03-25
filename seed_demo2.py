"""
seed_demo2.py — Load Demo #2 patient (PT-8839-CR) into the existing Patient KLM.

This script reads your 4 domain JSON files and loads all triples directly
into the same patient_klm.db used by Demo #1 (P-001). No existing data is
touched. PT-8839-CR will immediately be available via all existing GET endpoints.

Usage:
    python seed_demo2.py

    # Optional: point at a different DB
    PATIENT_KLM_DB_PATH=/path/to/other.db python seed_demo2.py


"""

import json
import hashlib
import os
import re
import sqlite3
import sys
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

PATIENT_ID = "PT-8839-CR"

# Edit these paths if your JSON files live elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO2_DIR = os.path.join(BASE_DIR, "demo_2") 

PATHS = {
    "patient":      os.path.join(DEMO2_DIR, "patient_triples.json"),
    "nephrology":   os.path.join(DEMO2_DIR, "nephrology_triples.json"),
    "cardiology":   os.path.join(DEMO2_DIR, "cardiology_triples.json"),
    "hypertension": os.path.join(DEMO2_DIR, "hypertension_triples.json"),
}

DB_PATH = os.environ.get(
    "PATIENT_KLM_DB_PATH",
    os.path.join(BASE_DIR, "data", "patient_klm.db")
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_id(head: str, relation: str, tail: str, timestamp: str) -> str:
    """Same ID generation as patient_klm_endpoint.py."""
    content = f"{head}{relation}{tail}{timestamp}"
    return "RT" + hashlib.md5(content.encode()).hexdigest()[:6].upper()


def strip_js_comments(text: str) -> str:
    """
    Remove JS-style // comments from JSON text.
    Your patient.json uses these — standard json.loads() rejects them.
    """
    # Remove // ... comments (not inside strings)
    text = re.sub(r'//[^\n]*', '', text)
    # Remove trailing commas before } or ] (common after comment removal)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def load_json_file(path: str, label: str) -> list[dict]:
    """
    Load a JSON file and return a flat list of triples.
    Handles two formats:
      - Bare array:           [ {...}, {...} ]
      - Wrapped object:       { "klm_id": "...", "triples": [ {...} ] }
    Also strips JS // comments.
    """
    if not os.path.exists(path):
        print(f"  [WARN] {label}: file not found at {path} — skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = strip_js_comments(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] {label}: JSON parse failed — {e}")
        return []

    # Unwrap if it's a dict with a "triples" key
    if isinstance(data, dict):
        triples = data.get("triples", [])
    elif isinstance(data, list):
        triples = data
    else:
        print(f"  [WARN] {label}: unexpected JSON structure — skipping")
        return []

    print(f"  {label}: found {len(triples)} triples")
    return triples


def normalise_triple(t: dict, domain: str) -> tuple | None:
    """
    Convert a raw triple dict into the 9-tuple expected by the DB schema:
    (triple_id, head, relation, tail, confidence, evidence_level,
     source, timestamp, klm_source)

    Fills in sensible defaults for missing fields.
    """
    head      = t.get("head", "").strip()
    relation  = t.get("relation", "").strip()
    tail      = str(t.get("tail", "")).strip()

    if not head or not relation or not tail:
        return None  # skip incomplete triples

    timestamp      = t.get("timestamp", datetime.now().strftime("%Y-%m-%d"))
    confidence     = float(t.get("confidence", 0.85))
    evidence_level = t.get("evidence_level", "II")   # default II if missing
    source         = t.get("source", domain.upper())
    klm_source     = f"patient_klm_{domain}"          # e.g. patient_klm_cardiology

    triple_id = make_id(head, relation, tail, timestamp)

    return (triple_id, head, relation, tail,
            confidence, evidence_level, source, timestamp, klm_source)


def store_triples(conn: sqlite3.Connection, tuples: list[tuple]) -> tuple[int, int]:
    """Insert triples; return (inserted, skipped) counts."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Demo #2 KLM Seed — Patient {PATIENT_ID}")
    print(f"  Database: {DB_PATH}")
    print("=" * 60)

    # Verify DB exists
    if not os.path.exists(DB_PATH):
        print(f"\n[ERROR] Database not found at {DB_PATH}")
        print("  Run your existing run_pipeline.py first to initialise the DB,")
        print("  then re-run this script.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    total_inserted = 0
    total_skipped  = 0

    # Load each domain file
    for domain, path in PATHS.items():
        print(f"\nLoading {domain} ...")
        raw_triples = load_json_file(path, domain)
        tuples = [normalise_triple(t, domain) for t in raw_triples]
        ins, skp = store_triples(conn, tuples)
        print(f"  → stored {ins}, skipped {skp}")
        total_inserted += ins
        total_skipped  += skp

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    print(f"  Done!")
    print(f"  Total triples stored : {total_inserted}")
    print(f"  Total skipped        : {total_skipped}")
    print(f"\n  Patient {PATIENT_ID} is now available via:")
    print(f"    GET /patient/{PATIENT_ID}")
    print(f"    GET /patient/{PATIENT_ID}/timeline")
    print(f"    GET /patient/{PATIENT_ID}/genomics")
    print("=" * 60)


if __name__ == "__main__":
    main()
