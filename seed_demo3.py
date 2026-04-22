"""
This patient has a dermatology-focused profile:
  - CDKN2A / MC1R variants (elevated melanoma susceptibility)
  - FKBP5 / COMT variants (HPA-axis dysregulation + stress-cortisol coupling)
  - Two EHR visits (2024-11-12 and 2025-11-15) showing disease progression
    from a latent pre-symptomatic state to an active stress-driven melanoma risk

Run:
    python seed_demo3.py
"""

import sqlite3
import json
import os
import sys

DB_PATH = os.environ.get(
    "PATIENT_KLM_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_klm.db")
)

# Domain KLM files for PT-9921
DEMO3_FILES = [
    "dermatology_dna.json",
    "dermatology_ehr.json",
]


def load_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def seed(db_path: str, files: list[str]):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at: {db_path}")
        print("        Run run_pipeline.py first to initialise the database.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    total_inserted = 0
    for fname in files:
        fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if not os.path.exists(fpath):
            print(f"[WARN]  {fname} not found at {fpath}, skipping.")
            continue

        triples = load_json(fpath)
        rows = [
            (
                t["triple_id"],
                t["head"],
                t["relation"],
                t["tail"],
                t["confidence"],
                t["evidence_level"],
                t["source"],
                t["timestamp"],
                t["klm_source"],
            )
            for t in triples
        ]

        conn.executemany(
            """
            INSERT OR REPLACE INTO triples
              (triple_id, head, relation, tail, confidence,
               evidence_level, source, timestamp, klm_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        print(f"[OK]    {fname}: {len(rows)} triples seeded.")
        total_inserted += len(rows)

    conn.close()
    print(f"\n[DONE]  PT-9921 (P-004) seeded — {total_inserted} total triples.")
    print(f"        Verify with:  GET /patient/PT-9921")
    print(f"        Domain view:  GET /patient/PT-9921/domain/dermatology")
    print(f"        Genomics:     GET /patient/PT-9921/genomics")


if __name__ == "__main__":
    print(f"Seeding Demo 3 — PT-9921 (P-004) into: {DB_PATH}\n")
    seed(DB_PATH, DEMO3_FILES)
