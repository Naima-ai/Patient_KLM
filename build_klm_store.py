"""
Step 4: Build the KLM Triple Store
Uses SQLite for persistence + fast querying.
The Meta Model will query this store to retrieve relevant triples.
"""

import json
import sqlite3
import os
from datetime import datetime


DB_PATH = "data/patient_klm.db"


def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS triples (
        triple_id       TEXT PRIMARY KEY,
        head            TEXT NOT NULL,
        relation        TEXT NOT NULL,
        tail            TEXT NOT NULL,
        confidence      REAL NOT NULL,
        evidence_level  TEXT NOT NULL,
        source          TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        klm_source      TEXT NOT NULL DEFAULT 'patient_klm'
    );

    -- Indexes for common Meta Model query patterns
    CREATE INDEX IF NOT EXISTS idx_head       ON triples(head);
    CREATE INDEX IF NOT EXISTS idx_relation   ON triples(relation);
    CREATE INDEX IF NOT EXISTS idx_tail       ON triples(tail);
    CREATE INDEX IF NOT EXISTS idx_timestamp  ON triples(timestamp);
    CREATE INDEX IF NOT EXISTS idx_klm_source ON triples(klm_source);
    CREATE INDEX IF NOT EXISTS idx_confidence ON triples(confidence);

    -- Full-text search over head + tail content
    CREATE VIRTUAL TABLE IF NOT EXISTS triples_fts USING fts5(
        triple_id UNINDEXED,
        head,
        relation,
        tail,
        content='triples',
        content_rowid='rowid'
    );

    CREATE TRIGGER IF NOT EXISTS triples_ai AFTER INSERT ON triples BEGIN
        INSERT INTO triples_fts(rowid, triple_id, head, relation, tail)
        VALUES (new.rowid, new.triple_id, new.head, new.relation, new.tail);
    END;
    """)
    conn.commit()
    print("Schema created")


def load_triples(conn, triples_path: str):
    with open(triples_path) as f:
        data = json.load(f)

    triples = data["triples"]
    inserted = 0
    skipped = 0

    for t in triples:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO triples
                (triple_id, head, relation, tail, confidence, evidence_level, source, timestamp, klm_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                t["triple_id"],
                t["head"],
                t["relation"],
                t["tail"],
                t["confidence"],
                t["evidence_level"],
                t["source"],
                t["timestamp"],
                t["klm_source"]
            ))
            inserted += 1
        except Exception as e:
            print(f"    Skipped {t.get('triple_id','?')}: {e}")
            skipped += 1

    conn.commit()
    print(f" Loaded {inserted} triples ({skipped} skipped)")
    return inserted


def verify_store(conn):
    count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    relations = conn.execute(
        "SELECT relation, COUNT(*) as cnt FROM triples GROUP BY relation ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    sources = conn.execute(
        "SELECT klm_source, COUNT(*) as cnt FROM triples GROUP BY klm_source"
    ).fetchall()

    print(f"\n📊 KLM Store Statistics:")
    print(f"   Total triples: {count}")
    print(f"\n   Top relations:")
    for rel, cnt in relations:
        print(f"     {rel}: {cnt}")
    print(f"\n   By KLM source:")
    for src, cnt in sources:
        print(f"     {src}: {cnt}")


def export_summary(conn):
    """Export a summary JSON that the Meta Model can use for initial context."""
    rows = conn.execute(
        "SELECT * FROM triples ORDER BY timestamp, triple_id"
    ).fetchall()

    cols = ["triple_id", "head", "relation", "tail", "confidence",
            "evidence_level", "source", "timestamp", "klm_source"]

    triples_list = [dict(zip(cols, row)) for row in rows]

    # Group by timeline stage
    timeline = {}
    for t in triples_list:
        year = t["timestamp"][:4]
        if year not in timeline:
            timeline[year] = []
        timeline[year].append(t)

    summary = {
        "klm_id": "patient_klm_1",
        "patient_id": "P-001",
        "exported_at": datetime.now().isoformat(),
        "total_triples": len(triples_list),
        "db_path": DB_PATH,
        "timeline_summary": {
            year: {
                "count": len(ts),
                "relations": list(set(t["relation"] for t in ts))
            }
            for year, ts in timeline.items()
        },
        "all_triples": triples_list
    }

    with open("data/klm_export.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n Full export saved to data/klm_export.json")
    return summary


def main():
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    print("\nLoading triples into store...")
    load_triples(conn, "data/triples.json")

    verify_store(conn)
    export_summary(conn)

    conn.close()
    print(f"\n KLM store ready at {DB_PATH}")


if __name__ == "__main__":
    main()
