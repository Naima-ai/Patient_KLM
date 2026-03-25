"""
Patient Knowledge Model (KLM) API Interface

Provides structured access to patient knowledge stored as triples.
Supports:
- Full patient retrieval
- Domain-specific filtering (klm_source)
- Timeline, genomics, labs, medications
- Semantic search
- Agent-ready context formatting
"""

import sqlite3
from typing import Optional, List, Dict
from datetime import datetime


class PatientKLM:
    """Patient Knowledge Model query interface."""

    def __init__(self, db_path: str = "data/patient_klm.db"):
        self.db_path = db_path
        self._conn = None

    # ─────────────────────────────────────────────
    # Connection Handling
    # ─────────────────────────────────────────────

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _rows_to_triples(self, rows) -> List[Dict]:
        return [dict(row) for row in rows]

    # ─────────────────────────────────────────────
    # Core Retrieval
    # ─────────────────────────────────────────────

    def get_by_patient(
        self,
        patient_id: str,
        relation: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """Retrieve all triples for a patient."""
        conn = self._get_conn()

        if relation:
            rows = conn.execute("""
                SELECT * FROM triples
                WHERE head = ? AND relation = ? AND confidence >= ?
                ORDER BY timestamp
            """, (patient_id, relation, min_confidence)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM triples
                WHERE head = ? AND confidence >= ?
                ORDER BY timestamp
            """, (patient_id, min_confidence)).fetchall()

        return self._rows_to_triples(rows)

    def get_by_patient_and_source(
        self,
        patient_id: str,
        klm_source: str,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """Retrieve triples filtered by KLM source (for specialist agents)."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ?
              AND klm_source = ?
              AND confidence >= ?
            ORDER BY timestamp
        """, (patient_id, klm_source, min_confidence)).fetchall()

        return self._rows_to_triples(rows)

    def get_by_patient_multi_source(
        self,
        patient_id: str,
        sources: List[str]
    ) -> List[Dict]:
        """Retrieve triples from multiple domains."""
        conn = self._get_conn()

        placeholders = ",".join(["?"] * len(sources))

        rows = conn.execute(f"""
            SELECT * FROM triples
            WHERE head = ?
              AND klm_source IN ({placeholders})
            ORDER BY timestamp
        """, [patient_id] + sources).fetchall()

        return self._rows_to_triples(rows)

    # ─────────────────────────────────────────────
    # Clinical Views
    # ─────────────────────────────────────────────

    def get_disease_timeline(self, patient_id: str) -> List[Dict]:
        """Disease progression timeline."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ?
              AND relation IN (
                'disease_progression_stage',
                'diagnosed_with',
                'has_imaging_finding',
                'has_lab_value'
              )
            ORDER BY timestamp
        """, (patient_id,)).fetchall()

        return self._rows_to_triples(rows)

    def get_genomic_context(self, patient_id: str) -> List[Dict]:
        """Genomic / DNA-related triples."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM triples
            WHERE (
                head = ?
                OR head LIKE '%variant%'
                OR head LIKE 'FRAG-%'
            )
            AND relation IN (
                'carries_genetic_variant',
                'predisposes_to',
                'has_polygenic_risk_score',
                'has_pharmacogenomic_profile',
                'has_dna_fragment',
                'contains_mutation_site',
                'has_family_history'
            )
            ORDER BY timestamp
        """, (patient_id,)).fetchall()

        return self._rows_to_triples(rows)

    def get_latest_labs(self, patient_id: str) -> List[Dict]:
        """Most recent lab values."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ? AND relation = 'has_lab_value'
            ORDER BY timestamp DESC
            LIMIT 20
        """, (patient_id,)).fetchall()

        return self._rows_to_triples(rows)

    def get_medications(self, patient_id: str) -> List[Dict]:
        """Current medications."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ? AND relation = 'prescribed_medication'
            ORDER BY timestamp DESC
        """, (patient_id,)).fetchall()

        return self._rows_to_triples(rows)

    # ─────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Semantic / keyword search."""
        conn = self._get_conn()

        try:
            rows = conn.execute("""
                SELECT t.* FROM triples t
                INNER JOIN triples_fts fts ON t.rowid = fts.rowid
                WHERE triples_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
        except Exception:
            like = f"%{query}%"
            rows = conn.execute("""
                SELECT * FROM triples
                WHERE head LIKE ? OR relation LIKE ? OR tail LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (like, like, like, limit)).fetchall()

        return self._rows_to_triples(rows)

    # ─────────────────────────────────────────────
    # Agent Context
    # ─────────────────────────────────────────────

    def get_agent_context(self, patient_id: str) -> Dict:
        """Structured context for LLM agents."""

        timeline = self.get_disease_timeline(patient_id)
        genomic = self.get_genomic_context(patient_id)
        labs = self.get_latest_labs(patient_id)
        meds = self.get_medications(patient_id)
        all_triples = self.get_by_patient(patient_id, min_confidence=0.8)

        stages = [
            {"date": t["timestamp"], "stage": t["tail"]}
            for t in timeline
            if t["relation"] == "disease_progression_stage"
        ]

        genetic_risks = list(set([
            t["tail"] if t["relation"] != "has_polygenic_risk_score"
            else f"PRS: {t['tail']}"
            for t in genomic
            if t["relation"] in ["predisposes_to", "has_polygenic_risk_score"]
        ]))

        return {
            "patient_id": patient_id,
            "generated_at": datetime.now().isoformat(),
            "disease_progression": stages,
            "genetic_risks": genetic_risks,
            "medications": [t["tail"] for t in meds],
            "recent_labs": [t["tail"] for t in labs[:10]],
            "total_triples": len(all_triples),
            "all_triples": all_triples,
            "genomic_triples": genomic
        }

    def to_prompt_context(self, patient_id: str, max_triples: int = 40) -> str:
        """Format context as LLM-ready prompt string."""

        ctx = self.get_agent_context(patient_id)

        lines = [
            f"=== PATIENT: {patient_id} ===",
            f"Generated: {ctx['generated_at']}",
            "",
            "## Disease Timeline"
        ]

        for s in ctx["disease_progression"]:
            lines.append(f"[{s['date']}] {s['stage']}")

        lines += ["", "## Genetic Risks"]
        for r in ctx["genetic_risks"]:
            lines.append(f"- {r}")

        lines += ["", "## Medications"]
        for m in ctx["medications"]:
            lines.append(f"- {m}")

        lines += ["", "## Recent Labs"]
        for l in ctx["recent_labs"]:
            lines.append(f"- {l}")

        lines += ["", "## Top Knowledge Triples"]

        sorted_triples = sorted(
            ctx["all_triples"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:max_triples]

        for t in sorted_triples:
            lines.append(
                f"[{t['triple_id']}] {t['head']} --[{t['relation']}]--> {t['tail']} "
                f"(conf={t['confidence']:.2f}, {t['timestamp']})"
            )

        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────

    def stats(self) -> Dict:
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

        relations = conn.execute("""
            SELECT relation, COUNT(*) 
            FROM triples 
            GROUP BY relation 
            ORDER BY COUNT(*) DESC
        """).fetchall()

        return {
            "total_triples": total,
            "relations": {r[0]: r[1] for r in relations}
        }


# ─────────────────────────────────────────────
# Demo Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Patient KLM API Test ===\n")

    klm = PatientKLM("data/patient_klm.db")

    print("Stats:")
    print(klm.stats())

    print("\nTimeline:")
    for t in klm.get_disease_timeline("PT-8839-CR")[:5]:
        print(t)

    print("\nSearch: hypertension")
    for t in klm.search("hypertension")[:5]:
        print(t)

    print("\nAgent Context Preview:")
    print(klm.to_prompt_context("PT-8839-CR")[:1000])

    klm.close()
