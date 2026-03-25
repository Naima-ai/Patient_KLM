"""
This is the interface that the Meta Model, Nephrologist Agent, 
and Patient Agent use to retrieve knowledge.

Usage:
  from klm_api import PatientKLM
  klm = PatientKLM("data/patient_klm.db")
  
  # Get all triples for patient
  triples = klm.get_by_patient("P-001")
  
  # Get disease progression
  progression = klm.get_disease_timeline("P-001")
  
  # Get genomic risk factors
  genomic = klm.get_genomic_context("P-001")
  
  # Semantic search (free text)
  results = klm.search("kidney tumor symptoms")
  
  # Full patient context for agent prompt injection
  context = klm.get_agent_context("P-001")
"""

import json
import sqlite3
from typing import Optional
from datetime import datetime


class PatientKLM:
    """Patient Knowledge Model query interface."""

    def __init__(self, db_path: str = "data/patient_klm.db"):
        self.db_path = db_path
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _rows_to_triples(self, rows) -> list[dict]:
        return [dict(row) for row in rows]

    def get_by_patient(self, patient_id, relation=None, min_confidence=0.0):
        conn = self._get_conn()
        
        # First get all tails from patient's genetic variant triples
        linked_heads = conn.execute("""
            SELECT DISTINCT substr(tail, 1, instr(tail||':', ':') - 1)
            FROM triples
            WHERE head = ? AND relation = 'carries_genetic_variant'
        """, (patient_id,)).fetchall()
        linked = [row[0] for row in linked_heads]
    
        # Build query to include patient + linked entity triples
        placeholders = ",".join("?" * len(linked))
        query = f"""
            SELECT * FROM triples
            WHERE (head = ? OR head IN ({placeholders}))
            AND confidence >= ?
            ORDER BY timestamp
        """ if linked else """
            SELECT * FROM triples
            WHERE head = ? AND confidence >= ?
            ORDER BY timestamp
        """
        
        params = [patient_id] + linked + [min_confidence] if linked else [patient_id, min_confidence]
        rows = conn.execute(query, params).fetchall()
        return self._rows_to_triples(rows)

    def get_disease_timeline(self, patient_id: str) -> list[dict]:
        """Get disease progression triples ordered by timestamp."""
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

    def get_genomic_context(self, patient_id: str) -> list[dict]:
        """Get all genomic / DNA-related triples."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM triples
            WHERE (head = ? OR head LIKE '%variant%' OR head LIKE 'FRAG-%')
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

    def get_latest_labs(self, patient_id: str) -> list[dict]:
        """Get the most recent lab values."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ? AND relation = 'has_lab_value'
            ORDER BY timestamp DESC
            LIMIT 20
        """, (patient_id,)).fetchall()
        return self._rows_to_triples(rows)

    def get_medications(self, patient_id: str) -> list[dict]:
        """Get current medications."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ? AND relation = 'prescribed_medication'
            ORDER BY timestamp DESC
        """, (patient_id,)).fetchall()
        return self._rows_to_triples(rows)

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text semantic search across all triple content."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT t.* FROM triples t
                INNER JOIN triples_fts fts ON t.rowid = fts.rowid
                WHERE triples_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return self._rows_to_triples(rows)
        except Exception:
            # Fallback: LIKE search
            like = f"%{query}%"
            rows = conn.execute("""
                SELECT * FROM triples
                WHERE head LIKE ? OR relation LIKE ? OR tail LIKE ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (like, like, like, limit)).fetchall()
            return self._rows_to_triples(rows)

    def get_triples_by_relation(self, relation: str) -> list[dict]:
        """Get all triples with a specific relation type."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM triples WHERE relation = ?
            ORDER BY confidence DESC
        """, (relation,)).fetchall()
        return self._rows_to_triples(rows)

    def get_high_confidence_triples(
        self,
        patient_id: str,
        threshold: float = 0.9
    ) -> list[dict]:
        """Get only high-confidence triples for the agent context."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM triples
            WHERE head = ? AND confidence >= ?
            ORDER BY timestamp, confidence DESC
        """, (patient_id, threshold)).fetchall()
        return self._rows_to_triples(rows)

    def get_agent_context(self, patient_id: str) -> dict:
        """
        Build a structured context object for injection into agent prompts.
        This is what the Patient Agent and Nephrologist Agent receive.
        """
        timeline = self.get_disease_timeline(patient_id)
        genomic = self.get_genomic_context(patient_id)
        latest_labs = self.get_latest_labs(patient_id)
        medications = self.get_medications(patient_id)
        all_triples = self.get_by_patient(patient_id, min_confidence=0.8)

        # Summarize disease stages
        stages = []
        for t in timeline:
            if t["relation"] == "disease_progression_stage":
                stages.append({
                    "date": t["timestamp"],
                    "stage": t["tail"]
                })

        # Extract genetic risk factors
        genetic_risks = []
        for t in genomic:
            if t["relation"] == "predisposes_to":
                genetic_risks.append(t["tail"])
            elif t["relation"] == "has_polygenic_risk_score":
                genetic_risks.append(f"PRS: {t['tail']}")

        context = {
            "patient_id": patient_id,
            "context_generated_at": datetime.now().isoformat(),
            "klm_source": "patient_klm",
            "disease_progression": stages,
            "genetic_risk_factors": list(set(genetic_risks)),
            "current_medications": [t["tail"] for t in medications],
            "recent_labs": [t["tail"] for t in latest_labs[:10]],
            "total_knowledge_triples": len(all_triples),
            "all_triples": all_triples,
            "genomic_triples": genomic,
        }
        return context

    def to_prompt_context(self, patient_id: str, max_triples: int = 40) -> str:
        """
        Format the KLM context as a string suitable for injection
        into an LLM prompt (for Patient or Nephrologist Agent).
        """
        ctx = self.get_agent_context(patient_id)

        lines = [
            f"=== PATIENT KNOWLEDGE MODEL: {patient_id} ===",
            f"Generated: {ctx['context_generated_at']}",
            "",
            "## Disease Progression Timeline",
        ]
        for stage in ctx["disease_progression"]:
            lines.append(f"  [{stage['date']}] {stage['stage']}")

        lines += ["", "## Genetic Risk Factors"]
        for risk in ctx["genetic_risk_factors"]:
            lines.append(f"  - {risk}")

        lines += ["", "## Current Medications"]
        for med in ctx["current_medications"]:
            lines.append(f"  - {med}")

        lines += ["", "## Recent Lab Values"]
        for lab in ctx["recent_labs"]:
            lines.append(f"  - {lab}")

        lines += ["", "## Knowledge Triples (top by confidence)"]
        sorted_triples = sorted(
            ctx["all_triples"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:max_triples]

        for t in sorted_triples:
            lines.append(
                f"  [{t['triple_id']}] {t['head']} --[{t['relation']}]--> {t['tail']} "
                f"(conf={t['confidence']:.2f}, {t['timestamp']})"
            )

        return "\n".join(lines)

    def stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        relations = conn.execute(
            "SELECT relation, COUNT(*) FROM triples GROUP BY relation ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {
            "total_triples": total,
            "relations": {r[0]: r[1] for r in relations}
        }

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ─────────────────────────────────────────────
# Demo / test when run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Patient KLM API Test ===\n")
    klm = PatientKLM("data/patient_klm.db")

    print(" Store stats:")
    stats = klm.stats()
    print(f"  Total triples: {stats['total_triples']}")
    for rel, cnt in list(stats["relations"].items())[:8]:
        print(f"  {rel}: {cnt}")

    print("\nDisease timeline:")
    for t in klm.get_disease_timeline("P-001"):
        print(f"  [{t['timestamp']}] {t['relation']}: {t['tail'][:80]}")

    print("\n Genomic context:")
    for t in klm.get_genomic_context("P-001"):
        print(f"  {t['relation']}: {t['tail'][:90]}")

    print("\n Search test: 'tumor'")
    for t in klm.search("tumor")[:5]:
        print(f"  {t['head']} -> {t['tail'][:80]}")

    print("\n Agent context string (first 60 lines):")
    ctx_str = klm.to_prompt_context("P-001")
    for line in ctx_str.split("\n")[:60]:
        print(" ", line)

    klm.close()
