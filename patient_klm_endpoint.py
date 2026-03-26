"""
patient_klm_endpoint.py
Simple REST API for the Patient KLM.

Endpoints:
  GET  /patient/{patient_id}           — all patient triples as JSON (for prompt injection)
  GET  /patient/{patient_id}/timeline  — disease progression timeline
  GET  /patient/{patient_id}/genomics  — DNA and genetic profile
  POST /patient                        — add a new patient with full profile
  POST /patient/{patient_id}/visit     — add a new EHR visit for existing patient
  POST /triple                         — add any custom triple directly

Run:
  pip install fastapi uvicorn
  python patient_klm_endpoint.py

Docs: http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sqlite3
import hashlib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klm_api import PatientKLM

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Patient KLM API",
    description="Patient Knowledge Model — EHR + DNA triples for agent prompt injection",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = os.environ.get(
    "PATIENT_KLM_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_klm.db")
)


def get_klm():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="Database not found. Run run_pipeline.py first.")
    return PatientKLM(DB_PATH)


def get_conn():
    return sqlite3.connect(DB_PATH)


def make_id(head, relation, tail, date):
    content = f"{head}{relation}{tail}{date}"
    return "RT" + hashlib.md5(content.encode()).hexdigest()[:6].upper()


def store_triples(triples: list):
    conn = get_conn()
    try:
        conn.executemany("""
            INSERT OR REPLACE INTO triples
            (triple_id, head, relation, tail, confidence,
             evidence_level, source, timestamp, klm_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, triples)
        conn.commit()
    finally:
        conn.close()


# ── Input models ──────────────────────────────────────────────────────────────

class NewPatient(BaseModel):
    """
    Full new patient profile.
    Only patient_id and name are required — everything else is optional.
    """
    patient_id: str
    name: str
    dob: Optional[str] = None
    sex: Optional[str] = None
    ethnicity: Optional[str] = None
    blood_type: Optional[str] = None
    baseline_conditions: list[str] = []
    # Optional first visit
    visit_date: Optional[str] = None
    symptoms: list[str] = []
    vitals: dict = {}
    lab_results: dict = {}
    diagnosis_codes: list[str] = []
    medications: list[str] = []
    imaging: dict = {}
    clinical_notes: Optional[str] = None
    # Optional genomic variants
    genetic_variants: list[dict] = []
    # e.g. [{"gene": "VHL", "variant_id": "rs123", "clinical_significance": "pathogenic", "associated_condition": "RCC"}]


class NewVisit(BaseModel):
    """A new EHR visit for an existing patient."""
    visit_date: str
    symptoms: list[str] = []
    vitals: dict = {}
    lab_results: dict = {}
    diagnosis_codes: list[str] = []
    medications: list[str] = []
    imaging: dict = {}
    clinical_notes: Optional[str] = None


class CustomTriple(BaseModel):
    """Any custom triple — use when the other endpoints don't cover your case."""
    patient_id: str
    head: str
    relation: str
    tail: str
    confidence: float = 0.85
    evidence_level: str = "III"
    source: str
    timestamp: Optional[str] = None
    klm_source: str = "patient_klm"


# ── Helper ────────────────────────────────────────────────────────────────────

def visit_to_triples(pid, visit_date, symptoms, vitals, lab_results,
                     diagnosis_codes, medications, imaging, clinical_notes, source):
    triples = []

    def t(head, relation, tail, confidence=0.95, evidence_level="II"):
        return (make_id(head, relation, tail, visit_date),
                head, relation, tail, confidence, evidence_level,
                source, visit_date, "patient_klm")

    for symptom in symptoms:
        triples.append(t(pid, "has_symptom", symptom))

    for key, value in vitals.items():
        triples.append(t(pid, "has_vital", f"{key}:{value}", 0.99, "I"))

    lab_map = {
        "creatinine_mg_dl": ("creatinine", "mg/dL"),
        "egfr_ml_min":      ("eGFR", "mL/min/1.73m²"),
        "bun_mg_dl":        ("BUN", "mg/dL"),
        "hemoglobin_g_dl":  ("hemoglobin", "g/dL"),
    }
    for key, value in lab_results.items():
        label, unit = lab_map.get(key, (key, ""))
        triples.append(t(pid, "has_lab_value", f"{label}:{value} {unit}".strip(), 0.99, "I"))

    for dx in diagnosis_codes:
        triples.append(t(pid, "diagnosed_with", dx))

    for med in medications:
        triples.append(t(pid, "prescribed_medication", med))

    if imaging.get("type", "none") != "none" and imaging.get("findings"):
        triples.append(t(pid, "has_imaging_finding",
                         f"{imaging['type']}:{imaging['findings'][:120]}"))

    if clinical_notes:
        triples.append(t(pid, "has_clinical_note", clinical_notes[:200], 0.85, "III"))

    return triples


# ── READ endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "service": "Patient KLM", "docs": "/docs"}


@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    """
    All triples for a patient as JSON — ready for agent prompt injection.
    This is the main read endpoint.

    Example:
        GET /patient/P-001
    """
    klm = get_klm()
    triples = klm.get_by_patient(patient_id)
    klm.close()
    if not triples:
        raise HTTPException(status_code=404, detail=f"No data found for {patient_id}")
    return {
        "patient_id": patient_id,
        "total_triples": len(triples),
        "triples": triples
    }


@app.get("/patient/{patient_id}/timeline")
def get_timeline(patient_id: str):
    """
    Disease progression timeline — diagnoses, labs, imaging ordered by date.

    Example:
        GET /patient/P-001/timeline
    """
    klm = get_klm()
    result = klm.get_disease_timeline(patient_id)
    klm.close()
    return {"patient_id": patient_id, "timeline": result}


@app.get("/patient/{patient_id}/genomics")
def get_genomics(patient_id: str):
    """
    DNA and genetic profile — variants, risk scores, pharmacogenomics.

    Example:
        GET /patient/P-001/genomics
    """
    klm = get_klm()
    result = klm.get_genomic_context(patient_id)
    klm.close()
    return {"patient_id": patient_id, "genomics": result}


@app.get("/patient/{patient_id}/domain/{domain}")
def get_by_domain(patient_id: str, domain: str):
    """
    Search all triples for a patient filtered by clinical domain.

    Available domains:
      pathology      — disease-level knowledge triples (from pathology KLM)
      cardiology     — cardiology-related triples (klm_source contains 'cardiology')
      nephrology     — nephrology-related triples
      hypertension   — hypertension-related triples
      genomics       — DNA and genetic triples
      ehr            — EHR visit triples only

    Domains are matched against the klm_source field AND tail/relation content,
    so you get results even when triples are stored under a combined source.

    Examples:
        GET /patient/P-003/domain/cardiology
        GET /patient/P-003/domain/pathology
        GET /patient/P-003/domain/hypertension
        GET /patient/P-001/domain/genomics
    """
    conn = get_conn()

    domain_lower = domain.lower()

    # Map domain keywords to what we search for in klm_source, relation, and tail
    DOMAIN_KEYWORDS = {
        "pathology":    ["pathology_klm", "pathology"],
        "cardiology":   ["cardiology", "cardiac", "heart", "atrial", "ventricular", "afib"],
        "nephrology":   ["nephrology", "renal", "kidney", "ckd", "egfr", "creatinine"],
        "hypertension": ["hypertension", "blood_pressure", "antihypertensive", "bp"],
        "genomics":     ["genomics", "variant", "dna", "gene"],
        "ehr":          ["ehr", "visit"],
    }

    keywords = DOMAIN_KEYWORDS.get(domain_lower)
    if not keywords:
        available = ", ".join(DOMAIN_KEYWORDS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{domain}'. Available: {available}"
        )

    # Build a query that checks klm_source, relation, and tail for any keyword
    like_clauses = " OR ".join(
        ["klm_source LIKE ?", "relation LIKE ?", "tail LIKE ?"] * len(keywords)
    )
    params = []
    for kw in keywords:
        params += [f"%{kw}%", f"%{kw}%", f"%{kw}%"]

    query = f"""
        SELECT * FROM triples
        WHERE head = ?
          AND ({like_clauses})
        ORDER BY timestamp
    """

    rows = conn.execute(query, [patient_id] + params).fetchall()
    conn.close()

    cols = ["triple_id", "head", "relation", "tail", "confidence",
            "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, row)) for row in rows]

    return {
        "patient_id": patient_id,
        "domain": domain_lower,
        "total_triples": len(triples),
        "triples": triples
    }


# ── WRITE endpoints ───────────────────────────────────────────────────────────

@app.post("/patient")
def add_patient(data: NewPatient):
    """
    Add a completely new patient to the KLM.
    Only patient_id and name are required. Everything else is optional.

    Minimal example:
    {
        "patient_id": "P-002",
        "name": "John Smith"
    }

    Full example:
    {
        "patient_id": "P-002",
        "name": "John Smith",
        "dob": "1975-06-20",
        "sex": "Male",
        "blood_type": "O+",
        "baseline_conditions": ["hypertension"],
        "visit_date": "2026-03-10",
        "symptoms": ["fatigue", "flank pain"],
        "lab_results": {"creatinine_mg_dl": 1.4, "egfr_ml_min": 65},
        "diagnosis_codes": ["ICD-10: N18.2"],
        "medications": ["Losartan 50mg daily"],
        "genetic_variants": [
            {
                "gene": "VHL",
                "variant_id": "rs123456",
                "clinical_significance": "pathogenic",
                "associated_condition": "Clear cell RCC"
            }
        ]
    }
    """
    pid = data.patient_id
    today = datetime.now().strftime("%Y-%m-%d")
    triples = []

    def t(head, relation, tail, confidence=0.99, evidence_level="I"):
        return (make_id(head, relation, tail, today),
                head, relation, tail, confidence, evidence_level,
                "DEMOGRAPHICS", today, "patient_klm")

    # Demographics
    triples.append(t(pid, "has_attribute", f"name:{data.name}"))
    if data.dob:        triples.append(t(pid, "has_attribute", f"dob:{data.dob}"))
    if data.sex:        triples.append(t(pid, "has_attribute", f"sex:{data.sex}"))
    if data.ethnicity:  triples.append(t(pid, "has_attribute", f"ethnicity:{data.ethnicity}"))
    if data.blood_type: triples.append(t(pid, "has_attribute", f"blood_type:{data.blood_type}"))
    for condition in data.baseline_conditions:
        triples.append(t(pid, "has_baseline_condition", condition, 0.95, "II"))

    # Initial visit
    visit_triples = []
    if data.visit_date:
        visit_triples = visit_to_triples(
            pid, data.visit_date, data.symptoms, data.vitals,
            data.lab_results, data.diagnosis_codes, data.medications,
            data.imaging, data.clinical_notes,
            source=f"EHR:visit_{data.visit_date}"
        )
        triples.extend(visit_triples)

    # Genomic variants
    genomic_count = 0
    for v in data.genetic_variants:
        gene        = v.get("gene", "unknown")
        variant_id  = v.get("variant_id", "unknown")
        significance= v.get("clinical_significance", "unknown")
        condition   = v.get("associated_condition", "unknown")
        triples.append((
            make_id(pid, "carries_genetic_variant", f"{gene}:{variant_id}", today),
            pid, "carries_genetic_variant", f"{gene}:{variant_id}:{significance}",
            0.88, "II", "GENOMICS", today, "patient_klm"
        ))
        if significance in ["pathogenic", "likely pathogenic"]:
            triples.append((
                make_id(f"{gene}_variant", "predisposes_to", condition, today),
                f"{gene}_variant", "predisposes_to", condition,
                0.85, "II", "GENOMICS", today, "patient_klm"
            ))
        genomic_count += 1

    store_triples(triples)
    return {
        "status": "stored",
        "patient_id": pid,
        "total_triples_stored": len(triples)
    }


@app.post("/patient/{patient_id}/visit")
def add_visit(patient_id: str, visit: NewVisit):
    """
    Add a new EHR visit for an existing patient.

    Example:
        POST /patient/P-001/visit
    {
        "visit_date": "2026-03-10",
        "symptoms": ["fatigue", "ankle swelling"],
        "vitals": {"blood_pressure": "145/90", "weight_kg": 78.5},
        "lab_results": {"creatinine_mg_dl": 1.8, "egfr_ml_min": 52},
        "diagnosis_codes": ["ICD-10: N18.3"],
        "medications": ["Amlodipine 5mg daily"],
        "imaging": {"type": "ultrasound", "findings": "Reduced kidney size bilateral"},
        "clinical_notes": "CKD progression noted"
    }
    """
    triples = visit_to_triples(
        patient_id, visit.visit_date, visit.symptoms, visit.vitals,
        visit.lab_results, visit.diagnosis_codes, visit.medications,
        visit.imaging, visit.clinical_notes,
        source=f"EHR:visit_{visit.visit_date}"
    )
    store_triples(triples)
    return {
        "status": "stored",
        "patient_id": patient_id,
        "visit_date": visit.visit_date,
        "triples_stored": len(triples)
    }


@app.post("/triple")
def add_triple(triple: CustomTriple):
    """
    Add any custom triple directly.
    Use this for anything not covered by the other endpoints
    e.g. risk factors, family history, social history, custom findings.

    Example:
        POST /triple
    {
        "patient_id": "P-001",
        "head": "P-001",
        "relation": "has_risk_factor",
        "tail": "smoking:20_pack_years",
        "confidence": 0.90,
        "evidence_level": "II",
        "source": "patient_intake_form",
        "timestamp": "2026-03-10"
    }
    """
    timestamp = triple.timestamp or datetime.now().strftime("%Y-%m-%d")
    triple_id = make_id(triple.head, triple.relation, triple.tail, timestamp)
    store_triples([(
        triple_id, triple.head, triple.relation, triple.tail,
        triple.confidence, triple.evidence_level, triple.source,
        timestamp, triple.klm_source
    )])
    return {"status": "stored", "triple_id": triple_id}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Patient KLM Endpoint...")
    print(f"Database: {DB_PATH}")
    print(f"Docs:     http://localhost:8001/docs\n")
    uvicorn.run("patient_klm_endpoint:app", host="0.0.0.0", port=8001, reload=False)
