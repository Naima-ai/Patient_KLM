"""
patient_klm_endpoint.py
Unified Patient KLM API — all endpoints on port 8001.

Patient KLM endpoints (existing):
  GET  /patient/{patient_id}                  — all triples for a patient
  GET  /patient/{patient_id}/timeline         — disease progression timeline
  GET  /patient/{patient_id}/genomics         — DNA and genetic profile
  GET  /patient/{patient_id}/domain/{domain}  — filter triples by clinical domain
  POST /patient                               — add a new patient
  POST /patient/{patient_id}/visit            — add a new EHR visit
  POST /triple                                — add any custom triple

KLM Builder endpoints (new, merged in):
  GET  /klm/list                              — list all custom KLMs
  POST /klm/create                            — create a new empty KLM
  POST /klm/{klm_name}/upload                 — upload a document, extract triples via Claude
  POST /klm/{klm_name}/triple                 — manually add a triple to a custom KLM
  POST /klm/{klm_name}/import_patients        — copy patients from patient_klm.db into a KLM
  GET  /klm/{klm_name}/all                    — all triples in a custom KLM
  GET  /klm/{klm_name}/query                  — search a custom KLM
  DELETE /klm/{klm_name}                      — delete a custom KLM
  GET  /patients/available                    — list patients available to import

Run locally:
  pip install fastapi uvicorn anthropic PyPDF2
  export ANTHROPIC_API_KEY=your_key   (optional — only needed for /upload)
  python patient_klm_endpoint.py

Docker:
  docker compose -f docker-compose.patient.yml up --build

Docs: http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from collections import Counter
import uvicorn
import sqlite3
import hashlib
import os
import json
import re
import anthropic
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klm_api import PatientKLM

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Patient KLM API",
    description=(
        "Patient Knowledge Model — EHR + DNA triples for agent prompt injection. "
        "Also includes KLM Builder: create custom Knowledge Models from any document."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Paths ─────────────────────────────────────────────────────────────────────

# Existing patient database (seeded patients: P-001, PT-8839-CR, PT-9921 etc.)
DB_PATH = os.environ.get(
    "PATIENT_KLM_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_klm.db")
)

# Custom KLMs created by the builder live in a sub-folder
KLM_DIR = os.environ.get(
    "KLM_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_klms")
)
os.makedirs(KLM_DIR, exist_ok=True)

# Anthropic key — optional, only needed for document upload triple extraction
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_id(head: str, relation: str, tail: str, date: str) -> str:
    content = f"{head}{relation}{tail}{date}"
    return "RT" + hashlib.md5(content.encode()).hexdigest()[:6].upper()


# ── Patient DB helpers ────────────────────────────────────────────────────────

def get_klm() -> PatientKLM:
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail="Database not found. Run run_pipeline.py / seed scripts first."
        )
    return PatientKLM(DB_PATH)


def get_patient_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def store_patient_triples(triples: list):
    conn = get_patient_conn()
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


# ── Custom KLM DB helpers ─────────────────────────────────────────────────────

def klm_path(klm_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", klm_name)
    return os.path.join(KLM_DIR, f"{safe}.db")


def klm_exists(klm_name: str) -> bool:
    return os.path.exists(klm_path(klm_name))


def get_klm_conn(klm_name: str) -> sqlite3.Connection:
    if not klm_exists(klm_name):
        raise HTTPException(
            status_code=404,
            detail=f"KLM '{klm_name}' not found. Create it first: POST /klm/create"
        )
    return sqlite3.connect(klm_path(klm_name))


def create_klm_schema(conn: sqlite3.Connection, klm_name: str, description: str):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klm_meta (
            klm_name    TEXT,
            description TEXT,
            created_at  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triples (
            triple_id      TEXT PRIMARY KEY,
            head           TEXT NOT NULL,
            relation       TEXT NOT NULL,
            tail           TEXT NOT NULL,
            confidence     REAL DEFAULT 0.85,
            evidence_level TEXT DEFAULT 'III',
            source         TEXT,
            timestamp      TEXT,
            klm_source     TEXT
        )
    """)
    conn.execute(
        "INSERT INTO klm_meta VALUES (?, ?, ?)",
        (klm_name, description, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()


def make_klm_id(head: str, relation: str, tail: str, ts: str) -> str:
    content = f"{head}{relation}{tail}{ts}"
    return "KT" + hashlib.md5(content.encode()).hexdigest()[:8].upper()


def insert_klm_triples(conn: sqlite3.Connection, triples: list):
    conn.executemany("""
        INSERT OR REPLACE INTO triples
        (triple_id, head, relation, tail, confidence,
         evidence_level, source, timestamp, klm_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, triples)
    conn.commit()


# ═════════════════════════════════════════════════════════════════════════════
# PATIENT KLM — INPUT MODELS
# ═════════════════════════════════════════════════════════════════════════════

class NewPatient(BaseModel):
    patient_id: str
    name: str
    dob: Optional[str] = None
    sex: Optional[str] = None
    ethnicity: Optional[str] = None
    blood_type: Optional[str] = None
    baseline_conditions: list[str] = []
    visit_date: Optional[str] = None
    symptoms: list[str] = []
    vitals: dict = {}
    lab_results: dict = {}
    diagnosis_codes: list[str] = []
    medications: list[str] = []
    imaging: dict = {}
    clinical_notes: Optional[str] = None
    genetic_variants: list[dict] = []


class NewVisit(BaseModel):
    visit_date: str
    symptoms: list[str] = []
    vitals: dict = {}
    lab_results: dict = {}
    diagnosis_codes: list[str] = []
    medications: list[str] = []
    imaging: dict = {}
    clinical_notes: Optional[str] = None


class CustomTriple(BaseModel):
    patient_id: str
    head: str
    relation: str
    tail: str
    confidence: float = 0.85
    evidence_level: str = "III"
    source: str
    timestamp: Optional[str] = None
    klm_source: str = "patient_klm"


# ── Visit → triples converter ──────────────────────────────────────────────────

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


# ═════════════════════════════════════════════════════════════════════════════
# KLM BUILDER — CLAUDE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a knowledge extraction engine.

Your job is to read the document text below and extract factual knowledge as triples.
Each triple has:
  - head:     the subject entity (a concept, name, drug, disease, guideline, etc.)
  - relation: the relationship verb (e.g. treats, causes, contraindicates, recommends,
              defined_as, associated_with, requires, measured_by, target_value)
  - tail:     the object (what the relation points to)

Rules:
- Extract 10 to 30 triples depending on document length
- Be specific — use the actual terms from the document
- Keep head and tail under 80 characters each
- Use snake_case for relation (e.g. first_line_treatment, not "first line treatment")
- Every triple must be a standalone fact — no pronouns, no "it", "this", "they"
- Assign a confidence score from 0.5 to 1.0 based on how clearly stated the fact is
- Assign an evidence_level: I (RCT/guideline), II (cohort study), III (expert opinion/review), IV (case report), V (inference)

Return ONLY a JSON array. No explanation. No markdown. No extra text.
Format exactly:
[
  {{
    "head": "...",
    "relation": "...",
    "tail": "...",
    "confidence": 0.90,
    "evidence_level": "II"
  }}
]

Document text:
__DOCUMENT_TEXT__
"""


def extract_triples_with_claude(text: str, source_name: str, klm_name: str) -> list:
    if not ANTHROPIC_API_KEY:
        ts  = datetime.now().strftime("%Y-%m-%d")
        tid = make_klm_id(source_name, "uploaded_to", klm_name, ts)
        return [(tid, source_name, "uploaded_to", klm_name,
                 0.99, "I", source_name, ts, klm_name)]

    client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    truncated = text[:24000]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.replace("__DOCUMENT_TEXT__", truncated)
        }]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    extracted = json.loads(raw)
    ts        = datetime.now().strftime("%Y-%m-%d")
    triples   = []

    for item in extracted:
        head     = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail     = str(item.get("tail", "")).strip()
        if not (head and relation and tail):
            continue
        tid = make_klm_id(head, relation, tail, ts)
        triples.append((
            tid, head, relation, tail,
            float(item.get("confidence", 0.85)),
            str(item.get("evidence_level", "III")),
            source_name, ts, klm_name
        ))
    return triples


def read_uploaded_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    content  = file.file.read()
    if filename.endswith(".pdf"):
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return content.decode("utf-8", errors="ignore")
    return content.decode("utf-8", errors="ignore")




class CreateKLM(BaseModel):
    klm_name:    str
    description: Optional[str] = "Custom knowledge model"


class KLMTriple(BaseModel):
    head:           str
    relation:       str
    tail:           str
    confidence:     float = 0.85
    evidence_level: str   = "III"
    source:         str   = "manual"


class ImportPatientRequest(BaseModel):
    patient_ids: list[str]
    domains:     list[str] = []



patient_router = APIRouter(tags=["Patient KLM"])
builder_router = APIRouter(prefix="/klm", tags=["KLM Builder"])




@patient_router.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    """All triples for a patient — ready for agent prompt injection."""
    klm    = get_klm()
    triples = klm.get_by_patient(patient_id)
    klm.close()
    if not triples:
        raise HTTPException(status_code=404, detail=f"No data found for {patient_id}")
    return {"patient_id": patient_id, "total_triples": len(triples), "triples": triples}


@patient_router.get("/patient/{patient_id}/timeline")
def get_timeline(patient_id: str):
    """Disease progression timeline ordered by date."""
    klm    = get_klm()
    result = klm.get_disease_timeline(patient_id)
    klm.close()
    return {"patient_id": patient_id, "timeline": result}


@patient_router.get("/patient/{patient_id}/genomics")
def get_genomics(patient_id: str):
    """DNA and genetic profile."""
    klm    = get_klm()
    result = klm.get_genomic_context(patient_id)
    klm.close()
    return {"patient_id": patient_id, "genomics": result}


@patient_router.get("/patient/{patient_id}/domain/{domain}")
def get_by_domain(patient_id: str, domain: str):
    """
    Filter triples by clinical domain.

    Available: pathology, cardiology, nephrology, hypertension, genomics, ehr, dermatology
    """
    DOMAIN_KEYWORDS = {
        "pathology":    ["pathology_klm", "pathology"],
        "cardiology":   ["cardiology", "cardiac", "heart", "atrial", "ventricular", "afib"],
        "nephrology":   ["nephrology", "renal", "kidney", "ckd", "egfr", "creatinine"],
        "hypertension": ["hypertension", "blood_pressure", "antihypertensive", "bp"],
        "genomics":     ["genomics", "variant", "dna", "gene"],
        "ehr":          ["ehr", "visit"],
        "dermatology":  ["dermatology", "nevi", "nevus", "melanoma", "lesion", "skin",
                         "cdkn2a", "mc1r", "fkbp5", "comt", "cortisol", "phq",
                         "atypical", "biopsy", "uv", "photoprotection"],
    }
    keywords = DOMAIN_KEYWORDS.get(domain.lower())
    if not keywords:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{domain}'. Available: {', '.join(DOMAIN_KEYWORDS)}"
        )
    like_clauses = " OR ".join(
        ["klm_source LIKE ?", "relation LIKE ?", "tail LIKE ?"] * len(keywords)
    )
    params = []
    for kw in keywords:
        params += [f"%{kw}%", f"%{kw}%", f"%{kw}%"]

    conn = get_patient_conn()
    rows = conn.execute(
        f"SELECT * FROM triples WHERE head = ? AND ({like_clauses}) ORDER BY timestamp",
        [patient_id] + params
    ).fetchall()
    conn.close()

    cols    = ["triple_id", "head", "relation", "tail", "confidence",
               "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, row)) for row in rows]
    return {"patient_id": patient_id, "domain": domain.lower(),
            "total_triples": len(triples), "triples": triples}



@patient_router.post("/patient")
def add_patient(data: NewPatient):
    """Add a completely new patient. Only patient_id and name are required."""
    pid   = data.patient_id
    today = datetime.now().strftime("%Y-%m-%d")
    triples = []

    def t(head, relation, tail, confidence=0.99, evidence_level="I"):
        return (make_id(head, relation, tail, today),
                head, relation, tail, confidence, evidence_level,
                "DEMOGRAPHICS", today, "patient_klm")

    triples.append(t(pid, "has_attribute", f"name:{data.name}"))
    if data.dob:        triples.append(t(pid, "has_attribute", f"dob:{data.dob}"))
    if data.sex:        triples.append(t(pid, "has_attribute", f"sex:{data.sex}"))
    if data.ethnicity:  triples.append(t(pid, "has_attribute", f"ethnicity:{data.ethnicity}"))
    if data.blood_type: triples.append(t(pid, "has_attribute", f"blood_type:{data.blood_type}"))
    for condition in data.baseline_conditions:
        triples.append(t(pid, "has_baseline_condition", condition, 0.95, "II"))

    if data.visit_date:
        triples.extend(visit_to_triples(
            pid, data.visit_date, data.symptoms, data.vitals,
            data.lab_results, data.diagnosis_codes, data.medications,
            data.imaging, data.clinical_notes,
            source=f"EHR:visit_{data.visit_date}"
        ))

    for v in data.genetic_variants:
        gene         = v.get("gene", "unknown")
        variant_id   = v.get("variant_id", "unknown")
        significance = v.get("clinical_significance", "unknown")
        condition    = v.get("associated_condition", "unknown")
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

    store_patient_triples(triples)
    return {"status": "stored", "patient_id": pid, "total_triples_stored": len(triples)}


@patient_router.post("/patient/{patient_id}/visit")
def add_visit(patient_id: str, visit: NewVisit):
    """Add a new EHR visit for an existing patient."""
    triples = visit_to_triples(
        patient_id, visit.visit_date, visit.symptoms, visit.vitals,
        visit.lab_results, visit.diagnosis_codes, visit.medications,
        visit.imaging, visit.clinical_notes,
        source=f"EHR:visit_{visit.visit_date}"
    )
    store_patient_triples(triples)
    return {"status": "stored", "patient_id": patient_id,
            "visit_date": visit.visit_date, "triples_stored": len(triples)}


@patient_router.post("/triple")
def add_triple(triple: CustomTriple):
    """Add any custom triple — family history, risk factors, social history, etc."""
    timestamp = triple.timestamp or datetime.now().strftime("%Y-%m-%d")
    triple_id = make_id(triple.head, triple.relation, triple.tail, timestamp)
    store_patient_triples([(
        triple_id, triple.head, triple.relation, triple.tail,
        triple.confidence, triple.evidence_level, triple.source,
        timestamp, triple.klm_source
    )])
    return {"status": "stored", "triple_id": triple_id}



@builder_router.get("/list")
def list_klms():
    """List all custom KLMs that have been created."""
    dbs    = [f.replace(".db", "") for f in os.listdir(KLM_DIR) if f.endswith(".db")]
    result = []
    for name in dbs:
        try:
            conn  = sqlite3.connect(klm_path(name))
            meta  = conn.execute("SELECT description, created_at FROM klm_meta LIMIT 1").fetchone()
            count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            conn.close()
            result.append({
                "klm_name":    name,
                "description": meta[0] if meta else "",
                "created_at":  meta[1] if meta else "",
                "triple_count": count
            })
        except Exception:
            result.append({"klm_name": name, "triple_count": "unknown"})
    return {"klms": result}


@builder_router.post("/create")
def create_klm(data: CreateKLM):
    """
    Create a new empty KLM.

    Example:
    { "klm_name": "cardiology_guidelines_2026", "description": "ACC/AHA guidelines" }
    """
    if klm_exists(data.klm_name):
        raise HTTPException(
            status_code=400,
            detail=f"KLM '{data.klm_name}' already exists. Use a different name."
        )
    conn = sqlite3.connect(klm_path(data.klm_name))
    create_klm_schema(conn, data.klm_name, data.description or "")
    conn.close()
    return {
        "status":      "created",
        "klm_name":    data.klm_name,
        "description": data.description,
        "next_steps": [
            f"Upload a document:        POST /klm/{data.klm_name}/upload",
            f"Import patients:          POST /klm/{data.klm_name}/import_patients",
            f"Add a triple manually:    POST /klm/{data.klm_name}/triple",
            f"Read all triples (agent): GET  /klm/{data.klm_name}/all",
        ]
    }


@builder_router.post("/{klm_name}/upload")
async def upload_document(
    klm_name:     str,
    file:         UploadFile = File(...),
    source_label: str        = Form(default="")
):
    """
    Upload a document (.txt, .pdf, .md, .json, .csv) and extract triples via Claude.

    Requires ANTHROPIC_API_KEY to be set — otherwise stores a placeholder triple.
    """
    if not klm_exists(klm_name):
        raise HTTPException(status_code=404,
                            detail=f"KLM '{klm_name}' not found. POST /klm/create first.")
    source  = source_label or file.filename or "uploaded_document"
    text    = read_uploaded_file(file)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

    triples = extract_triples_with_claude(text, source, klm_name)
    conn    = get_klm_conn(klm_name)
    insert_klm_triples(conn, triples)
    conn.close()

    return {
        "status":            "extracted_and_stored",
        "klm_name":          klm_name,
        "file":              file.filename,
        "triples_extracted": len(triples),
        "next_step":         f"GET /klm/{klm_name}/all"
    }


@builder_router.post("/{klm_name}/triple")
def add_klm_triple(klm_name: str, triple: KLMTriple):
    """Manually add a single triple to a custom KLM."""
    ts  = datetime.now().strftime("%Y-%m-%d")
    tid = make_klm_id(triple.head, triple.relation, triple.tail, ts)
    conn = get_klm_conn(klm_name)
    insert_klm_triples(conn, [(
        tid, triple.head, triple.relation, triple.tail,
        triple.confidence, triple.evidence_level, triple.source, ts, klm_name
    )])
    conn.close()
    return {"status": "stored", "triple_id": tid, "klm_name": klm_name}


@builder_router.get("/list/patients_available")
def list_available_patients():
    """
    List all patients in patient_klm.db with their domain breakdown.
    Use this before calling import_patients to see what is available.
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503,
                            detail=f"patient_klm.db not found at {DB_PATH}.")
    conn        = sqlite3.connect(DB_PATH)
    patient_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT head FROM triples ORDER BY head"
    ).fetchall()]
    result = []
    for pid in patient_ids:
        domain_rows = conn.execute(
            "SELECT klm_source, COUNT(*) FROM triples WHERE head=? GROUP BY klm_source",
            (pid,)
        ).fetchall()
        domains = {r[0]: r[1] for r in domain_rows}
        result.append({"patient_id": pid, "total_triples": sum(domains.values()),
                        "domains": domains})
    conn.close()
    return {"source_db": DB_PATH, "total_patients": len(result), "patients": result}


@builder_router.post("/{klm_name}/import_patients")
def import_patients(klm_name: str, data: ImportPatientRequest):
    """
    Copy one or more patients from patient_klm.db into a custom KLM.

    Import all domains:
    { "patient_ids": ["P-001", "PT-8839-CR"] }

    Import specific domains only:
    { "patient_ids": ["PT-8839-CR"], "domains": ["patient_klm_nephrology"] }
    """
    if not klm_exists(klm_name):
        raise HTTPException(status_code=404,
                            detail=f"KLM '{klm_name}' not found. POST /klm/create first.")
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503,
                            detail=f"patient_klm.db not found at {DB_PATH}.")
    if not data.patient_ids:
        raise HTTPException(status_code=400, detail="patient_ids cannot be empty.")

    src_conn  = sqlite3.connect(DB_PATH)
    all_rows  = []
    not_found = []

    for pid in data.patient_ids:
        if not src_conn.execute(
            "SELECT 1 FROM triples WHERE head=? LIMIT 1", (pid,)
        ).fetchone():
            not_found.append(pid)
            continue

        if data.domains:
            ph   = ",".join("?" * len(data.domains))
            rows = src_conn.execute(
                f"""SELECT triple_id, head, relation, tail, confidence,
                           evidence_level, source, timestamp, klm_source
                    FROM triples WHERE head=? AND klm_source IN ({ph})
                    ORDER BY timestamp""",
                [pid] + data.domains
            ).fetchall()
        else:
            rows = src_conn.execute(
                """SELECT triple_id, head, relation, tail, confidence,
                          evidence_level, source, timestamp, klm_source
                   FROM triples WHERE head=? ORDER BY timestamp""",
                (pid,)
            ).fetchall()
        all_rows.extend(rows)

    src_conn.close()

    if not all_rows and not_found:
        raise HTTPException(
            status_code=404,
            detail=f"Patients not found in patient_klm.db: {not_found}"
        )

    dest_conn = get_klm_conn(klm_name)
    insert_klm_triples(dest_conn, all_rows)
    dest_conn.close()

    per_patient = dict(Counter(row[1] for row in all_rows))
    return {
        "status":           "imported",
        "klm_name":         klm_name,
        "total_imported":   len(all_rows),
        "per_patient":      per_patient,
        "not_found":        not_found,
        "domains_filtered": data.domains if data.domains else "all",
        "next_step":        f"GET /klm/{klm_name}/all"
    }


@builder_router.get("/{klm_name}/all")
def get_all(klm_name: str):
    """All triples in a custom KLM — inject directly into an agent system prompt."""
    conn = get_klm_conn(klm_name)
    rows = conn.execute("SELECT * FROM triples ORDER BY timestamp").fetchall()
    meta = conn.execute("SELECT description FROM klm_meta LIMIT 1").fetchone()
    conn.close()
    cols    = ["triple_id", "head", "relation", "tail", "confidence",
               "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, r)) for r in rows]
    return {"klm_name": klm_name, "description": meta[0] if meta else "",
            "total_triples": len(triples), "triples": triples}


@builder_router.get("/{klm_name}/query")
def query_klm(
    klm_name: str,
    keyword:  Optional[str] = None,
    relation: Optional[str] = None,
    head:     Optional[str] = None,
    limit:    int = 50
):
    """
    Search a custom KLM.

    Examples:
        GET /klm/my_klm/query?keyword=hypertension
        GET /klm/my_klm/query?relation=first_line_treatment
        GET /klm/my_klm/query?head=metformin&limit=20
    """
    conn       = get_klm_conn(klm_name)
    conditions = []
    params     = []

    if keyword:
        conditions.append("(head LIKE ? OR relation LIKE ? OR tail LIKE ?)")
        params += [f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"]
    if relation:
        conditions.append("relation LIKE ?")
        params.append(f"%{relation}%")
    if head:
        conditions.append("head LIKE ?")
        params.append(f"%{head}%")

    where   = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows    = conn.execute(
        f"SELECT * FROM triples {where} ORDER BY confidence DESC LIMIT ?",
        params + [limit]
    ).fetchall()
    conn.close()

    cols    = ["triple_id", "head", "relation", "tail", "confidence",
               "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, r)) for r in rows]
    return {"klm_name": klm_name,
            "query": {"keyword": keyword, "relation": relation, "head": head},
            "total_results": len(triples), "triples": triples}


@builder_router.delete("/{klm_name}")
def delete_klm(klm_name: str):
    """Permanently delete a custom KLM and all its triples."""
    path = klm_path(klm_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"KLM '{klm_name}' not found.")
    os.remove(path)
    return {"status": "deleted", "klm_name": klm_name}


app.include_router(patient_router)
app.include_router(builder_router)


@app.get("/")
def root():
    return {
        "status":  "ok",
        "service": "Patient KLM + KLM Builder",
        "version": "2.0.0",
        "port":    8001,
        "docs":    "/docs",
        "patient_endpoints": [
            "GET  /patient/{id}",
            "GET  /patient/{id}/timeline",
            "GET  /patient/{id}/genomics",
            "GET  /patient/{id}/domain/{domain}",
            "POST /patient",
            "POST /patient/{id}/visit",
            "POST /triple",
        ],
        "klm_builder_endpoints": [
            "GET  /klm/list",
            "GET  /klm/list/patients_available",
            "POST /klm/create",
            "POST /klm/{name}/upload",
            "POST /klm/{name}/triple",
            "POST /klm/{name}/import_patients",
            "GET  /klm/{name}/all",
            "GET  /klm/{name}/query",
            "DELETE /klm/{name}",
        ]
    }



if __name__ == "__main__":
    print("Starting Patient KLM + KLM Builder — unified API")
    print(f"Patient DB:  {DB_PATH}")
    print(f"Custom KLMs: {KLM_DIR}")
    print(f"Docs:        http://localhost:8001/docs\n")
    if not ANTHROPIC_API_KEY:
        print("NOTE: ANTHROPIC_API_KEY not set — document upload stores placeholder triples only.\n")
    uvicorn.run("patient_klm_endpoint:app", host="0.0.0.0", port=8001, reload=False)
