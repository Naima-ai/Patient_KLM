"""
klm_builder.py
Custom KLM Builder — create a brand new Knowledge Model from your own documents.

What this adds on top of patient_klm_endpoint.py:
  POST /klm/create              — create a new named KLM (fresh SQLite database)
  POST /klm/{klm_name}/upload   — upload a text/PDF document and extract triples
  POST /klm/{klm_name}/triple   — manually add any triple to your KLM
  GET  /klm/{klm_name}/query    — query your KLM (keyword or relation filter)
  GET  /klm/{klm_name}/all      — get every triple in your KLM (for agent injection)
  GET  /klm/list                — list all KLMs you have created

How it works:
  1. You create a KLM with a name (e.g. "cardiology_guidelines")
  2. You upload documents — the API reads the text and uses Claude to extract
     (head, relation, tail) triples automatically
  3. Those triples are stored in a SQLite database named after your KLM
  4. Any agent can then GET /klm/{klm_name}/all and inject the triples as context

Run:
  pip install fastapi uvicorn anthropic pypdf2
  export ANTHROPIC_API_KEY=your_key
  python klm_builder.py

Docs: http://localhost:8002/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sqlite3
import hashlib
import os
import json
import re
import anthropic
from datetime import datetime

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="KLM Builder API",
    description="Create your own Knowledge Model from any document",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# All KLM databases live in this folder
KLM_DIR = os.environ.get(
    "KLM_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "custom_klms")
)
os.makedirs(KLM_DIR, exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── Database helpers ──────────────────────────────────────────────────────────

def klm_path(klm_name: str) -> str:
    """Returns the SQLite file path for a given KLM name."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", klm_name)
    return os.path.join(KLM_DIR, f"{safe}.db")


def klm_exists(klm_name: str) -> bool:
    return os.path.exists(klm_path(klm_name))


def get_conn(klm_name: str) -> sqlite3.Connection:
    if not klm_exists(klm_name):
        raise HTTPException(
            status_code=404,
            detail=f"KLM '{klm_name}' not found. Create it first with POST /klm/create"
        )
    return sqlite3.connect(klm_path(klm_name))


def create_schema(conn: sqlite3.Connection, klm_name: str, description: str):
    """Create the triples table and a metadata table in a new KLM database."""
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


def make_id(head: str, relation: str, tail: str, ts: str) -> str:
    content = f"{head}{relation}{tail}{ts}"
    return "KT" + hashlib.md5(content.encode()).hexdigest()[:8].upper()


def insert_triples(conn: sqlite3.Connection, triples: list):
    conn.executemany("""
        INSERT OR REPLACE INTO triples
        (triple_id, head, relation, tail, confidence,
         evidence_level, source, timestamp, klm_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, triples)
    conn.commit()


# ── Claude triple extractor ───────────────────────────────────────────────────

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
  {
    "head": "...",
    "relation": "...",
    "tail": "...",
    "confidence": 0.90,
    "evidence_level": "II"
  }
]

Document text:
{text}
"""


def extract_triples_with_claude(text: str, source_name: str, klm_name: str) -> list:
    """
    Sends document text to Claude and gets back a list of triples.
    Falls back to empty list if API key is missing.
    """
    if not ANTHROPIC_API_KEY:
        # No API key — return a placeholder triple so the KLM still works
        ts = datetime.now().strftime("%Y-%m-%d")
        tid = make_id(source_name, "uploaded_to", klm_name, ts)
        return [(tid, source_name, "uploaded_to", klm_name,
                 0.99, "I", source_name, ts, klm_name)]

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Truncate to keep within context limits (roughly 6000 words)
    truncated = text[:24000] if len(text) > 24000 else text

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(text=truncated)
        }]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude wraps in ```json
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    extracted = json.loads(raw)

    ts = datetime.now().strftime("%Y-%m-%d")
    triples = []
    for item in extracted:
        head     = str(item.get("head", "")).strip()
        relation = str(item.get("relation", "")).strip()
        tail     = str(item.get("tail", "")).strip()
        if not (head and relation and tail):
            continue
        confidence     = float(item.get("confidence", 0.85))
        evidence_level = str(item.get("evidence_level", "III"))
        tid = make_id(head, relation, tail, ts)
        triples.append((
            tid, head, relation, tail,
            confidence, evidence_level, source_name, ts, klm_name
        ))

    return triples


def read_uploaded_file(file: UploadFile) -> str:
    """
    Read text from an uploaded file.
    Supports .txt, .md, .json, .pdf (basic extraction), .csv
    """
    filename = file.filename.lower()
    content  = file.file.read()

    if filename.endswith(".pdf"):
        # Basic PDF text extraction — no extra dependencies
        try:
            import PyPDF2
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            return text
        except ImportError:
            # PyPDF2 not installed — try reading as raw text
            return content.decode("utf-8", errors="ignore")

    # For everything else — decode as UTF-8
    return content.decode("utf-8", errors="ignore")


# ── Input models ──────────────────────────────────────────────────────────────

class CreateKLM(BaseModel):
    klm_name:    str
    description: Optional[str] = "Custom knowledge model"


class ManualTriple(BaseModel):
    head:           str
    relation:       str
    tail:           str
    confidence:     float = 0.85
    evidence_level: str   = "III"
    source:         str   = "manual"


class QueryRequest(BaseModel):
    keyword:  Optional[str] = None   # searches head, relation, tail
    relation: Optional[str] = None   # filter by exact relation
    head:     Optional[str] = None   # filter by exact head entity
    limit:    int = 100


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "KLM Builder", "docs": "/docs"}


@app.get("/klm/list")
def list_klms():
    """List all KLMs that have been created."""
    dbs = [f.replace(".db", "") for f in os.listdir(KLM_DIR) if f.endswith(".db")]
    result = []
    for name in dbs:
        try:
            conn = sqlite3.connect(klm_path(name))
            meta = conn.execute("SELECT description, created_at FROM klm_meta LIMIT 1").fetchone()
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


@app.post("/klm/create")
def create_klm(data: CreateKLM):
    """
    Create a brand new empty KLM with a name and description.

    Example:
    {
        "klm_name": "cardiology_guidelines_2026",
        "description": "ACC/AHA heart failure guidelines and evidence base"
    }
    """
    if klm_exists(data.klm_name):
        raise HTTPException(
            status_code=400,
            detail=f"KLM '{data.klm_name}' already exists. Use a different name."
        )
    conn = sqlite3.connect(klm_path(data.klm_name))
    create_schema(conn, data.klm_name, data.description or "")
    conn.close()
    return {
        "status":      "created",
        "klm_name":    data.klm_name,
        "description": data.description,
        "db_path":     klm_path(data.klm_name),
        "next_step":   f"Upload documents with POST /klm/{data.klm_name}/upload"
    }


@app.post("/klm/{klm_name}/upload")
async def upload_document(
    klm_name: str,
    file: UploadFile = File(...),
    source_label: str = Form(default="")
):
    """
    Upload a document (.txt, .pdf, .md, .json, .csv) and automatically
    extract knowledge triples into the KLM using Claude.

    - file:         the document to upload
    - source_label: optional label for the source (e.g. "ACC_AHA_2026_guidelines")

    Claude reads the document and extracts (head, relation, tail) triples.
    These are stored in your KLM immediately and are ready for agent use.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    Without it, a placeholder triple is stored so the KLM still functions.
    """
    if not klm_exists(klm_name):
        raise HTTPException(
            status_code=404,
            detail=f"KLM '{klm_name}' not found. Create it first with POST /klm/create"
        )

    source = source_label or file.filename or "uploaded_document"
    text   = read_uploaded_file(file)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

    triples = extract_triples_with_claude(text, source, klm_name)

    conn = get_conn(klm_name)
    insert_triples(conn, triples)
    conn.close()

    return {
        "status":           "extracted_and_stored",
        "klm_name":         klm_name,
        "file":             file.filename,
        "source_label":     source,
        "triples_extracted": len(triples),
        "next_step":        f"Query your KLM with GET /klm/{klm_name}/all"
    }


@app.post("/klm/{klm_name}/triple")
def add_triple(klm_name: str, triple: ManualTriple):
    """
    Manually add a single triple to your KLM.
    Use this when you want to add a specific fact that the document extractor missed.

    Example:
    {
        "head": "metformin",
        "relation": "contraindicates",
        "tail": "eGFR < 30 mL/min",
        "confidence": 0.98,
        "evidence_level": "I",
        "source": "BNF_2026"
    }
    """
    ts  = datetime.now().strftime("%Y-%m-%d")
    tid = make_id(triple.head, triple.relation, triple.tail, ts)
    conn = get_conn(klm_name)
    insert_triples(conn, [(
        tid, triple.head, triple.relation, triple.tail,
        triple.confidence, triple.evidence_level, triple.source, ts, klm_name
    )])
    conn.close()
    return {"status": "stored", "triple_id": tid, "klm_name": klm_name}


@app.get("/klm/{klm_name}/all")
def get_all(klm_name: str):
    """
    Get every triple in this KLM — ready to inject directly into an agent system prompt.

    Usage in your agent:
        import requests
        triples = requests.get("http://localhost:8002/klm/my_klm/all").json()["triples"]
        system_prompt = f"Use this knowledge: {triples}"
    """
    conn = get_conn(klm_name)
    rows = conn.execute("SELECT * FROM triples ORDER BY timestamp").fetchall()
    meta = conn.execute("SELECT description FROM klm_meta LIMIT 1").fetchone()
    conn.close()
    cols = ["triple_id", "head", "relation", "tail", "confidence",
            "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, r)) for r in rows]
    return {
        "klm_name":     klm_name,
        "description":  meta[0] if meta else "",
        "total_triples": len(triples),
        "triples":      triples
    }


@app.get("/klm/{klm_name}/query")
def query_klm(
    klm_name: str,
    keyword:  Optional[str] = None,
    relation: Optional[str] = None,
    head:     Optional[str] = None,
    limit:    int = 50
):
    """
    Search your KLM by keyword, relation, or head entity.

    Examples:
        GET /klm/cardiology_guidelines/query?keyword=hypertension
        GET /klm/cardiology_guidelines/query?relation=first_line_treatment
        GET /klm/cardiology_guidelines/query?head=metformin
        GET /klm/cardiology_guidelines/query?keyword=heart+failure&limit=20
    """
    conn = get_conn(klm_name)
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

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows  = conn.execute(
        f"SELECT * FROM triples {where} ORDER BY confidence DESC LIMIT ?",
        params + [limit]
    ).fetchall()
    conn.close()

    cols    = ["triple_id", "head", "relation", "tail", "confidence",
               "evidence_level", "source", "timestamp", "klm_source"]
    triples = [dict(zip(cols, r)) for r in rows]
    return {
        "klm_name":     klm_name,
        "query":        {"keyword": keyword, "relation": relation, "head": head},
        "total_results": len(triples),
        "triples":      triples
    }


@app.delete("/klm/{klm_name}")
def delete_klm(klm_name: str):
    """
    Permanently delete a KLM and all its triples.
    This cannot be undone.
    """
    path = klm_path(klm_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"KLM '{klm_name}' not found.")
    os.remove(path)
    return {"status": "deleted", "klm_name": klm_name}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting KLM Builder API...")
    print(f"KLM storage: {KLM_DIR}")
    print(f"Docs:        http://localhost:8002/docs\n")

    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set.")
        print("         Document upload will store placeholder triples only.")
        print("         Set the key to enable automatic triple extraction.\n")

    uvicorn.run("klm_builder:app", host="0.0.0.0", port=8002, reload=False)
