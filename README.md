To run: 

docker compose -f docker-compose.patient.yml up --build


The Patient KLM is a knowledge store for individual patient data. It stores patient EHR records and DNA/genomic profiles as
structured knowledge triples in a SQLite database. These triples can be retrieved at any time and injected as context into an agent's system prompt,just personalised per patient.

The KLM currently contains 4 patients across:

## Patients

| ID | Domain | Conditions |
|---|---|---|
| `P-001` | Nephrology | Renal cell carcinoma, post-nephrectomy CKD |
| `PT-8839-CR` | Hypertension · CKD · Cardiology | Essential hypertension, CKD Stage 2, cardiovascular risk factors |
| `PT-9921` | Dermatology | CDKN2A/MC1R melanoma susceptibility, FKBP5/COMT stress-cortisol axis dysregulation, progressive atypical nevi |


All patient knowledge is stored as triples in the format:
Each triple also carries a confidence score, evidence level, source, and timestamp. The agent pulls these triples and uses them as factual context when reasoning about the patient.


-patient.json - patient_klm_patient- Any agent — full profile, DNA, EHR
-nephrology.json - patient_klm_nephrology -Nephrologist Agent
-cardiology.json - patient_klm_cardiology -Cardiology Agent
-hypertension.json - patient_klm_hypertension -Hypertension Agent



Loading Demo 2 (PT-8839-CR)
Before starting the API, seed the new patient into the database:
bashpython seed_demo2.py
This reads the four domain JSON files (patient.json, nephrology.json, cardiology.json, hypertension.json) and loads all triples for PT-8839-CR into the existing patient_klm.db. Both Demo 1 and Demo 2 patients are then available from the same running API.

Using the Domain KLMs with Specialist Agents
Demo 2 introduces four domain KLM files for the same patient. All triples land in the same SQLite database but are tagged with a klm_source value so each specialist agent can pull only what it needs:

There are two ways to use these in an agent:
Option 1 — Full context via the API (simplest)
Call GET /patient/PT-8839-CR and pass all triples into the system prompt. The agent reasons across all domains at once. Best for a general-purpose agent or when you want the full clinical picture.
code:

import requests

triples = requests.get("http://localhost:8001/patient/PT-8839-CR").json()["triples"]

system_prompt = f"""
You are a specialist clinical AI agent. Use the following patient knowledge
as ground truth when reasoning about this patient.

PATIENT KNOWLEDGE:
{triples}
"""
Option 2 — Domain-filtered context (for specialist agents)
Use klm_api.py directly to filter triples by klm_source before injecting them. This keeps the nephrologist agent focused on nephrology triples, the cardiology agent on cardiology triples, and so on.

code:

from klm_api import PatientKLM

klm = PatientKLM("data/patient_klm.db")
conn = klm._get_conn()

# Nephrologist Agent — only nephrology triples
nephro_triples = conn.execute(
    "SELECT * FROM triples WHERE head = ? AND klm_source = ?",
    ("PT-8839-CR", "patient_klm_nephrology")
).fetchall()

# Cardiology Agent — only cardiology triples
cardio_triples = conn.execute(
    "SELECT * FROM triples WHERE head = ? AND klm_source = ?",
    ("PT-8839-CR", "patient_klm_cardiology")
).fetchall()

klm.close()
You can also combine domains. For example, give the hypertension agent both hypertension and nephrology context since the conditions are clinically linked:
pythontriples = conn.execute("""
    SELECT * FROM triples
    WHERE head = ?
      AND klm_source IN ('patient_klm_hypertension', 'patient_klm_nephrology')
    ORDER BY timestamp
""", ("PT-8839-CR",)).fetchall()



ENDPOINTS

The KLM exposes 6 endpoints — 3 for reading data, 3 for writing new data.

-GET /patient/{patient_id}

    Returns all triples for a patient as JSON. This is the main endpoint used by the agent. Pass the returned triples directly as context into the Nephrologist Agent's system prompt so it knows the full patient history.
    
-GET /patient/{patient_id}/timeline
    Returns the disease progression timeline, diagnoses, lab values, and imaging findings ordered by date. Shows how the patient's condition evolved over time.
    
-GET /patient/{patient_id}/genomics
    Returns the DNA and genetic profile
    
-POST /patient

    Adds a completely new patient to the KLM. Only patient_id and name are required. Optionally include demographics, a first EHR visit, and genomic variants all in one call. Everything is automatically converted to triples and stored immediately.
    
{

            "patient_id": "P-002",
            "name": "user-1",
            "dob": "1980-03-15",
            "sex": "Male",
            "blood_type": "B+",
            "baseline_conditions": ["type 2 diabetes", "hypertension"],
            "visit_date": "2026-03-10",
            "symptoms": ["fatigue", "frequent urination"],
            "lab_results": {"creatinine_mg_dl": 1.6, "egfr_ml_min": 58},
            "diagnosis_codes": ["ICD-10: N18.3"],
            "medications": ["Metformin 500mg daily"],
            "genetic_variants": [
              {
                "gene": "VHL",
                "variant_id": "rs123456",
                "clinical_significance": "pathogenic",
                "associated_condition": "Clear cell RCC"
              }
            ]
          }'


-POST /patient/{patient_id}/visit

    Adds a new EHR visit for an existing patient. Use this when a real patient has a new clinic appointment. All fields are automatically converted to triples and immediately available on the next GET call.
    
{
            "visit_date": "2026-03-10",
            "symptoms": ["increased fatigue", "mild hematuria"],
            "vitals": {"blood_pressure": "138/88", "weight_kg": 67.2},
            "lab_results": {"creatinine_mg_dl": 1.4, "egfr_ml_min": 61},
            "diagnosis_codes": ["ICD-10: C64 - RCC follow-up"],
            "medications": ["Sunitinib 50mg daily"],
            "clinical_notes": "Post-nephrectomy follow-up, stable but monitoring eGFR decline"
          }'

-POST /triple

    Adds any single custom triple directly. Use this for anything not covered by the other endpoints — family history, social history, risk factors, allergy notes, or any custom clinical finding.
    
{
            "patient_id": "P-001",
            "head": "P-001",
            "relation": "has_family_history",
            "tail": "maternal uncle: renal cell carcinoma",
            "confidence": 0.95,
            "evidence_level": "II",
            "source": "patient_intake_form",
            "timestamp": "2026-03-10"
          }'

Part 2 — KLM Builder
Create your own Knowledge Models from any document or by combining existing patients. Useful for building guideline KLMs, research KLMs, or mixed patient + document KLMs for specialist agents.
Step 1 — Create a new KLM
bashPOST /klm/create
{
    "klm_name": "cardiology_guidelines_2026",
    "description": "ACC/AHA heart failure guidelines"
}
Step 2a — Upload a document (auto-extracts triples via Claude)
bashPOST /klm/cardiology_guidelines_2026/upload
# Attach a .pdf, .txt, .md, or .csv file
# Requires ANTHROPIC_API_KEY to be set
Step 2b — Import existing patients into the KLM
bash# See what patients are available first
GET /klm/list/patients_available

# Import all triples for two patients
POST /klm/cardiology_guidelines_2026/import_patients
{
    "patient_ids": ["P-001", "PT-8839-CR"]
}

# Or import only specific domains
POST /klm/cardiology_guidelines_2026/import_patients
{
    "patient_ids": ["PT-8839-CR"],
    "domains": ["patient_klm_nephrology", "patient_klm_cardiology"]
}
Step 2c — Add a triple manually
bashPOST /klm/cardiology_guidelines_2026/triple
{
    "head": "ACE inhibitor",
    "relation": "first_line_treatment",
    "tail": "heart failure with reduced ejection fraction",
    "confidence": 0.98,
    "evidence_level": "I",
    "source": "ACC_AHA_2022"
}
Step 3 — Use the KLM in an agent
pythonimport requests

triples = requests.get(
    "http://localhost:8001/klm/cardiology_guidelines_2026/all"
).json()["triples"]

system_prompt = f"""You are a cardiology AI agent.
Use this knowledge as ground truth when reasoning.

KNOWLEDGE BASE:
{triples}
"""
Other KLM endpoints
bash# List all custom KLMs
GET /klm/list

# Search within a KLM
GET /klm/cardiology_guidelines_2026/query?keyword=hypertension
GET /klm/cardiology_guidelines_2026/query?relation=first_line_treatment
GET /klm/cardiology_guidelines_2026/query?head=metformin&limit=10

# Delete a KLM
DELETE /klm/cardiology_guidelines_2026
