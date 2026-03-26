"""
generate_pathology_p003.py
Generates pathology knowledge triples for P-003's conditions.
Split into two batches to avoid token limits.

Output: data/p003_pathology_triples.json
"""

import json
import os
import anthropic

client = anthropic.Anthropic()


TRIPLE_TEMPLATE = """
Return ONLY a valid JSON array (no markdown, no wrapper object) like:
[
  {
    "triple_id": "PATH-P003-XXX",
    "head": "...",
    "relation": "...",
    "tail": "...",
    "confidence": 0.97,
    "evidence_level": "I",
    "source": "KDIGO 2024",
    "timestamp": "2025-01-01",
    "klm_source": "pathology_klm"
  }
]

Use these relation types:
  is_defined_as, has_diagnostic_criterion, has_first_line_treatment,
  is_caused_by, causes, has_complication, is_managed_with,
  has_contraindication, requires_dose_adjustment_in, has_biomarker,
  has_target_value, is_risk_factor_for, has_prevalence_in,
  is_associated_with, has_staging_criterion

Confidence 0.95-0.99 for guideline facts, 0.85-0.94 for associations.
Evidence levels: I = RCT/guideline, II = observational, III = expert consensus.
"""


def call_claude(prompt):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def generate_batch_1():
    """Hypertension + CKD triples (PATH-P003-001 to PATH-P003-030)."""
    prompt = f"""
You are a clinical knowledge extraction system generating pathology knowledge triples
from clinical guidelines (KDIGO 2024, ESC 2023, ACC/AHA 2022, JNC-8).

Generate exactly 30 triples covering:

HYPERTENSION (12 triples, IDs PATH-P003-001 to PATH-P003-012):
- Diagnostic threshold: systolic >=140 mmHg or diastolic >=90 mmHg
- First-line treatments: ACE inhibitors, ARBs, CCBs, thiazide diuretics
- Target BP in CKD patients: <130/80 per KDIGO
- Hypertension as cause of CKD progression
- Complications: LVH, stroke, MI, retinopathy, CKD
- Resistant hypertension: BP uncontrolled on 3+ agents including diuretic

CKD (18 triples, IDs PATH-P003-013 to PATH-P003-030):
- Staging: G1 eGFR>90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 <15
- Albuminuria staging: A1 <30, A2 30-300, A3 >300 mg/g
- Causes: diabetic nephropathy, hypertensive nephrosclerosis, glomerulonephritis
- Complications: anaemia, metabolic acidosis, hyperkalemia, renal osteodystrophy
- Progression risk factors: proteinuria, uncontrolled BP, diabetes
- Management: RAAS blockade, dietary protein restriction 0.8g/kg/day, sodium <2g/day
- CKD cardiovascular risk: 10x higher CVD mortality vs general population
- Dialysis indications: eGFR <10, uraemic symptoms

{TRIPLE_TEMPLATE}
"""
    return call_claude(prompt)


def generate_batch_2():
    """AFib + CRS + LVH + Anaemia triples (PATH-P003-031 to PATH-P003-060)."""
    prompt = f"""
You are a clinical knowledge extraction system generating pathology knowledge triples
from clinical guidelines (KDIGO 2024, ESC 2023, ACC/AHA 2022).

Generate exactly 30 triples covering:

ATRIAL FIBRILLATION IN CKD (8 triples, IDs PATH-P003-031 to PATH-P003-038):
- AFib prevalence in CKD: 3-4x higher than general population
- Anticoagulation challenge: bleeding risk vs thromboembolic risk
- Apixaban preferred NOAC when eGFR <30
- Warfarin avoided in CKD due to calciphylaxis and variable INR
- AFib increases stroke risk 5x in CKD patients
- Rate control target: resting HR <110 bpm

CARDIORENAL SYNDROME TYPE 4 (8 triples, IDs PATH-P003-039 to PATH-P003-046):
- Definition: chronic kidney disease causing chronic cardiac dysfunction
- Pathophysiology: neurohormonal activation, volume overload, uremic toxins
- Biomarkers: BNP >100 pg/mL, NT-proBNP >300 pg/mL
- Management: loop diuretics, RAAS blockade, fluid restriction
- SGLT2 inhibitors reduce CRS progression
- 50% 5-year mortality

LVH (7 triples, IDs PATH-P003-047 to PATH-P003-053):
- LVH definition: LV mass index >115 g/m2 men, >95 g/m2 women
- Independent CV risk factor: doubles MI and sudden death risk
- Caused by: pressure overload (hypertension), volume overload (CKD)
- Regression: RAAS blockade reduces LV mass 10-15%
- Diastolic dysfunction: impaired LV relaxation
- Concentric LVH: worst prognosis subtype

ANAEMIA OF CKD (7 triples, IDs PATH-P003-054 to PATH-P003-060):
- Definition: Hb <13 g/dL men, <12 g/dL women with CKD
- Primary cause: reduced EPO production by damaged kidneys
- Contributing: iron deficiency, chronic inflammation
- ESA therapy target: Hb 10-11.5 g/dL (avoid >13 — increased CV risk)
- IV iron preferred over oral in CKD stages 4-5
- Ferritin target: >200 ng/mL, transferrin saturation >20% before ESA
- Untreated anaemia worsens LVH and cardiac outcomes

{TRIPLE_TEMPLATE}
"""
    return call_claude(prompt)


def main():
    print("Generating P-003 pathology knowledge triples...")

    print("  Generating batch 1 (hypertension + CKD)...")
    batch1 = generate_batch_1()
    print(f"  ✅ Batch 1: {len(batch1)} triples")

    print("  Generating batch 2 (AFib + CRS + LVH + anaemia)...")
    batch2 = generate_batch_2()
    print(f"  ✅ Batch 2: {len(batch2)} triples")

    all_triples = batch1 + batch2

    data = {
        "klm_id": "pathology_klm_p003",
        "domain": "cardiology_nephrology_hypertension",
        "source_guidelines": ["KDIGO 2024", "ESC 2023", "ACC/AHA 2022", "JNC-8"],
        "generated_date": "2025-01-01",
        "triples": all_triples
    }

    os.makedirs("data", exist_ok=True)
    output_path = "data/p003_pathology_triples.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ P-003 pathology triples saved to {output_path}")
    print(f"   Total triples: {len(all_triples)}")
    return data


if __name__ == "__main__":
    main()