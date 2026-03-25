"""
generate_pathology_p003.py
Generates pathology knowledge triples for P-003's conditions:
  - Hypertension (essential + secondary to CKD)
  - Chronic Kidney Disease (stages 3-4)
  - Atrial Fibrillation
  - Cardiorenal Syndrome type 4
  - Left Ventricular Hypertrophy

These are disease-level knowledge triples (not patient-specific).
They come from clinical guidelines: KDIGO, ESC, JNC-8, ACC/AHA.

Output: data/p003_pathology_triples.json
"""

import json
import os
import anthropic

client = anthropic.Anthropic()


def generate_pathology_triples():
    prompt = """
You are a clinical knowledge extraction system. Generate pathology knowledge triples
covering the following conditions as they appear together in cardiorenal syndrome:

1. Essential Hypertension
2. Chronic Kidney Disease (CKD stages 3-4)
3. Atrial Fibrillation in CKD patients
4. Cardiorenal Syndrome type 4
5. Left Ventricular Hypertrophy secondary to hypertension
6. Anaemia of Chronic Kidney Disease

These are DISEASE-LEVEL knowledge triples from clinical guidelines (KDIGO 2024, ESC 2023, ACC/AHA 2022).
They are NOT patient-specific — they describe the diseases themselves.

Return ONLY a valid JSON object (no markdown) in this format:
{
  "klm_id": "pathology_klm_p003",
  "domain": "cardiology_nephrology_hypertension",
  "source_guidelines": ["KDIGO 2024", "ESC 2023", "ACC/AHA 2022", "JNC-8"],
  "generated_date": "2025-01-01",
  "triples": [
    {
      "triple_id": "PATH-P003-001",
      "head": "...",
      "relation": "...",
      "tail": "...",
      "confidence": 0.0,
      "evidence_level": "I",
      "source": "KDIGO 2024",
      "timestamp": "2025-01-01",
      "klm_source": "pathology_klm"
    }
  ]
}

Generate at least 60 triples covering:

HYPERTENSION (10+ triples):
- Diagnostic thresholds (systolic >=140 or diastolic >=90)
- First-line treatments (ACE inhibitors, ARBs, CCBs, thiazides)
- Target BP in CKD patients (<130/80 per KDIGO)
- Hypertension as a cause of CKD progression
- Hypertension complications (LVH, stroke, MI, retinopathy)
- Resistant hypertension definition and management

CKD (15+ triples):
- CKD staging by eGFR (G1-G5) and albuminuria (A1-A3)
- CKD causes: diabetic nephropathy, hypertensive nephrosclerosis, glomerulonephritis
- CKD complications: anaemia, metabolic acidosis, hyperkalemia, renal osteodystrophy
- CKD progression risk factors: proteinuria, uncontrolled BP, diabetes
- CKD management: RAAS blockade, dietary protein restriction, sodium restriction
- CKD-specific drug dose adjustments
- CKD and cardiovascular risk (10x higher CVD mortality)
- Dialysis indications (eGFR <10, symptoms)

ATRIAL FIBRILLATION IN CKD (10+ triples):
- AFib prevalence in CKD (3-4x higher than general population)
- Anticoagulation challenges in CKD (bleeding risk vs thromboembolic risk)
- NOAC dosing in CKD (Apixaban preferred for eGFR <30)
- AFib and stroke risk in CKD
- Rate vs rhythm control in CKD

CARDIORENAL SYNDROME (10+ triples):
- CRS type 4 definition (chronic kidney disease causing chronic cardiac dysfunction)
- Pathophysiology (neurohormonal activation, volume overload, uremic toxins)
- BNP/NT-proBNP as biomarkers
- Management principles (loop diuretics, RAAS blockade, fluid balance)
- CRS and mortality

LVH (8+ triples):
- LVH as an independent CV risk factor
- Echocardiographic criteria for LVH
- LVH regression with BP treatment
- Diastolic dysfunction in LVH

ANAEMIA OF CKD (7+ triples):
- Definition (Hb <13 g/dL men, <12 g/dL women with CKD)
- Causes: reduced EPO production, iron deficiency, inflammation
- Treatment: ESA therapy target Hb 10-11.5 g/dL
- Iron supplementation in CKD

Use these relation types:
  is_defined_as, has_diagnostic_criterion, has_first_line_treatment,
  is_caused_by, causes, has_complication, is_managed_with,
  has_contraindication, requires_dose_adjustment_in, has_biomarker,
  has_target_value, is_risk_factor_for, has_prevalence_in,
  is_associated_with, has_staging_criterion

Set confidence 0.95-0.99 for guideline-backed facts, 0.85-0.94 for associations.
Evidence levels: I = RCT/guideline, II = observational, III = expert consensus.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    print("Generating P-003 pathology knowledge triples...")
    data = generate_pathology_triples()

    os.makedirs("data", exist_ok=True)
    output_path = "data/p003_pathology_triples.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    count = len(data.get("triples", []))
    print(f"\n✅ P-003 pathology triples saved to {output_path}")
    print(f"   Total triples: {count}")
    return data


if __name__ == "__main__":
    main()
