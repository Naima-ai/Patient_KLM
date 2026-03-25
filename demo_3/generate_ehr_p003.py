"""
generate_ehr_p003.py
Generates synthetic EHR records for P-003 over 3 years.

Progression:
  Year 1 (2023) — Hypertension diagnosed, early CKD signs, mild cardiac strain
  Year 2 (2024) — CKD stage 3 confirmed, left ventricular hypertrophy, worsening BP control
  Year 3 (2025) — CKD stage 4, atrial fibrillation onset, cardiorenal syndrome

Output: data/p003_ehr_records.json
"""

import json
import os
import anthropic

client = anthropic.Anthropic()

PATIENT_META = {
    "patient_id": "P-003",
    "name": "demo-user-3",
    "dob": "1965-08-22",
    "sex": "Male",
    "ethnicity": "South Asian",
    "blood_type": "B+",
    "baseline_conditions": [
        "obesity (BMI 31)",
        "type 2 diabetes (diet-controlled)",
        "family history of hypertension and coronary artery disease (father)"
    ]
}

YEAR_PROMPTS = [
    {
        "year": 2023,
        "stage": "hypertension_early_ckd",
        "visits": ["2023-02-14", "2023-08-30"],
        "instruction": (
            "Patient is newly diagnosed with hypertension (BP 155/95). "
            "Early CKD signs: eGFR 68-72, creatinine 1.3-1.4 mg/dL, mild microalbuminuria. "
            "ECG shows mild left ventricular hypertrophy. No chest pain. "
            "Started on ACE inhibitor (Lisinopril 10mg) and low-sodium diet advice. "
            "Mild ankle oedema noted. Fasting glucose borderline elevated."
        )
    },
    {
        "year": 2024,
        "stage": "ckd_stage3_lvh",
        "visits": ["2024-03-05", "2024-10-17"],
        "instruction": (
            "CKD progressing to stage 3b: eGFR drops to 42-48, creatinine 1.8-2.0 mg/dL. "
            "Persistent proteinuria (300-500 mg/day). "
            "Echocardiogram confirms left ventricular hypertrophy and diastolic dysfunction. "
            "BP poorly controlled despite Lisinopril — add Amlodipine 5mg. "
            "Haemoglobin declining (11.2 g/dL), anaemia of CKD suspected. "
            "Potassium elevated at 5.2 mEq/L. Referral to nephrology."
        )
    },
    {
        "year": 2025,
        "stage": "ckd_stage4_afib_cardiorenal",
        "visits": ["2025-01-20", "2025-07-08"],
        "instruction": (
            "CKD stage 4: eGFR 22-28, creatinine 2.8-3.2 mg/dL. "
            "New onset atrial fibrillation detected on Holter monitor. "
            "Cardiorenal syndrome type 4 diagnosed. "
            "Started on anticoagulation (Apixaban, renally dosed). "
            "Haemoglobin 9.8 g/dL — erythropoiesis-stimulating agent discussed. "
            "Fluid overload, BNP elevated. BP now 168/100 on triple therapy. "
            "Dialysis access planning initiated. Urgent nephrology + cardiology co-management."
        )
    }
]


def generate_ehr_visit(patient_meta, year_info, visit_date):
    prompt = f"""
You are a nephrology/cardiology EHR system. Generate a realistic structured EHR entry for:

Patient: {json.dumps(patient_meta, indent=2)}
Visit Date: {visit_date}
Clinical Stage: {year_info['stage']}
Clinical Guidance: {year_info['instruction']}

Return ONLY a valid JSON object (no markdown) with this exact structure:
{{
  "visit_id": "V-{visit_date.replace('-','')}",
  "patient_id": "{patient_meta['patient_id']}",
  "visit_date": "{visit_date}",
  "visit_type": "outpatient nephrology-cardiology",
  "chief_complaint": "...",
  "vitals": {{
    "blood_pressure": "...",
    "heart_rate": 0,
    "temperature_celsius": 0.0,
    "weight_kg": 0.0,
    "bmi": 0.0
  }},
  "symptoms": ["..."],
  "physical_exam_findings": ["..."],
  "lab_results": {{
    "creatinine_mg_dl": 0.0,
    "egfr_ml_min": 0,
    "bun_mg_dl": 0,
    "potassium_meq_l": 0.0,
    "sodium_meq_l": 0,
    "hemoglobin_g_dl": 0.0,
    "bnp_pg_ml": 0,
    "hba1c_percent": 0.0,
    "urinalysis": {{
      "protein": "...",
      "blood": "...",
      "albumin_creatinine_ratio": "...",
      "specific_gravity": "..."
    }}
  }},
  "imaging": {{
    "type": "none | echocardiogram | renal_ultrasound | ECG | CT",
    "findings": "..."
  }},
  "diagnosis_codes": ["ICD-10: ..."],
  "medications": ["..."],
  "clinical_notes": "...",
  "follow_up_plan": "..."
}}
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    all_records = {"patient": PATIENT_META, "ehr_visits": []}

    for year_info in YEAR_PROMPTS:
        print(f"\n=== Generating Year {year_info['year']} ({year_info['stage']}) ===")
        for visit_date in year_info["visits"]:
            print(f"  Generating visit: {visit_date}")
            record = generate_ehr_visit(PATIENT_META, year_info, visit_date)
            record["clinical_stage"] = year_info["stage"]
            all_records["ehr_visits"].append(record)
            print(f"  ✅ Visit {visit_date} generated")

    os.makedirs("data", exist_ok=True)
    output_path = "data/p003_ehr_records.json"
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n✅ P-003 EHR records saved to {output_path}")
    print(f"   Total visits: {len(all_records['ehr_visits'])}")
    return all_records


if __name__ == "__main__":
    main()
