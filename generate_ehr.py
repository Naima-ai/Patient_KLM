"""
Progression: Year 1 (healthy) → Year 2 (early symptoms) → Year 3 (kidney tumor)
"""

import json
import anthropic
from datetime import datetime

client = anthropic.Anthropic()

PATIENT_META = {
    "patient_id": "P-001",
    "name": "demo-user-1",
    "dob": "1978-04-12",
    "sex": "Female",
    "ethnicity": "North African",
    "blood_type": "A+",
    "baseline_conditions": ["hypertension (mild, diet-controlled)", "family history of renal cell carcinoma (maternal uncle)"]
}

YEAR_PROMPTS = [
    {
        "year": 2022,
        "stage": "healthy",
        "visits": ["2022-03-10", "2022-09-15"],
        "instruction": "Patient is healthy. Labs are normal. No complaints beyond mild seasonal fatigue. Annual check and a follow-up. Include normal eGFR >90, creatinine ~0.9 mg/dL, normal urinalysis."
    },
    {
        "year": 2023,
        "stage": "early_symptoms",
        "visits": ["2023-02-20", "2023-11-08"],
        "instruction": "Early abnormalities begin. Patient reports occasional flank discomfort and mild hematuria. eGFR slightly declining (75-80). Creatinine edging up to 1.1-1.2. Mild proteinuria. No mass detected yet. Doctor notes to monitor."
    },
    {
        "year": 2024,
        "stage": "kidney_tumor",
        "visits": ["2024-04-03", "2024-10-22"],
        "instruction": "Condition progresses. CT scan reveals a 3.2cm right renal mass suspicious for renal cell carcinoma. eGFR declined to 58. Creatinine 1.5 mg/dL. Hematuria worsening. Patient referred to urology. Biopsy confirms clear cell RCC stage T1b. Medications updated."
    }
]

def generate_ehr_visit(patient_meta, year_info, visit_date):
    prompt = f"""
You are a nephrology EHR system. Generate a realistic, structured Electronic Health Record entry for:

Patient: {json.dumps(patient_meta, indent=2)}
Visit Date: {visit_date}
Clinical Stage: {year_info['stage']}
Clinical Guidance: {year_info['instruction']}

Return ONLY a valid JSON object (no markdown) with this exact structure:
{{
  "visit_id": "V-{visit_date.replace('-','')}",
  "patient_id": "{patient_meta['patient_id']}",
  "visit_date": "{visit_date}",
  "visit_type": "outpatient nephrology",
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
    "urinalysis": {{
      "protein": "...",
      "blood": "...",
      "rbc_per_hpf": "...",
      "specific_gravity": "..."
    }}
  }},
  "imaging": {{
    "type": "none | ultrasound | CT",
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
    # Strip markdown fences if present
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

    output_path = "data/ehr_records.json"
    import os; os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n✅ EHR records saved to {output_path}")
    print(f"   Total visits: {len(all_records['ehr_visits'])}")
    return all_records


if __name__ == "__main__":
    main()
