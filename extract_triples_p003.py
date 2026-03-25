"""
extract_triples_p003.py
Extracts knowledge triples from P-003 EHR and genomic profile.

Two methods:
  1. Deterministic — from structured JSON fields (labs, vitals, meds, diagnoses)
  2. LLM-assisted — from clinical notes (calls Claude to extract semantic triples)

Output: data/p003_patient_triples.json
"""

import json
import os
import anthropic

client = anthropic.Anthropic()

PATIENT_ID = "P-003"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_triple(triple_id, head, relation, tail, confidence, evidence_level, source, timestamp):
    return {
        "triple_id": triple_id,
        "head": head,
        "relation": relation,
        "tail": tail,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "source": source,
        "timestamp": timestamp,
        "klm_source": "patient_klm"
    }


def extract_ehr_triples(ehr_data):
    triples = []
    counter = [1]

    def tid():
        t = f"T{counter[0]:04d}"
        counter[0] += 1
        return t

    pid = PATIENT_ID

    # Patient demographics
    patient = ehr_data.get("patient", {})
    for field, relation in [
        ("dob", "has_attribute"),
        ("sex", "has_attribute"),
        ("ethnicity", "has_attribute"),
        ("blood_type", "has_attribute"),
    ]:
        if patient.get(field):
            triples.append(make_triple(
                tid(), pid, relation, f"{field}:{patient[field]}",
                0.99, "I", "DEMOGRAPHICS", "2023-01-01"
            ))

    for condition in patient.get("baseline_conditions", []):
        triples.append(make_triple(
            tid(), pid, "has_baseline_condition", condition,
            0.95, "II", "DEMOGRAPHICS", "2023-01-01"
        ))

    for visit in ehr_data.get("ehr_visits", []):
        vid = visit.get("visit_id", "unknown")
        date = visit.get("visit_date", "")
        source = f"EHR:{vid}"
        stage = visit.get("clinical_stage", "")

        # Disease stage
        if stage:
            triples.append(make_triple(
                tid(), pid, "disease_progression_stage", stage,
                0.95, "II", source, date
            ))

        # Vitals
        vitals = visit.get("vitals", {})
        if vitals.get("blood_pressure"):
            triples.append(make_triple(
                tid(), pid, "has_vital",
                f"blood_pressure:{vitals['blood_pressure']}",
                0.99, "I", source, date
            ))
        if vitals.get("heart_rate"):
            triples.append(make_triple(
                tid(), pid, "has_vital",
                f"heart_rate:{vitals['heart_rate']} bpm",
                0.99, "I", source, date
            ))
        if vitals.get("weight_kg"):
            triples.append(make_triple(
                tid(), pid, "has_vital",
                f"weight:{vitals['weight_kg']} kg",
                0.99, "I", source, date
            ))

        # Lab results
        labs = visit.get("lab_results", {})
        lab_map = {
            "creatinine_mg_dl":  ("creatinine", "mg/dL"),
            "egfr_ml_min":       ("eGFR", "mL/min/1.73m²"),
            "bun_mg_dl":         ("BUN", "mg/dL"),
            "potassium_meq_l":   ("potassium", "mEq/L"),
            "sodium_meq_l":      ("sodium", "mEq/L"),
            "hemoglobin_g_dl":   ("hemoglobin", "g/dL"),
            "bnp_pg_ml":         ("BNP", "pg/mL"),
            "hba1c_percent":     ("HbA1c", "%"),
        }
        for key, (label, unit) in lab_map.items():
            if labs.get(key):
                triples.append(make_triple(
                    tid(), pid, "has_lab_value",
                    f"{label}:{labs[key]} {unit}".strip(),
                    0.99, "I", source, date
                ))

        # Urinalysis
        ua = labs.get("urinalysis", {})
        if ua.get("albumin_creatinine_ratio"):
            triples.append(make_triple(
                tid(), pid, "has_lab_value",
                f"albumin_creatinine_ratio:{ua['albumin_creatinine_ratio']}",
                0.99, "I", source, date
            ))
        if ua.get("protein"):
            triples.append(make_triple(
                tid(), pid, "has_lab_value",
                f"urinalysis_protein:{ua['protein']}",
                0.99, "I", source, date
            ))

        # Symptoms
        for symptom in visit.get("symptoms", []):
            triples.append(make_triple(
                tid(), pid, "has_symptom", symptom,
                0.9, "II", source, date
            ))

        # Diagnoses
        for dx in visit.get("diagnosis_codes", []):
            triples.append(make_triple(
                tid(), pid, "diagnosed_with", dx,
                0.95, "II", source, date
            ))

        # Medications
        for med in visit.get("medications", []):
            triples.append(make_triple(
                tid(), pid, "prescribed_medication", med,
                0.95, "II", source, date
            ))

        # Imaging
        imaging = visit.get("imaging", {})
        if imaging.get("type", "none") != "none" and imaging.get("findings"):
            triples.append(make_triple(
                tid(), pid, "has_imaging_finding",
                f"{imaging['type']}:{imaging['findings'][:150]}",
                0.95, "II", source, date
            ))

    return triples, counter[0]


def extract_llm_triples(ehr_data, start_counter):
    """Use Claude to extract additional triples from clinical notes."""
    triples = []
    counter = [start_counter]

    def tid():
        t = f"T{counter[0]:04d}"
        counter[0] += 1
        return t

    for visit in ehr_data.get("ehr_visits", []):
        notes = visit.get("clinical_notes", "")
        if not notes:
            continue

        date = visit.get("visit_date", "")
        source = f"EHR:{visit.get('visit_id', 'unknown')}"

        prompt = f"""
Extract 4-6 clinical knowledge triples from this nephrology/cardiology clinical note.

Note: {notes}
Patient ID: {PATIENT_ID}
Visit Date: {date}

Return ONLY a JSON array of triples. No markdown. Format:
[
  {{
    "head": "{PATIENT_ID}",
    "relation": "relation_type",
    "tail": "value or finding",
    "confidence": 0.85,
    "evidence_level": "II"
  }}
]

Use relations like: has_clinical_finding, has_risk_factor, has_symptom,
diagnosed_with, prescribed_medication, requires_monitoring,
has_complication, responds_to_treatment, has_contraindication
"""
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        extracted = json.loads(raw.strip())
        for t in extracted:
            triples.append(make_triple(
                tid(),
                t.get("head", PATIENT_ID),
                t.get("relation", "has_clinical_finding"),
                t.get("tail", ""),
                t.get("confidence", 0.85),
                t.get("evidence_level", "III"),
                source,
                date
            ))
        print(f"  ✅ Extracted {len(extracted)} triples from visit {date}")

    return triples


def extract_genomic_triples(genomic_data, start_counter):
    triples = []
    counter = [start_counter]
    pid = PATIENT_ID

    def tid():
        t = f"T{counter[0]:04d}"
        counter[0] += 1
        return t

    date = genomic_data.get("profiling_date", "2023-02-14")

    for variant in genomic_data.get("genetic_variants", []):
        gene = variant.get("gene", "")
        variant_id = variant.get("variant_id", "")
        significance = variant.get("clinical_significance", "")
        condition = variant.get("associated_condition", "")
        effect = variant.get("effect", "")

        triples.append(make_triple(
            tid(), pid, "carries_genetic_variant",
            f"{gene}:{variant_id}:{significance}",
            0.99, "I", "GENOMICS", date
        ))

        if significance in ["likely_pathogenic", "pathogenic", "risk_factor"]:
            triples.append(make_triple(
                tid(), f"{gene}_variant", "predisposes_to", condition,
                0.85, "II", "GENOMICS", date
            ))

        if effect:
            triples.append(make_triple(
                tid(), f"{gene}_variant", "has_functional_effect", effect[:120],
                0.85, "II", "GENOMICS", date
            ))

    # PRS scores
    for condition, prs in genomic_data.get("polygenic_risk_scores", {}).items():
        if isinstance(prs, dict) and prs.get("percentile"):
            triples.append(make_triple(
                tid(), pid, "has_polygenic_risk_score",
                f"{condition}:{prs['percentile']}th_percentile — {prs.get('interpretation','')}",
                0.9, "II", "GENOMICS", date
            ))

    # Pharmacogenomics
    for drug, profile in genomic_data.get("pharmacogenomics", {}).items():
        triples.append(make_triple(
            tid(), pid, "has_pharmacogenomic_profile",
            f"{drug}:{profile}",
            0.9, "II", "GENOMICS", date
        ))

    # DNA fragments
    for frag in genomic_data.get("dna_fragments", []):
        frag_id = frag.get("fragment_id", "")
        gene_region = frag.get("gene_region", "")
        length = frag.get("length_bp", "")
        triples.append(make_triple(
            tid(), pid, "has_dna_fragment",
            f"{frag_id}:{gene_region}:{length}bp",
            0.99, "I", "GENOMICS", date
        ))
        for site in frag.get("mutation_sites", []):
            triples.append(make_triple(
                tid(), frag_id, "contains_mutation_site", site,
                0.99, "I", "GENOMICS", date
            ))

    # Family history
    family_note = genomic_data.get("family_history_genetic_notes", "")
    if family_note:
        triples.append(make_triple(
            tid(), pid, "has_family_history", family_note[:200],
            0.9, "II", "GENOMICS", date
        ))

    return triples


def main():
    print("=== Extracting P-003 Knowledge Triples ===\n")

    ehr_data = load_json("data/p003_ehr_records.json")
    genomic_data = load_json("data/p003_genomic_profile.json")

    print("Extracting deterministic EHR triples...")
    ehr_triples, next_counter = extract_ehr_triples(ehr_data)
    print(f"  ✅ {len(ehr_triples)} deterministic triples extracted")

    print("\nExtracting LLM-assisted triples from clinical notes...")
    llm_triples = extract_llm_triples(ehr_data, next_counter)
    next_counter += len(llm_triples)
    print(f"  ✅ {len(llm_triples)} LLM triples extracted")

    print("\nExtracting genomic triples...")
    genomic_triples = extract_genomic_triples(genomic_data, next_counter)
    print(f"  ✅ {len(genomic_triples)} genomic triples extracted")

    all_triples = ehr_triples + llm_triples + genomic_triples
    output = {
        "patient_id": PATIENT_ID,
        "total_triples": len(all_triples),
        "triples": all_triples
    }

    os.makedirs("data", exist_ok=True)
    output_path = "data/p003_patient_triples.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ P-003 triples saved to {output_path}")
    print(f"   Total: {len(all_triples)} triples")
    return output


if __name__ == "__main__":
    main()
