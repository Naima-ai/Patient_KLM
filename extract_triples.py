"""
Triple schema:
{
  "triple_id": "T0001",
  "head": "entity_1",
  "relation": "relationship_type",
  "tail": "entity_2",
  "confidence": 0.85,
  "evidence_level": "III",
  "source": "dataset_or_document",
  "timestamp": "YYYY-MM-DD",
  "klm_source": "patient_klm"
}
"""

import json
import anthropic
import os
from datetime import datetime

client = anthropic.Anthropic()



def extract_triples_from_ehr_visit(visit: dict, triple_counter: list) -> list:
    """Extract structured triples directly from an EHR visit record."""
    triples = []
    patient_id = visit["patient_id"]
    visit_date = visit["visit_date"]
    stage = visit.get("clinical_stage", "unknown")
    source = f"EHR:{visit['visit_id']}"

    def next_id():
        triple_counter[0] += 1
        return f"T{triple_counter[0]:04d}"

    # Patient has_symptom
    for symptom in visit.get("symptoms", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_symptom",
            "tail": symptom,
            "confidence": 0.95,
            "evidence_level": "II",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    # Patient has_lab_value
    labs = visit.get("lab_results", {})
    lab_map = {
        "creatinine_mg_dl": ("creatinine", "mg/dL"),
        "egfr_ml_min": ("eGFR", "mL/min/1.73m²"),
        "bun_mg_dl": ("BUN", "mg/dL"),
        "hemoglobin_g_dl": ("hemoglobin", "g/dL"),
    }
    for key, (name, unit) in lab_map.items():
        if key in labs and labs[key] is not None:
            triples.append({
                "triple_id": next_id(),
                "head": patient_id,
                "relation": "has_lab_value",
                "tail": f"{name}:{labs[key]} {unit}",
                "confidence": 0.99,
                "evidence_level": "I",
                "source": source,
                "timestamp": visit_date,
                "klm_source": "patient_klm"
            })

    # Patient has_blood_pressure
    bp = visit.get("vitals", {}).get("blood_pressure")
    if bp:
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_vital",
            "tail": f"blood_pressure:{bp}",
            "confidence": 0.99,
            "evidence_level": "I",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    # Patient diagnosed_with
    for dx in visit.get("diagnosis_codes", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "diagnosed_with",
            "tail": dx,
            "confidence": 0.95,
            "evidence_level": "II",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    # Patient prescribed_medication
    for med in visit.get("medications", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "prescribed_medication",
            "tail": med,
            "confidence": 0.95,
            "evidence_level": "II",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    # Imaging findings
    imaging = visit.get("imaging", {})
    if imaging.get("type", "none") != "none" and imaging.get("findings"):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_imaging_finding",
            "tail": f"{imaging['type']}:{imaging['findings'][:120]}",
            "confidence": 0.92,
            "evidence_level": "II",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    # Disease stage progression triple
    stage_map = {
        "healthy": "CKD_stage:normal",
        "early_symptoms": "CKD_stage:early_abnormality",
        "kidney_tumor": "renal_mass:confirmed_RCC_T1b"
    }
    if stage in stage_map:
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "disease_progression_stage",
            "tail": stage_map[stage],
            "confidence": 0.90,
            "evidence_level": "II",
            "source": source,
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })

    return triples


def extract_triples_from_genomics(genomic_profile: dict, triple_counter: list) -> list:
    """Extract structured triples from genomic profile."""
    triples = []
    patient_id = genomic_profile["patient_id"]
    profiling_date = genomic_profile.get("profiling_date", "2024-05-15")
    source = f"GENOMICS:{genomic_profile['genomic_profile_id']}"

    def next_id():
        triple_counter[0] += 1
        return f"T{triple_counter[0]:04d}"

    # Gene variants
    for variant in genomic_profile.get("variants", []):
        gene = variant.get("gene", "unknown")
        significance = variant.get("clinical_significance", "unknown")
        condition = variant.get("associated_condition", "unknown")

        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "carries_genetic_variant",
            "tail": f"{gene}:{variant.get('variant_id','unknown')}:{significance}",
            "confidence": 0.88,
            "evidence_level": "II",
            "source": source,
            "timestamp": profiling_date,
            "klm_source": "patient_klm"
        })

        if significance in ["pathogenic", "likely pathogenic"]:
            triples.append({
                "triple_id": next_id(),
                "head": f"{gene}_variant",
                "relation": "predisposes_to",
                "tail": condition,
                "confidence": 0.85,
                "evidence_level": "II",
                "source": source,
                "timestamp": profiling_date,
                "klm_source": "patient_klm"
            })

    # Polygenic risk scores
    for prs in genomic_profile.get("polygenic_risk_scores", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_polygenic_risk_score",
            "tail": f"{prs['trait']}:score={prs['score']}:percentile={prs['percentile']}",
            "confidence": 0.82,
            "evidence_level": "III",
            "source": source,
            "timestamp": profiling_date,
            "klm_source": "patient_klm"
        })

    # Pharmacogenomics
    for pgx in genomic_profile.get("pharmacogenomics", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_pharmacogenomic_profile",
            "tail": f"{pgx['gene']}:{pgx.get('drug_implication','')}",
            "confidence": 0.88,
            "evidence_level": "II",
            "source": source,
            "timestamp": profiling_date,
            "klm_source": "patient_klm"
        })

    # DNA fragment references
    for frag in genomic_profile.get("dna_sequences", []):
        triples.append({
            "triple_id": next_id(),
            "head": patient_id,
            "relation": "has_dna_fragment",
            "tail": f"{frag['fragment_id']}:{frag['gene_region']}",
            "confidence": 0.95,
            "evidence_level": "I",
            "source": source,
            "timestamp": profiling_date,
            "klm_source": "patient_klm"
        })

        if frag.get("mutation_site") and "absent" not in frag["mutation_site"].lower():
            triples.append({
                "triple_id": next_id(),
                "head": frag["fragment_id"],
                "relation": "contains_mutation_site",
                "tail": frag["mutation_site"],
                "confidence": 0.90,
                "evidence_level": "I",
                "source": source,
                "timestamp": profiling_date,
                "klm_source": "patient_klm"
            })

    return triples



def extract_triples_from_clinical_notes_llm(visit: dict, triple_counter: list) -> list:
    """Use Claude to extract triples from unstructured clinical notes."""
    note = visit.get("clinical_notes", "")
    if not note or len(note) < 20:
        return []

    patient_id = visit["patient_id"]
    visit_date = visit["visit_date"]
    start_id = triple_counter[0] + 1

    prompt = f"""
Extract knowledge triples from this clinical note for patient {patient_id}, visit date {visit_date}.

Clinical note:
"{note}"

Return ONLY a valid JSON array (no markdown) of triples with this exact schema:
[
  {{
    "head": "entity_or_patient_id",
    "relation": "relationship_verb_snake_case",
    "tail": "target_entity_or_value",
    "confidence": 0.85,
    "evidence_level": "III"
  }}
]

Rules:
- Use patient ID "{patient_id}" as head when the triple is about the patient
- Keep relations as snake_case verbs (e.g., has_risk_factor, shows_clinical_finding, requires_follow_up)
- Extract 3-6 triples maximum
- Focus on clinically meaningful relationships only
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

    llm_triples = json.loads(raw.strip())
    result = []
    for t in llm_triples:
        triple_counter[0] += 1
        result.append({
            "triple_id": f"T{triple_counter[0]:04d}",
            "head": t["head"],
            "relation": t["relation"],
            "tail": t["tail"],
            "confidence": t.get("confidence", 0.75),
            "evidence_level": t.get("evidence_level", "III"),
            "source": f"EHR_NOTES:{visit['visit_id']}",
            "timestamp": visit_date,
            "klm_source": "patient_klm"
        })
    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Load EHR and genomic data
    with open("data/ehr_records.json") as f:
        ehr_data = json.load(f)
    with open("data/genomic_profile.json") as f:
        genomic_data = json.load(f)

    all_triples = []
    counter = [0]  # mutable counter passed by reference

    # Patient metadata triples
    pm = ehr_data["patient"]
    counter[0] += 1
    all_triples.append({
        "triple_id": f"T{counter[0]:04d}",
        "head": pm["patient_id"],
        "relation": "has_attribute",
        "tail": f"dob:{pm['dob']}",
        "confidence": 1.0,
        "evidence_level": "I",
        "source": "EHR:DEMOGRAPHICS",
        "timestamp": "2022-01-01",
        "klm_source": "patient_klm"
    })
    counter[0] += 1
    all_triples.append({
        "triple_id": f"T{counter[0]:04d}",
        "head": pm["patient_id"],
        "relation": "has_attribute",
        "tail": f"sex:{pm['sex']}",
        "confidence": 1.0,
        "evidence_level": "I",
        "source": "EHR:DEMOGRAPHICS",
        "timestamp": "2022-01-01",
        "klm_source": "patient_klm"
    })
    counter[0] += 1
    all_triples.append({
        "triple_id": f"T{counter[0]:04d}",
        "head": pm["patient_id"],
        "relation": "has_family_history",
        "tail": "renal_cell_carcinoma:maternal_uncle",
        "confidence": 0.95,
        "evidence_level": "II",
        "source": "EHR:DEMOGRAPHICS",
        "timestamp": "2022-01-01",
        "klm_source": "patient_klm"
    })

    # EHR visit triples
    print(f"\nProcessing {len(ehr_data['ehr_visits'])} EHR visits...")
    for visit in ehr_data["ehr_visits"]:
        structured = extract_triples_from_ehr_visit(visit, counter)
        all_triples.extend(structured)

        print(f"  Extracting LLM triples from notes: {visit['visit_id']}")
        llm = extract_triples_from_clinical_notes_llm(visit, counter)
        all_triples.extend(llm)
        print(f"  {visit['visit_id']}: {len(structured)} structured + {len(llm)} LLM triples")

    # Genomic triples
    print("\nExtracting genomic triples...")
    genomic_triples = extract_triples_from_genomics(genomic_data, counter)
    all_triples.extend(genomic_triples)
    print(f"  {len(genomic_triples)} genomic triples")

    # Save
    output = {
        "klm_id": "patient_klm_1",
        "patient_id": "P-001",
        "generated_at": datetime.now().isoformat(),
        "total_triples": len(all_triples),
        "triples": all_triples
    }

    os.makedirs("data", exist_ok=True)
    with open("data/triples.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Knowledge triples saved to data/triples.json")
    print(f"   Total triples: {len(all_triples)}")

    return output


if __name__ == "__main__":
    main()
