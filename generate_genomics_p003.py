"""
generate_genomics_p003.py
Generates synthetic genomic/DNA profile for P-003.

Relevant genes:
  - ACE (I/D polymorphism) — hypertension and CKD risk
  - APOL1 (G1/G2 variants) — CKD risk in some populations
  - AGT (M235T) — angiotensinogen, hypertension
  - SCN5A — cardiac arrhythmia / atrial fibrillation predisposition
  - NPHS2 — podocin, proteinuria / nephrotic spectrum
  - CYP2C9 / VKORC1 — warfarin/anticoagulant metabolism
  - KCNQ1 — cardiac conduction

Output: data/p003_genomic_profile.json
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
    "conditions": ["hypertension", "CKD stage 4", "atrial fibrillation", "cardiorenal syndrome"]
}


def generate_genomic_profile():
    prompt = f"""
You are a clinical genomics system. Generate a detailed synthetic genomic profile for:

Patient: {json.dumps(PATIENT_META, indent=2)}

The patient has hypertension, CKD stage 4, atrial fibrillation, and cardiorenal syndrome.
Generate a clinically realistic genomic profile that partially explains these conditions.

Return ONLY a valid JSON object (no markdown) with this exact structure:
{{
  "patient_id": "P-003",
  "profiling_date": "2023-02-14",
  "platform": "clinical_exome_panel",
  "genetic_variants": [
    {{
      "gene": "ACE",
      "variant_id": "rs4646994",
      "zygosity": "homozygous",
      "clinical_significance": "risk_factor",
      "associated_condition": "hypertension, CKD progression",
      "effect": "D allele homozygous — higher ACE activity, associated with faster CKD progression and LVH",
      "evidence_level": "II"
    }},
    {{
      "gene": "AGT",
      "variant_id": "rs699",
      "zygosity": "heterozygous",
      "clinical_significance": "likely_pathogenic",
      "associated_condition": "essential hypertension",
      "effect": "M235T variant — increased angiotensinogen levels, elevated BP risk",
      "evidence_level": "II"
    }},
    {{
      "gene": "SCN5A",
      "variant_id": "rs generate a realistic variant ID",
      "zygosity": "heterozygous",
      "clinical_significance": "VUS",
      "associated_condition": "atrial fibrillation predisposition",
      "effect": "Sodium channel variant, borderline reduced conduction velocity",
      "evidence_level": "III"
    }},
    {{
      "gene": "APOL1",
      "variant_id": "rs73885319",
      "zygosity": "heterozygous",
      "clinical_significance": "risk_factor",
      "associated_condition": "CKD, focal segmental glomerulosclerosis",
      "effect": "G1 variant carrier — moderate increased CKD risk",
      "evidence_level": "II"
    }},
    {{
      "gene": "CYP2C9",
      "variant_id": "rs1799853",
      "zygosity": "heterozygous",
      "clinical_significance": "pharmacogenomic",
      "associated_condition": "anticoagulant metabolism",
      "effect": "*2 allele — intermediate metaboliser, dose adjustment needed for warfarin",
      "evidence_level": "I"
    }},
    {{
      "gene": "NPHS2",
      "variant_id": "generate realistic variant ID",
      "zygosity": "heterozygous",
      "clinical_significance": "VUS",
      "associated_condition": "proteinuria, podocyte dysfunction",
      "effect": "Podocin variant contributing to persistent proteinuria",
      "evidence_level": "III"
    }}
  ],
  "polygenic_risk_scores": {{
    "hypertension_prs": {{
      "score": 0.0,
      "percentile": 0,
      "interpretation": "..."
    }},
    "ckd_progression_prs": {{
      "score": 0.0,
      "percentile": 0,
      "interpretation": "..."
    }},
    "atrial_fibrillation_prs": {{
      "score": 0.0,
      "percentile": 0,
      "interpretation": "..."
    }}
  }},
  "pharmacogenomics": {{
    "ACE_inhibitor_response": "...",
    "anticoagulant_metabolism": "...",
    "statin_metabolism": "...",
    "diuretic_response": "..."
  }},
  "dna_fragments": [
    {{
      "fragment_id": "FRAG-P003-001",
      "gene_region": "ACE intron 16",
      "sequence_excerpt": "CTGGAGACCACTCCCATCCTTTCT",
      "length_bp": 287,
      "mutation_sites": ["rs4646994 deletion polymorphism at position 144"]
    }},
    {{
      "fragment_id": "FRAG-P003-002",
      "gene_region": "AGT exon 2",
      "sequence_excerpt": "ATGCAGCGCATCCTGGTCATC",
      "length_bp": 198,
      "mutation_sites": ["rs699 C>T at position 89"]
    }},
    {{
      "fragment_id": "FRAG-P003-003",
      "gene_region": "SCN5A exon 28",
      "sequence_excerpt": "GAGCTGGTGGCCTTTGTGAAC",
      "length_bp": 312,
      "mutation_sites": ["heterozygous missense at position 156"]
    }}
  ],
  "family_history_genetic_notes": "Father with CAD and hypertension. Paternal uncle with CKD. Pattern consistent with polygenic cardiovascular-renal risk inheritance.",
  "clinical_summary": "..."
}}

Fill in all numeric values and text fields with clinically realistic data for this patient.
Make the polygenic risk scores consistent with the patient's conditions (hypertension PRS >85th percentile, CKD PRS >75th percentile, AFib PRS >70th percentile).
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    print("Generating P-003 genomic profile...")
    profile = generate_genomic_profile()

    os.makedirs("data", exist_ok=True)
    output_path = "data/p003_genomic_profile.json"
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n✅ P-003 genomic profile saved to {output_path}")
    print(f"   Variants: {len(profile.get('genetic_variants', []))}")
    print(f"   DNA fragments: {len(profile.get('dna_fragments', []))}")
    return profile


if __name__ == "__main__":
    main()
