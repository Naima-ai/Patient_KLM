"""
Focuses on nephrology-relevant genetic markers:
  - VHL gene mutations (clear cell RCC)
  - PBRM1 (chromatin remodeling, RCC)
  - BAP1 (tumor suppressor, RCC aggressiveness)
  - CKD predisposition variants (APOL1, UMOD)
  - Hypertension-related (ACE, AGT)
"""

import json
import anthropic
import os
import random

client = anthropic.Anthropic()


def generate_genomic_profile(patient_meta):
    prompt = """
You are a clinical genomics system. Generate a synthetic but clinically plausible genomic profile for a nephrology patient.

Patient context:
- Female, 46 years old, North African ethnicity
- Family history: maternal uncle had renal cell carcinoma
- Developed clear cell RCC (right kidney, T1b) after 2 years of subtle symptoms
- Has mild hypertension

Generate a JSON genomic profile (no markdown) with the following structure. Use realistic but SYNTHETIC (non-real-patient) data:

{
  "genomic_profile_id": "GEN-001",
  "patient_id": "P-001",
  "profiling_date": "2024-05-15",
  "profiling_method": "Targeted NGS Panel (NephroGenome v2.1)",
  "variants": [
    {
      "gene": "VHL",
      "variant_id": "rs...",
      "hgvs_notation": "c.XXX>X",
      "zygosity": "heterozygous | homozygous",
      "clinical_significance": "pathogenic | likely pathogenic | VUS | benign",
      "associated_condition": "Clear cell renal cell carcinoma",
      "evidence_level": "strong | moderate | limited",
      "allele_frequency_population": 0.001
    }
  ],
  "polygenic_risk_scores": [
    {
      "trait": "Renal Cell Carcinoma",
      "score": 0.78,
      "percentile": 84,
      "interpretation": "..."
    }
  ],
  "pharmacogenomics": [
    {
      "gene": "CYP3A4",
      "variant": "...",
      "drug_implication": "Normal metabolism of sunitinib (RCC first-line therapy)"
    }
  ],
  "ancestry_composition": {
    "North_African": 0.72,
    "Middle_Eastern": 0.18,
    "Sub_Saharan_African": 0.10
  },
  "mitochondrial_haplogroup": "...",
  "dna_fragment_ids": ["FRAG-001", "FRAG-002", "FRAG-003"],
  "summary_clinical_interpretation": "..."
}

Include variants for these genes: VHL, PBRM1, BAP1, APOL1, UMOD, ACE.
Make VHL pathogenic heterozygous (consistent with familial RCC predisposition).
Make PBRM1 a VUS.
All other values should be clinically reasonable for this patient profile.
Return ONLY the JSON object.
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def generate_synthetic_dna_fragments():
    """Generate a few simplified synthetic DNA sequence fragments
    representing regions around the VHL and PBRM1 genes."""
    bases = ['A', 'T', 'C', 'G']
    fragments = []

    # VHL gene region fragment (chromosome 3p25.3) - synthetic
    vhl_fragment = {
        "fragment_id": "FRAG-001",
        "gene_region": "VHL (chr3:10183515-10183615)",
        "sequence_length": 100,
        "sequence": "ATGGAGCCTTACCTGTATGCAGGTGCAGACCAGCGGGAGCTGCAGCAGAGCTGCCTGCGCTTCGTGGTGGATGCCAAGATCAAGTTCAACCCC",
        "mutation_site": "position 67: C>T (p.Pro67Ser)",
        "clinical_note": "Synthetic fragment representing VHL exon 2 region with pathogenic variant"
    }

    # PBRM1 gene region fragment (chromosome 3p21.3) - synthetic
    pbrm1_fragment = {
        "fragment_id": "FRAG-002",
        "gene_region": "PBRM1 (chr3:52579100-52579200)",
        "sequence_length": 100,
        "sequence": "GAAGTTCAGCAGTTGCAGCAGCAGCAGCAGCAGCAACAGCAGCAGCAGCAGCAGCAGCAGCAGCAGAAGCTGAAGCAGCAGCAGCAGCAGCAG",
        "mutation_site": "position 43: G>A (intronic, potential splicing effect)",
        "clinical_note": "Synthetic fragment representing PBRM1 intron/exon boundary; VUS"
    }

    # APOL1 gene region fragment - synthetic (kidney disease predisposition)
    apol1_fragment = {
        "fragment_id": "FRAG-003",
        "gene_region": "APOL1 (chr22:36265860-36265960)",
        "sequence_length": 100,
        "sequence": "ATGGGAGAGAAAGAGCTGCAGAAGGAGATCAACAAGCAGCAGAAGAAGCTGCAGAAGCAGAAGGAGCTGCAGAAGCAGAAGCTGAAGCAGAAG",
        "mutation_site": "G1 risk variant absent; G2 risk variant absent (low-risk APOL1 genotype)",
        "clinical_note": "Synthetic APOL1 fragment; North African ancestry, G1/G2 risk variants checked - not detected"
    }

    return [vhl_fragment, pbrm1_fragment, apol1_fragment]


def main():
    print("=== Generating Synthetic Genomic Profile ===")
    profile = generate_genomic_profile({})
    print("  Genomic variants generated")

    print("  Adding synthetic DNA fragments...")
    profile["dna_sequences"] = generate_synthetic_dna_fragments()

    os.makedirs("data", exist_ok=True)
    output_path = "data/genomic_profile.json"
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n Genomic profile saved to {output_path}")
    print(f"   Variants: {len(profile.get('variants', []))}")
    print(f"   DNA fragments: {len(profile.get('dna_sequences', []))}")
    return profile


if __name__ == "__main__":
    main()
