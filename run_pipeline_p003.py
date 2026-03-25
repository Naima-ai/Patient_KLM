"""
run_pipeline_p003.py
Runs all generation steps for P-003 then seeds the DB.

Steps:
  1. Generate EHR records (3 years, 6 visits)
  2. Generate genomic profile
  3. Generate pathology triples (CKD, hypertension, AFib, CRS, LVH)
  4. Extract patient knowledge triples from EHR + genomics
  5. Seed everything into patient_klm.db

Run from the patient_klm/ directory:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo_3/run_pipeline_p003.py
"""

import os
import sys
import time
import importlib.util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO3_DIR = os.path.join(BASE_DIR, "demo_3")

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DEMO3_DIR)


def check_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    print("✅ API key found")


def run_step(name, filepath):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print('='*60)
    spec = importlib.util.spec_from_file_location("module", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.main()
    time.sleep(1)
    return result


if __name__ == "__main__":
    check_env()
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    run_step(
        "1. Generate P-003 EHR Records (3 years — hypertension → CKD → cardiorenal)",
        os.path.join(DEMO3_DIR, "generate_ehr_p003.py")
    )
    run_step(
        "2. Generate P-003 Genomic Profile",
        os.path.join(DEMO3_DIR, "generate_genomics_p003.py")
    )
    run_step(
        "3. Generate Pathology Knowledge Triples (CKD, hypertension, AFib, CRS)",
        os.path.join(DEMO3_DIR, "generate_pathology_p003.py")
    )
    run_step(
        "4. Extract P-003 Patient Knowledge Triples",
        os.path.join(DEMO3_DIR, "extract_triples_p003.py")
    )
    run_step(
        "5. Seed P-003 into patient_klm.db",
        os.path.join(DEMO3_DIR, "seed_p003.py")
    )

    print(f"\n{'='*60}")
    print("✅ P-003 Pipeline Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(os.path.join(BASE_DIR, "data"))):
        if "p003" in f:
            path = os.path.join(BASE_DIR, "data", f)
            size = os.path.getsize(path)
            print(f"  data/{f} ({size:,} bytes)")

    print("\nP-003 is now available via the endpoint:")
    print("  GET /patient/P-003")
    print("  GET /patient/P-003/timeline")
    print("  GET /patient/P-003/genomics")
