

import os
import sys
import time

def check_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(" ANTHROPIC_API_KEY not set. Export it first:")
        print("   export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)
    print(" API key found")

def run_step(name, module_path):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print('='*60)
    import importlib.util
    spec = importlib.util.spec_from_file_location("module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.main()
    time.sleep(1)
    return result

if __name__ == "__main__":
    check_env()
    os.makedirs("data", exist_ok=True)

    run_step("1. Generate Synthetic EHR Records (3 years)", "generate_ehr.py")
    run_step("2. Generate Synthetic Genomic Profile", "generate_genomics.py")
    run_step("3. Extract Knowledge Triples", "extract_triples.py")
    run_step("4. Build KLM Triple Store (SQLite)", "build_klm_store.py")

    print(f"\n{'='*60}")
    print(" Test KLM Query API")
    print('='*60)
    os.system("klm_api.py")

    print(f"\n{'='*60}")
    print("✅ Patient KLM Pipeline Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for f in sorted(os.listdir("data")):
        path = f"data/{f}"
        size = os.path.getsize(path)
        print(f"  {path} ({size:,} bytes)")

    print("\nNext steps:")
    print("  - Import PatientKLM from klm_api.py in your agent code")
    print("  - Use klm.to_prompt_context(patient_id) to inject context into agent prompts")
    print("  - Use klm.get_agent_context(patient_id) for structured JSON context")
    print("  - Query data/patient_klm.db directly from the Meta Model")
