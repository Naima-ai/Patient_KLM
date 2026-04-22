"""
    from dermatology_agent import DermatologyAgent
    agent = DermatologyAgent(patient_id="PT-9921")
    response = agent.ask("Is this patient's melanoma risk currently stable?")
    print(response)

Requirements:
    pip install anthropic requests
    The Patient KLM API must be running on localhost:8001
    (or set PATIENT_KLM_URL env var to point elsewhere).
"""

import os
import json
import requests
from anthropic import Anthropic

# ── Config ────────────────────────────────────────────────────────────────────

KLM_BASE_URL = os.environ.get("PATIENT_KLM_URL", "http://localhost:8001")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

DERMATOLOGY_KLM_SOURCES = [
    "patient_klm_dermatology",
]

SYSTEM_PROMPT_TEMPLATE = """
You are a specialist Dermatology AI Agent integrated with a Patient Knowledge
and Learning Model (KLM). Your role is to reason about skin health, melanoma
risk, photobiology, and the psychodermatological axis for a specific patient,
using the structured clinical knowledge triples provided below as ground truth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT ID: {patient_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATIENT KNOWLEDGE (KLM triples — treat as verified clinical facts):
{knowledge_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLINICAL SCOPE:
You specialise in the following domains for this patient:

1. MELANOMA RISK STRATIFICATION
   — Interpret CDKN2A (p16INK4a) and MC1R genomic variants
   — Track nevi count and atypical lesion evolution across visits
   — Classify current risk level (latent / escalating / active)
   — Recommend surveillance intervals and biopsy triggers

2. PHOTOBIOLOGY & UV VULNERABILITY
   — MC1R homozygous T/T → inefficient eumelanin; quantify solar risk
   — Advise on UV protection protocols (SPF, clothing, timing)
   — Assess cumulative UV damage risk relative to occupational/lifestyle exposure

3. PSYCHODERMATOLOGICAL AXIS
   — FKBP5-mediated HPA-axis dysregulation → sustained cortisol elevation
   — Cortisol-driven T-cell exhaustion → reduced immunosurveillance of
     melanocytic lesions
   — COMT Val158Met → stress-amplified neuroendocrine output
   — Correlate PHQ-9 / stress index with dermatological trajectory
   — Recommend stress-reduction interventions as part of skin cancer prevention

4. LESION MONITORING PROTOCOL
   — Baseline: 42 nevi, 0 atypical (2024-11-12)
   — Follow-up: 46 nevi, 3 atypical, under monitoring (2025-11-15)
   — Flag any changes requiring urgent dermatoscopy or excision
   — Suggest total body photography / reflectance confocal microscopy if warranted

5. INTEGRATED RISK TRAJECTORY
   — Combine genomic burden + cortisol trajectory + lesion data
   — Classify system state (stable / pre-chaotic / chaotic)
   — Generate personalised risk narrative for both clinician and patient

COMMUNICATION STYLE:
- For clinician queries: use precise medical terminology, cite confidence levels
  and evidence grades from the KLM triples where relevant.
- For patient-facing queries: use plain language, avoid alarming framing,
  focus on actionable steps.
- Always ground recommendations in the KLM data — do not extrapolate beyond
  what the triples support without explicitly flagging uncertainty.
- If a question falls outside dermatology, cardiology, nephrology, or
  psychodermatology, flag it and suggest the appropriate specialist agent.

IMPORTANT DISCLAIMER:
This agent is a decision-support tool, not a substitute for a board-certified
dermatologist. All clinical decisions must be confirmed by a licensed physician.
"""


# ── KLM fetch ────────────────────────────────────────────────────────────────

def fetch_dermatology_triples(patient_id: str) -> list[dict]:
    """
    Pull only dermatology KLM triples for this patient from the API.
    Falls back to the full patient endpoint if domain filtering returns nothing.
    """
    url = f"{KLM_BASE_URL}/patient/{patient_id}/domain/dermatology"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("total_triples", 0) > 0:
                return data["triples"]
    except requests.RequestException:
        pass

    # Fallback: filter full patient context by klm_source
    url = f"{KLM_BASE_URL}/patient/{patient_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        all_triples = resp.json().get("triples", [])
        return [
            t for t in all_triples
            if t.get("klm_source") in DERMATOLOGY_KLM_SOURCES
        ]
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach Patient KLM API at {KLM_BASE_URL}: {exc}\n"
            "Make sure the API is running and the patient has been seeded."
        ) from exc


def format_knowledge_block(triples: list[dict]) -> str:
    """Render triples as a compact, readable text block for the system prompt."""
    if not triples:
        return "  [No dermatology triples found for this patient]"

    lines = []
    for t in sorted(triples, key=lambda x: x.get("timestamp", "")):
        conf  = t.get("confidence", "?")
        evlvl = t.get("evidence_level", "?")
        ts    = t.get("timestamp", "unknown")
        lines.append(
            f"  [{ts}] [{evlvl}] [conf:{conf:.2f}]  "
            f"{t['head']}  —[{t['relation']}]→  {t['tail']}"
        )
    return "\n".join(lines)


# ── Agent ────────────────────────────────────────────────────────────────────

class DermatologyAgent:
    """
    Stateful, multi-turn Dermatology Specialist Agent.

    Each instance is scoped to a single patient. The KLM triples are fetched
    once at initialisation and injected into the system prompt so every turn
    has full context without repeat API calls.

    Example:
        agent = DermatologyAgent("PT-9921")
        print(agent.ask("Summarise this patient's melanoma risk."))
        print(agent.ask("What would you recommend at the next visit?"))
    """

    def __init__(self, patient_id: str, verbose: bool = False):
        self.patient_id = patient_id
        self.verbose    = verbose
        self.client     = Anthropic()
        self.history: list[dict] = []

        # Fetch and build context once
        triples = fetch_dermatology_triples(patient_id)
        if verbose:
            print(f"[DermatologyAgent] Loaded {len(triples)} triples for {patient_id}")

        knowledge_block  = format_knowledge_block(triples)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            patient_id=patient_id,
            knowledge_block=knowledge_block,
        )

    def ask(self, question: str) -> str:
        """
        Send a question to the agent and return the response string.
        Conversation history is maintained across calls.
        """
        self.history.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            system=self.system_prompt,
            messages=self.history,
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear conversation history (keep patient context)."""
        self.history = []

    def summary(self) -> str:
        """Ask the agent for a concise one-paragraph patient summary."""
        return self.ask(
            "Please provide a concise clinical summary of this patient's "
            "current dermatological risk status, integrating genomic, "
            "psychosomatic, and lesion-level data."
        )


# ── Standalone entrypoint ─────────────────────────────────────────────────────

def main():
    """
    Interactive REPL for the Dermatology Agent.
    Run:  python dermatology_agent.py
    """
    patient_id = input("Patient ID (default: PT-9921): ").strip() or "PT-9921"

    print(f"\nInitialising Dermatology Agent for {patient_id}...")
    try:
        agent = DermatologyAgent(patient_id=patient_id, verbose=True)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return

    print("\nDermatology Agent ready. Type your question or 'quit' to exit.\n")
    print("─" * 70)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "reset":
            agent.reset()
            print("[History cleared]")
            continue

        response = agent.ask(question)
        print(f"\nAgent: {response}")

    print("\nSession ended.")


if __name__ == "__main__":
    main()
