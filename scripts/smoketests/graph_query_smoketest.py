"""
Medical Records RAG Demo — Graph Query Smoke Test Script

Runs sample clinical queries corresponding to SRS §4.2 use cases against the
live Neo4j knowledge graph and prints retrieved facts with evidence tracking.

Maps to BUILD_GUIDE Task 2.4.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(_ROOT_DIR / ".env")

from graph_backend.retrieve import retrieve


def run_smoketest():
    print("=" * 70)
    print("        GRAPHRAG BACKEND RETRIEVAL SMOKE TEST (SRS §4.2)        ")
    print("=" * 70)

    test_queries = [
        {
            "use_case": "UC-1 (Individual: Lab Trends)",
            "question": "What is my hemoglobin and platelet trend?",
            "patient_ids": ["patient_a"],
        },
        {
            "use_case": "UC-2 (Individual: Diagnosis List)",
            "question": "What diseases and cancer diagnoses was I suffering from?",
            "patient_ids": ["patient_a"],
        },
        {
            "use_case": "UC-3 (Individual: Medication & Chemo History)",
            "question": "What chemotherapy and medications have I received and when?",
            "patient_ids": ["patient_a"],
        },
        {
            "use_case": "UC-6 (Group: Biomarker & Staging Status across all patients)",
            "question": "Which patients have HER2 and hormone receptor status documented?",
            "patient_ids": ["patient_a", "patient_b"],
        },
    ]

    for idx, test in enumerate(test_queries, 1):
        print(f"\n--- [{idx}/{len(test_queries)}] {test['use_case']} ---")
        print(f"Query: \"{test['question']}\"")
        print(f"Patients: {test['patient_ids']}")

        facts = retrieve(question=test["question"], patient_ids=test["patient_ids"])
        print(f"\nRetrieved {len(facts)} fact(s):")
        if not facts:
            print("  (No facts retrieved)")
        for f in facts:
            p_id = f.get("patient_id", "")
            ev_id = f.get("evidence_id", "")
            conf = f.get("confidence", 0.9)
            text = f.get("text", "")
            print(f"  • [{p_id}] {text}")
            print(f"    └─ Evidence: {ev_id} (Confidence: {conf:.2f})")

    print("\n" + "=" * 70)
    print("            SMOKE TEST EXECUTION COMPLETE             ")
    print("=" * 70)


if __name__ == "__main__":
    run_smoketest()
