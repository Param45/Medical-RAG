"""
Medical Records RAG Demo — Orchestration Smoke Test (Task 4.1 Verification)

Runs answer_question(...) for one query from EACH use case (UC-1 through UC-6)
across BOTH backends (GraphRAG and PageIndex), and prints the grounded answers
and citation lists.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from orchestrate import answer_question
from patients import all_patient_ids, get_display_label

SMOKETEST_CASES = [
    {
        "id": "UC-1",
        "title": "Individual: Lab Trend / Timeline",
        "question": "What is my health progress and hemoglobin trend across multiple reports?",
        "mode": "individual",
        "patients": ["patient_a"],
    },
    {
        "id": "UC-2",
        "title": "Individual: Diagnoses / Medical History",
        "question": "What primary disease and tumor grade was I diagnosed with?",
        "mode": "individual",
        "patients": ["patient_a"],
    },
    {
        "id": "UC-3",
        "title": "Individual: Chemotherapy & Medication History",
        "question": "What chemotherapy medications and doses did I receive?",
        "mode": "individual",
        "patients": ["patient_b"],
    },
    {
        "id": "UC-4",
        "title": "Group: Single Patient Treatment Details",
        "question": "Give me the surgical and medical oncology treatment details for Patient B.",
        "mode": "group",
        "patients": ["patient_b"],
    },
    {
        "id": "UC-5",
        "title": "Group: Multi-Patient Comparison",
        "question": "Compare the diagnosis and receptor biomarker status of Patient A vs Patient B.",
        "mode": "group",
        "patients": ["patient_a", "patient_b"],
    },
    {
        "id": "UC-6",
        "title": "Group: All Patients Cohort Query",
        "question": "Which patients have metastatic disease or HER2-positive status documented?",
        "mode": "group",
        "patients": all_patient_ids(),
    },
]


def run_smoketest():
    print("=" * 80)
    print("ORCHESTRATION LAYER SMOKE TEST (SRS §8.1, §8.2, BUILD_GUIDE Task 4.1)")
    print("Testing 6 Use Cases across BOTH Backends (GraphRAG & PageIndex)")
    print("=" * 80)

    backends = ["graph", "pageindex"]

    for case in SMOKETEST_CASES:
        uc_id = case["id"]
        title = case["title"]
        q = case["question"]
        mode = case["mode"]
        patients = case["patients"]

        print(f"\n" + "#" * 80)
        print(f"USE CASE {uc_id}: {title}")
        print(f"Question: \"{q}\"")
        print(f"Mode: {mode} | Patients: {', '.join(f'{get_display_label(p)} ({p})' for p in patients)}")
        print("#" * 80)

        for backend in backends:
            backend_label = "GraphRAG (Neo4j)" if backend == "graph" else "PageIndex (Tree)"
            print(f"\n--- Backend: {backend_label} ---")
            t0 = time.time()
            try:
                res = answer_question(
                    question=q,
                    mode=mode,
                    selected_patients=patients,
                    backend=backend,
                )
                dt = time.time() - t0
                print(f"Execution time: {dt:.2f}s")
                print(f"Answer:\n{res['answer']}\n")
                print(f"Valid Citations ({len(res['citations'])}):")
                for c in res['citations']:
                    print(f"  - {c}")
            except Exception as exc:
                print(f"Execution failed: {exc}")

            # Pacing delay between calls to respect LLM rate limits
            time.sleep(3.0)

    print("\n" + "=" * 80)
    print("Orchestration smoke test complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoketest()
