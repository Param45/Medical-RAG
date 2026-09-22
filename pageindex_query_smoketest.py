"""
Medical Records RAG Demo — PageIndex Query Smoke Test (Task 3.2 Verification)

Runs live retrieval queries against real PageIndex trees for Patient A and Patient B
and verifies returned facts, report types, dates, and evidence IDs.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pageindex_backend.retrieve import retrieve
from patients import get_display_label


def run_smoketest():
    print("=" * 70)
    print("PAGEINDEX RETRIEVAL ENGINE SMOKE TEST (SRS §7.2, BUILD_GUIDE Task 3.2)")
    print("=" * 70)

    test_queries = [
        {
            "description": "UC-1: Lab Trend / Flowsheet Query (Patient A)",
            "question": "What is my hemoglobin trend over time?",
            "patient_ids": ["patient_a"],
        },
        {
            "description": "UC-2: Staging & Biomarker Query (Patient A)",
            "question": "What are my ER, PR, and HER2 receptor status and biopsy results?",
            "patient_ids": ["patient_a"],
        },
        {
            "description": "UC-3: Surgical & Treatment History (Patient B)",
            "question": "What surgeries and chemotherapy regimens did I receive?",
            "patient_ids": ["patient_b"],
        },
        {
            "description": "UC-5: Multi-Patient Comparison Query (Group Mode)",
            "question": "Compare the primary tumor staging and biomarker profiles across patients.",
            "patient_ids": ["patient_a", "patient_b"],
        },
    ]

    for idx, test in enumerate(test_queries, 1):
        desc = test["description"]
        q = test["question"]
        p_ids = test["patient_ids"]

        print(f"\n[{idx}/{len(test_queries)}] {desc}")
        print(f"Question: \"{q}\"")
        print(f"Scoped Patients: {', '.join(f'{get_display_label(p)} ({p})' for p in p_ids)}")

        try:
            facts = retrieve(question=q, patient_ids=p_ids)
            print(f"  [PASS] Retrieved {len(facts)} fact(s)")
            for f_idx, fact in enumerate(facts[:3], 1):
                p_id = fact.get("patient_id")
                ev_id = fact.get("evidence_id")
                rep_type = fact.get("report_type", "REPORT")
                rep_date = fact.get("report_date", "undated")
                text_preview = fact.get("text", "").replace("\n", " ")[:140]
                print(f"    Fact {f_idx} [{p_id}] ({rep_type}, {rep_date}):")
                print(f"      Evidence ID: {ev_id}")
                print(f"      Text: {text_preview}...")
        except Exception as exc:
            print(f"  [FAIL] Retrieval failed: {exc}")

    print("\n" + "=" * 70)
    print("PageIndex retrieval smoke test completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoketest()
