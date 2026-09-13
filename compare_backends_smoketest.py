"""
Medical Records RAG Demo — Backend Comparison Smoke Test

Runs the same clinical questions through both:
1. GraphRAG (Neo4j Cypher knowledge graph queries)
2. PageIndex (LLM reasoning traversal over hierarchical index tree)

Compares facts, citations, retrieval precision, and clinical coverage side-by-side.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph_backend.retrieve import retrieve as graph_retrieve
from pageindex_backend.retrieve import retrieve as pageindex_retrieve
from patients import get_display_label


COMPARISON_QUERIES = [
    {
        "id": "Q1",
        "use_case": "UC-1: Laboratory Trend (Flowsheet Data)",
        "question": "What is my hemoglobin trend over time?",
        "patient_ids": ["patient_a"],
    },
    {
        "id": "Q2",
        "use_case": "UC-2: Staging, Histopathology & Receptor Biomarkers",
        "question": "What is my primary cancer diagnosis, histological grade, and ER/PR/HER2 receptor status?",
        "patient_ids": ["patient_a"],
    },
    {
        "id": "Q3",
        "use_case": "UC-3: Surgical & Chemotherapy History",
        "question": "What surgeries and chemotherapy medications were administered?",
        "patient_ids": ["patient_b"],
    },
]


def run_comparison():
    print("=" * 80)
    print("SIDE-BY-SIDE BACKEND COMPARISON: GRAPHRAG (NEO4J) vs PAGEINDEX (TREE)")
    print("=" * 80)

    for item in COMPARISON_QUERIES:
        qid = item["id"]
        uc = item["use_case"]
        q = item["question"]
        p_ids = item["patient_ids"]

        print(f"\n" + "#" * 80)
        print(f"QUERY {qid} — {uc}")
        print(f"Question: \"{q}\"")
        print(f"Patient(s): {', '.join(f'{get_display_label(p)} ({p})' for p in p_ids)}")
        print("#" * 80)

        # 1. GraphRAG Retrieval
        print(f"\n[1] GRAPHRAG BACKEND (Neo4j Cypher / Templates)")
        t0 = time.time()
        try:
            graph_facts = graph_retrieve(question=q, patient_ids=p_ids)
            dt_graph = time.time() - t0
            print(f"  ✓ Retrieved {len(graph_facts)} fact(s) in {dt_graph:.2f}s")
            for idx, fact in enumerate(graph_facts[:4], 1):
                ev = fact.get("evidence_id", "N/A")
                conf = fact.get("confidence", 1.0)
                txt = fact.get("text", "")
                print(f"    Fact {idx} (evidence: {ev}, conf: {conf:.2f}):")
                print(f"      {txt}")
            if len(graph_facts) > 4:
                print(f"      ... and {len(graph_facts) - 4} more facts")
        except Exception as exc:
            print(f"  ✗ GraphRAG retrieval failed: {exc}")
            graph_facts = []

        # Rate limit pacing
        time.sleep(3.0)

        # 2. PageIndex Retrieval
        print(f"\n[2] PAGEINDEX BACKEND (LLM Tree Traversal)")
        t1 = time.time()
        try:
            pageindex_facts = pageindex_retrieve(question=q, patient_ids=p_ids)
            dt_pi = time.time() - t1
            print(f"  ✓ Retrieved {len(pageindex_facts)} fact(s) in {dt_pi:.2f}s")
            for idx, fact in enumerate(pageindex_facts[:3], 1):
                ev = fact.get("evidence_id", "N/A")
                rep_type = fact.get("report_type", "REPORT")
                rep_date = fact.get("report_date", "undated")
                snippet = fact.get("text", "").replace("\n", " ")[:150]
                print(f"    Fact {idx} [{rep_type}, {rep_date}] (evidence: {ev}):")
                print(f"      \"{snippet}...\"")
            if len(pageindex_facts) > 3:
                print(f"      ... and {len(pageindex_facts) - 3} more pages")
        except Exception as exc:
            print(f"  ✗ PageIndex retrieval failed: {exc}")
            pageindex_facts = []

        # Rate limit pacing between questions
        time.sleep(3.0)

    print("\n" + "=" * 80)
    print("Comparison execution complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
