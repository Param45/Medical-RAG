"""
Medical Records RAG Demo — Retrieval Orchestration & Answer Generation (SRS §8)

Single entry-point function `answer_question(question, mode, selected_patients, backend)`
that dispatches to GraphRAG or PageIndex retrieval, generates a cited answer via LLM,
and performs a grounding check on citations (SRS FR-8.1.1–FR-8.2.3).

Maps to BUILD_GUIDE Task 4.1.
"""

import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from llm_client import chat
from patients import get_display_label
import graph_backend.retrieve as graph_retriever
import pageindex_backend.retrieve as pageindex_retriever

MEDICAL_DISCLAIMER = (
    "Disclaimer: This information is retrieved directly from documented medical records "
    "and does not constitute medical advice or clinical diagnosis."
)


def extract_and_filter_citations(
    answer_text: str,
    valid_evidence_ids: Set[str],
) -> Tuple[str, List[str]]:
    """
    Grounding check (SRS FR-8.2.2):
    Extracts citation markers like `[patient_a__report_0001__page_1__chunk_0]` from answer text,
    verifies them against the set of evidence IDs present in retrieved facts,
    and returns the cleaned answer text and list of valid cited evidence IDs in appearance order.
    """
    if not answer_text:
        return "", []

    # Find all bracketed tokens that match evidence_id structure
    # Matches patterns like [patient_a__report_0001__page_1__chunk_0] or [patient_b__report_0002__page_2__chunk_0]
    citation_pattern = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")

    found_citations: List[str] = []
    seen_valid: Set[str] = set()

    for match in citation_pattern.finditer(answer_text):
        cid = match.group(1)
        if cid in valid_evidence_ids:
            if cid not in seen_valid:
                seen_valid.add(cid)
                found_citations.append(cid)

    return answer_text, found_citations


def generate_answer(
    question: str,
    facts: List[Dict[str, Any]],
    mode: str = "individual",
    selected_patients: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Generates a cited clinical answer from retrieved facts (SRS §8.2, FR-8.2.1–FR-8.2.3).

    Args:
        question: User clinical question.
        facts: List of retrieved fact dicts (`[{"text": ..., "patient_id": ..., "evidence_id": ...}]`).
        mode: "individual" or "group".
        selected_patients: Optional list of patient IDs.

    Returns:
        Tuple of `(answer_text, citations_list)`.
    """
    if not facts:
        return "Not documented in the available records.", []

    # Collect valid evidence IDs for grounding check
    valid_evidence_ids: Set[str] = {
        f.get("evidence_id") for f in facts if f.get("evidence_id")
    }

    # Format facts for the prompt
    fact_lines = []
    for idx, f in enumerate(facts, 1):
        p_id = f.get("patient_id", "unknown")
        p_label = get_display_label(p_id)
        ev_id = f.get("evidence_id", "N/A")
        txt = (f.get("text") or "").strip()
        # Clean redundant inner newlines for prompt readability
        cleaned_snippet = " ".join(txt.split())[:1000]
        fact_lines.append(f"Fact {idx} [Patient: {p_label} ({p_id})] [Citation ID: {ev_id}]:\n\"{cleaned_snippet}\"\n")

    facts_block = "\n".join(fact_lines)[:15000]

    # Distinct patient IDs in retrieved facts
    patient_ids_in_facts = list({f.get("patient_id") for f in facts if f.get("patient_id")})
    is_multi_patient = mode.lower() == "group" and len(patient_ids_in_facts) > 1

    group_structure_instruction = ""
    if is_multi_patient:
        group_structure_instruction = (
            "IMPORTANT: Because multiple patients are selected, structure your response clearly PER PATIENT "
            "(e.g., separate sections or bullet points for Patient A and Patient B). "
            "Never blend statements from different patients into one uncited sentence (SRS FR-8.2.3).\n"
        )

    system_prompt = (
        "You are an expert clinical oncology documentation assistant. "
        "Your task is to answer the user's clinical question using ONLY the provided facts.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. Answer ONLY based on the facts provided below. Do not invent, extrapolate, or assume facts not explicitly stated.\n"
        "2. If the facts are insufficient to fully answer the question, state clearly: 'Not documented in the available records.'\n"
        "3. Ground every factual claim with an inline citation using the exact Citation ID in square brackets, e.g., "
        "[patient_a__report_0001__page_1__chunk_0]. Place citation markers directly after the claim.\n"
        "4. If the question appears to ask for clinical advice, disease prognosis, or treatment recommendations, "
        "answer strictly with what is documented in the records and append this standard disclaimer sentence: "
        f"'{MEDICAL_DISCLAIMER}'.\n"
        f"{group_structure_instruction}"
    )

    user_prompt = (
        f"Clinical Question: \"{question}\"\n\n"
        f"--- RETRIEVED EVIDENCE FACTS ---\n"
        f"{facts_block}\n"
        f"--------------------------------\n\n"
        "Provide a concise, cited clinical answer:"
    )

    try:
        raw_answer = chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            task="inference",
            max_retries=2,
            retry_delay_seconds=4.0,
        ).strip()
    except Exception as exc:
        print(f"Warning: generate_answer LLM call failed: {exc}")
        return (
            f"An error occurred during answer generation ({exc}). Retrieved facts are available in evidence store.",
            list(valid_evidence_ids),
        )

    # Perform grounding check on citations (SRS FR-8.2.2)
    cleaned_answer, valid_citations = extract_and_filter_citations(raw_answer, valid_evidence_ids)

    return cleaned_answer, valid_citations


def answer_question(
    question: str,
    mode: str,
    selected_patients: List[str],
    backend: str,
) -> Dict[str, Any]:
    """
    Main orchestration entry point (SRS §8.1, FR-8.1.1).

    Args:
        question: User's clinical query.
        mode: 'individual' or 'group'.
        selected_patients: List of patient IDs (e.g. ['patient_a'] or ['patient_a', 'patient_b']).
        backend: 'graph' or 'pageindex'.

    Returns:
        Dict matching SRS FR-8.1.1 return contract:
        `{"answer": str, "citations": list[str], "backend_used": str}`

    Raises:
        ValueError: If mode or backend is invalid, or selected_patients is empty.
    """
    norm_mode = (mode or "").strip().lower()
    norm_backend = (backend or "").strip().lower()

    if norm_mode not in ("individual", "group"):
        raise ValueError(
            f"Invalid mode '{mode}'. Mode must be either 'individual' or 'group'."
        )

    if norm_backend not in ("graph", "pageindex"):
        raise ValueError(
            f"Invalid backend '{backend}'. Backend must be either 'graph' or 'pageindex'."
        )

    if not selected_patients:
        return {
            "answer": "No patient selected.",
            "citations": [],
            "backend_used": backend,
        }

    # Dispatch to appropriate backend (SRS FR-8.1.2)
    if norm_backend == "graph":
        facts = graph_retriever.retrieve(question, selected_patients)
    else:  # pageindex
        facts = pageindex_retriever.retrieve(question, selected_patients)

    # Generate grounded, cited answer (SRS FR-8.2.1–FR-8.2.3)
    answer_text, citations = generate_answer(
        question=question,
        facts=facts,
        mode=norm_mode,
        selected_patients=selected_patients,
    )

    return {
        "answer": answer_text,
        "citations": citations,
        "backend_used": backend,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_q = sys.argv[1]
    else:
        test_q = "What is my diagnosis and treatment history?"

    result = answer_question(
        question=test_q,
        mode="individual",
        selected_patients=["patient_a"],
        backend="pageindex",
    )
    print(f"Backend Used: {result['backend_used']}")
    print(f"Answer:\n{result['answer']}")
    print(f"Citations: {result['citations']}")
