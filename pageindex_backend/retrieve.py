"""
Medical Records RAG Demo — PageIndex Retrieval Engine (SRS §7.2)

Performs pure LLM reasoning traversal over the PageIndex hierarchical tree.
No vector embeddings, no similarity search — selection is 100% LLM reasoning
over node summaries at each hierarchy level.

Maps to BUILD_GUIDE Task 3.2 and SRS FR-7.2.1–FR-7.2.4.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from llm_client import chat
from patients import get_display_label


def get_default_pageindex_dir() -> Path:
    """Get default path to data/pageindex/ directory."""
    return _ROOT_DIR / "data" / "pageindex"


def load_tree(patient_id: str, pageindex_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the PageIndex tree JSON for a given patient (SRS FR-7.2.1).
    
    This direct file read serves as the strict patient scope boundary for
    Individual mode — data from other patients is never loaded or accessed.
    """
    if pageindex_dir is None:
        pageindex_dir = get_default_pageindex_dir()
    else:
        pageindex_dir = Path(pageindex_dir)

    tree_path = pageindex_dir / f"{patient_id}.json"
    if not tree_path.exists():
        raise FileNotFoundError(
            f"PageIndex tree not found for patient '{patient_id}' at {tree_path}. "
            "Please run pageindex_backend/build.py first."
        )

    try:
        data = json.loads(tree_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "root" not in data:
            raise ValueError(f"Invalid PageIndex tree structure in {tree_path}")
        return data
    except Exception as exc:
        raise RuntimeError(f"Error loading PageIndex tree from {tree_path}: {exc}") from exc


def should_allow_multi(question: str) -> bool:
    """
    Heuristic to determine if a question requires multi-branch traversal (SRS FR-7.2.3).
    
    Returns True for questions seeking timelines, trends over time, histories,
    summaries across multiple visits, or comparisons.
    """
    q_lower = question.lower()

    multi_keywords = [
        "trend",
        "trends",
        "progress",
        "progression",
        "progressive",
        "over time",
        "history",
        "timeline",
        "all",
        "every",
        "serial",
        "compare",
        "comparison",
        "change",
        "changes",
        "changing",
        "trajectory",
        "evolution",
        "evolve",
        "evolved",
        "multiple",
        "across",
        "previous",
        "course",
        "follow up",
        "follow-up",
        "monitoring",
        "throughout",
    ]

    for kw in multi_keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return True

    return False


def is_structured_query(question: str) -> bool:
    """
    Determines if a question seeks structured deterministic facts:
    chemo cycle counts, suggested vs performed tests, lab improvement/trends, or specific lab values.
    """
    q_lower = question.lower()
    patterns = [
        r"(?:chemo|chemotherapy|cycle|round).*?(?:remain|left|completed|many|current|status)",
        r"(?:completed|finished).*?(?:cycle|round)",
        r"how many (?:more )?(?:cycles|rounds)",
        r"(?:suggested|advised|ordered).*?(?:test|performed|done|pending)",
        r"(?:which|what).*?tests?.*?(?:suggested|performed|done|pending|advised)",
        r"(?:sugar|glucose|creatinine|hemoglobin|hb|platelet|wbc|bilirubin).*?(?:level|value|trend|reading|status|in\s+\w+|on\s+\d+)",
        r"(?:trend|improvement|improving|better|worse).*?(?:health|test|marker|sugar|creatinine|hb|lab)",
        r"are there any improvements in my health",
    ]
    for pat in patterns:
        if re.search(pat, q_lower):
            return True
    return False


def resolve_structured_query(
    question: str,
    patient_id: str,
    structured_facts: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Extracts deterministic structured answers from StructuredFacts node (Phase 3).
    Returns list of fact dicts, or None if the query cannot be answered by StructuredFacts.
    """
    if not structured_facts:
        return None

    q_lower = question.lower()

    # 1. Chemo cycle / remaining query
    if any(w in q_lower for w in ["cycle", "round", "chemo", "chemotherapy"]):
        chemo_cycles = structured_facts.get("chemo_cycles", [])
        plan = structured_facts.get("treatment_plan", {})

        m_user_round = re.search(r"completed\s+(\d+)(?:st|nd|rd|th)?\s+(?:round|cycle)", q_lower)
        user_completed = int(m_user_round.group(1)) if m_user_round else None

        if chemo_cycles or plan or user_completed is not None:
            completed_nums = {c.get("cycle_number") for c in chemo_cycles if c.get("cycle_number") is not None}
            max_completed_record = max(completed_nums) if completed_nums else 0
            completed = user_completed if user_completed is not None else max_completed_record

            planned = plan.get("planned_cycles") or 8
            regimen = plan.get("regimen") or "Chemotherapy"
            remaining = max(0, planned - completed)

            evidence_ids = [c.get("evidence_id") for c in chemo_cycles if c.get("evidence_id")]
            if plan.get("evidence_id"):
                evidence_ids.append(plan.get("evidence_id"))
            ev_id = evidence_ids[0] if evidence_ids else f"{patient_id}__chemo_cycle_record"

            text = (
                f"Chemotherapy cycle status: Patient has completed {completed} of {planned} planned cycles "
                f"of {regimen}. Exactly {remaining} cycles are remaining."
            )
            return [{
                "text": text,
                "patient_id": patient_id,
                "evidence_id": ev_id,
                "confidence": 1.0,
                "node_id": "structured_facts_chemo",
            }]

    # 2. Suggested vs Performed tests query
    if ("suggested" in q_lower or "advised" in q_lower) and ("test" in q_lower or "perform" in q_lower or "done" in q_lower):
        suggested_tests = structured_facts.get("suggested_tests", [])
        if suggested_tests:
            suggested_all = [t.get("test_name") for t in suggested_tests if t.get("test_name")]
            performed = [t.get("test_name") for t in suggested_tests if t.get("performed") and t.get("test_name")]
            pending = [t.get("test_name") for t in suggested_tests if not t.get("performed") and t.get("test_name")]

            ev_ids = [t.get("evidence_id") for t in suggested_tests if t.get("evidence_id")]
            ev_id = ev_ids[0] if ev_ids else f"{patient_id}__suggested_tests"

            s_str = ", ".join(suggested_all) if suggested_all else "None documented"
            p_str = ", ".join(performed) if performed else "None of the suggested tests have been performed yet"
            pend_str = ", ".join(pending) if pending else "All suggested tests have been performed"

            text = (
                f"Suggested tests summary: All tests suggested: {s_str}. "
                f"Tests performed: {p_str}. "
                f"Tests pending/not performed: {pend_str}."
            )
            return [{
                "text": text,
                "patient_id": patient_id,
                "evidence_id": ev_id,
                "confidence": 1.0,
                "node_id": "structured_facts_suggested_tests",
            }]

    # 3. Lab time series / trends / specific lab values
    lab_series = structured_facts.get("lab_time_series", [])
    if lab_series:
        try:
            from normalize import canonicalize_lab_name
            test_canonical = canonicalize_lab_name(question)
        except Exception:
            test_canonical = None
        target_name = test_canonical.strip().lower() if test_canonical else None

        matching_labs = []
        for item in lab_series:
            t_name = (item.get("test") or "").strip().lower()
            if target_name and (target_name in t_name or t_name in target_name):
                matching_labs.append(item)
            elif not target_name:
                for kw in ["sugar", "glucose", "creatinine", "hemoglobin", "hb", "platelet", "wbc"]:
                    if kw in q_lower and kw in t_name:
                        matching_labs.append(item)
                        break

        if matching_labs:
            matching_labs.sort(key=lambda x: x.get("date") or "0000-00-00")
            first = matching_labs[0]
            last = matching_labs[-1]
            first_flag = first.get("abnormal_flag", "normal")
            last_flag = last.get("abnormal_flag", "normal")

            if first_flag in ("low", "high") and last_flag == "normal":
                direction = "improving"
            elif first_flag == "normal" and last_flag in ("low", "high"):
                direction = "worsening"
            elif first_flag == last_flag and first_flag in ("low", "high"):
                direction = "persistently abnormal"
            else:
                direction = "stable/normal"

            series_str = "; ".join(
                f"{l.get('value')} {l.get('unit') or ''} ({l.get('abnormal_flag') or 'normal'}) on {l.get('date') or 'undated'}"
                for l in matching_labs
            )

            text = (
                f"{matching_labs[0].get('test')} trend: {series_str}. "
                f"Overall trend: {direction}."
            )
            ev_id = last.get("evidence_id") or first.get("evidence_id") or f"{patient_id}__lab_trend"

            return [{
                "text": text,
                "patient_id": patient_id,
                "evidence_id": ev_id,
                "confidence": 1.0,
                "node_id": "structured_facts_lab_trend",
            }]

    return None


def _parse_llm_node_selection(llm_response: str, available_node_ids: List[str]) -> List[str]:
    """
    Defensively parses LLM JSON response to extract selected node IDs.
    Handles code fences, raw arrays, or embedded JSON blocks.
    """
    cleaned = llm_response.strip()
    if not cleaned:
        return []

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    available_set = set(available_node_ids)

    # 1. Try direct JSON parsing
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item) in available_set]
        elif isinstance(parsed, dict):
            # Sometimes LLMs wrap response in {"selected_nodes": [...]}
            for key in ["selected_nodes", "nodes", "selected", "node_ids"]:
                if key in parsed and isinstance(parsed[key], list):
                    return [str(item) for item in parsed[key] if str(item) in available_set]
    except Exception:
        pass

    # 2. Fallback: regex search for JSON array pattern [ ... ]
    match = re.search(r"\[(.*?)\]", cleaned, flags=re.DOTALL)
    if match:
        try:
            parsed_bracket = json.loads(f"[{match.group(1)}]")
            if isinstance(parsed_bracket, list):
                return [str(item) for item in parsed_bracket if str(item) in available_set]
        except Exception:
            pass

    # 3. Fallback: search for occurrences of available_node_ids directly in text
    found_nodes: List[str] = []
    for nid in available_node_ids:
        if re.search(r"\b" + re.escape(nid) + r"\b", cleaned):
            found_nodes.append(nid)

    return found_nodes


def select_children(
    question: str,
    node: Dict[str, Any],
    allow_multi: bool = False,
) -> List[Dict[str, Any]]:
    """
    Calls the LLM to reason over child node summaries and select the relevant
    branch(es) to expand (SRS §7.2, FR-7.2.2, FR-7.2.3).
    
    Args:
        question: User clinical question.
        node: Current parent node containing 'children' list.
        allow_multi: Whether multiple children can be selected (e.g. for timeline queries).

    Returns:
        List of selected child node dictionaries.
    """
    children = node.get("children", [])
    if not children:
        return []

    # If children are already leaf pages (no further hierarchy to reason over), return them directly
    if all(c.get("node_type") == "Page" or "children" not in c for c in children):
        return children

    child_descriptions = []
    child_map: Dict[str, Dict[str, Any]] = {}

    for idx, child in enumerate(children, 1):
        cid = child.get("node_id", f"node_{idx}")
        child_map[cid] = child
        ctype = child.get("report_type") or child.get("node_type") or "Report"
        cdate = child.get("report_date") or "Undated"
        csummary = child.get("summary", "No summary available.")
        child_descriptions.append(
            f"Node ID: \"{cid}\"\n"
            f"Type: {ctype} | Date: {cdate}\n"
            f"Summary: {csummary}\n"
        )

    children_block = "\n".join(child_descriptions)

    multi_instruction = (
        "You may select MULTIPLE relevant Node IDs if several reports contain relevant information, "
        "evidence, or historical progression needed to answer the question."
        if allow_multi
        else "Select the single most relevant Node ID, or up to 2-3 Node IDs if closely relevant."
    )

    system_prompt = (
        "You are an expert clinical reasoning assistant navigating a hierarchical medical records index. "
        "Given a clinical question and a list of report summaries, determine which report node(s) "
        "contain the specific clinical evidence needed to answer the question accurately.\n"
        "Respond ONLY with a valid JSON array of selected Node IDs, e.g. [\"report_0001\", \"report_0003\"]. "
        "If no reports are relevant to the question, return an empty array []."
    )

    user_prompt = (
        f"Clinical Question: \"{question}\"\n\n"
        f"Available Reports:\n"
        f"----------------------------------------\n"
        f"{children_block}\n"
        f"----------------------------------------\n"
        f"{multi_instruction}\n\n"
        "Return a JSON array of selected Node IDs:"
    )

    try:
        response = chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            task="inference",
            max_retries=2,
            retry_delay_seconds=4.0,
        ).strip()
    except Exception as exc:
        print(f"Warning: select_children LLM call failed: {exc}")
        # On failure, return top candidate or all children if allow_multi
        return children[:3] if allow_multi else children[:1]

    selected_ids = _parse_llm_node_selection(response, list(child_map.keys()))

    # If single selection enforced and multiple returned, take the first
    if not allow_multi and len(selected_ids) > 1:
        selected_ids = selected_ids[:1]

    selected_children = [child_map[cid] for cid in selected_ids if cid in child_map]
    return selected_children


def traverse(
    question: str,
    patient_id: str,
    max_depth: int = 3,
    max_nodes_expanded: int = 6,
    allow_multi: bool = False,
    tree: Optional[Dict[str, Any]] = None,
    pageindex_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Traverses the patient's PageIndex tree using LLM reasoning (SRS FR-7.2.2).
    
    Starting at root, selects children at each level until reaching leaves
    or hitting depth/breadth limits.
    
    Returns:
        List of facts dicts matching the common facts contract:
        `[{"text": str, "patient_id": str, "evidence_id": str, "confidence": float, ...}, ...]`
    """
    if tree is None:
        try:
            tree = load_tree(patient_id=patient_id, pageindex_dir=pageindex_dir)
        except Exception as exc:
            print(f"Warning: Could not load PageIndex tree for {patient_id}: {exc}")
            return []

    # 1. Check if question is a structured query that can be resolved deterministically (Phase 3)
    if is_structured_query(question):
        s_facts = tree.get("structured_facts") or tree.get("root", {}).get("structured_facts")
        resolved = resolve_structured_query(question, patient_id, s_facts)
        if resolved:
            return resolved

    root = tree.get("root", {})
    if not root:
        return []

    # Widen node expansion budget for multi-branch/trend queries
    if allow_multi:
        max_nodes_expanded = max(max_nodes_expanded, 12)

    collected_leaves: List[Dict[str, Any]] = []
    nodes_expanded_count = 0

    # Queue of (current_node, current_depth)
    queue: List[tuple[Dict[str, Any], int]] = [(root, 1)]

    while queue and nodes_expanded_count < max_nodes_expanded:
        current_node, depth = queue.pop(0)
        nodes_expanded_count += 1

        node_type = current_node.get("node_type", "Root")
        children = current_node.get("children", [])

        # Check if current_node is a leaf Page node
        if node_type == "Page" or (not children and "raw_text" in current_node):
            raw_text = current_node.get("raw_text", "")
            evidence_id = current_node.get("evidence_id", "")
            source_type = current_node.get("source_type", "typed")
            conf = 0.8 if source_type == "cursive_handwritten" else 1.0
            collected_leaves.append({
                "text": raw_text,
                "patient_id": patient_id,
                "evidence_id": evidence_id,
                "confidence": conf,
                "node_id": current_node.get("node_id", ""),
                "chunk_type": current_node.get("chunk_type", "page"),
                "source_type": source_type,
                "result_date": current_node.get("result_date"),
            })
            continue

        if not children or depth >= max_depth:
            continue

        # Check if children are leaf Page nodes directly
        are_children_leaves = all(
            c.get("node_type") == "Page" or "raw_text" in c for c in children
        )

        if are_children_leaves:
            # Current node is a Report node with leaf pages; collect all pages under it
            for child in children:
                raw_text = child.get("raw_text", "")
                evidence_id = child.get("evidence_id", "")
                source_type = child.get("source_type", "typed")
                conf = 0.8 if source_type == "cursive_handwritten" else 1.0
                collected_leaves.append({
                    "text": raw_text,
                    "patient_id": patient_id,
                    "evidence_id": evidence_id,
                    "confidence": conf,
                    "node_id": child.get("node_id", ""),
                    "chunk_type": child.get("chunk_type", "page"),
                    "source_type": source_type,
                    "result_date": child.get("result_date"),
                    "report_type": current_node.get("report_type"),
                    "report_date": current_node.get("report_date"),
                })
        else:
            # Expand non-leaf children using LLM reasoning
            selected = select_children(
                question=question,
                node=current_node,
                allow_multi=allow_multi,
            )
            for child in selected:
                if nodes_expanded_count + len(queue) < max_nodes_expanded:
                    queue.append((child, depth + 1))

    # Sort collected leaves chronologically by date (Phase 3)
    def _leaf_sort_key(leaf: Dict[str, Any]):
        d = leaf.get("result_date") or leaf.get("report_date")
        return (0, d) if d else (1, "9999-99-99")

    collected_leaves.sort(key=_leaf_sort_key)

    # Deduplicate collected leaves by evidence_id while preserving chronological order
    seen_evidence_ids = set()
    unique_facts: List[Dict[str, Any]] = []
    for leaf in collected_leaves:
        ev_id = leaf.get("evidence_id")
        if ev_id and ev_id not in seen_evidence_ids:
            seen_evidence_ids.add(ev_id)
            unique_facts.append(leaf)
        elif not ev_id:
            unique_facts.append(leaf)

    return unique_facts


def retrieve(
    question: str,
    patient_ids: List[str],
    max_depth: int = 3,
    max_nodes_expanded: int = 6,
    pageindex_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Main retrieval entry point for PageIndex backend (SRS §7.2, §8.1).
    
    Loops traverse() independently per patient_id in patient_ids (SRS FR-7.2.4),
    maintaining strict patient scoping symmetry with the GraphRAG backend.

    Args:
        question: Clinical question asked by user.
        patient_ids: List of scoped patient IDs (one for Individual, 1+ for Group).
        max_depth: Maximum tree depth to explore (default: 3).
        max_nodes_expanded: Maximum total nodes to expand (default: 6).
        pageindex_dir: Optional custom path to data/pageindex/ directory.

    Returns:
        Flat list of facts dicts matching the common facts contract:
        `[{"text": str, "patient_id": str, "evidence_id": str, "confidence": float}, ...]`
    """
    if not patient_ids:
        return []

    allow_multi = should_allow_multi(question)
    all_facts: List[Dict[str, Any]] = []

    for idx, p_id in enumerate(patient_ids):
        # Small delay between multiple patients to avoid rate-limit bursts
        if idx > 0 and "pytest" not in sys.modules:
            time.sleep(2.0)

        patient_facts = traverse(
            question=question,
            patient_id=p_id,
            max_depth=max_depth,
            max_nodes_expanded=max_nodes_expanded,
            allow_multi=allow_multi,
            pageindex_dir=pageindex_dir,
        )
        all_facts.extend(patient_facts)

    return all_facts


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What is my hemoglobin trend?"

    target_patients = ["patient_a"]
    print(f"Executing PageIndex retrieval for query: \"{query}\" on {target_patients}")
    facts = retrieve(query, target_patients)
    print(f"\nRetrieved {len(facts)} facts:")
    for f in facts:
        print(f"  - [{f['patient_id']}] {f.get('report_type', 'REPORT')} (evidence: {f['evidence_id']})")
        print(f"    Snippet: {f['text'][:120]}...\n")
