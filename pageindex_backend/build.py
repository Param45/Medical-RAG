"""
Medical Records RAG Demo — PageIndex Tree Build (SRS §7.1, §12.2)

Builds a 3-level tree per patient (Patient -> Report -> Page) with
LLM-generated summaries at non-leaf nodes. Persisted as local JSON files
under data/pageindex/{patient_id}.json.

Maps to BUILD_GUIDE Task 3.1.
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure .env is loaded
load_dotenv()

from llm_client import chat
from patients import all_patient_ids, get_display_label


def get_default_reports_dir() -> Path:
    """Get default path to data/reports/ directory."""
    return _ROOT_DIR / "data" / "reports"


def get_default_pageindex_dir() -> Path:
    """Get default path to data/pageindex/ directory."""
    return _ROOT_DIR / "data" / "pageindex"


def generate_report_summary(
    report_id: str,
    report_type: str,
    report_date: Optional[str],
    text: str,
) -> str:
    """
    Generate a 2-4 sentence clinical summary for a single report (SRS FR-7.1.2).
    Calls llm_client.chat once with concatenated text of the report's chunks.
    """
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return f"{report_type} report dated {report_date or 'undated'} with no extractable text."

    # Truncate text if excessively long to stay well within LLM context
    truncated_text = cleaned_text[:12000]

    system_prompt = (
        "You are an expert clinical oncology documentation summarizer. "
        "Summarize the given clinical report into 2 to 4 concise, factual sentences. "
        "Highlight primary diagnoses, procedures, key lab/pathology values, biomarker status, "
        "or clinical findings documented in the text. "
        "You MUST explicitly preserve every lab value, date, dose, cycle number, and staging code present in the text verbatim. "
        "Never paraphrase numbers into vague language like 'some lab values were recorded'. "
        "Preserve the Impression and Comparison sentences from radiology and pathology reports verbatim. "
        "Do NOT invent or extrapolate facts not present in the provided text."
    )

    user_prompt = (
        f"Report ID: {report_id}\n"
        f"Report Type: {report_type}\n"
        f"Report Date: {report_date or 'Unknown / Undated'}\n\n"
        f"--- REPORT TEXT ---\n"
        f"{truncated_text}\n"
        f"-------------------\n\n"
        "Provide a 2 to 4 sentence summary of this report:"
    )

    try:
        summary = chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            task="build",
            max_retries=2,
            retry_delay_seconds=4.0,
        ).strip()
        return summary if summary else f"{report_type} ({report_date or 'undated'})."
    except Exception as exc:
        print(f"Warning: Failed to generate summary for {report_id}: {exc}")
        # Fallback heuristic summary on error
        first_few = " ".join(cleaned_text.split()[:40])
        return f"{report_type} ({report_date or 'undated'}): {first_few}..."


def generate_root_summary(
    patient_id: str,
    report_nodes: List[Dict[str, Any]],
) -> str:
    """
    Generate a 3-5 sentence overall patient history overview (SRS FR-7.1.2).
    Calls llm_client.chat once with concatenated report summaries.
    """
    display_label = get_display_label(patient_id)
    if not report_nodes:
        return f"Medical records for {display_label} ({patient_id}) with no reports documented."

    summaries_text_list = []
    for rep in report_nodes:
        rep_id = rep.get("node_id", "report")
        rep_type = rep.get("report_type", "REPORT")
        rep_date = rep.get("report_date") or "undated"
        rep_sum = rep.get("summary", "")
        summaries_text_list.append(f"- [{rep_id}] {rep_type} ({rep_date}): {rep_sum}")

    summaries_block = "\n".join(summaries_text_list)[:15000]

    system_prompt = (
        "You are an expert clinical summarizer. Summarize the patient's complete medical history "
        "into a concise 3 to 5 sentence clinical overview based on the provided report summaries. "
        "Synthesize primary diagnoses, staging/biomarkers, surgical and medical oncology treatments "
        "(chemotherapy, surgeries), imaging trends, and current disease status. "
        "Do not invent details not present in the report summaries."
    )

    user_prompt = (
        f"Patient: {display_label} ({patient_id})\n\n"
        f"--- REPORT SUMMARIES ---\n"
        f"{summaries_block}\n"
        f"------------------------\n\n"
        "Provide a 3 to 5 sentence clinical overview of this patient:"
    )

    try:
        summary = chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            task="build",
            max_retries=2,
            retry_delay_seconds=4.0,
        ).strip()
        return summary if summary else f"Clinical record overview for {display_label}."
    except Exception as exc:
        print(f"Warning: Failed to generate root summary for {patient_id}: {exc}")
        return f"Clinical record overview for {display_label} ({patient_id}) across {len(report_nodes)} reports."


def extract_impression_and_comparison(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract IMPRESSION and COMPARISON sections from report text (Phase 3).
    Especially useful for radiology and pathology reports.
    """
    if not text:
        return None, None

    impression = None
    comparison = None

    imp_match = re.search(
        r'(?:IMPRESSION|CONCLUSION|OPINION)\s*[:\-]\s*(.*?)(?=\n\s*(?:RECOMMENDATION|PLAN|COMPARISON|TECHNIQUE|FINDINGS|[A-Z\s]{4,}:)|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if imp_match:
        impression = imp_match.group(1).strip()

    comp_match = re.search(
        r'(?:COMPARISON|COMPARED WITH|COMPARED TO)\s*[:\-]\s*(.*?)(?=\n\s*(?:IMPRESSION|CONCLUSION|TECHNIQUE|FINDINGS|RECOMMENDATION|[A-Z\s]{4,}:)|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if comp_match:
        comparison = comp_match.group(1).strip()

    return impression, comparison


def build_structured_facts(
    patient_id: str,
    chunks_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a StructuredFacts node for the patient (Phase 3).
    Pre-computes structured medical facts (chemo cycles, treatment plans,
    lab trends, suggested vs performed tests, medications).
    Attempts to read from Neo4j; falls back to parsing chunks_data if Neo4j is offline.
    """
    structured_facts: Dict[str, Any] = {
        "node_id": f"{patient_id}_structured_facts",
        "node_type": "StructuredFacts",
        "summary": (
            "Pre-computed structured medical facts: lab trends, chemo cycles, "
            "suggested vs performed tests, medication list, and treatment plan."
        ),
        "lab_time_series": [],
        "chemo_cycles": [],
        "treatment_plan": {},
        "suggested_tests": [],
        "current_medications": [],
    }

    # Attempt to populate from Neo4j
    neo4j_success = False
    try:
        from graph_backend.build import get_driver
        driver = get_driver()
        with driver.session() as session:
            # 1. Chemo administrations
            c_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:HAS_CHEMO_ADMIN]->(ca:ChemoAdministration)
                RETURN ca.cycle_number AS cycle_number, ca.regimen AS regimen,
                       ca.date AS administration_date, r.evidence_id AS evidence_id
                ORDER BY ca.cycle_number ASC, ca.date ASC
                """,
                pid=patient_id,
            )
            for rec in c_res:
                structured_facts["chemo_cycles"].append({
                    "cycle_number": rec.get("cycle_number"),
                    "regimen": rec.get("regimen"),
                    "date": rec.get("administration_date"),
                    "evidence_id": rec.get("evidence_id"),
                })

            # 2. Treatment plan
            tp_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:HAS_TREATMENT_PLAN]->(tp:TreatmentPlan)
                RETURN tp.regimen AS regimen, tp.planned_cycles AS planned_cycles, r.evidence_id AS evidence_id
                LIMIT 1
                """,
                pid=patient_id,
            )
            tp_rec = tp_res.single()
            if tp_rec:
                structured_facts["treatment_plan"] = {
                    "regimen": tp_rec.get("regimen"),
                    "planned_cycles": tp_rec.get("planned_cycles"),
                    "evidence_id": tp_rec.get("evidence_id"),
                }

            # 3. Lab time series
            l_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:HAS_RESULT]->(lr:LabResult)-[:OF_TEST]->(lt:LabTest)
                RETURN lt.canonical_name AS test, lr.value AS value, lr.unit AS unit,
                       coalesce(lr.result_date, lr.date, r.report_date) AS date,
                       lr.abnormal_flag AS abnormal_flag, r.evidence_id AS evidence_id
                ORDER BY date ASC, lt.canonical_name ASC
                """,
                pid=patient_id,
            )
            for rec in l_res:
                structured_facts["lab_time_series"].append({
                    "test": rec.get("test"),
                    "value": rec.get("value"),
                    "unit": rec.get("unit"),
                    "date": rec.get("date"),
                    "abnormal_flag": rec.get("abnormal_flag"),
                    "evidence_id": rec.get("evidence_id"),
                })

            # 4. Suggested vs performed
            s_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:SUGGESTS_TEST]->(lt:LabTest)
                RETURN lt.canonical_name AS test_name, r.report_date AS suggested_date, r.evidence_id AS evidence_id
                """,
                pid=patient_id,
            )
            suggested_list = [
                {"test_name": r.get("test_name"), "suggested_date": r.get("suggested_date"), "evidence_id": r.get("evidence_id")}
                for r in s_res
            ]
            perf_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:HAS_RESULT]->(lr:LabResult)-[:OF_TEST]->(lt:LabTest)
                RETURN DISTINCT toLower(lt.canonical_name) AS performed_test
                """,
                pid=patient_id,
            )
            perf_set = {r.get("performed_test") for r in perf_res if r.get("performed_test")}
            for s in suggested_list:
                t_name = s.get("test_name") or ""
                s["performed"] = t_name.strip().lower() in perf_set
            structured_facts["suggested_tests"] = suggested_list

            # 5. Medications
            m_res = session.run(
                """
                MATCH (p:Patient {patient_id: $pid})-[:HAS_REPORT]->(r:Report)-[:PRESCRIBES]->(m:Medication)
                RETURN m.canonical_name AS name, r.report_date AS start_date, r.evidence_id AS evidence_id
                """,
                pid=patient_id,
            )
            for rec in m_res:
                structured_facts["current_medications"].append({
                    "name": rec.get("name"),
                    "dose": rec.get("dose"),
                    "start_date": rec.get("start_date"),
                    "evidence_id": rec.get("evidence_id"),
                })
        neo4j_success = True
    except Exception:
        neo4j_success = False

    # Fallback to chunk scanning if Neo4j wasn't available or had no data
    if not neo4j_success or not structured_facts["chemo_cycles"]:
        for item in chunks_data:
            ev = item.get("evidence_record", {})
            raw_text = ev.get("raw_text", "")
            ev_id = ev.get("evidence_id", "")
            res_date = ev.get("result_date") or ev.get("report_date")

            # Chemo cycle regex
            chemo_m = re.search(r"(?:cycle|round)\s*[:#\-]?\s*(\d+)\s*(?:of\s*(\d+))?", raw_text, re.IGNORECASE)
            if chemo_m:
                c_num = int(chemo_m.group(1))
                p_cycles = int(chemo_m.group(2)) if chemo_m.group(2) else None
                if not any(c.get("cycle_number") == c_num for c in structured_facts["chemo_cycles"]):
                    structured_facts["chemo_cycles"].append({
                        "cycle_number": c_num,
                        "regimen": None,
                        "date": res_date,
                        "evidence_id": ev_id,
                    })
                if p_cycles and not structured_facts["treatment_plan"]:
                    structured_facts["treatment_plan"] = {
                        "regimen": None,
                        "planned_cycles": p_cycles,
                        "evidence_id": ev_id,
                    }

    return structured_facts


def build_tree_for_patient(
    patient_id: str,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build the 3-level PageIndex tree for a patient (SRS §7.1, §12.2).
    
    Tree structure:
    Patient (root) -> Report -> Page/Row (leaf)
    Plus synthetic StructuredFacts node at root level.

    Args:
        patient_id: Identifier of the patient (e.g., 'patient_a')
        reports_dir: Optional path to data/reports/ directory

    Returns:
        Dict matching SRS §12.2 PageIndex tree schema.
    """
    if reports_dir is None:
        reports_dir = get_default_reports_dir()
    else:
        reports_dir = Path(reports_dir)

    spans_path = reports_dir / f"{patient_id}.json"
    chunks_path = reports_dir / f"{patient_id}_chunks.json"

    if not spans_path.exists():
        raise FileNotFoundError(f"Report spans file not found: {spans_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Canonical chunks file not found: {chunks_path}")

    report_spans_data = json.loads(spans_path.read_text(encoding="utf-8"))
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Index chunks by report_id
    chunks_by_report: Dict[str, List[Dict[str, Any]]] = {}
    for item in chunks_data:
        ev = item.get("evidence_record", {})
        r_id = ev.get("report_id")
        if r_id:
            chunks_by_report.setdefault(r_id, []).append(item)

    report_nodes: List[Dict[str, Any]] = []

    # Iterate over report spans sequentially
    for span in report_spans_data:
        report_id = span.get("report_id", "")
        report_type = span.get("report_type", "REPORT")
        report_date = span.get("report_date")

        matching_chunks = chunks_by_report.get(report_id, [])

        # Build leaf Page / Row nodes
        page_nodes: List[Dict[str, Any]] = []
        page_texts: List[str] = []

        # Keep track of page node_ids to ensure uniqueness
        seen_page_ids: Dict[str, int] = {}

        for chunk_item in matching_chunks:
            ev = chunk_item.get("evidence_record", {})
            page_num = ev.get("page_number", 1)
            raw_text = ev.get("raw_text", "")
            evidence_id = ev.get("evidence_id", "")
            chunk_type = ev.get("chunk_type", "page")
            source_type = ev.get("source_type", "typed")
            result_date = ev.get("result_date")

            chunk_id = chunk_item.get("chunk_id", "")
            if chunk_type == "row" and chunk_id:
                base_node_id = f"{report_id}_p{page_num}_{chunk_id}"
            else:
                base_node_id = f"{report_id}_page_{page_num}"

            if base_node_id in seen_page_ids:
                seen_page_ids[base_node_id] += 1
                leaf_node_id = f"{base_node_id}_chunk_{seen_page_ids[base_node_id]}"
            else:
                seen_page_ids[base_node_id] = 0
                leaf_node_id = base_node_id

            page_node = {
                "node_id": leaf_node_id,
                "node_type": "Page",
                "chunk_type": chunk_type,
                "source_type": source_type,
                "result_date": result_date,
                "page_number": page_num,
                "raw_text": raw_text,
                "evidence_id": evidence_id,
            }
            page_nodes.append(page_node)
            if raw_text:
                page_texts.append(raw_text)

        # Concatenate text from all child chunks for LLM report summary
        concat_report_text = "\n\n".join(page_texts)

        # Extract impression and comparison for radiology/pathology
        impression, comparison = extract_impression_and_comparison(concat_report_text)

        # Generate report summary via LLM (SRS FR-7.1.2)
        report_summary = generate_report_summary(
            report_id=report_id,
            report_type=report_type,
            report_date=report_date,
            text=concat_report_text,
        )

        report_node = {
            "node_id": report_id,
            "node_type": "Report",
            "report_type": report_type,
            "report_date": report_date,
            "summary": report_summary,
            "impression": impression,
            "comparison": comparison,
            "children": page_nodes,
        }
        report_nodes.append(report_node)

        # Small pacing sleep to prevent rate-limit bursts on free tiers
        if "pytest" not in sys.modules:
            time.sleep(1.5)

    # Build StructuredFacts synthetic node
    structured_facts_node = build_structured_facts(patient_id, chunks_data)

    # Generate root summary via LLM across all report summaries (SRS FR-7.1.2)
    root_summary = generate_root_summary(patient_id, report_nodes)

    tree = {
        "patient_id": patient_id,
        "structured_facts": structured_facts_node,
        "root": {
            "node_id": "root",
            "summary": root_summary,
            "structured_facts": structured_facts_node,
            "children": report_nodes,
        },
    }

    return tree


def save_tree(
    patient_id: str,
    tree: Dict[str, Any],
    pageindex_dir: Optional[Path] = None,
) -> Path:
    """
    Save the PageIndex tree to data/pageindex/{patient_id}.json (SRS FR-7.1.4).
    """
    if pageindex_dir is None:
        pageindex_dir = get_default_pageindex_dir()
    else:
        pageindex_dir = Path(pageindex_dir)

    pageindex_dir.mkdir(parents=True, exist_ok=True)
    out_file = pageindex_dir / f"{patient_id}.json"
    out_file.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def build_all_pageindexes(
    reports_dir: Optional[Path] = None,
    pageindex_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build and save PageIndex trees sequentially for all registered patients (SRS §7.1).
    """
    patient_ids = all_patient_ids()
    results: Dict[str, Dict[str, Any]] = {}

    print("=" * 60)
    print("STAGE: PAGEINDEX TREE BUILD (SRS §7.1)")
    print(f"Target patients: {', '.join(patient_ids)}")
    print("=" * 60)

    start_time = time.time()

    for idx, pid in enumerate(patient_ids, 1):
        print(f"\n[{idx}/{len(patient_ids)}] Building PageIndex tree for {get_display_label(pid)} ({pid})...")
        t0 = time.time()
        tree = build_tree_for_patient(pid, reports_dir=reports_dir)
        saved_path = save_tree(pid, tree, pageindex_dir=pageindex_dir)
        dt = time.time() - t0

        report_count = len(tree.get("root", {}).get("children", []))
        total_pages = sum(
            len(r.get("children", [])) for r in tree.get("root", {}).get("children", [])
        )

        print(f"  ✓ Saved tree to {saved_path} in {dt:.1f}s")
        print(f"  ✓ Root summary: {tree.get('root', {}).get('summary')[:100]}...")
        print(f"  ✓ Reports: {report_count}, Pages/Leaves: {total_pages}")
        results[pid] = tree

    total_time = time.time() - start_time
    print(f"\nPageIndex tree build completed for all patients in {total_time:.1f}s.")
    return results


if __name__ == "__main__":
    build_all_pageindexes()
