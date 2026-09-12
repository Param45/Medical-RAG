"""
Medical Records RAG Demo — End-to-End Phase 1 Smoke Test & Runner (Task 1.8)

Executes the full Common Core pipeline start to finish:
1. Ingestion: PDF -> Page Images (ingest.py)
2. OCR: Bilingual Text Extraction (ocr.py)
3. Report Splitting: Document Boundary Detection (split_reports.py)
4. Chunking & Normalization: Evidence Record Generation (chunk.py)

Generates a unified Phase 1 execution summary table across all patients.
Maps to BUILD_GUIDE Task 1.8 and SRS §13.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chunk import chunk_all_patients, Chunk
from evidence_store import load_evidence
from ingest import ingest_all
from ocr import ocr_all_pages
from patients import all_patient_ids, get_display_label
from split_reports import split_all_reports, ReportSpan


def print_stage_banner(stage_num: int, title: str) -> None:
    """Print a prominent banner separating pipeline stages."""
    print("\n" + "=" * 80)
    print(f"=== STAGE {stage_num}: {title.upper()} ===")
    print("=" * 80 + "\n")


def run_phase1_pipeline() -> Dict[str, Any]:
    """
    Run the complete Phase 1 pipeline sequentially for all patients.
    Returns a dictionary of execution metrics per patient.
    """
    start_time = time.time()

    print("\n" + "#" * 80)
    print("# Medical Records RAG Demo — Phase 1 Common Core Pipeline")
    print("#" * 80)

    # -------------------------------------------------------------
    # STAGE 1: PDF Ingestion -> Page Images
    # -------------------------------------------------------------
    print_stage_banner(1, "PDF Ingestion -> Page Images (ingest.py)")
    ingest_results = ingest_all()

    # -------------------------------------------------------------
    # STAGE 2: OCR -> Page JSONs
    # -------------------------------------------------------------
    print_stage_banner(2, "Bilingual OCR Extraction (ocr.py)")
    for patient_id in all_patient_ids():
        label = get_display_label(patient_id)
        print(f"Running OCR for {patient_id} ({label})...")
        ocr_all_pages(patient_id)

    # -------------------------------------------------------------
    # STAGE 3: Report Boundary Splitting
    # -------------------------------------------------------------
    print_stage_banner(3, "Report Boundary Splitting (split_reports.py)")
    reports_by_patient = split_all_reports()

    # -------------------------------------------------------------
    # STAGE 4: Chunking, Normalization & Evidence Store
    # -------------------------------------------------------------
    print_stage_banner(4, "Chunking, Normalization & Evidence Store (chunk.py)")
    chunks_by_patient = chunk_all_patients()

    elapsed = round(time.time() - start_time, 2)

    # -------------------------------------------------------------
    # STAGE 5: Final Summary Table
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# === PHASE 1 PIPELINE COMPLETE: SUMMARY ===")
    print(f"# Elapsed time: {elapsed} seconds")
    print("#" * 80 + "\n")

    summary_data: Dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "patients": {},
        "totals": {
            "total_pages": 0,
            "total_reports": 0,
            "total_chunks": 0,
            "total_evidence": 0,
            "total_entities": 0,
        },
    }

    print(
        f"{'Patient ID':<14} | {'Display Label':<16} | {'Pages':<8} | {'Reports':<10} | {'Chunks':<8} | {'Evidence':<10} | {'Entities':<10}"
    )
    print("-" * 90)

    for pid in all_patient_ids():
        label = get_display_label(pid)
        page_paths = ingest_results.get(pid, [])
        pages_count = len(page_paths) if page_paths else len(list((Path("data/pages") / pid).glob("page_*.png")))
        reports_count = len(reports_by_patient.get(pid, []))
        chunks = chunks_by_patient.get(pid, [])
        chunks_count = len(chunks)
        evidence_records = load_evidence(pid)
        evidence_count = len(evidence_records)
        entities_count = sum(len(c.normalized_entities) for c in chunks)

        summary_data["patients"][pid] = {
            "display_label": label,
            "pages": pages_count,
            "reports": reports_count,
            "chunks": chunks_count,
            "evidence": evidence_count,
            "entities": entities_count,
        }

        summary_data["totals"]["total_pages"] += pages_count
        summary_data["totals"]["total_reports"] += reports_count
        summary_data["totals"]["total_chunks"] += chunks_count
        summary_data["totals"]["total_evidence"] += evidence_count
        summary_data["totals"]["total_entities"] += entities_count

        print(
            f"{pid:<14} | {label:<16} | {pages_count:<8} | {reports_count:<10} | {chunks_count:<8} | {evidence_count:<10} | {entities_count:<10}"
        )

    totals = summary_data["totals"]
    print("-" * 90)
    print(
        f"{'TOTAL':<14} | {'All Patients':<16} | {totals['total_pages']:<8} | {totals['total_reports']:<10} | {totals['total_chunks']:<8} | {totals['total_evidence']:<10} | {totals['total_entities']:<10}"
    )
    print("=" * 90)

    # Save summary report to data/_phase1_summary.md
    summary_md_path = Path("data") / "_phase1_summary.md"
    summary_md_content = f"""# Medical Records RAG Demo — Phase 1 Pipeline Summary

- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Elapsed Execution Time**: {elapsed}s

| Patient ID | Display Label | Pages | Reports Detected | Total Chunks | Evidence Records | Normalized Entities |
|---|---|---|---|---|---|---|
"""
    for pid, pdata in summary_data["patients"].items():
        summary_md_content += f"| `{pid}` | {pdata['display_label']} | {pdata['pages']} | {pdata['reports']} | {pdata['chunks']} | {pdata['evidence']} | {pdata['entities']} |\n"

    summary_md_content += (
        f"| **TOTAL** | **All Patients** | **{totals['total_pages']}** | **{totals['total_reports']}** | "
        f"**{totals['total_chunks']}** | **{totals['total_evidence']}** | **{totals['total_entities']}** |\n"
    )
    summary_md_path.write_text(summary_md_content, encoding="utf-8")
    print(f"\nPipeline summary saved to: {summary_md_path.resolve()}\n")

    return summary_data


if __name__ == "__main__":
    run_phase1_pipeline()
