"""
Medical Records RAG Demo — Full Ingestion + OCR Pipeline Runner (Task 1.4)

Orchestrates ingest_all() and ocr_all_pages() sequentially across all registered
patient PDFs, and generates a structured summary table saved to data/ocr/_summary.md.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from ingest import ingest_all
from ocr import ocr_all_pages, PageOCRResult
from patients import all_patient_ids, get_display_label

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_ocr_summary(
    ocr_dir: str | Path = "data/ocr",
    patient_results: Dict[str, List[PageOCRResult]] = None,
) -> str:
    """
    Computes statistics across all patient OCR results and returns a formatted markdown table.
    """
    summary_lines = [
        "# Medical Records RAG Demo — OCR Pipeline Summary",
        f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Patient ID | Display Label | Total Pages | Tabular Pages (`is_table`) | Handwritten | Bilingual/Hindi (`devanagari`/`mixed`) | Low Conf (< 0.5) | Duplicates (`duplicate_of`) | Avg Confidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    total_all_pages = 0
    total_tables = 0
    total_hw = 0
    total_hindi = 0
    total_low_conf = 0
    total_dups = 0
    all_confs: List[float] = []

    for patient_id in all_patient_ids():
        results = (patient_results or {}).get(patient_id, [])
        label = get_display_label(patient_id)

        total_pages = len(results)
        table_count = sum(1 for r in results if r.is_table)
        hw_count = sum(1 for r in results if r.is_handwritten)
        hindi_count = sum(1 for r in results if r.script in ("devanagari", "mixed"))
        low_conf_count = sum(1 for r in results if r.confidence < 0.5)
        dup_count = sum(1 for r in results if r.duplicate_of is not None)
        confs = [r.confidence for r in results]
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0

        total_all_pages += total_pages
        total_tables += table_count
        total_hw += hw_count
        total_hindi += hindi_count
        total_low_conf += low_conf_count
        total_dups += dup_count
        all_confs.extend(confs)

        summary_lines.append(
            f"| `{patient_id}` | {label} | {total_pages} | {table_count} | {hw_count} | {hindi_count} | {low_conf_count} | {dup_count} | {avg_conf:.1%} |"
        )

    overall_avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0
    summary_lines.append(
        f"| **TOTAL** | **All Patients** | **{total_all_pages}** | **{total_tables}** | **{total_hw}** | **{total_hindi}** | **{total_low_conf}** | **{total_dups}** | **{overall_avg_conf:.1%}** |"
    )

    return "\n".join(summary_lines)


def run_pipeline() -> None:
    """
    Execute full pipeline: Ingest -> OCR -> Summary Generation.
    """
    print("=" * 60)
    print("STEP 1: Ingesting PDFs to Page Images...")
    print("=" * 60)
    ingest_all()

    print("\n" + "=" * 60)
    print("STEP 2: Running Sequential OCR across Patients...")
    print("=" * 60)
    all_results: Dict[str, List[PageOCRResult]] = {}
    for patient_id in all_patient_ids():
        print(f"\n---> Starting OCR for: {patient_id} ({get_display_label(patient_id)})")
        results = ocr_all_pages(patient_id=patient_id)
        all_results[patient_id] = results

    print("\n" + "=" * 60)
    print("STEP 3: Generating OCR Summary Report...")
    print("=" * 60)
    summary_md = generate_ocr_summary(patient_results=all_results)
    
    # Save summary report to data/ocr/_summary.md
    summary_path = Path("data/ocr/_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\n" + summary_md + "\n")
    print(f"[SUCCESS] Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    run_pipeline()
