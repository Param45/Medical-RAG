"""
Medical Records RAG — Temporary User Report & Session Sandbox Manager

Manages ephemeral single-report uploads, end-to-end ingestion into the temporary
isolated Neo4j database (af2857f2), and complete cleanup upon session termination.
"""

import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from chunk import chunk_patient
from evidence_store import load_evidence
from graph_backend.build import (
    TEMP_NEO4J_CONFIG,
    build_graph_for_patient,
    get_driver,
    run_schema_init,
)
from ingest import render_pdf_to_pages
from ocr import ocr_all_pages
from pageindex_backend.build import build_pageindex_for_patient
from patients import get_display_label, register_patient, unregister_patient
from split_reports import split_patient_reports

TEMP_PATIENT_ID = "temp_user_report"


def get_temp_patient_paths(patient_id: str = TEMP_PATIENT_ID) -> Dict[str, Path]:
    """Returns all disk paths associated with a given temporary patient."""
    return {
        "raw_pdf": _ROOT_DIR / "data" / "raw" / f"{patient_id}.pdf",
        "pages_dir": _ROOT_DIR / "data" / "pages" / patient_id,
        "ocr_dir": _ROOT_DIR / "data" / "ocr" / patient_id,
        "reports_json": _ROOT_DIR / "data" / "reports" / f"{patient_id}.json",
        "chunks_json": _ROOT_DIR / "data" / "reports" / f"{patient_id}_chunks.json",
        "evidence_json": _ROOT_DIR / "data" / "evidence" / f"{patient_id}.json",
        "pageindex_json": _ROOT_DIR / "data" / "pageindex" / f"{patient_id}.json",
    }


def wipe_temp_neo4j_database() -> int:
    """
    Clears all nodes and relationships from the temporary isolated Neo4j database (af2857f2).
    Returns count of deleted nodes/relationships or 0.
    """
    try:
        driver = get_driver(
            uri=TEMP_NEO4J_CONFIG["uri"],
            username=TEMP_NEO4J_CONFIG["username"],
            password=TEMP_NEO4J_CONFIG["password"],
        )
        with driver.session(database=TEMP_NEO4J_CONFIG.get("database")) as session:
            res = session.run("MATCH (n) DETACH DELETE n")
            summary = res.consume()
            nodes_deleted = summary.counters.nodes_deleted
        driver.close()
        return nodes_deleted
    except Exception as exc:
        print(f"[TempSession] Warning: Failed to wipe temporary Neo4j database: {exc}")
        return 0


def cleanup_temp_patient(patient_id: str = TEMP_PATIENT_ID) -> Dict[str, Any]:
    """
    Deletes all temporary report files, OCR caches, chunk stores, PageIndex trees,
    clears the temporary Neo4j database, and unregisters the patient.
    """
    paths = get_temp_patient_paths(patient_id)
    deleted_files = []

    # 1. Remove raw PDF
    if paths["raw_pdf"].exists():
        try:
            paths["raw_pdf"].unlink()
            deleted_files.append(str(paths["raw_pdf"]))
        except Exception as e:
            print(f"[TempSession] Error deleting {paths['raw_pdf']}: {e}")

    # 2. Remove rendered pages directory
    if paths["pages_dir"].exists():
        try:
            shutil.rmtree(paths["pages_dir"], ignore_errors=True)
            deleted_files.append(str(paths["pages_dir"]))
        except Exception as e:
            print(f"[TempSession] Error deleting {paths['pages_dir']}: {e}")

    # 3. Remove OCR directory
    if paths["ocr_dir"].exists():
        try:
            shutil.rmtree(paths["ocr_dir"], ignore_errors=True)
            deleted_files.append(str(paths["ocr_dir"]))
        except Exception as e:
            print(f"[TempSession] Error deleting {paths['ocr_dir']}: {e}")

    # 4. Remove single JSON files
    for k in ["reports_json", "chunks_json", "evidence_json", "pageindex_json"]:
        p = paths[k]
        if p.exists():
            try:
                p.unlink()
                deleted_files.append(str(p))
            except Exception as e:
                print(f"[TempSession] Error deleting {p}: {e}")

    # 5. Clear temporary Neo4j database
    nodes_deleted = wipe_temp_neo4j_database()

    # 6. Unregister patient
    unregister_patient(patient_id)

    return {
        "patient_id": patient_id,
        "deleted_files_count": len(deleted_files),
        "neo4j_nodes_deleted": nodes_deleted,
    }


def purge_all_temporary_data() -> None:
    """
    Scans workspace data folders for any lingering temporary session files and purges them.
    Registered as an atexit hook.
    """
    data_dir = _ROOT_DIR / "data"
    if not data_dir.exists():
        return

    # Scan and clean data/raw
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for f in raw_dir.glob("temp_*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass

    # Scan and clean data/pages, data/ocr
    for sub in ["pages", "ocr"]:
        sdir = data_dir / sub
        if sdir.exists():
            for d in sdir.iterdir():
                if d.is_dir() and d.name.startswith("temp_"):
                    shutil.rmtree(d, ignore_errors=True)

    # Scan and clean json stores
    for sub in ["reports", "evidence", "pageindex"]:
        sdir = data_dir / sub
        if sdir.exists():
            for f in sdir.glob("temp_*.*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    wipe_temp_neo4j_database()


# Register cleanup at Python process exit
atexit.register(purge_all_temporary_data)


def ingest_user_report(
    pdf_bytes: bytes,
    original_filename: str = "report.pdf",
    patient_id: str = TEMP_PATIENT_ID,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs the full end-to-end ingestion pipeline for a single user-uploaded PDF:
    1. Resets any previous session data for this patient ID.
    2. Writes PDF to data/raw/{patient_id}.pdf.
    3. Ingests PDF into high-res PNG pages (ingest.render_pdf_to_pages).
    4. Runs MinerU OCR extraction (ocr.ocr_all_pages).
    5. Splits report sections by clinical boundaries (split_reports.split_patient_reports).
    6. Chunks & normalizes clinical entities into evidence store (chunk.chunk_patient).
    7. Connects to isolated temporary Neo4j instance (af2857f2), initializes schema,
       and builds knowledge graph (graph_backend.build).
    8. Builds hierarchical PageIndex document tree (pageindex_backend.build).
    9. Registers patient in registry.

    Returns execution summary metrics dictionary.
    """
    def log(msg: str):
        print(f"[TempReport Ingestion] {msg}")
        if progress_callback:
            progress_callback(msg)

    start_time = time.time()

    # Step 1: Pre-cleanup
    log("Step 1/6: Initializing isolated temporary workspace & cleaning previous session records...")
    cleanup_temp_patient(patient_id)

    # Step 2: Save uploaded PDF to data/raw/
    paths = get_temp_patient_paths(patient_id)
    paths["raw_pdf"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw_pdf"].write_bytes(pdf_bytes)

    # Step 3: Render PDF pages to PNG
    log("Step 2/6: Converting medical report PDF pages to high-resolution scans...")
    page_paths = render_pdf_to_pages(
        pdf_path=paths["raw_pdf"],
        patient_id=patient_id,
        output_dir=_ROOT_DIR / "data" / "pages",
        dpi=250,
    )
    pages_count = len(page_paths)
    log(f"Rendered {pages_count} page{'s' if pages_count != 1 else ''}.")

    # Step 4: OCR extraction
    log("Step 3/6: Performing bilingual OCR, layout parsing, and deduplication...")
    ocr_results = ocr_all_pages(
        patient_id=patient_id,
        pages_dir=_ROOT_DIR / "data" / "pages",
        ocr_dir=_ROOT_DIR / "data" / "ocr",
    )
    log(f"Extracted OCR text for {len(ocr_results)} pages.")

    # Step 5: Report boundary splitting
    log("Step 4/6: Detecting clinical document boundaries, headers, and dates...")
    reports = split_patient_reports(patient_id=patient_id)
    paths["reports_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["reports_json"].write_text(
        json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    reports_count = len(reports)
    log(f"Identified {reports_count} report section{'s' if reports_count != 1 else ''}.")

    # Step 6: Chunking, normalization & evidence store
    log("Step 5/6: Generating chunks, normalizing clinical terminology & building evidence store...")
    chunks = chunk_patient(
        patient_id=patient_id,
        ocr_dir=_ROOT_DIR / "data" / "ocr",
        reports_dir=_ROOT_DIR / "data" / "reports",
        evidence_dir=_ROOT_DIR / "data" / "evidence",
        pages_dir=_ROOT_DIR / "data" / "pages",
    )
    chunks_count = len(chunks)
    evidence_count = len(load_evidence(patient_id))
    log(f"Generated {chunks_count} chunks and {evidence_count} evidence records.")

    # Step 7: Temporary Neo4j GraphRAG build
    log(f"Step 6/6: Initializing isolated Neo4j database ({TEMP_NEO4J_CONFIG['username']}) & constructing knowledge graph...")
    driver = get_driver(
        uri=TEMP_NEO4J_CONFIG["uri"],
        username=TEMP_NEO4J_CONFIG["username"],
        password=TEMP_NEO4J_CONFIG["password"],
    )
    try:
        run_schema_init(driver)
        graph_stats = build_graph_for_patient(patient_id=patient_id, driver=driver)
    finally:
        driver.close()

    triples_written = graph_stats.get("triples_written", 0)
    log(f"Constructed knowledge graph with {triples_written} triples in temporary Neo4j database.")

    # Step 8: PageIndex tree build
    log("Finalizing: Building hierarchical PageIndex tree with LLM clinical summaries...")
    pi_tree = build_pageindex_for_patient(patient_id=patient_id)

    # Step 9: Register in patients dictionary
    clean_name = Path(original_filename).stem.replace("_", " ").title()
    display_label = f"My Report ({clean_name})"
    register_patient(patient_id=patient_id, label=display_label, is_temp=True)

    elapsed = round(time.time() - start_time, 2)
    log(f"Report ingestion successfully completed in {elapsed}s.")

    return {
        "patient_id": patient_id,
        "display_label": display_label,
        "elapsed_seconds": elapsed,
        "pages_count": pages_count,
        "reports_count": reports_count,
        "chunks_count": chunks_count,
        "evidence_count": evidence_count,
        "triples_written": triples_written,
        "pageindex_ready": pi_tree is not None,
    }
