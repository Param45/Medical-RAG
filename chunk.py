"""
Medical Records RAG Demo — Chunking & Evidence Creation (SRS §5.5)

Chunks report pages into evidence-producing units. Each chunk maps to
exactly one EvidenceRecord and carries normalized clinical entities.
Maps to BUILD_GUIDE Task 1.7.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evidence_store import EvidenceRecord, save_evidence_list
from normalize import NormalizedEntity, normalize_chunk_text
from patients import all_patient_ids, get_display_label
from split_reports import ReportSpan


@dataclass
class Chunk:
    """
    In-memory chunk containing its EvidenceRecord and extracted NormalizedEntity list (SRS §5.5).
    """
    evidence_record: EvidenceRecord
    normalized_entities: List[NormalizedEntity]
    chunk_type: str = "page"  # "page" | "row" | "table_summary"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "evidence_record": self.evidence_record.to_dict(),
            "normalized_entities": [e.to_dict() for e in self.normalized_entities],
            "chunk_type": self.chunk_type,
        }
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            evidence_record=EvidenceRecord.from_dict(d.get("evidence_record", {})),
            normalized_entities=[
                NormalizedEntity.from_dict(e) for e in d.get("normalized_entities", [])
            ],
            chunk_type=str(d.get("chunk_type", "page")),
            metadata=d.get("metadata"),
        )


def source_type_for_page(ocr_result: Dict[str, Any]) -> str:
    """
    Map OCR metadata flags (is_table, is_handwritten) to source_type (SRS FR-5.5.4).
    
    Returns:
    - "tabular_handwritten" if table and handwritten
    - "cursive_handwritten" if handwritten
    - "typed" otherwise (default)
    """
    is_table = bool(ocr_result.get("is_table", False))
    is_handwritten = bool(ocr_result.get("is_handwritten", False))

    if is_table and is_handwritten:
        return "tabular_handwritten"
    elif is_handwritten:
        return "cursive_handwritten"
    else:
        return "typed"


def chunk_report(
    patient_id: str,
    report_span: Union[ReportSpan, Dict[str, Any]],
    ocr_dir: Optional[Path] = None,
    pages_dir: Optional[Path] = None,
    dict_dir: Optional[Path] = None,
) -> List[Chunk]:
    """
    Chunk a single report span into one or more Chunk / EvidenceRecord units (SRS FR-5.5.1).
    
    - For standard/prose reports: 1 chunk per non-duplicate page.
    - For ONCOLOGY_FLOWSHEET reports: if structured rows exist in OCR output, emits 1 chunk per
      dated row plus 1 chunk for the whole table summary; otherwise emits 1 chunk per page.
    """
    if isinstance(report_span, dict):
        span = ReportSpan.from_dict(report_span)
    else:
        span = report_span

    base_dir = Path(__file__).parent
    if ocr_dir is None:
        patient_ocr_dir = base_dir / "data" / "ocr" / patient_id
    else:
        patient_ocr_dir = Path(ocr_dir) / patient_id

    chunks: List[Chunk] = []

    for page_num in range(span.page_start, span.page_end + 1):
        ocr_file = patient_ocr_dir / f"page_{page_num}.json"
        if not ocr_file.exists():
            continue

        try:
            page_data = json.loads(ocr_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to read {ocr_file}: {e}")
            continue

        # Skip near-duplicate pages per SRS FR-5.2.5
        if page_data.get("duplicate_of") is not None:
            continue

        raw_text = str(page_data.get("raw_text", ""))
        confidence = float(page_data.get("confidence", 1.0))
        script = str(page_data.get("script", "latin"))
        source_type = source_type_for_page(page_data)
        page_img_rel = f"data/pages/{patient_id}/page_{page_num}.png"

        # Check if this page is a flowsheet table with structured rows
        is_flowsheet = span.report_type == "ONCOLOGY_FLOWSHEET"
        table_rows = page_data.get("table_rows", [])

        if is_flowsheet and table_rows and isinstance(table_rows, list):
            # 1. Emit one chunk per row (per date/record)
            for row_idx, row in enumerate(table_rows):
                row_text = str(row) if not isinstance(row, dict) else " | ".join(f"{k}: {v}" for k, v in row.items())
                row_evidence_id = f"{patient_id}__{span.report_id}__page_{page_num}__chunk_{row_idx + 1}"
                row_entities = normalize_chunk_text(
                    row_text, script=script, base_confidence=confidence, dict_dir=dict_dir
                )
                row_record = EvidenceRecord(
                    evidence_id=row_evidence_id,
                    patient_id=patient_id,
                    report_id=span.report_id,
                    report_type=span.report_type,
                    report_date=span.report_date,
                    page_number=page_num,
                    raw_text=row_text,
                    source_type=source_type,
                    confidence=confidence,
                    page_image_path=page_img_rel,
                )
                chunks.append(
                    Chunk(
                        evidence_record=row_record,
                        normalized_entities=row_entities,
                        chunk_type="row",
                        metadata={"row_index": row_idx},
                    )
                )

            # 2. Plus one chunk for the whole table summary
            summary_evidence_id = f"{patient_id}__{span.report_id}__page_{page_num}__chunk_0"
            summary_entities = normalize_chunk_text(
                raw_text, script=script, base_confidence=confidence, dict_dir=dict_dir
            )
            summary_record = EvidenceRecord(
                evidence_id=summary_evidence_id,
                patient_id=patient_id,
                report_id=span.report_id,
                report_type=span.report_type,
                report_date=span.report_date,
                page_number=page_num,
                raw_text=raw_text,
                source_type=source_type,
                confidence=confidence,
                page_image_path=page_img_rel,
            )
            chunks.append(
                Chunk(
                    evidence_record=summary_record,
                    normalized_entities=summary_entities,
                    chunk_type="table_summary",
                )
            )
        else:
            # Default / Prose report case: exactly 1 chunk per page
            evidence_id = f"{patient_id}__{span.report_id}__page_{page_num}__chunk_0"
            entities = normalize_chunk_text(
                raw_text, script=script, base_confidence=confidence, dict_dir=dict_dir
            )
            record = EvidenceRecord(
                evidence_id=evidence_id,
                patient_id=patient_id,
                report_id=span.report_id,
                report_type=span.report_type,
                report_date=span.report_date,
                page_number=page_num,
                raw_text=raw_text,
                source_type=source_type,
                confidence=confidence,
                page_image_path=page_img_rel,
            )
            chunks.append(
                Chunk(
                    evidence_record=record,
                    normalized_entities=entities,
                    chunk_type="page",
                )
            )

    return chunks


def chunk_patient(
    patient_id: str,
    reports_dir: Optional[Path] = None,
    ocr_dir: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    pages_dir: Optional[Path] = None,
    dict_dir: Optional[Path] = None,
) -> List[Chunk]:
    """
    Process all report spans for a patient, generating EvidenceRecords and Chunk records.
    Persists:
    - data/evidence/{patient_id}.json (flat list of EvidenceRecord dicts)
    - data/reports/{patient_id}_chunks.json (list of Chunk dicts with normalized entities)
    """
    base_dir = Path(__file__).parent
    if reports_dir is None:
        reports_dir = base_dir / "data" / "reports"
    else:
        reports_dir = Path(reports_dir)

    reports_file = reports_dir / f"{patient_id}.json"
    if not reports_file.exists():
        print(f"Warning: Reports file not found: {reports_file}")
        return []

    try:
        report_spans_data = json.loads(reports_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading reports file {reports_file}: {e}")
        return []

    all_chunks: List[Chunk] = []
    for span_dict in report_spans_data:
        report_chunks = chunk_report(
            patient_id=patient_id,
            report_span=span_dict,
            ocr_dir=ocr_dir,
            pages_dir=pages_dir,
            dict_dir=dict_dir,
        )
        all_chunks.extend(report_chunks)

    # 1. Save evidence records to data/evidence/{patient_id}.json
    evidence_records = [c.evidence_record for c in all_chunks]
    save_evidence_list(patient_id, evidence_records, evidence_dir=evidence_dir)

    # 2. Save full chunk records to data/reports/{patient_id}_chunks.json
    chunks_file = reports_dir / f"{patient_id}_chunks.json"
    chunks_data = [c.to_dict() for c in all_chunks]
    chunks_file.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return all_chunks


def chunk_all_patients() -> Dict[str, List[Chunk]]:
    """
    Run chunking across all registered patients and print summary table.
    """
    results: Dict[str, List[Chunk]] = {}

    print("\n" + "=" * 80)
    print("Medical Records RAG Demo — Chunking & Evidence Pipeline (Task 1.7)")
    print("=" * 80)

    for pid in all_patient_ids():
        label = get_display_label(pid)
        print(f"\nProcessing {pid} ({label})...")
        chunks = chunk_patient(pid)
        results[pid] = chunks

        total_chunks = len(chunks)
        total_entities = sum(len(c.normalized_entities) for c in chunks)
        print(f"Produced {total_chunks} chunks ({total_entities} normalized entities total).")
        print(f"Saved: data/evidence/{pid}.json")
        print(f"Saved: data/reports/{pid}_chunks.json")

    print("\n" + "=" * 80)
    print("Chunking & Evidence Summary")
    print("=" * 80)
    print(f"{'Patient ID':<15} | {'Display Label':<16} | {'Total Chunks':<14} | {'Normalized Entities':<20}")
    print("-" * 80)
    for pid, chunks in results.items():
        label = get_display_label(pid)
        total_chunks = len(chunks)
        total_entities = sum(len(c.normalized_entities) for c in chunks)
        print(f"{pid:<15} | {label:<16} | {total_chunks:<14} | {total_entities:<20}")
    print("-" * 80)

    return results


if __name__ == "__main__":
    chunk_all_patients()

