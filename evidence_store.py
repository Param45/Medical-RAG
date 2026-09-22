"""
Medical Records RAG Demo — Evidence Store (SRS §5.5)

Read/write helpers for local JSON evidence files under data/evidence/.
Each patient has one flat JSON list of EvidenceRecord objects.
Maps to BUILD_GUIDE Task 1.7.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class EvidenceRecord:
    """
    Evidence record schema (SRS §5.5.2).

    report_date: The date on the report letterhead / header.
    result_date: The date of the specific result within the report (e.g., per-column
                 date in flowsheet tables). Falls back to report_date if not available.
                 This distinction is critical for answering "sugar level at a specific
                 time of year" — flowsheet tables have per-column dates that differ
                 from the report letterhead date.
    """
    evidence_id: str
    patient_id: str
    report_id: str
    report_type: str
    report_date: Optional[str]
    page_number: int
    raw_text: str
    source_type: str  # "typed" | "tabular_handwritten" | "cursive_handwritten"
    confidence: float
    page_image_path: str
    result_date: Optional[str] = None  # Per-column / per-row date, distinct from report_date

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(d.get("evidence_id", "")),
            patient_id=str(d.get("patient_id", "")),
            report_id=str(d.get("report_id", "")),
            report_type=str(d.get("report_type", "")),
            report_date=d.get("report_date"),
            result_date=d.get("result_date"),
            page_number=int(d.get("page_number", 1)),
            raw_text=str(d.get("raw_text", "")),
            source_type=str(d.get("source_type", "typed")),
            confidence=float(d.get("confidence", 1.0)),
            page_image_path=str(d.get("page_image_path", "")),
        )



def get_default_evidence_dir() -> Path:
    """Get the default path to data/evidence/."""
    return Path(__file__).parent / "data" / "evidence"


def load_evidence(
    patient_id: str,
    evidence_dir: Optional[Path] = None
) -> List[EvidenceRecord]:
    """
    Load all EvidenceRecord objects for a given patient from data/evidence/{patient_id}.json.
    Returns empty list if file does not exist.
    """
    if evidence_dir is None:
        evidence_dir = get_default_evidence_dir()
    else:
        evidence_dir = Path(evidence_dir)

    file_path = evidence_dir / f"{patient_id}.json"
    if not file_path.exists():
        return []

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [EvidenceRecord.from_dict(item) for item in data]
    except Exception as e:
        print(f"Warning: Error reading evidence file {file_path}: {e}")
        return []

    return []


def save_evidence_list(
    patient_id: str,
    records: List[EvidenceRecord],
    evidence_dir: Optional[Path] = None
) -> Path:
    """
    Save list of EvidenceRecord objects to data/evidence/{patient_id}.json.
    Overwrites existing file.
    """
    if evidence_dir is None:
        evidence_dir = get_default_evidence_dir()
    else:
        evidence_dir = Path(evidence_dir)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    file_path = evidence_dir / f"{patient_id}.json"

    data = [rec.to_dict() for rec in records]
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def append_evidence(
    patient_id: str,
    record: EvidenceRecord,
    evidence_dir: Optional[Path] = None
) -> None:
    """
    Append an EvidenceRecord to data/evidence/{patient_id}.json (creating file if missing).
    If an evidence record with the same evidence_id already exists, it is updated.
    """
    records = load_evidence(patient_id, evidence_dir=evidence_dir)

    # Check if record already exists
    updated = False
    for i, existing in enumerate(records):
        if existing.evidence_id == record.evidence_id:
            records[i] = record
            updated = True
            break

    if not updated:
        records.append(record)

    save_evidence_list(patient_id, records, evidence_dir=evidence_dir)


def get_evidence_by_id(
    patient_id: str,
    evidence_id: str,
    evidence_dir: Optional[Path] = None
) -> Optional[EvidenceRecord]:
    """
    Retrieve an EvidenceRecord by its unique evidence_id from data/evidence/{patient_id}.json.
    """
    records = load_evidence(patient_id, evidence_dir=evidence_dir)
    for record in records:
        if record.evidence_id == evidence_id:
            return record
    return None

