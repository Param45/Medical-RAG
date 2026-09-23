"""
Medical Records RAG — Patient Registry (SRS §5.7)

Supports both the baseline demo patients (Patient A and Patient B) and dynamic
auto-discovery for any new patient records placed in data/raw/, data/evidence/,
or data/ocr/.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional

# Baseline demo patient registry
PATIENTS: Dict[str, str] = {
    "patient_a": "Patient A",
    "patient_b": "Patient B",
}


_TEMP_PATIENTS: set[str] = set()


def is_temp_patient(patient_id: str) -> bool:
    """Check if a patient_id corresponds to a temporary session patient."""
    if not patient_id:
        return False
    norm_id = patient_id.strip().lower()
    return norm_id.startswith("temp_") or norm_id.startswith("user_") or norm_id in _TEMP_PATIENTS


def format_patient_label(patient_id: str) -> str:
    """
    Format a patient_id like 'patient_c' or 'john_doe' into 'Patient C' or 'John Doe'.
    """
    if not patient_id:
        return "Unknown Patient"
    # Format 'patient_x' -> 'Patient X'
    match = re.match(r"^patient[_-]([a-zA-Z0-9]+)$", patient_id, re.IGNORECASE)
    if match:
        return f"Patient {match.group(1).upper()}"
    parts = patient_id.replace("_", " ").replace("-", " ").split()
    return " ".join(p.capitalize() for p in parts)


def register_patient(patient_id: str, label: Optional[str] = None, is_temp: bool = False) -> None:
    """
    Dynamically register a new patient ID and display label.
    """
    if not patient_id:
        return
    norm_id = patient_id.strip().lower()
    display = label or format_patient_label(norm_id)
    PATIENTS[norm_id] = display
    if is_temp or norm_id.startswith("temp_") or norm_id.startswith("user_"):
        _TEMP_PATIENTS.add(norm_id)


def unregister_patient(patient_id: str) -> None:
    """
    Remove a patient ID from the registry and temporary set.
    """
    if not patient_id:
        return
    norm_id = patient_id.strip().lower()
    PATIENTS.pop(norm_id, None)
    _TEMP_PATIENTS.discard(norm_id)


def discover_patients(base_dir: Optional[Path] = None) -> List[str]:
    """
    Auto-discover any patients present on disk in data/raw/, data/evidence/, or data/ocr/.
    Registers any newly discovered patients into the PATIENTS dictionary.
    Returns sorted list of all patient IDs.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "data"
    else:
        base_dir = Path(base_dir)

    discovered = set(PATIENTS.keys())

    # Check data/raw for PDFs
    raw_dir = base_dir / "raw"
    if raw_dir.exists():
        for pdf_file in raw_dir.glob("*.pdf"):
            pid = pdf_file.stem.strip().lower()
            if pid and not pid.startswith("_") and not pid.startswith("temp_") and not pid.startswith("user_"):
                discovered.add(pid)

    # Check data/evidence for JSON stores
    ev_dir = base_dir / "evidence"
    if ev_dir.exists():
        for ev_file in ev_dir.glob("*.json"):
            pid = ev_file.stem.strip().lower()
            if pid and not pid.startswith("_") and not pid.startswith("temp_") and not pid.startswith("user_"):
                discovered.add(pid)

    # Check data/ocr for directories
    ocr_dir = base_dir / "ocr"
    if ocr_dir.exists():
        for p_dir in ocr_dir.iterdir():
            if p_dir.is_dir() and not p_dir.name.startswith("_") and not p_dir.name.startswith("temp_") and not p_dir.name.startswith("user_"):
                discovered.add(p_dir.name.strip().lower())

    # Register any newly discovered patients
    for pid in sorted(discovered):
        if pid not in PATIENTS:
            register_patient(pid)

    return sorted(discovered)


def get_display_label(patient_id: str) -> str:
    """
    Get the human-readable display label for a given patient_id.
    If patient_id is not in the registry, falls back to returning the patient_id.
    """
    return PATIENTS.get(patient_id, str(patient_id))


def all_patient_ids() -> List[str]:
    """
    Return a list of all registered patient IDs.
    """
    return list(PATIENTS.keys())
