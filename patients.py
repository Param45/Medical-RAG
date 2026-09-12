"""
Medical Records RAG Demo — Patient Registry (SRS §5.7)

Static, hardcoded patient registry for the demo. Adding a third demo patient
means adding one entry to the PATIENTS dict here plus placing a PDF in
data/raw/ — no other code changes required.
"""

from typing import Dict, List

PATIENTS: Dict[str, str] = {
    "patient_a": "Patient A",
    "patient_b": "Patient B",
}


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
