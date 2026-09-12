"""
Medical Records RAG Demo — PDF Ingestion (SRS §5.1, FR-5.1.1–FR-5.1.4)

Reads PDFs from data/raw/, renders each page to a PNG image at 250 DPI under
data/pages/{patient_id}/page_{n}.png (1-indexed).
Sequential processing with progress logging.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
import shutil
import sys
from pdf2image import convert_from_path


def _find_poppler_path() -> Optional[str]:
    """
    Helper to locate poppler bin directory on Windows if not already on PATH.
    """
    if shutil.which("pdftoppm"):
        return None
    
    # Specific known Windows installation paths for poppler
    known_paths = [
        Path(r"C:\Program Files\poppler\bin"),
        Path(r"C:\Program Files (x86)\poppler\bin"),
        Path(r"C:\poppler\bin"),
        Path(r"C:\tools\poppler\bin"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "poppler" / "bin",
    ]
    for p in known_paths:
        if (p / "pdftoppm.exe").exists():
            return str(p)
    return None


def discover_patient_pdfs(raw_dir: str | Path = "data/raw") -> Dict[str, Path]:
    """
    Scans raw_dir for PDF files and returns a dictionary mapping patient_id to pdf Path.
    patient_id is derived from the filename stem (e.g. patient_a.pdf -> 'patient_a').
    Does not hardcode 'exactly two files'.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return {}

    pdf_map: Dict[str, Path] = {}
    for file_path in sorted(raw_path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            patient_id = file_path.stem.lower()
            pdf_map[patient_id] = file_path

    return pdf_map


def render_pdf_to_pages(
    pdf_path: Path,
    patient_id: str,
    output_dir: str | Path = "data/pages",
    dpi: int = 250,
    poppler_path: Optional[str] = None,
) -> List[Path]:
    """
    Renders each page of a PDF to a PNG image at the given DPI.
    Saves pages as data/pages/{patient_id}/page_{n}.png (1-indexed).
    Skips re-rendering if page PNG already exists.
    Returns the list of saved page PNG paths in order.
    """
    out_dir = Path(output_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if existing pages are already rendered
    existing_pages = sorted(
        out_dir.glob("page_*.png"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else 0
    )
    if existing_pages:
        print(f"[{patient_id}] {len(existing_pages)} pages already rendered (skipping convert_from_path)")
        return existing_pages

    if poppler_path is None:
        poppler_path = _find_poppler_path()

    # Convert PDF to list of PIL Images
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    total_pages = len(images)
    saved_paths: List[Path] = []

    for idx, img in enumerate(images, start=1):
        page_file = out_dir / f"page_{idx}.png"
        if not page_file.exists():
            img.save(str(page_file), "PNG")
        saved_paths.append(page_file)
        print(f"[{patient_id}] rendered page {idx}/{total_pages}")

    return saved_paths


def ingest_all(
    raw_dir: str | Path = "data/raw",
    output_base_dir: str | Path = "data/pages",
    dpi: int = 250,
) -> Dict[str, List[Path]]:
    """
    Discovers all patient PDFs in raw_dir and renders each PDF to PNG pages sequentially,
    one patient at a time, one page at a time.
    Returns mapping of patient_id to list of page image Paths.
    """
    pdf_map = discover_patient_pdfs(raw_dir=raw_dir)
    results: Dict[str, List[Path]] = {}

    for patient_id, pdf_path in pdf_map.items():
        print(f"Ingesting {patient_id} from {pdf_path}...")
        page_paths = render_pdf_to_pages(
            pdf_path=pdf_path,
            patient_id=patient_id,
            output_dir=output_base_dir,
            dpi=dpi,
        )
        results[patient_id] = page_paths

    return results


if __name__ == "__main__":
    ingest_all()
