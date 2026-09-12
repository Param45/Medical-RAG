"""
Medical Records RAG Demo — Report Boundary Splitting (SRS §5.3)

Splits OCR'd page sequences into individual report sections using
text-anchor heuristics tuned to the known sample PDFs.
Maps to BUILD_GUIDE Task 1.5.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from patients import all_patient_ids, get_display_label


@dataclass
class ReportSpan:
    """
    Metadata for a detected report section (SRS §5.3.2).
    """
    report_id: str
    report_type: str
    page_start: int
    page_end: int
    report_date: Optional[str] = None  # ISO format YYYY-MM-DD or None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReportSpan":
        return cls(
            report_id=str(d.get("report_id", "")),
            report_type=str(d.get("report_type", "")),
            page_start=int(d.get("page_start", 0)),
            page_end=int(d.get("page_end", 0)),
            report_date=d.get("report_date"),
        )


# Month name mapping for date extraction
MONTH_ABBREVIATIONS: Dict[str, str] = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def extract_report_date(text: str) -> Optional[str]:
    """
    Extract a report date from the first ~500 characters of a report's opening page.
    
    Supports: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY, DD-MON-YYYY, DD/MON/YYYY, DD MON YYYY,
              and 2-digit year variants (DD-MM-YY, DD/MM/YY, DD.MM.YY, DD-MON-YY).
              
    Explicit assumption per SRS §5.3.4 & Task 1.5:
    For 2-digit years, years 00–79 are assumed 20xx (2000–2079), and years 80–99
    are assumed 19xx (1980–1999).
    
    Returns ISO date string 'YYYY-MM-DD' or None if no confident date is matched.
    """
    if not text:
        return None

    prefix = text[:500]

    # Regex patterns for dates
    patterns = [
        # Numeric: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b",
        # Alphanumeric month: DD-MON-YYYY, DD/MON/YYYY, DD MON YYYY
        r"\b(\d{1,2})[-/\s.]([A-Za-z]{3})[-/\s.](\d{4})\b",
        # Numeric 2-digit year: DD-MM-YY, DD/MM/YY, DD.MM.YY
        r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})\b",
        # Alphanumeric month 2-digit year: DD-MON-YY, DD/MON/YY, DD MON YY
        r"\b(\d{1,2})[-/\s.]([A-Za-z]{3})[-/\s.](\d{2})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, prefix):
            g1, g2, g3 = match.group(1), match.group(2), match.group(3)
            try:
                day = int(g1)
                g2_upper = g2.upper()
                if g2_upper in MONTH_ABBREVIATIONS:
                    month = int(MONTH_ABBREVIATIONS[g2_upper])
                else:
                    month = int(g2)
                
                year = int(g3)
                # 2-digit year normalization (Assumption: 00-79 -> 20xx, 80-99 -> 19xx)
                if len(g3) == 2 or year < 100:
                    year = 2000 + year if year <= 79 else 1900 + year

                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2099:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except (ValueError, KeyError):
                continue

    return None


# Anchor rules: (report_type, matcher_func(header_text_upper, full_text_upper))
# Header is the first ~500 characters where document letterheads/titles appear.
REPORT_ANCHORS: List[Tuple[str, Callable[[str, str], bool]]] = [
    # 1. Discharge Summary
    (
        "DISCHARGE_SUMMARY",
        lambda h, f: ("DISCHARGE SUMMARY" in h or "DISCHARGE STANDARD TITLE" in h or ("DISCHARGE" in h and "SUMMARY" in h)),
    ),
    # 2. Anesthesia Record
    (
        "ANESTHESIA_RECORD",
        lambda h, f: any(k in h for k in ["ANAESTHESIA RECORD", "ANESTHESIA RECORD", "DEPARTMENT OF ANAESTHESIOLOGY", "PRE ANAESTHETIC", "PAC RECORD", "PAC FORM"]),
    ),
    # 3. Surgical Operative Note
    (
        "SURGICAL_OPERATIVE_NOTE",
        lambda h, f: ("OPERATIVE PROCEDURE" in h or "OPERATIVE NOTE" in h or ("SURGICAL ONCOLOGY" in h and any(k in h for k in ["OPERATIVE", "PROCEDURE", "TREATMENT PLAN"]))),
    ),
    # 4. Echocardiography
    (
        "ECHOCARDIOGRAPHY",
        lambda h, f: "ECHOCARDIOGRAPHY" in h,
    ),
    # 5. PET-CT (Department of Nuclear Medicine / PET-CT study / Positron Emission)
    (
        "RADIOLOGY_PET_CT",
        lambda h, f: any(k in h for k in ["PET-CT", "PET CT", "POSITRON EMISSION", "NUCLEAR MEDICINE AND PET", "OLECONY EMISSION", "WHOLE BODY EN-FANGEH"]),
    ),
    # 6. Bone Scan
    (
        "RADIOLOGY_BONE_SCAN",
        lambda h, f: any(k in h for k in ["MDP WHOLE BODY", "BONE SCAN REPORT", "MDP BONE SCAN", "WHOLE BODY BONE SCAN"]),
    ),
    # 7. Mammography
    (
        "RADIOLOGY_MAMMOGRAM",
        lambda h, f: any(k in h for k in ["MAMMOGRAPHY", "MAMMOGRAM", "MAMMTEPDY"]) and any(k in h for k in ["RADIOLOGY", "DEPARTMENT", "BILATERAL"]),
    ),
    # 8. CECT (Computed Tomography / Contrast enhanced)
    (
        "RADIOLOGY_CECT",
        lambda h, f: (
            ("RADIOLOGY" in h or "SETH DIAGNOSTICS" in h)
            and any(k in h for k in ["CECT", "CONTRAST", "COMPUTED TOMOGRAPHY", "WHOLE BODY CT", "CT SCAN"])
        ),
    ),
    # 9. USG (Ultrasound)
    (
        "RADIOLOGY_USG",
        lambda h, f: (
            ("RADIOLOGY" in h or "ULTRASOUND" in h or "USG" in h)
            and any(k in h for k in ["ULTRASOUND REPORT", "USG REPORT", "SONOGRAPHY"])
            and not any(k in h for k in ["PET", "HISTO", "CYTO", "MAMMOG"])
        ),
    ),
    # 10. Histopathology (Department of Pathology / Histopathology Report)
    (
        "HISTOPATHOLOGY",
        lambda h, f: (
            any(k in h for k in ["HISTOPATHOLOGY REPORT", "HISTOLOGY REPORT", "HISTO REPORT", "HISTO_REPORT", "HPE REPORT"])
            or ("DEPARTMENT OF PATHOLOGY" in h and "HISTO" in f)
        ) and "REPORT AWAITED" not in h,
    ),
    # 11. Cytopathology
    (
        "CYTOPATHOLOGY",
        lambda h, f: any(k in h for k in ["CYTOPATHOLOGY REPORT", "CYTOLOGY REPORT", "CYTO REPORT", "CYTOLOGY CARD PRINT", "CYTOLOGY ENT OF ALL INDIA"]),
    ),
    # 12. Chemo Drug Administration Record
    (
        "CHEMO_DRUG_ADMIN_RECORD",
        lambda h, f: any(k in h for k in ["DAYCARE DRUGS ADMINISTERED", "DRUGS ADMINISTERED"]),
    ),
    # 13. Consent Form (Bilingual English / Hindi)
    (
        "CONSENT_FORM",
        lambda h, f: ("सहमति पत्र" in h or "सहमति" in h or "CONSENT FORM" in h or "CONSENT" in h),
    ),
    # 14. Oncology Flowsheet
    (
        "ONCOLOGY_FLOWSHEET",
        lambda h, f: (
            any(k in h for k in ["FLOWSHEET", "FLOW SHEET", "ADULT MEDICAL ONCOLOGY", "CLINIC ADULT MEDICAL ONCOLOGY"])
            or ("MEDICAL" in h and "ONCOLOGY" in h)
            or ("ROTARY CANCER HOSPITAL" in h and ("CLINIC" in h or "ONCOLOGY" in h))
        ),
    ),
    # 15. General Lab Report (Hematology, Biochemistry, Diagnostics, etc.)
    (
        "LAB_REPORT_GENERAL",
        lambda h, f: (
            any(k in h for k in ["HAEMATOLOGY", "HEMATOLOGY", "BIOCHEMISTRY", "SETH DIAGNOSTICS", "SRL IC REPORT", "DEPARTMENT OF PATHOLOGY", "MODULES/LABORATORY"])
            and not any(k in h for k in ["HISTOPATHOLOGY", "CYTOPATHOLOGY", "HISTOLOGY", "CYTOLOGY"])
        ),
    ),
]


def match_report_anchor(text: str) -> Optional[str]:
    """
    Match OCR text against defined report anchors (case-insensitive).
    Normalizes whitespace and checks header section (first ~500 chars) as well as full text.
    Returns report_type string or None.
    """
    if not text:
        return None
    # Normalize whitespace to ensure multi-word anchors match across line breaks
    header_upper = " ".join(text[:500].split()).upper()
    full_upper = " ".join(text.split()).upper()
    for report_type, matcher in REPORT_ANCHORS:
        if matcher(header_upper, full_upper):
            return report_type
    return None


def split_patient_reports(patient_id: str, ocr_dir: Optional[Path] = None) -> List[ReportSpan]:
    """
    Split a patient's OCR pages into ReportSpan sections (SRS §5.3.2).
    
    1. Reads all OCR JSONs under data/ocr/{patient_id}/ in page order.
    2. Skips near-duplicate pages marked with 'duplicate_of'.
    3. Identifies report boundaries using text anchors; unmatched pages extend the current report.
    4. Extracts report dates from opening pages.
    """
    if ocr_dir is None:
        ocr_dir = Path(__file__).parent / "data" / "ocr" / patient_id
    else:
        ocr_dir = Path(ocr_dir) / patient_id

    if not ocr_dir.exists():
        print(f"Warning: OCR directory not found: {ocr_dir}")
        return []

    # Load all page JSONs sorted by page number
    page_files = sorted(
        ocr_dir.glob("page_*.json"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else 0
    )

    reports: List[ReportSpan] = []
    current_report: Optional[ReportSpan] = None

    for page_file in page_files:
        try:
            data = json.loads(page_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading {page_file}: {e}")
            continue

        # Skip near-duplicate pages
        if data.get("duplicate_of") is not None:
            continue

        page_num = int(data.get("page_number", 0))
        raw_text = str(data.get("raw_text", ""))

        matched_type = match_report_anchor(raw_text)
        page_date = extract_report_date(raw_text)

        if matched_type is not None:
            # Anchor matched -> close previous report (if any) and start a new one
            if current_report is not None:
                reports.append(current_report)
            
            current_report = ReportSpan(
                report_id=f"report_{page_num:04d}",
                report_type=matched_type,
                page_start=page_num,
                page_end=page_num,
                report_date=page_date,
            )
        else:
            # No anchor matched -> extend current report or start fallback
            if current_report is not None:
                current_report.page_end = page_num
                # If current report has no date yet, attempt extraction from continuation page
                if current_report.report_date is None and page_date is not None:
                    current_report.report_date = page_date
            else:
                current_report = ReportSpan(
                    report_id=f"report_{page_num:04d}",
                    report_type="CLINICIAN_PROGRESS_NOTE",
                    page_start=page_num,
                    page_end=page_num,
                    report_date=page_date,
                )

    if current_report is not None:
        reports.append(current_report)

    return reports


def save_patient_reports(patient_id: str, reports: List[ReportSpan], output_dir: Optional[Path] = None) -> Path:
    """
    Persist detected reports to data/reports/{patient_id}.json.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "reports"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{patient_id}.json"

    data = [r.to_dict() for r in reports]
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def split_all_reports() -> Dict[str, List[ReportSpan]]:
    """
    Process all registered patients, detect report boundaries, and save reports JSON.
    Prints a summary table to the console.
    """
    all_reports: Dict[str, List[ReportSpan]] = {}

    print("\n" + "=" * 80)
    print("Medical Records RAG Demo — Report Splitting Pipeline (Task 1.5)")
    print("=" * 80)

    for pid in all_patient_ids():
        display_label = get_display_label(pid)
        print(f"\nProcessing {pid} ({display_label})...")
        reports = split_patient_reports(pid)
        out_path = save_patient_reports(pid, reports)
        all_reports[pid] = reports

        print(f"Saved {len(reports)} detected reports to {out_path}")
        print("-" * 80)
        print(f"{'Report ID':<15} | {'Report Type':<26} | {'Pages':<10} | {'Date':<12}")
        print("-" * 80)
        for r in reports:
            date_str = r.report_date or "N/A"
            pages_str = f"{r.page_start} - {r.page_end}" if r.page_start != r.page_end else str(r.page_start)
            print(f"{r.report_id:<15} | {r.report_type:<26} | {pages_str:<10} | {date_str:<12}")
        print("-" * 80)

    return all_reports


if __name__ == "__main__":
    split_all_reports()

