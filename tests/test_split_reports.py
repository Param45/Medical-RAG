"""
Unit tests for Report Boundary Splitting (Task 1.5, SRS §5.3).
"""

import json
from pathlib import Path
import tempfile
import pytest

from split_reports import (
    ReportSpan,
    extract_report_date,
    match_report_anchor,
    save_patient_reports,
    split_all_reports,
    split_patient_reports,
)


class TestExtractReportDate:
    """Tests for extract_report_date regex and normalization."""

    def test_extract_dd_mm_yyyy_hyphen(self):
        text = "Report Header\nDate of Examination: 15-08-2021\nPatient Name: John Doe"
        assert extract_report_date(text) == "2021-08-15"

    def test_extract_dd_mm_yyyy_slash(self):
        text = "AIIMS Hospital\nReg. Date: 08/12/2016 Clinic Adult Medical Oncology"
        assert extract_report_date(text) == "2016-12-08"

    def test_extract_dd_mm_yyyy_dot(self):
        text = "Nuclear Medicine Study\nDate: 31.01.2018\nPatient Study ID: 123"
        assert extract_report_date(text) == "2018-01-31"

    def test_extract_dd_mon_yyyy(self):
        text = "SETH DIAGNOSTICS\nDate : 02-Dec-2016 By Dr. K ARYA"
        assert extract_report_date(text) == "2016-12-02"

    def test_extract_dd_mon_yyyy_slash(self):
        text = "DISCHARGE SUMMARY\nDOS 28/May/2014 DOD 31/May/2014"
        assert extract_report_date(text) == "2014-05-28"

    def test_extract_two_digit_year_20xx(self):
        # 00-79 maps to 2000-2079
        text = "Lab Date: 03-Dec-16 09:34 AM"
        assert extract_report_date(text) == "2016-12-03"

    def test_extract_two_digit_year_19xx(self):
        # 80-99 maps to 1980-1999
        text = "Archive record dated: 14-05-95"
        assert extract_report_date(text) == "1995-05-14"

    def test_extract_no_date_returns_none(self):
        text = "Department of Radiology\nNo date mentioned anywhere in header"
        assert extract_report_date(text) is None

    def test_extract_empty_string(self):
        assert extract_report_date("") is None


class TestMatchReportAnchor:
    """Tests for report anchor matching rules."""

    def test_match_pet_ct(self):
        text = "Department of Nuclear Medicine and PET All India Institute\n18F-FDG WHOLE BODY PET-CT STUDY"
        assert match_report_anchor(text) == "RADIOLOGY_PET_CT"

    def test_match_histopathology(self):
        text = "Department Of Pathology All India Institute\nHISTOPATHOLOGY REPORT"
        assert match_report_anchor(text) == "HISTOPATHOLOGY"

    def test_match_cytopathology(self):
        text = "Cytology Card Print /cyto report print result.php"
        assert match_report_anchor(text) == "CYTOPATHOLOGY"

    def test_match_flowsheet(self):
        text = "Dr. B.R. Ambedkar Institute Cancer Hospital Flowsheet Medical Oncology"
        assert match_report_anchor(text) == "ONCOLOGY_FLOWSHEET"

    def test_match_discharge_summary(self):
        text = "MEDICAL RECORD Progress NOTE LOCAL TITLE: ONCO IRCH DISCHARGE STANDARD TITLE: DISCHARGE SUMMARY"
        assert match_report_anchor(text) == "DISCHARGE_SUMMARY"

    def test_match_surgical_note(self):
        text = "SURGICAL ONCOLOGY, IRCH, AIIMS\nOperative Procedure, Treatment Plan"
        assert match_report_anchor(text) == "SURGICAL_OPERATIVE_NOTE"

    def test_match_anesthesia_record(self):
        text = "DEPARTMENT OF ANAESTHESIOLOGY\nANAESTHESIA RECORD"
        assert match_report_anchor(text) == "ANESTHESIA_RECORD"

    def test_match_echocardiography(self):
        text = "ECHOCARDIOGRAPHY REPORT\nDEPARTMENT OF CARDIOLOGY"
        assert match_report_anchor(text) == "ECHOCARDIOGRAPHY"

    def test_match_consent_form_hindi(self):
        text = "डा, संस्थान रोटरी अभा.आ.सं. अन्सारी नगर, नई दिल्ली सहमति पत्र"
        assert match_report_anchor(text) == "CONSENT_FORM"

    def test_match_general_lab(self):
        text = "HAEMATOLOGY / BIOCHEMISTRY LAB REPORT\nComplete Blood Count"
        assert match_report_anchor(text) == "LAB_REPORT_GENERAL"

    def test_match_no_anchor(self):
        text = "Patient was feeling much better today. Advised regular checkup in 2 weeks."
        assert match_report_anchor(text) is None


class TestReportSpanSerialization:
    """Tests for ReportSpan dataclass serialization."""

    def test_to_dict_and_from_dict(self):
        span = ReportSpan(
            report_id="report_0003",
            report_type="RADIOLOGY_PET_CT",
            page_start=3,
            page_end=4,
            report_date="2017-12-28",
        )
        d = span.to_dict()
        assert d["report_id"] == "report_0003"
        assert d["report_type"] == "RADIOLOGY_PET_CT"
        assert d["page_start"] == 3
        assert d["page_end"] == 4
        assert d["report_date"] == "2017-12-28"

        restored = ReportSpan.from_dict(d)
        assert restored == span


class TestSplitPatientReportsMock:
    """Tests for split_patient_reports on synthetic mock data."""

    def test_split_synthetic_pages(self, tmp_path):
        ocr_dir = tmp_path / "mock_patient"
        ocr_dir.mkdir(parents=True)

        pages = [
            {"page_number": 1, "raw_text": "Department of Nuclear Medicine and PET-CT\nDate: 01-01-2020"},
            {"page_number": 2, "raw_text": "Continuation of scanning impression, findings unremarkable..."},
            {"page_number": 3, "raw_text": "HISTOPATHOLOGY REPORT\nDate: 05-01-2020\nSpecimen: Biopsy"},
            {"page_number": 4, "raw_text": "Duplicate page", "duplicate_of": 3},
            {"page_number": 5, "raw_text": "Flowsheet Medical Oncology\nDate: 10-01-2020"},
        ]

        for p in pages:
            file_path = ocr_dir / f"page_{p['page_number']}.json"
            file_path.write_text(json.dumps(p), encoding="utf-8")

        reports = split_patient_reports("mock_patient", ocr_dir=tmp_path)
        assert len(reports) == 3

        assert reports[0].report_id == "report_0001"
        assert reports[0].report_type == "RADIOLOGY_PET_CT"
        assert reports[0].page_start == 1
        assert reports[0].page_end == 2
        assert reports[0].report_date == "2020-01-01"

        assert reports[1].report_id == "report_0003"
        assert reports[1].report_type == "HISTOPATHOLOGY"
        assert reports[1].page_start == 3
        assert reports[1].page_end == 3
        assert reports[1].report_date == "2020-01-05"

        assert reports[2].report_id == "report_0005"
        assert reports[2].report_type == "ONCOLOGY_FLOWSHEET"
        assert reports[2].page_start == 5
        assert reports[2].page_end == 5
        assert reports[2].report_date == "2020-01-10"


class TestSplitPatientReportsRealCorpus:
    """Tests against real OCR output from Task 1.4 for patient_a and patient_b."""

    @pytest.mark.parametrize("patient_id", ["patient_a", "patient_b"])
    def test_split_real_patient_reports(self, patient_id):
        reports = split_patient_reports(patient_id)
        assert len(reports) >= 5, f"Expected at least 5 reports for {patient_id}, got {len(reports)}"

        # Check required report types per BUILD_GUIDE Task 1.5 requirement 6:
        # At least one HISTOPATHOLOGY, one RADIOLOGY_PET_CT, and one ONCOLOGY_FLOWSHEET
        report_types = {r.report_type for r in reports}
        assert "HISTOPATHOLOGY" in report_types or "CYTOPATHOLOGY" in report_types, f"Missing pathology in {patient_id}"
        assert "HISTOPATHOLOGY" in report_types, f"Missing HISTOPATHOLOGY in {patient_id}"
        assert "RADIOLOGY_PET_CT" in report_types, f"Missing RADIOLOGY_PET_CT in {patient_id}"
        assert "ONCOLOGY_FLOWSHEET" in report_types, f"Missing ONCOLOGY_FLOWSHEET in {patient_id}"

        # Verify contiguous and valid page numbers
        prev_end = 0
        for r in reports:
            assert r.page_start <= r.page_end
            assert r.page_start == prev_end + 1 or r.page_start > prev_end
            assert r.report_id == f"report_{r.page_start:04d}"
            prev_end = r.page_end

    def test_save_patient_reports(self, tmp_path):
        reports = split_patient_reports("patient_a")
        out_file = save_patient_reports("patient_a", reports, output_dir=tmp_path)
        assert out_file.exists()
        saved_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(saved_data) == len(reports)
        assert saved_data[0]["report_id"] == reports[0].report_id

    def test_split_all_reports_execution(self):
        all_reports = split_all_reports()
        assert "patient_a" in all_reports
        assert "patient_b" in all_reports
        assert len(all_reports["patient_a"]) > 0
        assert len(all_reports["patient_b"]) > 0
