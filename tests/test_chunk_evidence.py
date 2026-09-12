"""
Unit tests for Chunking and Evidence Store (Task 1.7, SRS §5.5).
"""

import json
from pathlib import Path
import pytest

from chunk import (
    Chunk,
    chunk_all_patients,
    chunk_patient,
    chunk_report,
    source_type_for_page,
)
from evidence_store import (
    EvidenceRecord,
    append_evidence,
    get_evidence_by_id,
    load_evidence,
    save_evidence_list,
)
from split_reports import ReportSpan


class TestEvidenceStore:
    """Tests for evidence_store.py functions and EvidenceRecord data model."""

    def test_evidence_record_serialization(self):
        rec = EvidenceRecord(
            evidence_id="patient_a__report_0003__page_3__chunk_0",
            patient_id="patient_a",
            report_id="report_0003",
            report_type="RADIOLOGY_PET_CT",
            report_date="2017-12-28",
            page_number=3,
            raw_text="Sample PET-CT text content",
            source_type="typed",
            confidence=0.92,
            page_image_path="data/pages/patient_a/page_3.png",
        )
        d = rec.to_dict()
        assert d["evidence_id"] == "patient_a__report_0003__page_3__chunk_0"
        assert d["patient_id"] == "patient_a"
        assert d["confidence"] == 0.92

        restored = EvidenceRecord.from_dict(d)
        assert restored == rec

    def test_save_and_load_evidence_list(self, tmp_path):
        records = [
            EvidenceRecord(
                evidence_id=f"patient_test__report_0001__page_{i}__chunk_0",
                patient_id="patient_test",
                report_id="report_0001",
                report_type="HISTOPATHOLOGY",
                report_date="2020-01-01",
                page_number=i,
                raw_text=f"Page {i} text",
                source_type="typed",
                confidence=0.95,
                page_image_path=f"data/pages/patient_test/page_{i}.png",
            )
            for i in range(1, 4)
        ]
        out_file = save_evidence_list("patient_test", records, evidence_dir=tmp_path)
        assert out_file.exists()

        loaded = load_evidence("patient_test", evidence_dir=tmp_path)
        assert len(loaded) == 3
        assert loaded[0].evidence_id == records[0].evidence_id

    def test_append_evidence(self, tmp_path):
        rec1 = EvidenceRecord(
            evidence_id="patient_test__report_0001__page_1__chunk_0",
            patient_id="patient_test",
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            report_date="2020-01-01",
            page_number=1,
            raw_text="Page 1 text",
            source_type="typed",
            confidence=0.95,
            page_image_path="data/pages/patient_test/page_1.png",
        )
        append_evidence("patient_test", rec1, evidence_dir=tmp_path)
        assert len(load_evidence("patient_test", evidence_dir=tmp_path)) == 1

        # Append second record
        rec2 = EvidenceRecord(
            evidence_id="patient_test__report_0001__page_2__chunk_0",
            patient_id="patient_test",
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            report_date="2020-01-01",
            page_number=2,
            raw_text="Page 2 text",
            source_type="typed",
            confidence=0.90,
            page_image_path="data/pages/patient_test/page_2.png",
        )
        append_evidence("patient_test", rec2, evidence_dir=tmp_path)
        assert len(load_evidence("patient_test", evidence_dir=tmp_path)) == 2

        # Update existing record
        rec1_updated = EvidenceRecord(
            evidence_id="patient_test__report_0001__page_1__chunk_0",
            patient_id="patient_test",
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            report_date="2020-01-01",
            page_number=1,
            raw_text="Page 1 updated text",
            source_type="typed",
            confidence=0.98,
            page_image_path="data/pages/patient_test/page_1.png",
        )
        append_evidence("patient_test", rec1_updated, evidence_dir=tmp_path)
        loaded = load_evidence("patient_test", evidence_dir=tmp_path)
        assert len(loaded) == 2
        assert loaded[0].raw_text == "Page 1 updated text"
        assert loaded[0].confidence == 0.98

    def test_get_evidence_by_id(self, tmp_path):
        records = [
            EvidenceRecord(
                evidence_id="p1__r1__page_1__chunk_0",
                patient_id="p1",
                report_id="r1",
                report_type="RADIOLOGY_PET_CT",
                report_date="2019-05-10",
                page_number=1,
                raw_text="PET text",
                source_type="typed",
                confidence=0.95,
                page_image_path="data/pages/p1/page_1.png",
            )
        ]
        save_evidence_list("p1", records, evidence_dir=tmp_path)

        res = get_evidence_by_id("p1", "p1__r1__page_1__chunk_0", evidence_dir=tmp_path)
        assert res is not None
        assert res.report_type == "RADIOLOGY_PET_CT"

        non_existent = get_evidence_by_id("p1", "unknown__id", evidence_dir=tmp_path)
        assert non_existent is None


class TestSourceTypeHeuristics:
    """Tests for source_type_for_page mapping (SRS FR-5.5.4)."""

    def test_typed_prose(self):
        assert source_type_for_page({"is_table": False, "is_handwritten": False}) == "typed"

    def test_tabular_handwritten(self):
        assert source_type_for_page({"is_table": True, "is_handwritten": True}) == "tabular_handwritten"

    def test_cursive_handwritten(self):
        assert source_type_for_page({"is_table": False, "is_handwritten": True}) == "cursive_handwritten"

    def test_tabular_typed(self):
        assert source_type_for_page({"is_table": True, "is_handwritten": False}) == "typed"


class TestChunking:
    """Tests for chunk.py report chunking and patient orchestration."""

    def test_chunk_report_synthetic(self, tmp_path):
        ocr_dir = tmp_path / "ocr" / "patient_mock"
        ocr_dir.mkdir(parents=True)

        # Create 2 pages
        p1_data = {
            "page_number": 1,
            "raw_text": "DIAGNOSIS: IDC T2N0M0 post MRM. ER 8/8, PR 8/8.",
            "is_table": False,
            "is_handwritten": False,
            "confidence": 0.95,
        }
        (ocr_dir / "page_1.json").write_text(json.dumps(p1_data), encoding="utf-8")

        p2_data = {
            "page_number": 2,
            "raw_text": "Duplicate of page 1",
            "is_table": False,
            "is_handwritten": False,
            "confidence": 0.50,
            "duplicate_of": 1,
        }
        (ocr_dir / "page_2.json").write_text(json.dumps(p2_data), encoding="utf-8")

        span = ReportSpan(
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            page_start=1,
            page_end=2,
            report_date="2016-12-19",
        )

        chunks = chunk_report("patient_mock", span, ocr_dir=tmp_path / "ocr")
        # Page 2 duplicate is skipped -> exactly 1 chunk
        assert len(chunks) == 1
        c = chunks[0]
        assert c.evidence_record.evidence_id == "patient_mock__report_0001__page_1__chunk_0"
        assert c.evidence_record.source_type == "typed"
        assert len(c.normalized_entities) >= 1

    @pytest.mark.parametrize("patient_id", ["patient_a", "patient_b"])
    def test_real_patient_evidence_and_chunks(self, patient_id):
        """Verify real patient evidence files and chunk files produced by Task 1.7."""
        evidence_records = load_evidence(patient_id)
        assert len(evidence_records) >= 30, f"Expected >= 30 evidence records for {patient_id}"

        # Verify all evidence IDs follow SRS format {patient_id}__{report_id}__page_{n}__chunk_{i}
        for rec in evidence_records:
            assert rec.evidence_id.startswith(f"{patient_id}__report_")
            assert "__page_" in rec.evidence_id
            assert "__chunk_" in rec.evidence_id
            assert rec.source_type in {"typed", "tabular_handwritten", "cursive_handwritten"}
            assert 0.0 <= rec.confidence <= 1.0

            # Verify round-trip retrieval
            retrieved = get_evidence_by_id(patient_id, rec.evidence_id)
            assert retrieved is not None
            assert retrieved.evidence_id == rec.evidence_id

        # Verify chunk file
        chunks_file = Path("data/reports") / f"{patient_id}_chunks.json"
        assert chunks_file.exists()
        chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
        assert len(chunks_data) == len(evidence_records)
        assert all("normalized_entities" in c for c in chunks_data)
