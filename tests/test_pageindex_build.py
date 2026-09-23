"""
Tests for PageIndex Tree Build (SRS §7.1, §12.2, BUILD_GUIDE Task 3.1).
All LLM calls are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pageindex_backend.build import (
    build_tree_for_patient,
    generate_report_summary,
    generate_root_summary,
    save_tree,
)
from evidence_store import EvidenceRecord, save_evidence_list


class TestPageIndexSummaries:
    """Tests for report and root summary generation."""

    @patch("pageindex_backend.build.chat")
    def test_generate_report_summary_success(self, mock_chat):
        mock_chat.return_value = "Patient underwent CECT scan showing stable metastatic disease."
        summary = generate_report_summary(
            report_id="report_0001",
            report_type="RADIOLOGY_CECT",
            report_date="2018-03-20",
            text="Some report text",
        )
        assert summary == "Patient underwent CECT scan showing stable metastatic disease."
        mock_chat.assert_called_once()

    @patch("pageindex_backend.build.chat")
    def test_generate_report_summary_empty_text(self, mock_chat):
        summary = generate_report_summary(
            report_id="report_0001",
            report_type="RADIOLOGY_CECT",
            report_date="2018-03-20",
            text="",
        )
        assert "no extractable text" in summary
        mock_chat.assert_not_called()

    @patch("pageindex_backend.build.chat")
    def test_generate_root_summary_success(self, mock_chat):
        mock_chat.return_value = "Overall clinical summary of Patient A with breast cancer."
        report_nodes = [
            {
                "node_id": "report_0001",
                "report_type": "RADIOLOGY_CECT",
                "report_date": "2018-03-20",
                "summary": "Stable metastatic lesions.",
            }
        ]
        summary = generate_root_summary("patient_a", report_nodes)
        assert summary == "Overall clinical summary of Patient A with breast cancer."
        mock_chat.assert_called_once()

    @patch("pageindex_backend.build.chat")
    def test_generate_root_summary_empty_reports(self, mock_chat):
        summary = generate_root_summary("patient_a", [])
        assert "no reports documented" in summary
        mock_chat.assert_not_called()

    @patch("pageindex_backend.build.chat")
    def test_generate_report_summary_strips_thought_tags(self, mock_chat):
        mock_chat.return_value = "<thought>Thinking about patient scan...</thought>Patient underwent CECT scan showing stable metastatic disease."
        summary = generate_report_summary(
            report_id="report_0001",
            report_type="RADIOLOGY_CECT",
            report_date="2018-03-20",
            text="Some report text",
        )
        assert summary == "Patient underwent CECT scan showing stable metastatic disease."
        assert "<thought>" not in summary

    @patch("pageindex_backend.build.chat")
    def test_generate_root_summary_strips_thought_tags(self, mock_chat):
        mock_chat.return_value = "<thought>Synthesizing overview...</thought>Overall clinical summary of Patient A with breast cancer."
        report_nodes = [
            {
                "node_id": "report_0001",
                "report_type": "RADIOLOGY_CECT",
                "report_date": "2018-03-20",
                "summary": "Stable metastatic lesions.",
            }
        ]
        summary = generate_root_summary("patient_a", report_nodes)
        assert summary == "Overall clinical summary of Patient A with breast cancer."
        assert "<thought>" not in summary


class TestPageIndexTreeBuild:
    """Tests for 3-level tree construction and schema validation."""

    @pytest.fixture
    def synthetic_patient_dir(self, tmp_path):
        """Creates a temporary synthetic patient data fixture with 1 report and 2 pages."""
        patient_id = "patient_test"
        reports_dir = tmp_path / "reports"
        evidence_dir = tmp_path / "evidence"
        reports_dir.mkdir(parents=True)
        evidence_dir.mkdir(parents=True)

        # 1. Report spans
        spans = [
            {
                "report_id": "report_0001",
                "report_type": "HISTOPATHOLOGY",
                "page_start": 1,
                "page_end": 2,
                "report_date": "2018-01-15",
            }
        ]
        (reports_dir / f"{patient_id}.json").write_text(json.dumps(spans), encoding="utf-8")

        # 2. Chunks
        ev1 = EvidenceRecord(
            evidence_id=f"{patient_id}__report_0001__page_1__chunk_0",
            patient_id=patient_id,
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            report_date="2018-01-15",
            page_number=1,
            raw_text="Invasive Ductal Carcinoma Grade 3. Modified Radical Mastectomy specimen.",
            source_type="typed",
            confidence=0.98,
            page_image_path=f"data/pages/{patient_id}/page_1.png",
        )
        ev2 = EvidenceRecord(
            evidence_id=f"{patient_id}__report_0001__page_2__chunk_0",
            patient_id=patient_id,
            report_id="report_0001",
            report_type="HISTOPATHOLOGY",
            report_date="2018-01-15",
            page_number=2,
            raw_text="ER Positive 8/8, PR Positive 6/8, HER2 Negative 1+.",
            source_type="typed",
            confidence=0.95,
            page_image_path=f"data/pages/{patient_id}/page_2.png",
        )

        chunks = [
            {
                "evidence_record": ev1.to_dict(),
                "normalized_entities": [],
                "chunk_type": "page",
            },
            {
                "evidence_record": ev2.to_dict(),
                "normalized_entities": [],
                "chunk_type": "page",
            },
        ]
        (reports_dir / f"{patient_id}_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

        # 3. Evidence store records
        save_evidence_list(patient_id, [ev1, ev2], evidence_dir=evidence_dir)

        return {
            "patient_id": patient_id,
            "reports_dir": reports_dir,
            "evidence_dir": evidence_dir,
            "ev1": ev1,
            "ev2": ev2,
        }

    @patch("pageindex_backend.build.chat")
    def test_build_tree_shape_and_schema(self, mock_chat, synthetic_patient_dir):
        """Verify the 3-level tree matches SRS §12.2 schema."""
        mock_chat.side_effect = [
            "Histopathology report confirming IDC Grade 3, ER/PR positive, HER2 negative.",  # Report summary
            "Patient Test overview: Diagnosed with ER/PR+ IDC following MRM.",                # Root summary
        ]

        pid = synthetic_patient_dir["patient_id"]
        reports_dir = synthetic_patient_dir["reports_dir"]

        tree = build_tree_for_patient(pid, reports_dir=reports_dir)

        # Level 1: Root / Patient
        assert tree["patient_id"] == pid
        assert "root" in tree
        root = tree["root"]
        assert root["node_id"] == "root"
        assert root["summary"] == "Patient Test overview: Diagnosed with ER/PR+ IDC following MRM."
        assert len(root["children"]) == 1

        # Level 2: Report
        rep_node = root["children"][0]
        assert rep_node["node_id"] == "report_0001"
        assert rep_node["node_type"] == "Report"
        assert rep_node["report_type"] == "HISTOPATHOLOGY"
        assert rep_node["report_date"] == "2018-01-15"
        assert rep_node["summary"] == "Histopathology report confirming IDC Grade 3, ER/PR positive, HER2 negative."
        assert len(rep_node["children"]) == 2

        # Level 3: Leaf Pages
        p1 = rep_node["children"][0]
        assert p1["node_id"] == "report_0001_page_1"
        assert p1["node_type"] == "Page"
        assert "Invasive Ductal Carcinoma" in p1["raw_text"]
        assert p1["evidence_id"] == f"{pid}__report_0001__page_1__chunk_0"

        p2 = rep_node["children"][1]
        assert p2["node_id"] == "report_0001_page_2"
        assert p2["node_type"] == "Page"
        assert "ER Positive" in p2["raw_text"]
        assert p2["evidence_id"] == f"{pid}__report_0001__page_2__chunk_0"

    @patch("pageindex_backend.build.chat")
    def test_save_and_load_tree_roundtrip(self, mock_chat, synthetic_patient_dir, tmp_path):
        """Verify save_tree writes valid JSON to disk that matches expected schema."""
        mock_chat.return_value = "Mocked clinical summary."

        pid = synthetic_patient_dir["patient_id"]
        reports_dir = synthetic_patient_dir["reports_dir"]
        pageindex_dir = tmp_path / "pageindex"

        tree = build_tree_for_patient(pid, reports_dir=reports_dir)
        saved_file = save_tree(pid, tree, pageindex_dir=pageindex_dir)

        assert saved_file.exists()
        loaded_tree = json.loads(saved_file.read_text(encoding="utf-8"))

        assert loaded_tree["patient_id"] == pid
        assert loaded_tree["root"]["node_id"] == "root"
        assert len(loaded_tree["root"]["children"]) == 1
        assert len(loaded_tree["root"]["children"][0]["children"]) == 2

    def test_missing_files_raise_filenotfound(self, tmp_path):
        """Verify FileNotFoundError is raised if reports or chunks files are missing."""
        empty_dir = tmp_path / "empty_reports"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            build_tree_for_patient("non_existent_patient", reports_dir=empty_dir)
