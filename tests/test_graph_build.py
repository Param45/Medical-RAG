"""
Tests for Knowledge Graph Construction & Extraction (SRS §6.1, BUILD_GUIDE Task 2.3).
All tests mock the LLM and Neo4j driver.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from graph_backend.build import (
    extract_triples,
    write_triples_to_neo4j,
    build_graph_for_patient,
)


class TestExtractTriples:
    """Tests for extract_triples LLM response parsing and schema validation."""

    @patch("graph_backend.build.chat")
    def test_extract_triples_valid_json(self, mock_chat):
        mock_chat.return_value = json.dumps([
            {
                "subject_label": "Report",
                "subject_name": "report_0001",
                "relation": "STATES_DIAGNOSIS",
                "object_label": "Diagnosis",
                "object_name": "Invasive Ductal Carcinoma",
                "properties": {},
            },
            {
                "subject_label": "Patient",
                "subject_name": "patient_a",
                "relation": "UNDERWENT",
                "object_label": "Procedure",
                "object_name": "Modified Radical Mastectomy",
                "properties": {"date": "2018-03-20"},
            },
        ])

        report_meta = {
            "patient_id": "patient_a",
            "report_id": "report_0001",
            "report_type": "SURGICAL_NOTE",
            "report_date": "2018-03-20",
        }
        triples = extract_triples("raw chunk text", [], report_meta)

        assert len(triples) == 2
        assert triples[0]["relation"] == "STATES_DIAGNOSIS"
        assert triples[0]["object_name"] == "Invasive Ductal Carcinoma"
        assert triples[1]["relation"] == "UNDERWENT"
        assert triples[1]["object_name"] == "Modified Radical Mastectomy"

    @patch("graph_backend.build.chat")
    def test_extract_triples_code_fences(self, mock_chat):
        mock_chat.return_value = """```json
[
  {
    "subject_label": "Report",
    "subject_name": "report_0002",
    "relation": "ADMINISTERED",
    "object_label": "Medication",
    "object_name": "Paclitaxel",
    "properties": {"dose": "80mg/m2", "cycle": "1"}
  }
]
```"""
        report_meta = {"patient_id": "patient_a", "report_id": "report_0002"}
        triples = extract_triples("raw text", [], report_meta)

        assert len(triples) == 1
        assert triples[0]["relation"] == "ADMINISTERED"
        assert triples[0]["object_name"] == "Paclitaxel"
        assert triples[0]["properties"]["dose"] == "80mg/m2"

    @patch("graph_backend.build.chat")
    def test_extract_triples_empty_or_null(self, mock_chat):
        mock_chat.return_value = "[]"
        triples = extract_triples("random non-medical text", [], {})
        assert triples == []

        mock_chat.return_value = ""
        triples2 = extract_triples("empty response", [], {})
        assert triples2 == []

    @patch("graph_backend.build.chat")
    def test_extract_triples_malformed_json_graceful_recovery(self, mock_chat):
        mock_chat.return_value = "This is not JSON at all: {broken"
        triples = extract_triples("raw text", [], {})
        assert triples == []

    @patch("graph_backend.build.chat")
    def test_extract_triples_filters_invalid_relations(self, mock_chat):
        mock_chat.return_value = json.dumps([
            {
                "subject_label": "Report",
                "relation": "NON_EXISTENT_RELATION",
                "object_name": "Invalid",
            },
            {
                "subject_label": "Report",
                "relation": "STATES_DIAGNOSIS",
                "object_name": "Valid Diagnosis",
            },
        ])
        triples = extract_triples("raw text", [], {})
        assert len(triples) == 1
        assert triples[0]["relation"] == "STATES_DIAGNOSIS"


class TestWriteTriplesToNeo4j:
    """Tests for write_triples_to_neo4j Cypher generation and execution."""

    def test_write_triples_all_relation_types(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        report_meta = {
            "report_id": "report_0001",
            "report_type": "HISTOPATHOLOGY",
            "report_date": "2018-03-20",
        }

        sample_triples = [
            {
                "relation": "STATES_DIAGNOSIS",
                "object_name": "Breast Cancer",
            },
            {
                "relation": "UNDERWENT",
                "object_name": "MRM",
                "properties": {"date": "2018-03-20"},
            },
            {
                "relation": "ADMINISTERED",
                "object_label": "Medication",
                "object_name": "Trastuzumab",
                "properties": {"dose": "6mg/kg", "cycle": "2"},
            },
            {
                "relation": "ADMINISTERED",
                "object_label": "Regimen",
                "object_name": "AC Regimen",
            },
            {
                "relation": "CONTAINS",
                "subject_name": "AC Regimen",
                "object_name": "Doxorubicin",
            },
            {
                "relation": "HAS_RESULT",
                "object_name": "Hemoglobin",
                "properties": {"value": "12.5", "unit": "g/dL", "date": "2018-03-20"},
            },
            {
                "relation": "HAS_STAGING",
                "properties": {"t": "T2", "n": "N1", "m": "M0", "date": "2018-03-20"},
            },
            {
                "relation": "HAS_BIOMARKER",
                "properties": {"marker": "HER2", "value": "3+", "date": "2018-03-20"},
            },
        ]

        written = write_triples_to_neo4j(
            driver=mock_driver,
            patient_id="patient_a",
            report_meta=report_meta,
            triples=sample_triples,
            evidence_id="patient_a__report_0001__page_1__chunk_0",
            confidence=0.95,
        )

        assert written == len(sample_triples)
        # 1 base query (Patient+Report+HAS_REPORT) + 8 triple queries = 9 queries
        assert mock_session.run.call_count == 9


class TestBuildGraphForPatient:
    """Tests for build_graph_for_patient pipeline execution."""

    @patch("graph_backend.build.extract_triples")
    @patch("graph_backend.build.write_triples_to_neo4j")
    def test_build_graph_for_patient_with_mock_data(self, mock_write, mock_extract, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = reports_dir / "patient_a_chunks.json"

        sample_chunks = [
            {
                "evidence_record": {
                    "evidence_id": "patient_a__r1__p1__c0",
                    "patient_id": "patient_a",
                    "report_id": "report_0001",
                    "report_type": "RADIOLOGY",
                    "report_date": "2018-01-01",
                    "raw_text": "sample text",
                    "confidence": 0.99,
                },
                "normalized_entities": [],
            }
        ]
        chunks_file.write_text(json.dumps(sample_chunks), encoding="utf-8")

        mock_extract.return_value = [{"relation": "STATES_DIAGNOSIS", "object_name": "Cancer"}]
        mock_write.return_value = 1

        mock_driver = MagicMock()
        stats = build_graph_for_patient(patient_id="patient_a", driver=mock_driver, data_dir=tmp_path)

        assert stats["patient_id"] == "patient_a"
        assert stats["total_chunks"] == 1
        assert stats["triples_extracted"] == 1
        assert stats["triples_written"] == 1
