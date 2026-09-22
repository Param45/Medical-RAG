"""
Unit tests for Medical RAG Quality Improvements (Phases 1-5).
Tests canonicalization, abnormal flag computation, deterministic Cypher retrieval,
StructuredFacts node build, structured query resolution, and orchestration enhancements.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from normalize import canonicalize_lab_name, compute_abnormal_flag
from graph_backend.retrieve import (
    get_chemo_cycle_status,
    get_suggested_vs_performed,
    get_improvement_trend,
    get_specific_lab_value,
    classify_intent,
)
from pageindex_backend.build import (
    extract_impression_and_comparison,
    build_structured_facts,
)
from pageindex_backend.retrieve import (
    is_structured_query,
    resolve_structured_query,
)
from orchestrate import is_numeric_or_count_query


class TestNormalizationImprovements:
    """Phase 5 & 1 normalization and abnormal flag computation tests."""

    def test_canonicalize_lab_name_standard(self):
        assert canonicalize_lab_name("Hb") == "Hemoglobin"
        assert canonicalize_lab_name("S. Creatinine") == "Serum Creatinine"
        assert canonicalize_lab_name("SGPT (ALT)") == "SGPT (ALT)"
        assert canonicalize_lab_name("Random Blood Sugar") == "Random Blood Sugar"

    def test_canonicalize_lab_name_unknown(self):
        assert canonicalize_lab_name("Custom Novel Marker") == "Custom Novel Marker"
        assert canonicalize_lab_name("") == ""

    def test_compute_abnormal_flag_numeric_ranges(self):
        # Hemoglobin (12.0 - 17.5)
        assert compute_abnormal_flag("Hemoglobin", 10.5) == "low"
        assert compute_abnormal_flag("Hemoglobin", 13.5) == "normal"
        assert compute_abnormal_flag("Hemoglobin", 18.0) == "high"

        # Serum Creatinine (0.6 - 1.2)
        assert compute_abnormal_flag("Serum Creatinine", 1.8) == "high"
        assert compute_abnormal_flag("Serum Creatinine", 0.9) == "normal"
        assert compute_abnormal_flag("Serum Creatinine", 0.4) == "low"

    def test_compute_abnormal_flag_unknown_test(self):
        assert compute_abnormal_flag("UnknownTestX", 55.0) is None


class TestDeterministicGraphRetrieval:
    """Phase 2 deterministic graph retrieval tests."""

    def test_get_chemo_cycle_status_calculation(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # 1st query: ChemoAdministration (4 cycles completed)
        # 2nd query: TreatmentPlan (8 planned cycles)
        mock_session.run.side_effect = [
            [
                {"cycle_number": 1, "regimen": "EC", "date": "2018-02-01", "evidence_id": "ev_1", "confidence": 1.0},
                {"cycle_number": 2, "regimen": "EC", "date": "2018-02-22", "evidence_id": "ev_2", "confidence": 1.0},
                {"cycle_number": 3, "regimen": "EC", "date": "2018-03-15", "evidence_id": "ev_3", "confidence": 1.0},
                {"cycle_number": 4, "regimen": "EC", "date": "2018-04-05", "evidence_id": "ev_4", "confidence": 1.0},
            ],
            [
                {"regimen": "EC", "planned_cycles": 8, "evidence_id": "ev_plan", "confidence": 1.0},
            ],
        ]

        facts = get_chemo_cycle_status("patient_a", driver=mock_driver)
        assert len(facts) == 1
        fact_text = facts[0]["text"]
        assert "Completed 4 of 8 planned cycles" in fact_text
        assert "4 cycle(s) remaining" in fact_text

    def test_get_suggested_vs_performed_calculation(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # 1st query: Suggested tests (CBC, LFT, 2D ECHO)
        # 2nd query: Performed tests (CBC, LFT)
        # 3rd query: Undergone procedures
        mock_session.run.side_effect = [
            [
                {"test_name": "CBC", "suggested_date": "2018-01-10", "evidence_id": "ev_s1"},
                {"test_name": "LFT", "suggested_date": "2018-01-10", "evidence_id": "ev_s2"},
                {"test_name": "2D ECHO", "suggested_date": "2018-01-10", "evidence_id": "ev_s3"},
            ],
            [
                {"test_name": "CBC"},
                {"test_name": "LFT"},
            ],
            [],
        ]

        facts = get_suggested_vs_performed("patient_a", driver=mock_driver)
        assert len(facts) == 1
        fact_text = facts[0]["text"]
        assert "CBC" in fact_text
        assert "2D ECHO" in fact_text
        assert "Pending (suggested but not yet performed): 2D ECHO" in fact_text

    def test_get_improvement_trend_calculation(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_session.run.return_value = [
            {
                "test_name": "Hemoglobin",
                "value": 8.2,
                "unit": "g/dL",
                "date": "2018-01-10",
                "abnormal_flag": "low",
                "evidence_id": "ev_hb1",
                "confidence": 1.0,
            },
            {
                "test_name": "Hemoglobin",
                "value": 12.5,
                "unit": "g/dL",
                "date": "2018-05-20",
                "abnormal_flag": "normal",
                "evidence_id": "ev_hb2",
                "confidence": 1.0,
            },
        ]

        facts = get_improvement_trend("patient_a", test_name="Hemoglobin", driver=mock_driver)
        assert len(facts) == 1
        fact_text = facts[0]["text"]
        assert "8.2" in fact_text
        assert "12.5" in fact_text
        assert "IMPROVING" in fact_text

    def test_classify_intent_new_categories(self):
        assert classify_intent("Completed 4th round of chemo, how many more remaining?") == "chemo_cycle_status"
        assert classify_intent("What tests are suggested and which have been performed?") == "suggested_vs_performed"
        assert classify_intent("Are there any improvements in my health or hemoglobin trend?") == "improvement_trend"
        assert classify_intent("What was my sugar level on 2018-03-15?") == "specific_lab_value"


class TestPageIndexEnhancements:
    """Phase 3 PageIndex build and retrieval enhancements tests."""

    def test_extract_impression_and_comparison(self):
        text = (
            "TECHNIQUE: High resolution contrast enhanced CT.\n"
            "FINDINGS: Liver and spleen unremarkable.\n"
            "IMPRESSION: No evidence of metastatic disease.\n"
            "COMPARISON: Compared with prior study dated 12/01/2017."
        )
        impression, comparison = extract_impression_and_comparison(text)
        assert impression is not None
        assert "No evidence of metastatic disease" in impression
        assert comparison is not None
        assert "Compared with prior study" in comparison

    def test_is_structured_query(self):
        assert is_structured_query("completed 4th round of chemo, how many remaining?") is True
        assert is_structured_query("what all tests are suggested to me and which of these have been performed?") is True
        assert is_structured_query("what were my sugar levels at a specific time of the year?") is True
        assert is_structured_query("are there any improvements in my health?") is True
        assert is_structured_query("What is my diagnosis?") is False

    def test_resolve_structured_query_chemo(self):
        structured_facts = {
            "node_id": "patient_a_structured_facts",
            "chemo_cycles": [
                {"cycle_number": 1, "regimen": "EC", "evidence_id": "ev1"},
                {"cycle_number": 2, "regimen": "EC", "evidence_id": "ev2"},
                {"cycle_number": 3, "regimen": "EC", "evidence_id": "ev3"},
                {"cycle_number": 4, "regimen": "EC", "evidence_id": "ev4"},
            ],
            "treatment_plan": {"regimen": "EC", "planned_cycles": 8, "evidence_id": "ev_p"},
            "suggested_tests": [],
            "lab_time_series": [],
            "current_medications": [],
        }

        facts = resolve_structured_query(
            "completed 4th round of chemo, how many remaining?",
            "patient_a",
            structured_facts,
        )
        assert facts is not None
        assert len(facts) == 1
        assert "completed 4 of 8 planned cycles" in facts[0]["text"]
        assert "4 cycles are remaining" in facts[0]["text"]


class TestOrchestrationEnhancements:
    """Phase 4 orchestration numeric detection tests."""

    def test_is_numeric_or_count_query(self):
        assert is_numeric_or_count_query("How many cycles of chemotherapy are remaining?") is True
        assert is_numeric_or_count_query("What were my blood sugar levels?") is True
        assert is_numeric_or_count_query("Are there any improvements in my health?") is True
        assert is_numeric_or_count_query("Explain what my pathology report means.") is False
