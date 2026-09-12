"""
Tests for Graph Retrieval Engine (SRS §6.2, BUILD_GUIDE Task 2.4).
All tests mock the Neo4j driver and LLM client.
"""

from unittest.mock import MagicMock, patch
import pytest

from graph_backend.retrieve import (
    classify_intent,
    get_lab_trend,
    get_diagnoses,
    get_medication_history,
    get_staging_and_biomarkers,
    open_ended_query,
    retrieve,
)


class TestClassifyIntent:
    """Tests for classify_intent keyword matching and fallback."""

    def test_classify_lab_trend(self):
        assert classify_intent("What is my hemoglobin and platelet trend?") == "lab_trend"
        assert classify_intent("Show me creatinine and bilirubin lab values") == "lab_trend"

    def test_classify_diagnosis_list(self):
        assert classify_intent("What diseases was I suffering from in the last year?") == "diagnosis_list"
        assert classify_intent("List all my cancer diagnoses") == "diagnosis_list"

    def test_classify_medication_history(self):
        assert classify_intent("What chemotherapy and medications have I received?") == "medication_history"
        assert classify_intent("What surgery or procedure did I undergo?") == "medication_history"

    def test_classify_staging_biomarker(self):
        assert classify_intent("Which patients have HER2-positive status documented?") == "staging_biomarker"
        assert classify_intent("What is my TNM staging and ER/PR receptor status?") == "staging_biomarker"

    @patch("graph_backend.retrieve.chat")
    def test_classify_intent_llm_fallback(self, mock_chat):
        mock_chat.return_value = "lab_trend"
        res = classify_intent("Can you check the numbers from my blood draw?")
        assert res == "lab_trend"


class TestCypherTemplates:
    """Tests for Cypher template execution with mocked session."""

    def test_get_lab_trend(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_record = {
            "test_name": "Hemoglobin",
            "value": "11.0",
            "unit": "g/dL",
            "date": "2018-04-08",
            "report_id": "report_0002",
            "evidence_id": "patient_a__report_0002__page_2__chunk_0",
            "confidence": 0.95,
        }
        mock_session.run.return_value = [mock_record]

        facts = get_lab_trend("patient_a", test_name="hemoglobin", driver=mock_driver)
        assert len(facts) == 1
        assert "Hemoglobin" in facts[0]["text"]
        assert "11.0 g/dL" in facts[0]["text"]
        assert facts[0]["patient_id"] == "patient_a"
        assert facts[0]["evidence_id"] == "patient_a__report_0002__page_2__chunk_0"

    def test_get_diagnoses(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_record = {
            "diagnosis": "Invasive Ductal Carcinoma",
            "report_type": "HISTOPATHOLOGY",
            "date": "2018-03-20",
            "report_id": "report_0001",
            "evidence_id": "patient_a__report_0001__page_1__chunk_0",
            "confidence": 0.98,
        }
        mock_session.run.return_value = [mock_record]

        facts = get_diagnoses("patient_a", driver=mock_driver)
        assert len(facts) == 1
        assert "Invasive Ductal Carcinoma" in facts[0]["text"]
        assert facts[0]["evidence_id"] == "patient_a__report_0001__page_1__chunk_0"

    def test_get_medication_history(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_records_med = [
            {
                "name": "Paclitaxel",
                "dose": "80mg/m2",
                "cycle": "1",
                "date": "2018-04-08",
                "report_id": "report_0002",
                "evidence_id": "patient_a__r2__p2__c0",
                "confidence": 0.94,
            }
        ]
        # mock_session.run called 3 times: meds, regimens, procs
        mock_session.run.side_effect = [mock_records_med, [], []]

        facts = get_medication_history("patient_a", driver=mock_driver)
        assert len(facts) == 1
        assert "Paclitaxel" in facts[0]["text"]
        assert "80mg/m2" in facts[0]["text"]

    def test_get_staging_and_biomarkers(self):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        mock_bm_record = {
            "marker": "HER2",
            "value": "3+",
            "date": "2018-03-20",
            "report_id": "report_0001",
            "evidence_id": "patient_a__r1__p1__c0",
            "confidence": 0.99,
        }
        mock_st_record = {
            "t": "T2",
            "n": "N1",
            "m": "M0",
            "date": "2018-03-20",
            "report_id": "report_0001",
            "evidence_id": "patient_a__r1__p1__c0",
            "confidence": 0.99,
        }
        mock_session.run.side_effect = [[mock_bm_record], [mock_st_record]]

        facts = get_staging_and_biomarkers("patient_a", driver=mock_driver)
        assert len(facts) == 2
        assert "HER2" in facts[0]["text"]
        assert "T:T2 N:N1 M:M0" in facts[1]["text"]


class TestRetrieveOrchestration:
    """Tests for retrieve public entry point."""

    @patch("graph_backend.retrieve.get_diagnoses")
    def test_retrieve_individual_patient(self, mock_get_diag):
        mock_get_diag.return_value = [{"text": "Diagnosis Breast Cancer", "patient_id": "patient_a", "evidence_id": "ev1", "confidence": 0.9}]
        mock_driver = MagicMock()

        facts = retrieve("What diseases was I diagnosed with?", ["patient_a"], driver=mock_driver)
        assert len(facts) == 1
        assert facts[0]["patient_id"] == "patient_a"
        mock_get_diag.assert_called_once_with(patient_id="patient_a", driver=mock_driver)

    @patch("graph_backend.retrieve.get_staging_and_biomarkers")
    def test_retrieve_multiple_patients(self, mock_get_st_bm):
        mock_get_st_bm.side_effect = [
            [{"text": "Patient A HER2 3+", "patient_id": "patient_a", "evidence_id": "ev_a", "confidence": 0.9}],
            [{"text": "Patient B HER2 1+", "patient_id": "patient_b", "evidence_id": "ev_b", "confidence": 0.9}],
        ]
        mock_driver = MagicMock()

        facts = retrieve("Which patients have HER2-positive status?", ["patient_a", "patient_b"], driver=mock_driver)
        assert len(facts) == 2
        assert facts[0]["patient_id"] == "patient_a"
        assert facts[1]["patient_id"] == "patient_b"
        assert mock_get_st_bm.call_count == 2
