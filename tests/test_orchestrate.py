"""
Tests for Orchestration Layer & Answer Generation (SRS §8, BUILD_GUIDE Task 4.1).
All retrieval and LLM calls are mocked.
"""

from unittest.mock import MagicMock, patch
import pytest

from orchestrate import (
    MEDICAL_DISCLAIMER,
    answer_question,
    extract_and_filter_citations,
    generate_answer,
)


class TestValidationAndDispatch:
    """Tests for parameter validation and backend routing (SRS §8.1)."""

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            answer_question(
                question="What is my diagnosis?",
                mode="super_admin",
                selected_patients=["patient_a"],
                backend="graph",
            )

    def test_invalid_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            answer_question(
                question="What is my diagnosis?",
                mode="individual",
                selected_patients=["patient_a"],
                backend="sql_db",
            )

    @patch("orchestrate.graph_retriever.retrieve")
    @patch("orchestrate.generate_answer")
    def test_dispatch_to_graph_backend(self, mock_generate, mock_retrieve):
        mock_retrieve.return_value = [{"text": "Fact 1", "patient_id": "patient_a", "evidence_id": "ev_1"}]
        mock_generate.return_value = ("Answer from graph [ev_1]", ["ev_1"])

        res = answer_question(
            question="What is my diagnosis?",
            mode="individual",
            selected_patients=["patient_a"],
            backend="graph",
        )

        mock_retrieve.assert_called_once_with("What is my diagnosis?", ["patient_a"])
        assert res["backend_used"] == "graph"
        assert res["answer"] == "Answer from graph [ev_1]"
        assert res["citations"] == ["ev_1"]

    @patch("orchestrate.pageindex_retriever.retrieve")
    @patch("orchestrate.generate_answer")
    def test_dispatch_to_pageindex_backend(self, mock_generate, mock_retrieve):
        mock_retrieve.return_value = [{"text": "Fact 2", "patient_id": "patient_b", "evidence_id": "ev_2"}]
        mock_generate.return_value = ("Answer from pageindex [ev_2]", ["ev_2"])

        res = answer_question(
            question="What medications were given?",
            mode="individual",
            selected_patients=["patient_b"],
            backend="pageindex",
        )

        mock_retrieve.assert_called_once_with("What medications were given?", ["patient_b"])
        assert res["backend_used"] == "pageindex"
        assert res["answer"] == "Answer from pageindex [ev_2]"
        assert res["citations"] == ["ev_2"]

    def test_empty_selected_patients_returns_clean_response(self):
        res = answer_question(
            question="Any questions?",
            mode="individual",
            selected_patients=[],
            backend="graph",
        )
        assert res["citations"] == []
        assert "No patient selected" in res["answer"]


class TestGroundingAndAnswerGeneration:
    """Tests for citation filtering and prompt construction (SRS §8.2)."""

    def test_empty_facts_returns_not_documented(self):
        ans, cites = generate_answer("What is my diagnosis?", facts=[], mode="individual")
        assert "Not documented" in ans
        assert cites == []

    def test_extract_and_filter_citations_strips_hallucinations(self):
        valid_evidence = {
            "patient_a__report_0001__page_1__chunk_0",
            "patient_a__report_0002__page_2__chunk_0",
        }
        raw_llm_text = (
            "The patient was diagnosed with metastatic breast cancer [patient_a__report_0001__page_1__chunk_0]. "
            "She also had high blood pressure [hallucinated_citation_9999]. "
            "Her hemoglobin was 9.9 [patient_a__report_0002__page_2__chunk_0]."
        )

        cleaned, valid_cites = extract_and_filter_citations(raw_llm_text, valid_evidence)
        assert len(valid_cites) == 2
        assert "patient_a__report_0001__page_1__chunk_0" in valid_cites
        assert "patient_a__report_0002__page_2__chunk_0" in valid_cites
        assert "hallucinated_citation_9999" not in valid_cites

    @patch("orchestrate.chat")
    def test_generate_answer_individual_mode(self, mock_chat):
        mock_chat.return_value = (
            "Patient was diagnosed with invasive ductal carcinoma [patient_a__report_0001__page_1__chunk_0]."
        )
        facts = [
            {
                "text": "Histopathology confirms IDC Grade 3.",
                "patient_id": "patient_a",
                "evidence_id": "patient_a__report_0001__page_1__chunk_0",
            }
        ]

        ans, cites = generate_answer("What is my cancer diagnosis?", facts, mode="individual")
        assert "invasive ductal carcinoma" in ans
        assert cites == ["patient_a__report_0001__page_1__chunk_0"]

    @patch("orchestrate.chat")
    def test_generate_answer_group_mode_multi_patient_prompt(self, mock_chat):
        mock_chat.return_value = (
            "Patient A had ER+ disease [ev_a]. Patient B had HER2+ disease [ev_b]."
        )
        facts = [
            {"text": "Patient A findings", "patient_id": "patient_a", "evidence_id": "ev_a"},
            {"text": "Patient B findings", "patient_id": "patient_b", "evidence_id": "ev_b"},
        ]

        ans, cites = generate_answer("Compare patients", facts, mode="group")
        # Ensure system prompt instructed per-patient structuring
        args, kwargs = mock_chat.call_args
        system_arg = kwargs.get("system", "")
        assert "PER PATIENT" in system_arg
        assert cites == ["ev_a", "ev_b"]
