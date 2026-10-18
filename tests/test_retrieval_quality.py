"""
Tests for the retrieval-quality fixes:
lab test name resolution, multi-intent routing, read-only Cypher guard,
per-patient prompt budgeting, and unfabricated structured facts.
"""

from unittest.mock import MagicMock
import pytest

from normalize import resolve_lab_test_name
from graph_backend.retrieve import (
    assert_read_only_patient_scoped,
    classify_intent,
    classify_intents,
    get_improvement_trend,
)
from orchestrate import build_facts_block
from pageindex_backend.retrieve import resolve_structured_query


class TestResolveLabTestName:
    """Extraction puts values, units and row ids in the test-name slot."""

    @pytest.mark.parametrize("junk", [
        "1.10", "1.10 mg/dL", "10,200", "123 mg/dl", "26.22", "7992 /uL",
        "LabResult_001", "LabResult_3", "", "   ",
    ])
    def test_rejects_non_test_names(self, junk):
        assert resolve_lab_test_name(junk) is None

    @pytest.mark.parametrize("raw,expected", [
        ("Hemoglobin", "Hemoglobin"),
        ("Hb", "Hemoglobin"),
        ("hemoglobin_11.5", "Hemoglobin"),
        ("Hemoglobin Result: 11.5 gm/dl", "Hemoglobin"),
        ("S. Creatinine", "Serum Creatinine"),
        ("serum creatinine 1.10 mg/dl", "Serum Creatinine"),
        ("ANC=7992", "Absolute Neutrophil Count"),
        ("wbc_15/1/17", "White Blood Cell Count"),
        ("TOTAL_LEUCOCYTE_COUNT", "Total Leukocyte Count"),
    ])
    def test_merges_variants_onto_one_canonical_name(self, raw, expected):
        assert resolve_lab_test_name(raw) == expected

    def test_prefers_the_more_specific_test(self):
        # 'Mean Corp Hemoglobin (MCH)' contains 'Hemoglobin' but is a different test.
        assert resolve_lab_test_name("Mean Corp Hemoglobin (MCH) Result: 24.2 pg") == (
            "Mean Corpuscular Hemoglobin"
        )

    def test_keeps_a_plausible_name_missing_from_the_dictionary(self):
        assert resolve_lab_test_name("Reticulocyte Count") == "Reticulocyte Count"
        assert resolve_lab_test_name("RDW-CV Result: 16.4 %") == "RDW-CV"

    def test_variants_let_a_trend_form(self):
        """Unmerged spellings gave one point per node, so no trend was ever computed."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = [
            {"test_name": "Hemoglobin_8.2", "value": "8.2", "unit": "g/dL",
             "abnormal_flag": "low", "date": "2018-01-10",
             "evidence_id": "ev1", "confidence": 1.0},
            {"test_name": "Hb", "value": "12.5", "unit": "g/dL",
             "abnormal_flag": "normal", "date": "2018-05-20",
             "evidence_id": "ev2", "confidence": 1.0},
            {"test_name": "LabResult_001", "value": "999", "unit": "",
             "abnormal_flag": "", "date": "2018-06-01",
             "evidence_id": "ev3", "confidence": 1.0},
        ]

        facts = get_improvement_trend("patient_a", driver=mock_driver)
        text = facts[0]["text"]
        assert "Hemoglobin" in text
        assert "IMPROVING" in text
        assert "LabResult_001" not in text  # noise dropped
        assert "999" not in text


class TestMultiIntent:
    def test_compound_question_keeps_every_intent(self):
        intents = classify_intents("What is my diagnosis and treatment history?")
        assert "diagnosis_list" in intents
        assert "medication_history" in intents

    def test_single_intent_wrapper_is_unchanged(self):
        assert classify_intent("What is my diagnosis and treatment history?") == "diagnosis_list"

    def test_unmatched_question_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "graph_backend.retrieve.chat",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        assert classify_intents("Tell me about the weather") == ["open_ended"]


class TestReadOnlyCypherGuard:
    def test_allows_a_scoped_read(self):
        q = "MATCH (p:Patient) WHERE p.patient_id IN $patient_ids RETURN p.patient_id"
        assert assert_read_only_patient_scoped(q) == q

    @pytest.mark.parametrize("query", [
        "MATCH (n) WHERE n.patient_id IN $patient_ids DETACH DELETE n",
        "MATCH (p) WHERE p.patient_id IN $patient_ids SET p.x = 1 RETURN p",
        "MERGE (p:Patient {patient_id: 'x'}) RETURN p",
        "MATCH (p:Patient) RETURN p",                                   # unscoped
        "MATCH (p) WHERE p.patient_id IN $patient_ids RETURN p; DROP INDEX i",
        "MATCH (p) WHERE p.patient_id IN $patient_ids RETURN apoc.version()",
        "",
    ])
    def test_rejects_writes_and_unscoped_reads(self, query):
        with pytest.raises(ValueError):
            assert_read_only_patient_scoped(query)


class TestFactsBlockBudget:
    def test_every_patient_gets_a_share(self):
        facts = (
            [{"patient_id": "patient_a", "evidence_id": f"a{i}", "text": "A" * 900} for i in range(40)]
            + [{"patient_id": "patient_b", "evidence_id": f"b{i}", "text": "B" * 900} for i in range(40)]
        )
        block = build_facts_block(facts, budget=4000)

        assert "(patient_a)" in block
        assert "(patient_b)" in block
        assert len(block) <= 4000 + 2000  # one overshooting line per patient at most

    def test_empty_facts_give_an_empty_block(self):
        assert build_facts_block([]) == ""


class TestStructuredFactsAreNotFabricated:
    def test_unknown_planned_cycle_count_is_not_invented(self):
        facts = resolve_structured_query(
            "how many chemo cycles remain?",
            "patient_a",
            {"chemo_cycles": [{"cycle_number": 2, "evidence_id": "ev2"}], "treatment_plan": {}},
        )
        text = facts[0]["text"]
        assert "completed 2" in text
        assert "not documented" in text
        assert "of 8" not in text  # the old hardcoded default

    def test_no_citation_is_minted_when_none_exists(self):
        facts = resolve_structured_query(
            "how many chemo cycles remain?",
            "patient_a",
            {"chemo_cycles": [{"cycle_number": 2}], "treatment_plan": {}},
        )
        assert facts[0]["evidence_id"] == ""

    def test_each_test_gets_its_own_trend(self):
        structured = {
            "chemo_cycles": [],
            "treatment_plan": {},
            "suggested_tests": [],
            "lab_time_series": [
                {"test": "Hemoglobin", "value": "8.2", "unit": "g/dL",
                 "abnormal_flag": "low", "date": "2018-01-10", "evidence_id": "ev1"},
                {"test": "Hb", "value": "12.5", "unit": "g/dL",
                 "abnormal_flag": "normal", "date": "2018-05-20", "evidence_id": "ev2"},
                {"test": "S. Creatinine", "value": "1.1", "unit": "mg/dL",
                 "abnormal_flag": "normal", "date": "2018-01-10", "evidence_id": "ev3"},
                {"test": "LabResult_7", "value": "42", "unit": "",
                 "abnormal_flag": "", "date": "2018-01-10", "evidence_id": "ev4"},
            ],
        }

        facts = resolve_structured_query(
            "what are my hemoglobin and creatinine trends?", "patient_a", structured
        )
        by_name = {f["text"].split(":")[0]: f["text"] for f in facts}

        assert set(by_name) == {"Hemoglobin", "Serum Creatinine"}
        # Values from one test never appear under another's name.
        assert "1.1" not in by_name["Hemoglobin"]
        assert "8.2" not in by_name["Serum Creatinine"]
        assert "improving" in by_name["Hemoglobin"]

    def test_a_single_point_is_not_called_a_trend(self):
        facts = resolve_structured_query(
            "what is my hemoglobin trend?",
            "patient_a",
            {
                "chemo_cycles": [], "treatment_plan": {}, "suggested_tests": [],
                "lab_time_series": [
                    {"test": "Hemoglobin", "value": "8.2", "unit": "g/dL",
                     "abnormal_flag": "low", "date": "2018-01-10", "evidence_id": "ev1"},
                ],
            },
        )
        assert "not enough data points" in facts[0]["text"]
