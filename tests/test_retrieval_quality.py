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
from orchestrate import build_facts_block, extract_and_filter_citations
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


class TestCanonicalNamesAreNotCollapsed:
    """
    The junk repair in resolve_lab_test_name must never merge distinct analytes.

    'Mean Corpuscular Hemoglobin' (pg) and 'Mean Corpuscular Hemoglobin
    Concentration' (g/dL) both contain the token 'Hemoglobin'. Collapsing them into
    it produced one series of 11.5 gm/dl, 24.2 pg and 30.3 g/dL presented as a
    patient's haemoglobin, and an answer calling the trend "improving" when the only
    real haemoglobin value was flagged low.
    """

    @pytest.mark.parametrize("name", [
        "Hemoglobin",
        "Mean Corpuscular Hemoglobin",
        "Mean Corpuscular Hemoglobin Concentration",
        "Mean Corpuscular Volume",
        "Glycated Hemoglobin (HbA1c)",
        "Packed Cell Volume",
        "Platelet Count",
        "Serum Creatinine",
        "White Blood Cell Count",
        "Absolute Neutrophil Count",
        "Total Leukocyte Count",
    ])
    def test_a_canonical_name_is_returned_unchanged(self, name):
        assert resolve_lab_test_name(name) == name

    def test_the_three_hemoglobin_analytes_stay_distinct(self):
        names = {
            resolve_lab_test_name(n)
            for n in (
                "Hemoglobin",
                "Mean Corpuscular Hemoglobin",
                "Mean Corpuscular Hemoglobin Concentration",
            )
        }
        assert len(names) == 3, f"distinct tests were merged: {names}"

    def test_junk_repair_still_works(self):
        assert resolve_lab_test_name("hemoglobin_11.5") == "Hemoglobin"
        assert resolve_lab_test_name("Hb") == "Hemoglobin"
        assert resolve_lab_test_name("mean corp hemoglobin (mch) result: 24.2 pg") == (
            "Mean Corpuscular Hemoglobin"
        )
        assert resolve_lab_test_name("1.10") is None


class TestCitationFormsAreAllAccepted:
    """
    The evidence block labels each fact "[Citation ID: <id>]" and the model copies
    that wording, so a bare-id-only pattern discarded valid citations: two answers
    in a live run reported 0 citations while quoting real evidence ids.
    """

    VALID = {
        "patient_a__report_0001__page_1__chunk_0",
        "patient_b__report_0023__page_25__chunk_0",
    }

    @pytest.mark.parametrize("text,expected", [
        ("see [patient_a__report_0001__page_1__chunk_0].", 1),
        ("see [Citation ID: patient_a__report_0001__page_1__chunk_0].", 1),
        ("see [citation: patient_b__report_0023__page_25__chunk_0].", 1),
        ("see [Citations IDs: patient_a__report_0001__page_1__chunk_0; "
         "patient_b__report_0023__page_25__chunk_0].", 2),
        ("see [patient_a__report_0001__page_1__chunk_0], "
         "[patient_b__report_0023__page_25__chunk_0].", 2),
    ])
    def test_accepted_forms(self, text, expected):
        _, cites = extract_and_filter_citations(text, self.VALID)
        assert len(cites) == expected

    def test_unknown_ids_are_still_rejected(self):
        _, cites = extract_and_filter_citations(
            "see [made_up_evidence_id] and [Citation ID: also__not__real]", self.VALID
        )
        assert cites == []

    def test_each_id_is_reported_once(self):
        _, cites = extract_and_filter_citations(
            "a [patient_a__report_0001__page_1__chunk_0] "
            "b [Citation ID: patient_a__report_0001__page_1__chunk_0]", self.VALID
        )
        assert cites == ["patient_a__report_0001__page_1__chunk_0"]


class TestRecurrenceQuestionsReachDiagnoses:
    """
    "Has my cancer come back after surgery?" routed to medication_history alone,
    because "surgery" matched that list and plain "cancer" matched nothing — so
    STATES_DIAGNOSIS was never queried and the recurrence was reported as
    "Not documented in the available records" by both engines.
    """

    @pytest.mark.parametrize("question", [
        "Has my cancer come back after surgery, and what was found?",
        "Has my cancer recurred?",
        "Did the tumour relapse after my operation?",
        "Has the cancer spread anywhere?",
        "Is there any metastasis documented?",
    ])
    def test_diagnosis_intent_is_included(self, question):
        assert "diagnosis_list" in classify_intents(question)

    @pytest.mark.parametrize("question", [
        "Has my cancer come back after surgery?",
        "Did my carcinoma recur?",
        "Has the disease spread to the bones?",
    ])
    def test_pageindex_widens_traversal(self, question):
        from pageindex_backend.retrieve import should_allow_multi
        assert should_allow_multi(question) is True


class TestLexicalDocumentShortlisting:
    """
    PageIndex selects documents by their LLM-generated summary, so a report whose
    summary omits a detail can never be shortlisted however wide the traversal:
    asked whether the cancer had recurred it explored several reports and answered
    "Not documented" while 'Recurrent carcinoma' sat in one it never considered.
    """

    CHILDREN = [
        {"node_id": "r1", "summary": "Discharge summary after admission.",
         "children": [{"raw_text": "Patient admitted for observation. Stable."}]},
        {"node_id": "r2", "summary": "Follow-up visit.",
         "children": [{"raw_text": "Post lumpectomy recurrent lump left breast noted."}]},
        {"node_id": "r3", "summary": "Routine review.",
         "children": [{"raw_text": "Known case of bronchial asthma and seizure disorder."}]},
    ]

    def test_lay_wording_still_finds_the_clinical_record(self):
        from pageindex_backend.retrieve import lexically_relevant_children
        got = lexically_relevant_children(
            "Has my cancer come back after surgery?", self.CHILDREN, limit=2
        )
        assert "r2" in [c["node_id"] for c in got], (
            "'come back' must reach 'recurrent'"
        )

    def test_comorbidity_question_finds_the_comorbidity_report(self):
        from pageindex_backend.retrieve import lexically_relevant_children
        got = lexically_relevant_children(
            "Apart from cancer, what other medical conditions do I have?",
            self.CHILDREN, limit=2,
        )
        assert "r3" in [c["node_id"] for c in got]

    def test_no_distinctive_terms_shortlists_nothing(self):
        from pageindex_backend.retrieve import lexically_relevant_children
        assert lexically_relevant_children("What about it?", self.CHILDREN) == []

    def test_stopwords_alone_do_not_match(self):
        from pageindex_backend.retrieve import _routing_terms
        assert _routing_terms("what are the records for my report") == set()


class TestStructuredBranchesUnion:
    """
    A compound question matches more than one category. Returning at the first match
    answered only half of it: asked for receptor status AND chemotherapy, the engine
    reported receptors and silently dropped the drugs.
    """

    FACTS = {
        "chemo_cycles": [], "treatment_plan": {}, "suggested_tests": [],
        "lab_time_series": [],
        "biomarkers": [
            {"marker": "Human Epidermal Growth Factor Receptor 2",
             "value": "Positive (3+)", "date": "2014-05-28", "evidence_id": "ev1"},
        ],
        "current_medications": [
            {"name": "Epirubicin", "dose": "110 mg", "evidence_id": "ev2"},
        ],
    }

    def test_compound_question_returns_both_categories(self):
        facts = resolve_structured_query(
            "Compare the receptor status and the chemotherapy given.",
            "patient_b", self.FACTS,
        )
        nodes = " ".join(f["node_id"] for f in facts)
        assert "biomarker" in nodes and "medications" in nodes

    def test_her2_polarity_is_quoted_verbatim(self):
        facts = resolve_structured_query(
            "What is my HER2 status?", "patient_b", self.FACTS
        )
        text = " ".join(f["text"] for f in facts)
        assert "Positive (3+)" in text
        assert "negative" not in text.lower()

    def test_dose_absence_is_not_filled_in(self):
        facts = resolve_structured_query(
            "Which drugs was I given?", "patient_b",
            {**self.FACTS, "current_medications": [{"name": "Paclitaxel", "evidence_id": "e"}]},
        )
        text = " ".join(f["text"] for f in facts)
        assert "Paclitaxel" in text
        assert "mg" not in text
