"""
Unit tests for Terminology Normalization (Task 1.6, SRS §5.4).
"""

from pathlib import Path
import pytest

from normalize import (
    NormalizedEntity,
    dictionary_match,
    load_normalization_dictionaries,
    normalize_chunk_text,
    normalize_date,
    regex_parsers,
)


class TestDictionaryMatch:
    """Tests for oncology and Hindi dictionary matching (exact and fuzzy)."""

    def test_exact_oncology_abbreviations(self):
        cases = [
            ("MRM", "Modified Radical Mastectomy", "Procedure"),
            ("IDC", "Invasive Ductal Carcinoma", "Diagnosis"),
            ("EC", "Epirubicin + Cyclophosphamide", "Regimen"),
            ("5FU", "Fluorouracil", "Medication"),
            ("PET-CT", "Positron Emission Tomography - Computed Tomography", "Procedure"),
            ("IRCH", "Dr. B.R. Ambedkar Institute Rotary Cancer Hospital", "Organization"),
            ("ER", "Estrogen Receptor", "Biomarker"),
            ("PR", "Progesterone Receptor", "Biomarker"),
            ("HER2", "Human Epidermal Growth Factor Receptor 2", "Biomarker"),
            ("Ki67", "Ki-67 Proliferation Index", "Biomarker"),
            ("Hb", "Hemoglobin", "LabTest"),
            ("WBC", "White Blood Cell Count", "LabTest"),
            ("ANC", "Absolute Neutrophil Count", "LabTest"),
            ("ESR", "Erythrocyte Sedimentation Rate", "LabTest"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type
            assert res.method == "dictionary"

    def test_case_insensitivity(self):
        res = dictionary_match("mrm")
        assert res is not None
        assert res.normalized_term == "Modified Radical Mastectomy"

        res_idc = dictionary_match("idc")
        assert res_idc is not None
        assert res_idc.normalized_term == "Invasive Ductal Carcinoma"

    def test_hindi_dictionary_match(self):
        hindi_cases = [
            ("सहमति पत्र", "Consent Form", "DocumentSection"),
            ("रोगी के हस्ताक्षर", "Patient Signature", "DocumentSection"),
            ("दिनांक", "Date", "DocumentSection"),
            ("अस्पताल", "Hospital", "Organization"),
        ]
        for term, expected_canonical, expected_type in hindi_cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected Hindi match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type
            assert res.method == "dictionary"

    def test_fuzzy_matching(self):
        # Slight variations / typos
        res = dictionary_match("PACLITAXIL", min_similarity=0.85)
        assert res is not None
        assert res.normalized_term == "Paclitaxel"

    def test_unmatched_returns_none(self):
        # Non-medical random terms should return None, avoiding false positives
        unmatched_terms = ["xyz123", "random_string", "unrelated_word", "12345", ""]
        for term in unmatched_terms:
            assert dictionary_match(term) is None


class TestRegexParsers:
    """Tests for structured clinical regex parsers (TNM staging, biomarkers, grading)."""

    def test_tnm_staging_standard(self):
        text = "Patient diagnosed with carcinoma breast T2N0M0 post biopsy."
        entities = regex_parsers(text)
        staging = [e for e in entities if e.entity_type == "Staging"]
        assert len(staging) >= 1
        assert "T2N0M0" in staging[0].normalized_term
        assert staging[0].method == "regex"

    def test_tnm_staging_with_prefixes(self):
        cases = [
            ("Assessment: rT2N1M0 carcinoma left breast", "RT2N1M0"),
            ("Post-op: pT3N1aM0 invasive ductal carcinoma", "PT3N1AM0"),
            ("Staging: T3NoMo in 2004", "T3N0M0"),
            ("T1cN0Mx noted on PET", "T1CN0MX"),
        ]
        for text, expected_code in cases:
            entities = regex_parsers(text)
            staging = [e for e in entities if e.entity_type == "Staging"]
            assert len(staging) >= 1, f"Expected staging in: {text}"
            assert staging[0].metadata["staging_code"] == expected_code

    def test_biomarker_er_pr_scores(self):
        text = "IHC Report: ER 8/8, PR 0/8, Her2neu 1+"
        entities = regex_parsers(text)
        biomarkers = {e.metadata.get("marker"): e for e in entities if e.entity_type == "Biomarker"}
        
        assert "ER" in biomarkers
        assert biomarkers["ER"].metadata["value"] == "8/8"
        assert biomarkers["ER"].method == "regex"

        assert "PR" in biomarkers
        assert biomarkers["PR"].metadata["value"] == "0/8"

        assert "HER2" in biomarkers
        assert biomarkers["HER2"].metadata["value"] == "1+"

    def test_biomarker_her2_variants(self):
        cases = [
            ("HER2 3+", "3+"),
            ("HER-2/neu: 3+", "3+"),
            ("Her2neu 1+", "1+"),
            ("HER2 positive", "positive"),
        ]
        for text, expected_val in cases:
            entities = regex_parsers(text)
            her2_ents = [e for e in entities if e.metadata and e.metadata.get("marker") == "HER2"]
            assert len(her2_ents) >= 1, f"Expected HER2 match in: {text}"
            assert her2_ents[0].metadata["value"] == expected_val

    def test_biomarker_ki67(self):
        text = "Ki67 12-14% and Ki-67: 20%"
        entities = regex_parsers(text)
        ki67_ents = [e for e in entities if e.metadata and e.metadata.get("marker") == "Ki-67"]
        assert len(ki67_ents) >= 1
        assert "12-14%" in ki67_ents[0].metadata["value"]

    def test_nottingham_histologic_grade(self):
        text = "Modified Bloom Richardson Score: Total Score - 8, Grade 3"
        entities = regex_parsers(text)
        grading = [e for e in entities if e.entity_type == "Staging" and "Grade" in e.normalized_term]
        assert len(grading) >= 1
        assert grading[0].metadata["grade"] == "3"
        assert grading[0].metadata["score"] == "8"


class TestNormalizeChunkText:
    """End-to-end tests for normalize_chunk_text across sample paragraphs."""

    def test_normalize_clinical_report_chunk(self):
        raw_text = (
            "PATIENT WITH IDC NST T2N0M0.\\n"
            "UNDERWENT LEFT MRM UNDER ANAESTHESIA.\\n"
            "IHC STATUS: ER 8/8, PR 8/8, HER2 3+, Ki67 15%.\\n"
            "PLAN: ADJUVANT CHEMOTHERAPY WITH EC REGIMEN FOLLOWED BY PACLITAXEL."
        )
        entities = normalize_chunk_text(raw_text, base_confidence=0.95)
        terms = {e.normalized_term for e in entities}
        types = {e.entity_type for e in entities}

        assert "Invasive Ductal Carcinoma" in terms
        assert "Modified Radical Mastectomy" in terms
        assert "Epirubicin + Cyclophosphamide" in terms
        assert "Paclitaxel" in terms
        assert any("TNM Staging: T2N0M0" in t for t in terms)
        assert any("Estrogen Receptor: 8/8" in t for t in terms)
        assert any("HER2: 3+" in t for t in terms)
        assert "Biomarker" in types
        assert "Procedure" in types
        assert "Regimen" in types

    def test_normalize_hindi_consent_chunk(self):
        raw_text = "डा, संस्थान रोटरी अभा.आ.सं. सहमति पत्र रोगी के हस्ताक्षर"
        entities = normalize_chunk_text(raw_text, script="devanagari")
        terms = {e.normalized_term for e in entities}
        assert "Consent Form" in terms
        assert "Patient Signature" in terms

    def test_normalize_date_helper(self):
        assert normalize_date("Date: 28-May-2014") == "2014-05-28"
        assert normalize_date("Reg. Date: 08/12/2016") == "2016-12-08"


class TestNormalizedEntitySerialization:
    """Tests for NormalizedEntity dataclass serialization."""

    def test_serialization_roundtrip(self):
        ent = NormalizedEntity(
            raw_text="MRM",
            normalized_term="Modified Radical Mastectomy",
            entity_type="Procedure",
            method="dictionary",
            confidence=0.95,
            metadata={"source": "test"},
        )
        d = ent.to_dict()
        assert d["raw_text"] == "MRM"
        assert d["normalized_term"] == "Modified Radical Mastectomy"
        assert d["entity_type"] == "Procedure"

        restored = NormalizedEntity.from_dict(d)
        assert restored == ent
