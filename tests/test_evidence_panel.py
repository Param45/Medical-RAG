"""
Unit tests for Task 5.3: Evidence Panel in Streamlit UI (SRS §9.1.3, FR-9.1.3).
"""

from unittest.mock import MagicMock, patch
import pytest

from app import (
    get_confidence_badge_markdown,
    parse_patient_id_from_evidence_id,
    render_citations_list,
    render_evidence_expander,
)
from evidence_store import EvidenceRecord


def test_confidence_badge_markdown():
    """Verify confidence badge markdown matches SRS FR-9.1.3 thresholds."""
    # High confidence (>= 0.8)
    badge_high = get_confidence_badge_markdown(0.95)
    assert ":green[" in badge_high
    assert "High confidence" in badge_high
    assert "95%" in badge_high

    badge_boundary_high = get_confidence_badge_markdown(0.8)
    assert ":green[" in badge_boundary_high

    # Medium confidence (0.5 <= c < 0.8)
    badge_med = get_confidence_badge_markdown(0.65)
    assert ":orange[" in badge_med
    assert "Medium confidence" in badge_med
    assert "65%" in badge_med

    badge_boundary_med = get_confidence_badge_markdown(0.5)
    assert ":orange[" in badge_boundary_med

    # Low confidence (< 0.5)
    badge_low = get_confidence_badge_markdown(0.42)
    assert ":red[" in badge_low
    assert "Low confidence" in badge_low
    assert "42%" in badge_low


def test_parse_patient_id_from_evidence_id():
    """Verify extraction of patient_id from standard evidence_id strings."""
    assert parse_patient_id_from_evidence_id("patient_a__report_1__page_3__chunk_0") == "patient_a"
    assert parse_patient_id_from_evidence_id("patient_b__rep_flowsheet__page_10__chunk_2") == "patient_b"
    assert parse_patient_id_from_evidence_id("") == ""


@patch("app.get_evidence_by_id")
@patch("streamlit.expander")
@patch("streamlit.columns")
@patch("streamlit.tabs")
@patch("streamlit.image")
@patch("streamlit.code")
@patch("streamlit.markdown")
def test_render_evidence_expander_found(
    mock_md, mock_code, mock_img, mock_tabs, mock_cols, mock_expander, mock_get_ev
):
    """Verify rendering an evidence expander when the record is found."""
    # Setup mock context manager
    exp_cm = MagicMock()
    exp_cm.__enter__.return_value = exp_cm
    exp_cm.__exit__.return_value = None
    mock_expander.return_value = exp_cm

    col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
    for col in (col1, col2, col3):
        col.__enter__.return_value = col
        col.__exit__.return_value = None
    mock_cols.return_value = (col1, col2, col3)

    tab1, tab2 = MagicMock(), MagicMock()
    for tab in (tab1, tab2):
        tab.__enter__.return_value = tab
        tab.__exit__.return_value = None
    mock_tabs.return_value = (tab1, tab2)

    fake_record = EvidenceRecord(
        evidence_id="patient_a__report_1__page_1__chunk_0",
        patient_id="patient_a",
        report_id="report_1",
        report_type="HISTOPATHOLOGY",
        report_date="2023-01-15",
        page_number=1,
        raw_text="Infiltrating duct carcinoma, Grade III",
        source_type="typed",
        confidence=0.95,
        page_image_path="data/pages/patient_a/page_001.png",
    )
    mock_get_ev.return_value = fake_record

    render_evidence_expander("patient_a__report_1__page_1__chunk_0")

    mock_get_ev.assert_called_once_with(
        patient_id="patient_a",
        evidence_id="patient_a__report_1__page_1__chunk_0",
    )
    mock_expander.assert_called_once()
    mock_code.assert_called_once_with("Infiltrating duct carcinoma, Grade III", language=None)


@patch("app.get_evidence_by_id")
@patch("streamlit.expander")
@patch("streamlit.warning")
def test_render_evidence_expander_missing(mock_warning, mock_expander, mock_get_ev):
    """Verify rendering a warning when the evidence record is not found."""
    exp_cm = MagicMock()
    exp_cm.__enter__.return_value = exp_cm
    exp_cm.__exit__.return_value = None
    mock_expander.return_value = exp_cm

    mock_get_ev.return_value = None

    render_evidence_expander("patient_a__missing_id")
    mock_warning.assert_called_once()


@patch("app.render_evidence_expander")
@patch("streamlit.markdown")
def test_render_citations_list(mock_md, mock_render_exp):
    """Verify render_citations_list renders each citation."""
    citations = ["patient_a__chunk_0", "patient_a__chunk_1"]
    render_citations_list(citations)

    assert mock_render_exp.call_count == 2
    mock_render_exp.assert_any_call("patient_a__chunk_0")
    mock_render_exp.assert_any_call("patient_a__chunk_1")


@patch("app.render_evidence_expander")
def test_render_citations_list_empty(mock_render_exp):
    """Verify render_citations_list does nothing on empty list."""
    render_citations_list([])
    mock_render_exp.assert_not_called()


def test_extract_and_strip_evidence_brackets():
    """Verify extraction of evidence IDs and clean stripping of raw brackets."""
    from app import extract_evidence_ids_from_line, strip_evidence_brackets

    line = "Metastatic Breast Cancer [patient_a__report_0022__page_30__chunk_0, patient_a__report_0022__page_32__chunk_0]"
    ids = extract_evidence_ids_from_line(line)
    assert ids == ["patient_a__report_0022__page_30__chunk_0", "patient_a__report_0022__page_32__chunk_0"]

    clean = strip_evidence_brackets(line)
    assert clean == "Metastatic Breast Cancer"

    plain_line = "No citations here [just standard text]"
    assert extract_evidence_ids_from_line(plain_line) == []
    assert strip_evidence_brackets(plain_line) == "No citations here [just standard text]"



