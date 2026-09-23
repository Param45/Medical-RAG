"""
Unit tests for User Medical Report Upload & Ephemeral Session Sandbox.
Replaces legacy admin rebuild tests.
"""

from unittest.mock import MagicMock, patch
import pytest

from app import render_user_report_tab
from graph_backend.build import TEMP_NEO4J_CONFIG, get_driver_for_patient
from patients import PATIENTS, is_temp_patient, register_patient, unregister_patient
from temp_session import (
    TEMP_PATIENT_ID,
    cleanup_temp_patient,
    get_temp_patient_paths,
    ingest_user_report,
)


def test_is_temp_patient_and_registration():
    """Verify temporary patient registration and detection."""
    test_pid = "temp_test_patient"
    assert not is_temp_patient("patient_a")
    assert not is_temp_patient("patient_b")
    assert is_temp_patient(test_pid)

    register_patient(test_pid, "Test Temporary Patient", is_temp=True)
    assert is_temp_patient(test_pid)
    assert PATIENTS[test_pid] == "Test Temporary Patient"

    unregister_patient(test_pid)
    assert test_pid not in PATIENTS


@patch("graph_backend.build.get_driver")
def test_driver_routing_for_temp_patient(mock_get_driver):
    """Verify get_driver_for_patient routes to temporary Neo4j DB for temp patients."""
    # Test regular patient uses default .env driver
    get_driver_for_patient("patient_a")
    mock_get_driver.assert_called_with()

    mock_get_driver.reset_mock()

    # Test temp patient uses temporary credentials
    get_driver_for_patient("temp_user_report")
    mock_get_driver.assert_called_with(
        uri=TEMP_NEO4J_CONFIG["uri"],
        username=TEMP_NEO4J_CONFIG["username"],
        password=TEMP_NEO4J_CONFIG["password"],
    )


@patch("temp_session.cleanup_temp_patient")
@patch("temp_session.build_pageindex_for_patient")
@patch("temp_session.build_graph_for_patient")
@patch("temp_session.run_schema_init")
@patch("temp_session.get_driver")
@patch("temp_session.chunk_patient")
@patch("temp_session.split_patient_reports")
@patch("temp_session.ocr_all_pages")
@patch("temp_session.render_pdf_to_pages")
@patch("temp_session.load_evidence")
def test_ingest_user_report_pipeline(
    mock_load_ev,
    mock_render_pdf,
    mock_ocr,
    mock_split,
    mock_chunk,
    mock_get_driver,
    mock_run_schema,
    mock_build_graph,
    mock_build_pi,
    mock_cleanup,
):
    """Verify ingest_user_report runs all pipeline stages sequentially into isolated DB."""
    # Setup mocks
    mock_render_pdf.return_value = ["page_1.png", "page_2.png"]
    mock_ocr.return_value = [MagicMock(), MagicMock()]
    mock_report = MagicMock()
    mock_report.to_dict.return_value = {"report_id": "r1", "report_type": "LAB"}
    mock_split.return_value = [mock_report]
    mock_chunk.return_value = [MagicMock(), MagicMock(), MagicMock()]
    mock_load_ev.return_value = [MagicMock(), MagicMock()]
    
    mock_driver_instance = MagicMock()
    mock_get_driver.return_value = mock_driver_instance
    mock_build_graph.return_value = {"triples_written": 25}
    mock_build_pi.return_value = MagicMock()

    test_pdf_content = b"%PDF-1.4 dummy pdf content"
    progress = []

    try:
        summary = ingest_user_report(
            pdf_bytes=test_pdf_content,
            original_filename="sample_cbc_report.pdf",
            patient_id="temp_test_ingest",
            progress_callback=progress.append,
        )

        assert summary["pages_count"] == 2
        assert summary["chunks_count"] == 3
        assert summary["triples_written"] == 25
        assert summary["patient_id"] == "temp_test_ingest"
        assert "sample_cbc_report" in summary["display_label"].lower() or "Sample Cbc Report" in summary["display_label"]

        # Check driver was called with temporary Neo4j credentials
        mock_get_driver.assert_called_with(
            uri=TEMP_NEO4J_CONFIG["uri"],
            username=TEMP_NEO4J_CONFIG["username"],
            password=TEMP_NEO4J_CONFIG["password"],
        )
        mock_run_schema.assert_called_with(mock_driver_instance)
        mock_build_graph.assert_called_with(patient_id="temp_test_ingest", driver=mock_driver_instance)
        mock_driver_instance.close.assert_called_once()
        mock_build_pi.assert_called_with(patient_id="temp_test_ingest")

    finally:
        unregister_patient("temp_test_ingest")
        raw_p = get_temp_patient_paths("temp_test_ingest")["raw_pdf"]
        if raw_p.exists():
            raw_p.unlink()


@patch("temp_session.wipe_temp_neo4j_database")
def test_cleanup_temp_patient(mock_wipe_neo4j):
    """Verify cleanup_temp_patient removes disk paths and calls Neo4j purge."""
    mock_wipe_neo4j.return_value = 10
    test_pid = "temp_test_cleanup"

    # Create dummy files
    paths = get_temp_patient_paths(test_pid)
    paths["raw_pdf"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw_pdf"].write_text("dummy")
    paths["reports_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["reports_json"].write_text("[]")
    register_patient(test_pid, "Cleanup Test", is_temp=True)

    result = cleanup_temp_patient(test_pid)

    assert result["patient_id"] == test_pid
    assert result["neo4j_nodes_deleted"] == 10
    assert not paths["raw_pdf"].exists()
    assert not paths["reports_json"].exists()
    assert test_pid not in PATIENTS
    mock_wipe_neo4j.assert_called_once()
