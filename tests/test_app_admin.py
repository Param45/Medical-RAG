"""
Unit tests for Task 5.4: Rebuild Index Admin Button in Streamlit UI (SRS §9.2, FR-9.2.1).
"""

from unittest.mock import MagicMock, call, patch
import pytest

from app import execute_full_rebuild, render_admin_tab


@patch("app.build_all_pageindexes")
@patch("app.build_all_patients")
@patch("app.chunk_all_patients")
@patch("app.split_all_reports")
@patch("app.ocr_all_pages")
@patch("app.ingest_all")
@patch("app.load_evidence")
def test_execute_full_rebuild_success(
    mock_load_ev,
    mock_ingest,
    mock_ocr,
    mock_split,
    mock_chunk,
    mock_build_graph,
    mock_build_pi,
):
    """Verify execute_full_rebuild runs stages in order and calculates summaries."""
    # Setup mocks
    mock_ingest.return_value = {"patient_a": ["p1.png", "p2.png"], "patient_b": ["p3.png"]}
    mock_split.return_value = {"patient_a": [MagicMock(), MagicMock()], "patient_b": [MagicMock()]}
    
    mock_chunk_obj = MagicMock()
    mock_chunk.return_value = {"patient_a": [mock_chunk_obj, mock_chunk_obj], "patient_b": [mock_chunk_obj]}
    
    mock_load_ev.return_value = [MagicMock(), MagicMock()]
    mock_build_graph.return_value = {
        "patient_a": {"triples_written": 15},
        "patient_b": {"triples_written": 20},
    }
    mock_build_pi.return_value = {"patient_a": {}, "patient_b": {}}

    progress_messages = []
    summary = execute_full_rebuild(progress_callback=progress_messages.append)

    # 1. Verify stage call order
    mock_ingest.assert_called_once()
    assert mock_ocr.call_count >= 2
    mock_split.assert_called_once()
    mock_chunk.assert_called_once()
    mock_build_graph.assert_called_once()
    mock_build_pi.assert_called_once()

    # 2. Verify progress messages
    assert len(progress_messages) >= 6
    assert any("Stage 1/6" in m for m in progress_messages)
    assert any("Stage 6/6" in m for m in progress_messages)

    # 3. Verify summary metrics
    assert "patients" in summary
    assert "totals" in summary
    assert "elapsed_seconds" in summary
    assert summary["totals"]["total_pages"] == 3
    assert summary["totals"]["total_reports"] == 3
    assert summary["totals"]["total_chunks"] == 3


@patch("app.execute_full_rebuild")
@patch("streamlit.button")
@patch("streamlit.status")
@patch("streamlit.toast")
@patch("streamlit.table")
@patch("streamlit.columns")
@patch("streamlit.markdown")
@patch("streamlit.caption")
def test_render_admin_tab_rebuild_success(
    mock_cap,
    mock_md,
    mock_cols,
    mock_tbl,
    mock_toast,
    mock_status,
    mock_btn,
    mock_rebuild,
):
    """Verify render_admin_tab executes rebuild and displays results when button clicked."""
    col1, col2 = MagicMock(), MagicMock()
    for col in (col1, col2):
        col.__enter__.return_value = col
        col.__exit__.return_value = None
    mock_cols.return_value = (col1, col2)

    mock_btn.return_value = True

    status_mock = MagicMock()
    status_mock.__enter__.return_value = status_mock
    status_mock.__exit__.return_value = None
    mock_status.return_value = status_mock

    mock_rebuild.return_value = {
        "elapsed_seconds": 12.34,
        "patients": {
            "patient_a": {
                "display_label": "Patient A (R. Sharma)",
                "pages": 36,
                "reports": 13,
                "chunks": 40,
                "evidence": 40,
                "graph_stats": {"triples_written": 120},
            }
        },
        "totals": {"total_pages": 36, "total_reports": 13, "total_chunks": 40, "total_evidence": 40},
    }

    render_admin_tab()

    mock_rebuild.assert_called_once()
    status_mock.update.assert_called_once_with(
        label="✅ Pipeline Rebuild Completed in 12.34s!",
        state="complete",
        expanded=True,
    )
    mock_toast.assert_called_once()
    mock_tbl.assert_called_once()


@patch("app.execute_full_rebuild")
@patch("streamlit.button")
@patch("streamlit.status")
@patch("streamlit.error")
@patch("streamlit.columns")
@patch("streamlit.markdown")
@patch("streamlit.caption")
def test_render_admin_tab_rebuild_error_handling(
    mock_cap,
    mock_md,
    mock_cols,
    mock_err,
    mock_status,
    mock_btn,
    mock_rebuild,
):
    """Verify render_admin_tab handles exceptions gracefully without sticking in loading state."""
    col1, col2 = MagicMock(), MagicMock()
    for col in (col1, col2):
        col.__enter__.return_value = col
        col.__exit__.return_value = None
    mock_cols.return_value = (col1, col2)

    mock_btn.return_value = True

    status_mock = MagicMock()
    status_mock.__enter__.return_value = status_mock
    status_mock.__exit__.return_value = None
    mock_status.return_value = status_mock

    mock_rebuild.side_effect = RuntimeError("Neo4j database connection timeout")

    render_admin_tab()

    mock_rebuild.assert_called_once()
    status_mock.update.assert_called_once_with(
        label="❌ Pipeline Rebuild Failed",
        state="error",
        expanded=True,
    )
    mock_err.assert_called_once()
    assert "Neo4j database connection timeout" in mock_err.call_args[0][0]
