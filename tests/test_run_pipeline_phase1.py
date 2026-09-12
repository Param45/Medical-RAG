"""
Unit tests for End-to-End Phase 1 Pipeline Runner (Task 1.8).
"""

from pathlib import Path
import pytest

from run_pipeline_phase1 import run_phase1_pipeline


class TestPhase1PipelineRunner:
    """Tests for run_pipeline_phase1.py end-to-end execution."""

    def test_run_phase1_pipeline_execution(self):
        summary = run_phase1_pipeline()

        assert "totals" in summary
        assert "patients" in summary
        assert "patient_a" in summary["patients"]
        assert "patient_b" in summary["patients"]

        # Check patient_a metrics
        p_a = summary["patients"]["patient_a"]
        assert p_a["pages"] >= 30
        assert p_a["reports"] >= 10
        assert p_a["chunks"] >= 30
        assert p_a["evidence"] == p_a["chunks"]
        assert p_a["entities"] >= 50

        # Check patient_b metrics
        p_b = summary["patients"]["patient_b"]
        assert p_b["pages"] >= 35
        assert p_b["reports"] >= 10
        assert p_b["chunks"] >= 35
        assert p_b["evidence"] == p_b["chunks"]
        assert p_b["entities"] >= 50

        # Check total metrics
        totals = summary["totals"]
        assert totals["total_pages"] == p_a["pages"] + p_b["pages"]
        assert totals["total_reports"] == p_a["reports"] + p_b["reports"]
        assert totals["total_chunks"] == p_a["chunks"] + p_b["chunks"]
        assert totals["total_evidence"] == p_a["evidence"] + p_b["evidence"]

        # Check that artifacts exist on disk
        assert Path("data/reports/patient_a.json").exists()
        assert Path("data/reports/patient_b.json").exists()
        assert Path("data/evidence/patient_a.json").exists()
        assert Path("data/evidence/patient_b.json").exists()
        assert Path("data/reports/patient_a_chunks.json").exists()
        assert Path("data/reports/patient_b_chunks.json").exists()
        assert Path("data/_phase1_summary.md").exists()
