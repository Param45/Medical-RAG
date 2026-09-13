"""
Tests for Streamlit App Skeleton & Sidebar Resolution Logic (SRS §9.1, BUILD_GUIDE Task 5.1).
"""

import pytest
from app import (
    get_label_to_id_mapping,
    resolve_backend,
    resolve_selected_patients,
)
from patients import PATIENTS, all_patient_ids


class TestAppSidebarResolution:
    """Tests for patient and backend resolution helpers in app.py."""

    def test_label_to_id_mapping(self):
        mapping = get_label_to_id_mapping()
        assert mapping["Patient A"] == "patient_a"
        assert mapping["Patient B"] == "patient_b"

    def test_resolve_individual_mode(self):
        res_a = resolve_selected_patients("Individual", "Patient A")
        assert res_a == ["patient_a"]

        res_b = resolve_selected_patients("Individual", "Patient B")
        assert res_b == ["patient_b"]

        res_direct_id = resolve_selected_patients("Individual", "patient_a")
        assert res_direct_id == ["patient_a"]

    def test_resolve_group_mode_multiselect(self):
        res = resolve_selected_patients("Group", ["Patient A", "Patient B"])
        assert res == ["patient_a", "patient_b"]

        res_single = resolve_selected_patients("Group", ["Patient B"])
        assert res_single == ["patient_b"]

    def test_resolve_group_mode_all_patients_override(self):
        # 'All patients' checkbox overrides whatever is in multiselect
        res_all = resolve_selected_patients(
            mode="Group",
            selected_label_or_labels=["Patient A"],
            all_patients_checked=True,
        )
        assert res_all == all_patient_ids()
        assert "patient_a" in res_all
        assert "patient_b" in res_all

    def test_resolve_backend(self):
        assert resolve_backend("GraphRAG (Neo4j)") == "graph"
        assert resolve_backend("Graph") == "graph"
        assert resolve_backend("PageIndex") == "pageindex"
        assert resolve_backend("PageIndex (Tree)") == "pageindex"
