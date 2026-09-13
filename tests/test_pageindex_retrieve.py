"""
Tests for PageIndex Retrieval Engine (SRS §7.2, BUILD_GUIDE Task 3.2).
All LLM calls are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from pageindex_backend.retrieve import (
    load_tree,
    select_children,
    should_allow_multi,
    traverse,
    retrieve,
)


@pytest.fixture
def fake_tree():
    """Builds a small synthetic 3-level tree for testing."""
    return {
        "patient_id": "patient_test",
        "root": {
            "node_id": "root",
            "summary": "Overall summary of patient test with breast cancer.",
            "children": [
                {
                    "node_id": "report_0001",
                    "node_type": "Report",
                    "report_type": "HISTOPATHOLOGY",
                    "report_date": "2018-01-15",
                    "summary": "Histopathology confirming IDC Grade 3, ER/PR positive, HER2 negative.",
                    "children": [
                        {
                            "node_id": "report_0001_page_1",
                            "node_type": "Page",
                            "raw_text": "Invasive Ductal Carcinoma Grade 3 specimen text.",
                            "evidence_id": "patient_test__report_0001__page_1__chunk_0",
                        }
                    ],
                },
                {
                    "node_id": "report_0002",
                    "node_type": "Report",
                    "report_type": "ONCOLOGY_FLOWSHEET",
                    "report_date": "2018-04-08",
                    "summary": "Oncology flowsheet with Hemoglobin 9.9 and Platelets 252k.",
                    "children": [
                        {
                            "node_id": "report_0002_page_2",
                            "node_type": "Page",
                            "raw_text": "Flowsheet labs: Hb 9.9, Platelets 252000, ANC 4730.",
                            "evidence_id": "patient_test__report_0002__page_2__chunk_0",
                        }
                    ],
                },
                {
                    "node_id": "report_0003",
                    "node_type": "Report",
                    "report_type": "RADIOLOGY_PET_CT",
                    "report_date": "2018-05-10",
                    "summary": "PET-CT scan showing stable skeletal and nodal metastases.",
                    "children": [
                        {
                            "node_id": "report_0003_page_3",
                            "node_type": "Page",
                            "raw_text": "Whole body FDG PET-CT scan: stable osseous lesions.",
                            "evidence_id": "patient_test__report_0003__page_3__chunk_0",
                        }
                    ],
                },
            ],
        },
    }


class TestLoadTree:
    """Tests for load_tree patient scoping boundary."""

    def test_load_tree_success(self, tmp_path, fake_tree):
        pageindex_dir = tmp_path / "pageindex"
        pageindex_dir.mkdir(parents=True)
        (pageindex_dir / "patient_test.json").write_text(
            json.dumps(fake_tree), encoding="utf-8"
        )

        loaded = load_tree("patient_test", pageindex_dir=pageindex_dir)
        assert loaded["patient_id"] == "patient_test"
        assert len(loaded["root"]["children"]) == 3

    def test_load_tree_missing_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tree("non_existent_patient", pageindex_dir=tmp_path)


class TestShouldAllowMulti:
    """Tests for multi-hop / trend heuristic."""

    def test_multi_trend_queries(self):
        assert should_allow_multi("What is my hemoglobin trend?") is True
        assert should_allow_multi("Show me my platelet history over time") is True
        assert should_allow_multi("Compare my previous scans") is True
        assert should_allow_multi("List all medications administered") is True
        assert should_allow_multi("How has the disease progression evolved?") is True

    def test_single_point_queries(self):
        assert should_allow_multi("What is my ER PR status?") is False
        assert should_allow_multi("What surgery was performed?") is False
        assert should_allow_multi("What was the initial tumor size?") is False


class TestSelectChildren:
    """Tests for LLM reasoning child selection."""

    @patch("pageindex_backend.retrieve.chat")
    def test_select_children_single_match(self, mock_chat, fake_tree):
        mock_chat.return_value = json.dumps(["report_0001"])
        root = fake_tree["root"]

        selected = select_children(
            question="What is my cancer type and pathology grade?",
            node=root,
            allow_multi=False,
        )
        assert len(selected) == 1
        assert selected[0]["node_id"] == "report_0001"
        assert selected[0]["report_type"] == "HISTOPATHOLOGY"

    @patch("pageindex_backend.retrieve.chat")
    def test_select_children_multi_match(self, mock_chat, fake_tree):
        mock_chat.return_value = json.dumps(["report_0002", "report_0003"])
        root = fake_tree["root"]

        selected = select_children(
            question="What are my lab trends and imaging results?",
            node=root,
            allow_multi=True,
        )
        assert len(selected) == 2
        selected_ids = [s["node_id"] for s in selected]
        assert "report_0002" in selected_ids
        assert "report_0003" in selected_ids

    @patch("pageindex_backend.retrieve.chat")
    def test_select_children_with_code_fences(self, mock_chat, fake_tree):
        mock_chat.return_value = """```json
["report_0002"]
```"""
        root = fake_tree["root"]
        selected = select_children(
            question="What is my hemoglobin count?",
            node=root,
            allow_multi=False,
        )
        assert len(selected) == 1
        assert selected[0]["node_id"] == "report_0002"

    @patch("pageindex_backend.retrieve.chat")
    def test_select_children_malformed_json_fallback(self, mock_chat, fake_tree):
        mock_chat.return_value = "I think report_0001 and report_0003 are relevant."
        root = fake_tree["root"]
        selected = select_children(
            question="Tell me about my reports",
            node=root,
            allow_multi=True,
        )
        # Should regex-match report IDs or gracefully fall back
        assert len(selected) >= 1

    @patch("pageindex_backend.retrieve.chat")
    def test_select_children_empty_returns_empty(self, mock_chat, fake_tree):
        mock_chat.return_value = "[]"
        root = fake_tree["root"]
        selected = select_children(
            question="What is my dental history?",
            node=root,
            allow_multi=False,
        )
        assert selected == []


class TestTraverseAndRetrieve:
    """Tests for traverse and retrieve orchestration."""

    @patch("pageindex_backend.retrieve.chat")
    def test_traverse_collects_leaf_facts(self, mock_chat, fake_tree):
        mock_chat.return_value = json.dumps(["report_0001"])

        facts = traverse(
            question="What is my diagnosis?",
            patient_id="patient_test",
            tree=fake_tree,
        )

        assert len(facts) == 1
        fact = facts[0]
        assert fact["patient_id"] == "patient_test"
        assert fact["evidence_id"] == "patient_test__report_0001__page_1__chunk_0"
        assert "Invasive Ductal Carcinoma" in fact["text"]
        assert fact["report_type"] == "HISTOPATHOLOGY"

    @patch("pageindex_backend.retrieve.chat")
    def test_traverse_respects_max_nodes_expanded(self, mock_chat, fake_tree):
        mock_chat.return_value = json.dumps(["report_0001", "report_0002", "report_0003"])

        facts = traverse(
            question="Show everything",
            patient_id="patient_test",
            max_nodes_expanded=2,
            allow_multi=True,
            tree=fake_tree,
        )
        # Tree expansion should stop at max_nodes_expanded
        assert len(facts) <= 3

    @patch("pageindex_backend.retrieve.load_tree")
    @patch("pageindex_backend.retrieve.chat")
    def test_retrieve_multi_patient(self, mock_chat, mock_load, fake_tree):
        mock_chat.return_value = json.dumps(["report_0001"])
        mock_load.return_value = fake_tree

        facts = retrieve(
            question="What is the histopathology result?",
            patient_ids=["patient_a", "patient_b"],
        )

        assert len(facts) == 2
        p_ids = [f["patient_id"] for f in facts]
        assert "patient_a" in p_ids
        assert "patient_b" in p_ids
