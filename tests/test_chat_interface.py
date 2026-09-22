"""
Tests for Streamlit Chat Interface & Orchestration Wiring (SRS §9.1.2, BUILD_GUIDE Task 5.2).
"""

from unittest.mock import MagicMock, patch
import pytest

from app import (
    execute_query,
    init_session_state,
    clear_chat_history,
)


class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_streamlit():
    """Mocks Streamlit session state and chat components."""
    with patch("app.st") as mock_st:
        session_state = MockSessionState()
        mock_st.session_state = session_state
        mock_st.chat_message.return_value.__enter__ = MagicMock(return_value=mock_st)
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=mock_st)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=None)
        yield mock_st


class TestChatInterfaceWiring:
    """Tests for chat execution, session state mutation, and backend routing."""

    def test_init_and_clear_session_state(self, mock_streamlit):
        init_session_state()
        assert mock_streamlit.session_state.messages == []

        mock_streamlit.session_state.messages.append({"role": "user", "content": "Hi"})
        assert len(mock_streamlit.session_state.messages) == 1

        clear_chat_history()
        assert mock_streamlit.session_state.messages == []

    def test_execute_query_no_patients_shows_warning(self, mock_streamlit):
        init_session_state()
        execute_query(
            question="What is my diagnosis?",
            mode="Individual",
            selected_patients=[],
            backend="graph",
            compare_both=False,
        )
        mock_streamlit.warning.assert_called_once()
        assert len(mock_streamlit.session_state.messages) == 0

    @patch("app.answer_question")
    def test_execute_query_single_backend(self, mock_answer, mock_streamlit):
        init_session_state()
        mock_answer.return_value = {
            "answer": "Diagnosed with IDC Grade 3 [ev_1].",
            "citations": ["ev_1"],
            "backend_used": "graph",
        }

        execute_query(
            question="What is my diagnosis?",
            mode="Individual",
            selected_patients=["patient_a"],
            backend="graph",
            compare_both=False,
        )

        mock_answer.assert_called_once_with(
            question="What is my diagnosis?",
            mode="individual",
            selected_patients=["patient_a"],
            backend="graph",
        )

        messages = mock_streamlit.session_state.messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is my diagnosis?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["is_comparison"] is False
        assert messages[1]["content"] == "Diagnosed with IDC Grade 3 [ev_1]."
        assert messages[1]["citations"] == ["ev_1"]

    @patch("app.answer_question")
    def test_execute_query_comparison_mode(self, mock_answer, mock_streamlit):
        init_session_state()
        mock_answer.side_effect = [
            {"answer": "Graph answer [ev_g]", "citations": ["ev_g"], "backend_used": "graph"},
            {"answer": "PageIndex answer [ev_pi]", "citations": ["ev_pi"], "backend_used": "pageindex"},
        ]

        execute_query(
            question="What is my hemoglobin trend?",
            mode="Individual",
            selected_patients=["patient_a"],
            backend="graph",
            compare_both=True,
        )

        assert mock_answer.call_count == 2
        messages = mock_streamlit.session_state.messages
        assert len(messages) == 2
        assert messages[1]["is_comparison"] is True
        assert messages[1]["graph_result"]["answer"] == "Graph answer [ev_g]"
        assert messages[1]["pageindex_result"]["answer"] == "PageIndex answer [ev_pi]"

    @patch("app.answer_question")
    def test_execute_query_error_handling(self, mock_answer, mock_streamlit):
        init_session_state()
        mock_answer.side_effect = RuntimeError("Database connection timeout")

        execute_query(
            question="What is my diagnosis?",
            mode="Individual",
            selected_patients=["patient_a"],
            backend="graph",
            compare_both=False,
        )

        mock_streamlit.error.assert_called_once()
        assert "Database connection timeout" in str(mock_streamlit.error.call_args)

    @patch("app.answer_question")
    def test_execute_query_default_compare_both(self, mock_answer, mock_streamlit):
        init_session_state()
        mock_answer.side_effect = [
            {"answer": "Graph answer", "citations": [], "backend_used": "graph"},
            {"answer": "PageIndex answer", "citations": [], "backend_used": "pageindex"},
        ]

        execute_query(
            question="What is my condition?",
            mode="Individual",
            selected_patients=["patient_a"],
        )

        assert mock_answer.call_count == 2
        messages = mock_streamlit.session_state.messages
        assert len(messages) == 2
        assert messages[1]["is_comparison"] is True

