"""
Medical Records RAG Demo — Streamlit Web Application (SRS §9, BUILD_GUIDE Phase 5)

A locally hosted Streamlit application enabling clinical RAG over patient records:
- Dual Roles: Individual (patient self-lookup) & Group (hospital staff multi-patient review)
- Dual Backends: GraphRAG on Neo4j & PageIndex (pure LLM reasoning tree)
- Interactive Chat Interface with Evidence Tracking & Grounded Citations

Maps to BUILD_GUIDE Task 5.1 & Task 5.2 (Chat Interface Wired to Orchestration).
"""

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from orchestrate import answer_question
from patients import PATIENTS, all_patient_ids, get_display_label


# -----------------------------------------------------------------------------
# 1. Helper Functions (Modular & Testable)
# -----------------------------------------------------------------------------

def get_label_to_id_mapping() -> Dict[str, str]:
    """Returns a reverse dictionary mapping display labels to patient IDs."""
    return {label: pid for pid, label in PATIENTS.items()}


def resolve_selected_patients(
    mode: str,
    selected_label_or_labels: Any,
    all_patients_checked: bool = False,
) -> List[str]:
    """
    Resolves UI patient selection into a clean list of patient IDs (SRS FR-9.1.1, FR-8.1.1).
    
    - In Individual mode: Maps the single chosen display label to [patient_id].
    - In Group mode: If 'All patients' is checked, resolves to all registered patient IDs;
      otherwise maps selected multiselect labels to a list of patient IDs.
    """
    label_to_id = get_label_to_id_mapping()
    norm_mode = mode.strip().lower()

    if norm_mode == "individual":
        if isinstance(selected_label_or_labels, str):
            pid = label_to_id.get(selected_label_or_labels, selected_label_or_labels)
            return [pid] if pid else []
        elif isinstance(selected_label_or_labels, list) and selected_label_or_labels:
            pid = label_to_id.get(selected_label_or_labels[0], selected_label_or_labels[0])
            return [pid] if pid else []
        return []

    # Group mode
    if all_patients_checked:
        return all_patient_ids()

    if isinstance(selected_label_or_labels, list):
        resolved = []
        for label in selected_label_or_labels:
            pid = label_to_id.get(label, label)
            if pid:
                resolved.append(pid)
        return resolved
    elif isinstance(selected_label_or_labels, str) and selected_label_or_labels:
        pid = label_to_id.get(selected_label_or_labels, selected_label_or_labels)
        return [pid] if pid else []

    return []


def resolve_backend(backend_label: str) -> str:
    """
    Maps UI radio option to backend code identifier ('graph' or 'pageindex').
    """
    if "graph" in backend_label.lower():
        return "graph"
    return "pageindex"


def init_session_state() -> None:
    """
    Initializes session state variables for chat history and app lifecycle (SRS FR-9.1.4).
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat_history() -> None:
    """Clears the conversational history in the current session."""
    st.session_state.messages = []


# -----------------------------------------------------------------------------
# 2. Sidebar Component (SRS §9.1, FR-9.1.1)
# -----------------------------------------------------------------------------

def render_sidebar() -> Tuple[str, List[str], str, bool]:
    """
    Renders sidebar controls top-to-bottom exactly per SRS FR-9.1.1:
    1. Mode dropdown (Individual / Group)
    2. Patient selector (Single select or Multi-select + 'All patients' checkbox)
    3. Backend toggle (GraphRAG / PageIndex)
    4. Optional 'Compare both backends' checkbox

    Returns:
        Tuple of (mode, selected_patients, backend, compare_both).
    """
    st.sidebar.title("🏥 Medical Records RAG")
    st.sidebar.caption("Clinical Document Intelligence & Evidence Explorer")
    st.sidebar.markdown("---")

    # 1. Mode dropdown
    mode = st.sidebar.selectbox(
        "Mode",
        options=["Individual", "Group"],
        index=0,
        help="Individual mode scopes queries to one patient. Group mode allows multi-patient comparison.",
    )

    # 2. Patient selector
    display_names = list(PATIENTS.values())
    selected_patients: List[str] = []

    if mode == "Individual":
        selected_label = st.sidebar.selectbox(
            "Patient",
            options=display_names,
            index=0,
            help="Select the single patient whose medical records you wish to view.",
        )
        selected_patients = resolve_selected_patients(mode, selected_label)
    else:  # Group mode
        all_patients = st.sidebar.checkbox(
            "All patients",
            value=False,
            help="Select all registered patients in the hospital database.",
        )
        if all_patients:
            st.sidebar.info(f"Targeting all {len(all_patient_ids())} patients.")
            selected_patients = all_patient_ids()
        else:
            selected_labels = st.sidebar.multiselect(
                "Patients",
                options=display_names,
                default=display_names[:1] if display_names else [],
                help="Select one or more patients to query or compare.",
            )
            selected_patients = resolve_selected_patients(mode, selected_labels, all_patients_checked=False)

    st.sidebar.markdown("---")

    # 3. Backend toggle
    backend_choice = st.sidebar.radio(
        "Backend",
        options=["GraphRAG (Neo4j)", "PageIndex"],
        index=0,
        help="GraphRAG queries structured Neo4j knowledge graph. PageIndex traverses hierarchical document tree via LLM reasoning.",
    )
    backend = resolve_backend(backend_choice)

    # 4. Optional 'Compare both backends' checkbox
    compare_both = st.sidebar.checkbox(
        "Compare both backends",
        value=False,
        help="Executes queries simultaneously across both GraphRAG and PageIndex for side-by-side comparison.",
    )

    st.sidebar.markdown("---")

    # Clear chat affordance
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    # Scope Summary Badge in Sidebar
    st.sidebar.markdown("### 📋 Active Scope")
    st.sidebar.markdown(f"**Mode:** `{mode}`")
    st.sidebar.markdown(
        f"**Patients:** `{', '.join(get_display_label(p) for p in selected_patients) if selected_patients else 'None'}`"
    )
    st.sidebar.markdown(f"**Backend:** `{'GraphRAG + PageIndex' if compare_both else backend_choice}`")

    return mode, selected_patients, backend, compare_both


# -----------------------------------------------------------------------------
# 3. Chat Interface & Orchestration Component (SRS §9.1.2, Task 5.2)
# -----------------------------------------------------------------------------

def render_message_content(msg: Dict[str, Any]) -> None:
    """
    Renders an individual message from history, handling both single-backend
    and dual-backend comparison layouts.
    """
    role = msg.get("role", "user")

    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
            return

        # Assistant message rendering
        if msg.get("is_comparison"):
            # Dual-Backend Comparison Layout
            graph_res = msg.get("graph_result", {})
            pi_res = msg.get("pageindex_result", {})

            st.markdown("#### ⚖️ Side-by-Side Backend Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🕸️ GraphRAG (Neo4j)")
                st.markdown(graph_res.get("answer", "No response generated."))
                citations_g = graph_res.get("citations", [])
                if citations_g:
                    st.caption(f"📎 **{len(citations_g)} citation(s):** {', '.join(f'`{c}`' for c in citations_g)}")

            with col2:
                st.markdown("##### 🌲 PageIndex (Tree Traversal)")
                st.markdown(pi_res.get("answer", "No response generated."))
                citations_pi = pi_res.get("citations", [])
                if citations_pi:
                    st.caption(f"📎 **{len(citations_pi)} citation(s):** {', '.join(f'`{c}`' for c in citations_pi)}")

        else:
            # Single Backend Layout
            backend_used = msg.get("backend_used", "engine")
            backend_label = "GraphRAG (Neo4j)" if backend_used == "graph" else "PageIndex (Tree)"
            st.caption(f"⚙️ Generated via **{backend_label}**")
            st.markdown(msg.get("content", ""))

            citations = msg.get("citations", [])
            if citations:
                st.caption(f"📎 **Grounded Citations ({len(citations)}):** {', '.join(f'`{c}`' for c in citations)}")


def execute_query(
    question: str,
    mode: str,
    selected_patients: List[str],
    backend: str,
    compare_both: bool,
) -> None:
    """
    Executes user query through orchestrate.answer_question, manages UI spinners,
    and records results in session state (SRS FR-9.1.2).
    """
    if not selected_patients:
        st.warning("⚠️ Please select at least one patient in the sidebar before asking a question.")
        return

    # 1. Append and display user message
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Execute retrieval & answer generation
    with st.chat_message("assistant"):
        try:
            if compare_both:
                with st.spinner("Analyzing records with both GraphRAG and PageIndex engines..."):
                    graph_res = answer_question(
                        question=question,
                        mode=mode.lower(),
                        selected_patients=selected_patients,
                        backend="graph",
                    )
                    pi_res = answer_question(
                        question=question,
                        mode=mode.lower(),
                        selected_patients=selected_patients,
                        backend="pageindex",
                    )

                assistant_msg = {
                    "role": "assistant",
                    "is_comparison": True,
                    "graph_result": graph_res,
                    "pageindex_result": pi_res,
                    "mode": mode,
                    "selected_patients": selected_patients,
                }
                st.session_state.messages.append(assistant_msg)
                render_message_content(assistant_msg)

            else:
                backend_label = "GraphRAG (Neo4j)" if backend == "graph" else "PageIndex (Tree)"
                with st.spinner(f"Querying {backend_label} & verifying evidence citations..."):
                    res = answer_question(
                        question=question,
                        mode=mode.lower(),
                        selected_patients=selected_patients,
                        backend=backend,
                    )

                assistant_msg = {
                    "role": "assistant",
                    "is_comparison": False,
                    "content": res.get("answer", ""),
                    "citations": res.get("citations", []),
                    "backend_used": backend,
                    "mode": mode,
                    "selected_patients": selected_patients,
                }
                st.session_state.messages.append(assistant_msg)
                render_message_content(assistant_msg)

        except Exception as exc:
            st.error(f"❌ An error occurred during retrieval or answer generation: {exc}")


def render_chat_interface(
    mode: str,
    selected_patients: List[str],
    backend: str,
    compare_both: bool,
) -> None:
    """
    Renders the chat history, sample query buttons, and active chat input (SRS FR-9.1.2).
    """
    # Render existing conversation history
    for msg in st.session_state.messages:
        render_message_content(msg)

    # Chat Input Box
    prompt = st.chat_input("Ask a clinical question about the medical records...")
    if prompt:
        execute_query(
            question=prompt.strip(),
            mode=mode,
            selected_patients=selected_patients,
            backend=backend,
            compare_both=compare_both,
        )


# -----------------------------------------------------------------------------
# 4. Main Entry Point
# -----------------------------------------------------------------------------

def main():
    # Page configuration
    st.set_page_config(
        page_title="Medical Records RAG (Demo)",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state (SRS FR-9.1.4)
    init_session_state()

    # Render sidebar controls (SRS FR-9.1.1)
    mode, selected_patients, backend, compare_both = render_sidebar()

    # Header & Metric Badges
    st.title("Medical Records Retrieval-Augmented Generation")
    st.markdown(
        "Clinical question-answering across scanned hospital records, pathology reports, "
        "oncology flowsheets, and discharge summaries with **verifiable citation evidence**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Active Mode", value=mode)
    with col2:
        patient_str = ", ".join(get_display_label(p) for p in selected_patients) if selected_patients else "None"
        st.metric(label="Scoped Patient(s)", value=patient_str)
    with col3:
        backend_str = "Both (Comparison)" if compare_both else ("GraphRAG (Neo4j)" if backend == "graph" else "PageIndex (Tree)")
        st.metric(label="Active Engine", value=backend_str)

    st.markdown("---")

    # Render interactive chat interface (SRS FR-9.1.2, Task 5.2)
    render_chat_interface(
        mode=mode,
        selected_patients=selected_patients,
        backend=backend,
        compare_both=compare_both,
    )


if __name__ == "__main__":
    main()
