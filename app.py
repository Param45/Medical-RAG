"""
Medical Records RAG Demo — Streamlit Web Application (SRS §9, BUILD_GUIDE Phase 5)

A locally hosted Streamlit application enabling clinical RAG over patient records:
- Dual Roles: Individual (patient self-lookup) & Group (hospital staff multi-patient review)
- Dual Backends: GraphRAG on Neo4j & PageIndex (pure LLM reasoning tree)
- Interactive Chat Interface with Evidence Tracking & Grounded Citations

Maps to BUILD_GUIDE Tasks 5.1, 5.2 & 5.3 (Evidence Panel).
"""

import os
from pathlib import Path
import re
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

from chunk import chunk_all_patients
from evidence_store import EvidenceRecord, get_evidence_by_id, load_evidence
from graph_backend.build import build_all_patients, run_schema_init
from ingest import ingest_all
from ocr import ocr_all_pages
from orchestrate import answer_question
from pageindex_backend.build import build_all_pageindexes
from patients import PATIENTS, all_patient_ids, get_display_label, is_temp_patient
from split_reports import split_all_reports
from temp_session import (
    TEMP_PATIENT_ID,
    TEMP_NEO4J_CONFIG,
    cleanup_temp_patient,
    ingest_user_report,
)


# -----------------------------------------------------------------------------
# 1. Helper Functions & Evidence Store Lookups (SRS §9.1.3, Task 5.3)
# -----------------------------------------------------------------------------

def get_label_to_id_mapping() -> Dict[str, str]:
    """Returns a reverse dictionary mapping display labels to patient IDs."""
    return {label: pid for pid, label in PATIENTS.items()}


def get_chat_avatar(role: str) -> str:
    """Returns clean non-emoji Material Symbol avatar for chat messages."""
    return ":material/person:" if role == "user" else ":material/smart_toy:"



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
    Maps UI backend selection to code identifier ('graph', 'pageindex', or 'both').
    """
    lbl = backend_label.lower()
    if "both" in lbl or "compare" in lbl:
        return "both"
    if "graph" in lbl:
        return "graph"
    return "pageindex"


def get_confidence_badge_markdown(confidence: float) -> str:
    """
    Returns colored Streamlit markdown for confidence rating (SRS FR-9.1.3):
    - Green >= 0.8: High confidence
    - Yellow / Orange 0.5 <= confidence < 0.8: Medium confidence
    - Red < 0.5: Low confidence
    """
    conf_pct = f"{confidence * 100:.0f}%"
    if confidence >= 0.8:
        return f":green[● High confidence ({conf_pct})]"
    elif confidence >= 0.5:
        return f":orange[● Medium confidence ({conf_pct})]"
    else:
        return f":red[● Low confidence ({conf_pct})]"


def parse_patient_id_from_evidence_id(evidence_id: str) -> str:
    """
    Extracts the patient_id from an evidence_id formatted as `{patient_id}__...`.
    """
    if not evidence_id:
        return ""
    parts = evidence_id.split("__")
    return parts[0] if parts else ""


def init_session_state() -> None:
    """
    Initializes session state variables for chat history and app lifecycle (SRS FR-9.1.4).
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_messages" not in st.session_state:
        st.session_state.temp_messages = []
    if "temp_report_active" not in st.session_state:
        st.session_state.temp_report_active = False
    if "temp_report_summary" not in st.session_state:
        st.session_state.temp_report_summary = None


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
    st.sidebar.title("Medical Records RAG")
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

    # If temporary report was recently ingested, default to it
    default_patient_index = 0
    if TEMP_PATIENT_ID in PATIENTS:
        temp_label = PATIENTS[TEMP_PATIENT_ID]
        if temp_label in display_names:
            default_patient_index = display_names.index(temp_label)

    if mode == "Individual":
        selected_label = st.sidebar.selectbox(
            "Patient",
            options=display_names,
            index=default_patient_index,
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

    # Backend mode is fixed to dual comparison across both engines
    backend = "both"
    compare_both = True

    # Clear chat affordance
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    # Scope Summary Badge in Sidebar
    st.sidebar.markdown("### Active Scope")
    st.sidebar.markdown(f"**Mode:** `{mode}`")
    st.sidebar.markdown(
        f"**Patients:** `{', '.join(get_display_label(p) for p in selected_patients) if selected_patients else 'None'}`"
    )
    st.sidebar.markdown("**Backend:** `Compare both Backends`")

    # Hardware compute device indicator
    from device_utils import get_device_info
    dev_info = get_device_info()
    if dev_info["is_gpu_available"]:
        st.sidebar.markdown(f"**Compute:** `GPU ({dev_info['device_name']})`")
    else:
        st.sidebar.markdown("**Compute:** `CPU Mode`")

    # Temporary Neo4j indicator badge if active
    if any(is_temp_patient(p) for p in selected_patients):
        st.sidebar.info("Temporary Neo4j Sandbox: `af2857f2`")

    # End Temporary Session button if temporary report is active
    if TEMP_PATIENT_ID in PATIENTS:
        st.sidebar.markdown("---")
        if st.sidebar.button("End Session & Clear Report", use_container_width=True, help="Permanently wipes all uploaded files and temporary Neo4j records."):
            cleanup_temp_patient(TEMP_PATIENT_ID)
            st.session_state.temp_report_active = False
            st.session_state.temp_report_summary = None
            st.session_state.temp_messages = []
            st.toast("Temporary session data and Neo4j records permanently cleared.")
            st.rerun()

    return mode, selected_patients, backend, compare_both


# -----------------------------------------------------------------------------
# 3. Evidence Viewer Component (SRS §9.1.3, Task 5.3)
# -----------------------------------------------------------------------------

EVIDENCE_BRACKET_REGEX = re.compile(
    r"\[([a-zA-Z0-9_-]+__[a-zA-Z0-9_ -]+(?:__chunk_\d+)?(?:\s*,\s*[a-zA-Z0-9_-]+__[a-zA-Z0-9_ -]+(?:__chunk_\d+)?)*)\]"
)


def extract_evidence_ids_from_line(line: str) -> List[str]:
    """
    Extracts all clean evidence IDs from a line containing bracketed citations.
    """
    matches = EVIDENCE_BRACKET_REGEX.findall(line)
    cits: List[str] = []
    for m in matches:
        for item in m.split(","):
            cleaned = item.strip()
            if "__" in cleaned and cleaned not in cits:
                cits.append(cleaned)
    return cits


def strip_evidence_brackets(line: str) -> str:
    """
    Removes raw [patient_id__...] citation brackets from line text for clean reading.
    """
    return EVIDENCE_BRACKET_REGEX.sub("", line).rstrip()


def render_evidence_expander(evidence_id: str) -> None:
    """
    Renders an interactive evidence expander displaying the source page image,
    raw OCR text, report metadata, and confidence badge (SRS FR-9.1.3).
    """
    patient_id = parse_patient_id_from_evidence_id(evidence_id)
    record: Optional[EvidenceRecord] = None
    if patient_id:
        record = get_evidence_by_id(patient_id=patient_id, evidence_id=evidence_id)

    with st.expander(f"Source Evidence: `{evidence_id}`", expanded=False):
        if record is None:
            st.warning(f"Evidence record `{evidence_id}` could not be found in the local evidence store.")
            return

        # Top Metadata Banner
        meta_col1, meta_col2, meta_col3 = st.columns([1.5, 1.5, 1.2])
        with meta_col1:
            st.markdown(f"**Report Type:** `{record.report_type}`")
            st.markdown(f"**Patient:** `{get_display_label(record.patient_id)}` ({record.patient_id})")
        with meta_col2:
            date_str = record.report_date or "Undated"
            st.markdown(f"**Report Date:** `{date_str}`")
            st.markdown(f"**Page:** `{record.page_number}` (Source: `{record.source_type}`)")
        with meta_col3:
            st.markdown("**Confidence:**")
            st.markdown(get_confidence_badge_markdown(record.confidence))

        st.markdown("---")

        # Evidence Tabs: Raw OCR Text vs Source Page Scan
        tab_raw, tab_image = st.tabs(["Raw OCR Text", "Original Page Scan"])

        with tab_raw:
            st.caption("Verbatim extracted text from source document (no paraphrasing):")
            st.code(record.raw_text, language=None)

        with tab_image:
            image_rel_path = record.page_image_path
            image_abs_path = _ROOT_DIR / image_rel_path if image_rel_path else None

            if image_abs_path and image_abs_path.exists():
                st.image(
                    str(image_abs_path),
                    caption=f"{get_display_label(record.patient_id)} — Report {record.report_id} (Page {record.page_number})",
                    use_container_width=True,
                )
            else:
                st.info(f"Page scan image not found on disk at: `{image_rel_path}`")


def render_citations_list(citations: List[str]) -> None:
    """
    Renders an expandable list of source evidence panels for an assistant message.
    """
    if not citations:
        return

    st.markdown(f"##### Source Evidence Trail ({len(citations)} item{'s' if len(citations) > 1 else ''})")
    for ev_id in citations:
        render_evidence_expander(ev_id)


# -----------------------------------------------------------------------------
# 4. Chat Interface & Orchestration Component (SRS §9.1.2, §9.1.3)
# -----------------------------------------------------------------------------

def render_message_content(msg: Dict[str, Any]) -> None:
    """
    Renders an individual message from history, handling both single-backend
    and dual-backend comparison layouts.
    """
    role = msg.get("role", "user")

    with st.chat_message(role, avatar=get_chat_avatar(role)):
        if role == "user":
            st.markdown(msg.get("content", ""))
            return

        # Assistant message rendering
        if msg.get("is_comparison"):
            # Dual-Backend Comparison Layout
            graph_res = msg.get("graph_result", {})
            pi_res = msg.get("pageindex_result", {})

            st.markdown("#### Side-by-Side Backend Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### GraphRAG (Neo4j)")
                st.markdown(graph_res.get("answer", "No response generated."))
                citations_g = graph_res.get("citations", [])
                if citations_g:
                    st.caption(f"Citations ({len(citations_g)}): {', '.join(f'`{c}`' for c in citations_g)}")

            with col2:
                st.markdown("##### PageIndex (Tree Traversal)")
                st.markdown(pi_res.get("answer", "No response generated."))
                citations_pi = pi_res.get("citations", [])
                if citations_pi:
                    st.caption(f"Citations ({len(citations_pi)}): {', '.join(f'`{c}`' for c in citations_pi)}")

        else:
            # Single Backend Layout
            backend_used = msg.get("backend_used", "engine")
            backend_label = "GraphRAG (Neo4j)" if backend_used == "graph" else "PageIndex (Tree)"
            st.caption(f"Generated via **{backend_label}**")
            st.markdown(msg.get("content", ""))

            citations = msg.get("citations", [])
            if citations:
                st.caption(f"Citations ({len(citations)}): {', '.join(f'`{c}`' for c in citations)}")


def execute_query(
    question: str,
    mode: str,
    selected_patients: List[str],
    backend: str = "both",
    compare_both: bool = True,
) -> None:
    """
    Executes user query through orchestrate.answer_question, manages UI spinners,
    and records results in session state (SRS FR-9.1.2).
    """
    if not selected_patients:
        st.warning("Please select at least one patient in the sidebar before asking a question.")
        return

    # 1. Append and display user message
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user", avatar=get_chat_avatar("user")):
        st.markdown(question)

    # 2. Execute retrieval & answer generation
    with st.chat_message("assistant", avatar=get_chat_avatar("assistant")):
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
            st.error(f"An error occurred during retrieval or answer generation: {exc}")


def render_chat_interface(
    mode: str,
    selected_patients: List[str],
    backend: str = "both",
    compare_both: bool = True,
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
# 5. User Report Upload & Ephemeral Session Component
# -----------------------------------------------------------------------------

def render_user_report_tab() -> None:
    """
    Renders the 'Upload & Query My Report' section where users can upload their own medical
    report PDF, run the full document intelligence pipeline into an isolated temporary
    Neo4j database (af2857f2), query their report with grounded citations, and clean up
    the entire session upon completion.
    """
    st.markdown("### Upload & Analyze Your Medical Report")
    st.markdown(
        "Upload a scanned or digital medical report (PDF). The system executes the full end-to-end "
        "pipeline: OCR extraction, report boundary detection, clinical entity normalization, "
        "knowledge graph construction in a **temporary isolated Neo4j database**, and hierarchical "
        "reasoning tree generation."
    )

    # Privacy & Isolation Notice Card
    st.info(
        "🔒 **Session Sandbox & Data Privacy:** All uploaded reports, OCR text, and knowledge graphs "
        f"are stored strictly for the duration of this active session in an isolated Neo4j database "
        f"(`{TEMP_NEO4J_CONFIG['username']}`). Once your session ends, all files and graph records are "
        "permanently cleared from disk and the database."
    )

    is_report_active = st.session_state.get("temp_report_active", False) and (TEMP_PATIENT_ID in PATIENTS)

    if is_report_active:
        summary = st.session_state.get("temp_report_summary") or {}
        display_label = get_display_label(TEMP_PATIENT_ID)

        # Active Report Banner
        st.success(f"Active Medical Report: **{display_label}**")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Pages Extracted", summary.get("pages_count", "—"))
        with mcol2:
            st.metric("Reports Identified", summary.get("reports_count", "—"))
        with mcol3:
            st.metric("Clinical Chunks", summary.get("chunks_count", "—"))
        with mcol4:
            st.metric("Neo4j Triples (Isolated)", summary.get("triples_written", "—"))

        # Session Cleanup Button
        col_clean, _ = st.columns([1.5, 2])
        with col_clean:
            if st.button("End Session & Clear Report Data", type="secondary", use_container_width=True, help="Permanently wipes all files and graph records for this report."):
                with st.spinner("Clearing temporary session data and purging isolated Neo4j database..."):
                    cleanup_temp_patient(TEMP_PATIENT_ID)
                    st.session_state.temp_report_active = False
                    st.session_state.temp_report_summary = None
                    st.session_state.temp_messages = []
                st.toast("Temporary session data and Neo4j records permanently cleared.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Ask Questions About Your Report")
        st.caption("Ask questions about your diagnoses, test results, prescribed medications, or treatment plan.")

        # Suggested Questions
        st.markdown("**Sample Questions:**")
        sample_cols = st.columns(3)
        sample_q = None
        if sample_cols[0].button("What is my diagnosis?", use_container_width=True):
            sample_q = "What is my primary diagnosis and clinical condition?"
        if sample_cols[1].button("Are there abnormal lab results?", use_container_width=True):
            sample_q = "Are there any abnormal lab test results documented in my report?"
        if sample_cols[2].button("What treatments were given?", use_container_width=True):
            sample_q = "What treatments, medications, or chemotherapy regimens are documented?"

        # Render conversation history for this report
        if "temp_messages" not in st.session_state:
            st.session_state.temp_messages = []

        for msg in st.session_state.temp_messages:
            render_message_content(msg)

        # Question input
        prompt = st.chat_input("Ask a question about your uploaded report...", key="temp_chat_input")
        active_query = sample_q or prompt

        if active_query:
            # Append and display user message
            user_msg = {"role": "user", "content": active_query}
            st.session_state.temp_messages.append(user_msg)

            with st.chat_message("user", avatar=get_chat_avatar("user")):
                st.markdown(active_query)

            with st.chat_message("assistant", avatar=get_chat_avatar("assistant")):
                with st.spinner("Analyzing your report with GraphRAG (isolated Neo4j) and PageIndex engines..."):
                    try:
                        graph_res = answer_question(
                            question=active_query,
                            mode="individual",
                            selected_patients=[TEMP_PATIENT_ID],
                            backend="graph",
                        )
                        pi_res = answer_question(
                            question=active_query,
                            mode="individual",
                            selected_patients=[TEMP_PATIENT_ID],
                            backend="pageindex",
                        )

                        assistant_msg = {
                            "role": "assistant",
                            "is_comparison": True,
                            "graph_result": graph_res,
                            "pageindex_result": pi_res,
                            "mode": "individual",
                            "selected_patients": [TEMP_PATIENT_ID],
                        }
                        st.session_state.temp_messages.append(assistant_msg)
                        render_message_content(assistant_msg)

                        # Render citations expandable view
                        all_cits = list(dict.fromkeys(graph_res.get("citations", []) + pi_res.get("citations", [])))
                        if all_cits:
                            st.markdown("---")
                            render_citations_list(all_cits)

                    except Exception as exc:
                        st.error(f"Error analyzing report: {exc}")

    else:
        # Upload Form
        st.markdown("#### Step 1: Select Your Medical Report")
        uploaded_file = st.file_uploader(
            "Upload Medical Report PDF",
            type=["pdf"],
            help="Select a scanned or digital PDF document.",
        )

        if uploaded_file is not None:
            st.info(f"Selected file: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

            ingest_btn = st.button(
                "Process & Ingest Medical Report",
                type="primary",
                use_container_width=True,
                help="Runs full ingestion pipeline: OCR -> Splitting -> Chunking -> Temporary Neo4j Graph -> PageIndex.",
            )

            if ingest_btn:
                status_container = st.status("Initializing report ingestion pipeline...", expanded=True)
                try:
                    summary = ingest_user_report(
                        pdf_bytes=uploaded_file.getvalue(),
                        original_filename=uploaded_file.name,
                        patient_id=TEMP_PATIENT_ID,
                        progress_callback=status_container.write,
                    )
                    status_container.update(
                        label=f"Ingestion Completed in {summary['elapsed_seconds']}s!",
                        state="complete",
                        expanded=True,
                    )
                    st.session_state.temp_report_active = True
                    st.session_state.temp_report_summary = summary
                    st.toast("Report ingested and ready for clinical queries!")
                    st.rerun()
                except Exception as exc:
                    status_container.update(
                        label="Report Ingestion Failed",
                        state="error",
                        expanded=True,
                    )
                    st.error(f"An error occurred during report ingestion: {exc}")


# -----------------------------------------------------------------------------
# 6. Main Entry Point
# -----------------------------------------------------------------------------

def main():
    # Page configuration
    st.set_page_config(
        page_title="Medical Records RAG (Demo)",
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
        "Clinical question-answering across scanned hospital records, lab reports, "
        "clinical flowsheets, imaging studies, and discharge summaries with **verifiable citation evidence**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Active Mode", value=mode)
    with col2:
        patient_str = ", ".join(get_display_label(p) for p in selected_patients) if selected_patients else "None"
        st.metric(label="Scoped Patient(s)", value=patient_str)

    st.markdown("---")

    # Main Application Navigation: Clinical Chat vs User Report Upload
    tab_chat, tab_upload = st.tabs(["Clinical Chat", "Upload & Query My Report"])

    with tab_chat:
        render_chat_interface(
            mode=mode,
            selected_patients=selected_patients,
            backend=backend,
            compare_both=compare_both,
        )

    with tab_upload:
        render_user_report_tab()


if __name__ == "__main__":
    main()
