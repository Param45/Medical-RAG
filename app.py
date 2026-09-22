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
from patients import PATIENTS, all_patient_ids, get_display_label
from split_reports import split_all_reports


# -----------------------------------------------------------------------------
# 1. Helper Functions & Evidence Store Lookups (SRS §9.1.3, Task 5.3)
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


def render_evidence_popover_content(evidence_ids: List[str]) -> None:
    """
    Renders detailed evidence in a compact, attached popover container (SRS §9.1.3):
    - Metadata summary (Report Type, Date, Page, Source Type)
    - Confidence badge
    - Raw OCR text tab
    - Original page scan image tab
    """
    for idx, ev_id in enumerate(evidence_ids):
        if idx > 0:
            st.markdown("---")

        st.markdown(f"**Evidence ID:** `{ev_id}`")
        patient_id = parse_patient_id_from_evidence_id(ev_id)
        record: Optional[EvidenceRecord] = None
        if patient_id:
            record = get_evidence_by_id(patient_id=patient_id, evidence_id=ev_id)

        if record is None:
            st.warning(f"⚠️ Evidence record `{ev_id}` not found in evidence store.")
            continue

        # Metadata banner
        st.markdown(f"**Report Type:** `{record.report_type}` | **Page:** `{record.page_number}`")
        st.markdown(f"**Date:** `{record.report_date or 'Undated'}` | **Source:** `{record.source_type}`")
        st.markdown(f"**Confidence:** {get_confidence_badge_markdown(record.confidence)}")

        # Dual Evidence Tabs: Raw OCR vs Scanned Image
        tab_raw, tab_image = st.tabs(["📝 Raw OCR Text", "🖼️ Original Scan"])

        with tab_raw:
            st.caption("Verbatim extracted OCR text:")
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


def render_answer_with_evidence_popovers(answer_text: str) -> None:
    """
    Renders an answer where citation brackets [patient_id__...] are replaced by
    an interactive [📄 Evidence] button / popover next to each backed point (SRS §9.1.3).
    Clicking the button opens an attached pop-up box with source details without covering the screen.
    """
    if not answer_text:
        st.markdown("No response generated.")
        return

    lines = answer_text.splitlines()
    for line in lines:
        if not line.strip():
            st.write("")
            continue

        cits = extract_evidence_ids_from_line(line)
        if cits:
            cleaned_line = strip_evidence_brackets(line)
            col_text, col_btn = st.columns([0.82, 0.18])
            with col_text:
                st.markdown(cleaned_line)
            with col_btn:
                btn_label = f"📄 Evidence ({len(cits)})" if len(cits) > 1 else "📄 Evidence"
                with st.popover(btn_label, use_container_width=True):
                    render_evidence_popover_content(cits)
        else:
            st.markdown(line)


def render_evidence_expander(evidence_id: str) -> None:
    """
    Renders an interactive evidence expander displaying the source page image,
    raw OCR text, report metadata, and confidence badge (SRS FR-9.1.3).
    """
    patient_id = parse_patient_id_from_evidence_id(evidence_id)
    record: Optional[EvidenceRecord] = None
    if patient_id:
        record = get_evidence_by_id(patient_id=patient_id, evidence_id=evidence_id)

    with st.expander(f"📄 Source Evidence: `{evidence_id}`", expanded=False):
        if record is None:
            st.warning(f"⚠️ Evidence record `{evidence_id}` could not be found in the local evidence store.")
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
        tab_raw, tab_image = st.tabs(["📝 Raw OCR Text", "🖼️ Original Page Scan"])

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

    st.markdown(f"##### 🔍 Source Evidence Trail ({len(citations)} item{'s' if len(citations) > 1 else ''})")
    for ev_id in citations:
        render_evidence_expander(ev_id)


# -----------------------------------------------------------------------------
# 4. Chat Interface & Orchestration Component (SRS §9.1.2, §9.1.3)
# -----------------------------------------------------------------------------

def render_message_content(msg: Dict[str, Any]) -> None:
    """
    Renders an individual message from history, handling both single-backend
    and dual-backend comparison layouts along with interactive evidence popovers.
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
                render_answer_with_evidence_popovers(graph_res.get("answer", "No response generated."))

            with col2:
                st.markdown("##### 🌲 PageIndex (Tree Traversal)")
                render_answer_with_evidence_popovers(pi_res.get("answer", "No response generated."))

        else:
            # Single Backend Layout
            backend_used = msg.get("backend_used", "engine")
            backend_label = "GraphRAG (Neo4j)" if backend_used == "graph" else "PageIndex (Tree)"
            st.caption(f"⚙️ Generated via **{backend_label}**")
            render_answer_with_evidence_popovers(msg.get("content", ""))


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
    # Render existing conversation history or prompt suggestions
    if not st.session_state.messages:
        st.info("💡 **Clinical Question Suggestions across Domains:**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("- 🩺 **Diagnoses & History**: *What primary illnesses or medical conditions are documented?*")
            st.markdown("- 🩸 **Lab Trends & Vitals**: *What are the latest Hemoglobin, Creatinine, and Blood Sugar values?*")
        with col_s2:
            st.markdown("- 💊 **Medications & Therapy**: *What medications or treatment regimens were administered?*")
            st.markdown("- 🔬 **Tests & Diagnostics**: *What diagnostic procedures or scans were performed or advised?*")
    else:
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
# 5. Pipeline Rebuild & Admin Component (SRS §9.2, FR-9.2.1, Task 5.4)
# -----------------------------------------------------------------------------

def execute_full_rebuild(progress_callback: Optional[Any] = None) -> Dict[str, Any]:
    """
    Executes the entire Medical Records RAG pipeline end-to-end (SRS §13, Task 5.4):
    1. Ingest raw PDFs into page images (ingest.ingest_all)
    2. OCR extraction per patient (ocr.ocr_all_pages)
    3. Split reports by anchor detection (split_reports.split_all_reports)
    4. Chunking, normalization & evidence store (chunk.chunk_all_patients)
    5. GraphRAG: schema init & build knowledge graph in Neo4j (graph_backend.build.build_all_patients)
    6. PageIndex: build hierarchical trees (pageindex_backend.build.build_all_pageindexes)

    Returns a comprehensive execution metrics summary dictionary.
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    start_time = time.time()
    p_ids = all_patient_ids()

    # Stage 1: Ingestion
    log("Stage 1/6: Ingesting raw PDFs from `data/raw/` into page images...")
    ingest_res = ingest_all()

    # Stage 2: OCR
    log("Stage 2/6: Running OCR and layout analysis across all patient pages...")
    for pid in p_ids:
        ocr_all_pages(pid)

    # Stage 3: Split Reports
    log("Stage 3/6: Detecting report boundaries and anchoring document spans...")
    reports_res = split_all_reports()

    # Stage 4: Chunking & Evidence
    log("Stage 4/6: Generating text chunks, normalizing clinical terms, and updating evidence store...")
    chunks_res = chunk_all_patients()

    # Stage 5: GraphRAG Knowledge Graph Build
    log("Stage 5/6: Initializing Neo4j schema & constructing GraphRAG knowledge graph...")
    graph_res = build_all_patients()

    # Stage 6: PageIndex Tree Build
    log("Stage 6/6: Generating LLM summaries and building PageIndex hierarchical trees...")
    pageindex_res = build_all_pageindexes()

    elapsed = round(time.time() - start_time, 2)
    log(f"All stages completed successfully in {elapsed}s.")

    # Calculate summary metrics
    summary: Dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "patients": {},
        "totals": {
            "total_pages": 0,
            "total_reports": 0,
            "total_chunks": 0,
            "total_evidence": 0,
        },
    }

    for pid in p_ids:
        label = get_display_label(pid)
        page_paths = ingest_res.get(pid, [])
        pages_count = len(page_paths) if page_paths else len(list((_ROOT_DIR / "data" / "pages" / pid).glob("page_*.png")))
        reports_count = len(reports_res.get(pid, []))
        chunks = chunks_res.get(pid, [])
        chunks_count = len(chunks)
        evidence_records = load_evidence(pid)
        evidence_count = len(evidence_records)

        summary["patients"][pid] = {
            "display_label": label,
            "pages": pages_count,
            "reports": reports_count,
            "chunks": chunks_count,
            "evidence": evidence_count,
            "graph_stats": graph_res.get(pid, {}),
        }

        summary["totals"]["total_pages"] += pages_count
        summary["totals"]["total_reports"] += reports_count
        summary["totals"]["total_chunks"] += chunks_count
        summary["totals"]["total_evidence"] += evidence_count

    return summary


def render_admin_tab() -> None:
    """
    Renders the Administrator management tab with one-click pipeline rebuild (SRS FR-9.2.1).
    """
    st.markdown("### ⚙️ System Administration & Index Management")
    st.markdown(
        "Manage the underlying index stores and trigger complete end-to-end data pipeline rebuilds. "
        "Rebuilding re-processes all raw PDFs in `data/raw/`, refreshes OCR extraction, resets chunking & evidence stores, "
        "re-synchronizes the Neo4j Knowledge Graph, and regenerates PageIndex document trees."
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🔄 Full Pipeline Rebuild")
        st.caption(
            "This will sequentially execute all pipeline stages: Ingestion ➔ OCR ➔ Report Splitting ➔ "
            "Chunking ➔ GraphRAG (Neo4j) ➔ PageIndex Tree."
        )

        rebuild_btn = st.button(
            "🚀 Rebuild Index from data/raw/",
            type="primary",
            use_container_width=True,
            help="Re-runs all pipeline stages end-to-end for all patients.",
        )

        if rebuild_btn:
            status_container = st.status("Initializing end-to-end pipeline rebuild...", expanded=True)
            try:
                summary = execute_full_rebuild(progress_callback=status_container.write)
                status_container.update(
                    label=f"✅ Pipeline Rebuild Completed in {summary['elapsed_seconds']}s!",
                    state="complete",
                    expanded=True,
                )
                st.toast("✅ Index rebuilt successfully for all patients!", icon="🎉")

                # Display summary table
                st.markdown("#### 📊 Rebuild Execution Summary")
                summary_rows = []
                for pid, pdata in summary["patients"].items():
                    summary_rows.append({
                        "Patient ID": pid,
                        "Display Label": pdata["display_label"],
                        "Pages Extracted": pdata["pages"],
                        "Reports Detected": pdata["reports"],
                        "Chunks Created": pdata["chunks"],
                        "Evidence Records": pdata["evidence"],
                        "Graph Triples Written": pdata["graph_stats"].get("triples_written", "N/A"),
                    })

                st.table(summary_rows)

            except Exception as exc:
                status_container.update(
                    label="❌ Pipeline Rebuild Failed",
                    state="error",
                    expanded=True,
                )
                st.error(f"❌ An error occurred during index rebuilding: {exc}")

    with col2:
        st.markdown("#### 📁 Local Index Status")
        for pid in all_patient_ids():
            label = get_display_label(pid)
            ev_list = load_evidence(pid)
            pi_file = _ROOT_DIR / "data" / "pageindex" / f"{pid}.json"
            pi_exists = "✅ Available" if pi_file.exists() else "❌ Missing"

            st.markdown(f"**{label}** (`{pid}`)")
            st.markdown(f"- Evidence Records: `{len(ev_list)}`")
            st.markdown(f"- PageIndex Tree: `{pi_exists}`")
            st.markdown("")


# -----------------------------------------------------------------------------
# 6. Main Entry Point
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
        "Clinical question-answering across scanned hospital records, lab reports, "
        "clinical flowsheets, imaging studies, and discharge summaries with **verifiable citation evidence**."
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

    # Main Application Navigation: Chat Interface vs Admin Management (Task 5.4)
    tab_chat, tab_admin = st.tabs(["💬 Clinical Chat", "⚙️ Admin"])

    with tab_chat:
        render_chat_interface(
            mode=mode,
            selected_patients=selected_patients,
            backend=backend,
            compare_both=compare_both,
        )

    with tab_admin:
        render_admin_tab()


if __name__ == "__main__":
    main()
