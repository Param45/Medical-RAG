"""
Medical Records RAG Demo — Streamlit Web Application (SRS §9, BUILD_GUIDE Phase 5)

A locally hosted clinical workstation enabling multi-modal retrieval-augmented generation:
- Dual Roles: Individual (patient self-lookup) & Group (hospital staff multi-patient review)
- Dual Backends: GraphRAG on Neo4j & PageIndex (pure LLM reasoning tree)
- Interactive Chat Interface with Evidence Tracking & Grounded Citations
- Ephemeral Single-Report Sandbox for user-uploaded clinical PDF reports

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

import api_client
from device_utils import get_device_info
from patients import PATIENTS, all_patient_ids, get_display_label, is_temp_patient

# Local backend modules (active in local direct mode, optional on Streamlit Community Cloud)
try:
    from orchestrate import answer_question
except ImportError:
    answer_question = None

try:
    from evidence_store import EvidenceRecord, get_evidence_by_id, load_evidence
except ImportError:
    EvidenceRecord = None
    get_evidence_by_id = None
    load_evidence = None

try:
    from temp_session import (
        TEMP_PATIENT_ID,
        TEMP_NEO4J_CONFIG,
        cleanup_temp_patient,
        ingest_user_report,
    )
except ImportError:
    TEMP_PATIENT_ID = "temp_user_report"
    TEMP_NEO4J_CONFIG = {}
    cleanup_temp_patient = None
    ingest_user_report = None


# -----------------------------------------------------------------------------
# Deployment-Aware Execution Dispatchers (Local Direct vs Azure REST API)
# -----------------------------------------------------------------------------
def execute_answer_question(
    question: str,
    mode: str,
    selected_patients: List[str],
    backend: str,
) -> Dict[str, Any]:
    """Executes query either via remote Azure REST API or local orchestrator."""
    if api_client.is_remote_mode():
        return api_client.send_query(
            question=question,
            mode=mode,
            selected_patients=selected_patients,
            backend=backend,
        )
    elif answer_question is not None:
        return answer_question(
            question=question,
            mode=mode,
            selected_patients=selected_patients,
            backend=backend,
        )
    else:
        raise RuntimeError("No Azure backend configured and local orchestrator is unavailable.")


def execute_cleanup_temp_session(patient_id: str = TEMP_PATIENT_ID) -> Dict[str, Any]:
    """Wipes temporary session data either on Azure or locally."""
    if api_client.is_remote_mode():
        return api_client.cleanup_temp_session()
    elif cleanup_temp_patient is not None:
        return cleanup_temp_patient(patient_id=patient_id)
    return {"status": "skipped"}


def execute_ingest_user_report(
    pdf_bytes: bytes,
    original_filename: str = "report.pdf",
    patient_id: str = TEMP_PATIENT_ID,
    progress_callback=None,
) -> Dict[str, Any]:
    """Ingests uploaded PDF either by dispatching to Azure API or running local pipeline."""
    if api_client.is_remote_mode():
        if progress_callback:
            progress_callback("Uploading PDF to Azure Docker backend & running clinical pipeline...")
        resp = api_client.upload_temp_report(pdf_bytes=pdf_bytes, filename=original_filename)
        return resp.get("summary", resp)
    elif ingest_user_report is not None:
        return ingest_user_report(
            pdf_bytes=pdf_bytes,
            original_filename=original_filename,
            patient_id=patient_id,
            progress_callback=progress_callback,
        )
    else:
        raise RuntimeError("No Azure backend configured and local ingestion pipeline is unavailable.")


def execute_get_evidence_by_id(evidence_id: str, patient_id: Optional[str] = None) -> Any:
    """Retrieves evidence record either via Azure REST API or local evidence store."""
    if api_client.is_remote_mode():
        remote_data = api_client.fetch_evidence(evidence_id)
        if remote_data:
            class RemoteEvidence:
                def __init__(self, data: Dict[str, Any]):
                    for k, v in data.items():
                        setattr(self, k, v)
                    if not hasattr(self, "page_image_path"):
                        self.page_image_path = data.get("image_path")
            return RemoteEvidence(remote_data)
        return None
    elif get_evidence_by_id is not None and patient_id:
        return get_evidence_by_id(patient_id=patient_id, evidence_id=evidence_id)
    return None


# -----------------------------------------------------------------------------
# 0. Modern Clinical Workstation Stylesheet (CSS Design System)
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #0b0f17;
    --bg-card: #111827;
    --bg-card-hover: #172136;
    --bg-surface: #1e293b;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(14, 165, 233, 0.4);
    --primary: #0ea5e9;
    --primary-glow: rgba(14, 165, 233, 0.2);
    --graph-indigo: #6366f1;
    --graph-indigo-glow: rgba(99, 102, 241, 0.15);
    --tree-emerald: #10b981;
    --tree-emerald-glow: rgba(16, 185, 129, 0.15);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Global Typography & Background Override */
html, body, [class*="css"], .stApp {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

/* Hide default streamlit decor */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background-color: transparent !important;
    backdrop-filter: blur(8px);
}

/* Executive App Bar */
.med-app-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.15rem 1.4rem;
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
}

.med-app-bar-brand {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.med-brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px var(--primary-glow);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.med-brand-title {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.med-brand-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 0.15rem;
}

.med-pill-badge {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.3);
}

.med-pill-live {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.med-pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: med-pulse 2s infinite;
}

@keyframes med-pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* KPI Scope Grid */
.med-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.25rem;
}

.med-kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.med-kpi-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.med-kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.med-kpi-value {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    font-family: var(--font-sans);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.med-kpi-subtext {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
    font-family: var(--font-mono);
}

/* Dual Backend Comparison Cards */
.comparison-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.15rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.comparison-card.graph-card {
    border-top: 3px solid var(--graph-indigo);
}

.comparison-card.pageindex-card {
    border-top: 3px solid var(--tree-emerald);
}

.comparison-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--border-subtle);
}

.comparison-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.comparison-tag {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.18rem 0.5rem;
    border-radius: 6px;
    text-transform: uppercase;
}

.graph-tag {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.pageindex-tag {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.comparison-body {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e2e8f0;
    flex-grow: 1;
}

.comparison-citations-row {
    margin-top: 0.9rem;
    padding-top: 0.65rem;
    border-top: 1px solid var(--border-subtle);
}

/* Citation Pill Badges */
.cit-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.18rem 0.5rem;
    margin: 0.15rem 0.2rem;
    border-radius: 6px;
    background: rgba(14, 165, 233, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.25);
    transition: all 0.15s ease;
}

.cit-badge:hover {
    background: rgba(14, 165, 233, 0.2);
    border-color: rgba(14, 165, 233, 0.5);
    color: #7dd3fc;
}

/* Modern Tabs Styling */
div[data-testid="stTabs"] [role="tablist"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
}

div[data-testid="stTabs"] button[role="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.15rem !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(14, 165, 233, 0.1) !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.15);
}

/* Streamlit Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0d121f !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    background: rgba(17, 24, 39, 0.9) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 20px var(--primary-glow) !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 1.15rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    margin-bottom: 0.75rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 14px var(--primary-glow) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35) !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    background: var(--bg-surface) !important;
}

/* Sandbox Security Card */
.sandbox-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 10px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
}

.sandbox-icon {
    color: #38bdf8;
    flex-shrink: 0;
    margin-top: 0.15rem;
}

.sandbox-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.sandbox-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Pipeline Step Markers */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6rem;
    margin: 1rem 0 1.25rem 0;
}

.pipeline-step {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.65rem 0.5rem;
    text-align: center;
}

.pipeline-num {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: #38bdf8;
    font-weight: 700;
}

.pipeline-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 0.2rem;
}
</style>
"""


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
        return f":green[High confidence ({conf_pct})]"
    elif confidence >= 0.5:
        return f":orange[Medium confidence ({conf_pct})]"
    else:
        return f":red[Low confidence ({conf_pct})]"


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
    Renders clean executive sidebar controls:
    1. Operational Scope (Individual / Group)
    2. Patient selector
    3. Active Configuration & Database Indicator
    4. Session lifecycle actions
    """
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" stroke-width="2.2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            <span style="font-size: 1.05rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">MED-RAG</span>
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem;">
            Clinical Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Configuration & Scope")

    # 1. Mode dropdown
    mode = st.sidebar.selectbox(
        "Operational Mode",
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
            "Target Patient",
            options=display_names,
            index=default_patient_index,
            help="Select the single patient whose medical records you wish to view.",
        )
        selected_patients = resolve_selected_patients(mode, selected_label)
    else:  # Group mode
        all_patients = st.sidebar.checkbox(
            "Select all cohort patients",
            value=False,
            help="Select all registered patients in the hospital database.",
        )
        if all_patients:
            st.sidebar.info(f"Targeting all {len(all_patient_ids())} registered patients.")
            selected_patients = all_patient_ids()
        else:
            selected_labels = st.sidebar.multiselect(
                "Cohort Patients",
                options=display_names,
                default=display_names[:1] if display_names else [],
                help="Select one or more patients to query or compare.",
            )
            selected_patients = resolve_selected_patients(mode, selected_labels, all_patients_checked=False)

    st.sidebar.markdown("---")

    # Engine evaluation selection
    backend_choice = st.sidebar.selectbox(
        "Retrieval Engine",
        options=["Dual Comparison (Both Engines)", "GraphRAG (Neo4j)", "PageIndex (Tree Index)"],
        index=0,
        help="Choose whether to evaluate both engines side-by-side or run a faster single engine on CPU.",
    )
    backend = resolve_backend(backend_choice)
    compare_both = (backend == "both")

    # Clear chat affordance
    if st.sidebar.button("Clear Conversation", use_container_width=True):
        clear_chat_history()
        st.rerun()

    # Scope Summary Badge in Sidebar
    st.sidebar.markdown("### Active Scope")
    patient_display = ', '.join(get_display_label(p) for p in selected_patients) if selected_patients else 'None'
    dev_info = get_device_info()
    accel_badge = f"GPU ({dev_info['device_name']})" if dev_info["is_gpu_available"] else "CPU Mode"

    deployment_label = "Azure Docker Backend" if api_client.is_remote_mode() else "Local Direct"
    deployment_color = "#38bdf8" if api_client.is_remote_mode() else "#a855f7"

    st.sidebar.markdown(
        f"""
        <div style="background: rgba(17, 24, 39, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 0.85rem; font-size: 0.78rem; line-height: 1.6;">
            <div style="color: #94a3b8; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">Mode</div>
            <div style="color: #ffffff; font-weight: 600; margin-bottom: 0.4rem;">{mode}</div>
            <div style="color: #94a3b8; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">Patient(s)</div>
            <div style="color: #ffffff; font-weight: 600; margin-bottom: 0.4rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{patient_display}">{patient_display}</div>
            <div style="color: #94a3b8; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">Engine Evaluation</div>
            <div style="color: #38bdf8; font-weight: 600; margin-bottom: 0.4rem;">{"Dual Comparison" if compare_both else ("GraphRAG (Neo4j)" if backend == "graph" else "PageIndex (Tree)")}</div>
            <div style="color: #94a3b8; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">Architecture</div>
            <div style="color: {deployment_color}; font-weight: 600; margin-bottom: 0.4rem;">{deployment_label}</div>
            <div style="color: #94a3b8; font-size: 0.68rem; text-transform: uppercase; font-weight: 600;">Compute</div>
            <div style="color: {'#34d399' if dev_info['is_gpu_available'] else '#94a3b8'}; font-weight: 600;">{accel_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Temporary Neo4j indicator badge if active
    if any(is_temp_patient(p) for p in selected_patients):
        st.sidebar.markdown(
            """
            <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 6px; font-size: 0.72rem; color: #a5b4fc; font-family: var(--font-mono);">
                Isolated Sandbox: af2857f2
            </div>
            """,
            unsafe_allow_html=True,
        )

    # End Temporary Session button if temporary report is active
    if TEMP_PATIENT_ID in PATIENTS:
        st.sidebar.markdown("---")
        if st.sidebar.button("End Session & Purge Report", use_container_width=True, help="Permanently wipes all uploaded files and temporary Neo4j records."):
            execute_cleanup_temp_session(TEMP_PATIENT_ID)
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
    """Extracts all clean evidence IDs from a line containing bracketed citations."""
    matches = EVIDENCE_BRACKET_REGEX.findall(line)
    cits: List[str] = []
    for m in matches:
        for item in m.split(","):
            cleaned = item.strip()
            if "__" in cleaned and cleaned not in cits:
                cits.append(cleaned)
    return cits


def strip_evidence_brackets(line: str) -> str:
    """Removes raw [patient_id__...] citation brackets from line text for clean reading."""
    return EVIDENCE_BRACKET_REGEX.sub("", line).rstrip()


def render_evidence_expander(evidence_id: str) -> None:
    """
    Renders an interactive evidence expander displaying the source page image,
    raw OCR text, report metadata, and confidence badge (SRS FR-9.1.3).
    """
    patient_id = parse_patient_id_from_evidence_id(evidence_id)
    record = execute_get_evidence_by_id(evidence_id=evidence_id, patient_id=patient_id)

    with st.expander(f"Source Evidence: {evidence_id}", expanded=False):
        if record is None:
            st.warning(f"Evidence record {evidence_id} could not be found.")
            return

        # Top Metadata Banner
        meta_col1, meta_col2, meta_col3 = st.columns([1.5, 1.5, 1.2])
        with meta_col1:
            st.markdown(f"**Report Type:** `{getattr(record, 'report_type', 'Clinical Report')}`")
            st.markdown(f"**Patient:** `{get_display_label(record.patient_id)}` ({record.patient_id})")
        with meta_col2:
            date_str = getattr(record, "report_date", None) or "Undated"
            st.markdown(f"**Report Date:** `{date_str}`")
            st.markdown(f"**Page:** `{record.page_number}` (Source: `{getattr(record, 'source_type', 'document')}`)")
        with meta_col3:
            st.markdown("**Confidence:**")
            st.markdown(get_confidence_badge_markdown(getattr(record, "confidence", 1.0)))

        st.markdown("---")

        # Evidence Tabs: Raw OCR Text vs Source Page Scan
        tab_raw, tab_image = st.tabs(["Raw Verbatim OCR Text", "Original Page Scan"])

        with tab_raw:
            st.caption("Verbatim extracted text from source document (no paraphrasing):")
            st.code(getattr(record, "raw_text", getattr(record, "text", "")), language=None)

        with tab_image:
            img_b64 = getattr(record, "image_base64", None)
            if img_b64:
                import base64
                st.image(
                    base64.b64decode(img_b64),
                    caption=f"{get_display_label(record.patient_id)} — Report {record.report_id} (Page {record.page_number})",
                    use_container_width=True,
                )
            else:
                image_rel_path = getattr(record, "page_image_path", None)
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

    st.markdown(f"##### Verified Evidence Trail ({len(citations)} source document{'s' if len(citations) > 1 else ''})")
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

            col1, col2 = st.columns(2)

            with col1:
                citations_g = graph_res.get("citations", [])
                answer_g = graph_res.get("answer", "No response generated.")
                st.markdown(
                    f"""
                    <div class="comparison-card graph-card">
                        <div class="comparison-header">
                            <div class="comparison-title">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.2"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><circle cx="6" cy="18" r="3"/><line x1="8.5" y1="7.5" x2="15.5" y2="16.5"/><line x1="6" y1="9" x2="6" y2="15"/></svg>
                                <span>GraphRAG</span>
                            </div>
                            <span class="comparison-tag graph-tag">Neo4j Knowledge Graph</span>
                        </div>
                        <div class="comparison-body">
                            {answer_g}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                citations_pi = pi_res.get("citations", [])
                answer_pi = pi_res.get("answer", "No response generated.")
                st.markdown(
                    f"""
                    <div class="comparison-card pageindex-card">
                        <div class="comparison-header">
                            <div class="comparison-title">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                                <span>PageIndex</span>
                            </div>
                            <span class="comparison-tag pageindex-tag">Hierarchical Reasoning Tree</span>
                        </div>
                        <div class="comparison-body">
                            {answer_pi}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Combined verifiable evidence trail below the comparison cards
            all_cits = list(dict.fromkeys(citations_g + citations_pi))
            if all_cits:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                render_citations_list(all_cits)

        else:
            # Single Backend Layout
            backend_used = msg.get("backend_used", "engine")
            backend_label = "GraphRAG (Neo4j)" if backend_used == "graph" else "PageIndex (Tree)"
            st.caption(f"Generated via **{backend_label}**")
            st.markdown(msg.get("content", ""))

            citations = msg.get("citations", [])
            if citations:
                render_citations_list(citations)


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
                    graph_res = execute_answer_question(
                        question=question,
                        mode=mode.lower(),
                        selected_patients=selected_patients,
                        backend="graph",
                    )
                    pi_res = execute_answer_question(
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
                    res = execute_answer_question(
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
    st.markdown(
        """
        <div class="sandbox-card">
            <div class="sandbox-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
            </div>
            <div>
                <div class="sandbox-title">Session Sandbox & Ephemeral Privacy Notice</div>
                <div class="sandbox-desc">
                    Uploaded documents are processed entirely in an isolated runtime sandbox. All OCR text, extracted entities,
                    hierarchical reasoning trees, and GraphRAG nodes are saved to a temporary Neo4j database instance (<code>af2857f2</code>).
                    No data persists beyond this session. When you click <strong>End Session</strong> or exit, all artifacts are permanently purged.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_report_active = st.session_state.get("temp_report_active", False) and (TEMP_PATIENT_ID in PATIENTS)

    if is_report_active:
        summary = st.session_state.get("temp_report_summary") or {}
        display_label = get_display_label(TEMP_PATIENT_ID)

        # Active Report Banner
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 0.85rem 1.15rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #34d399; letter-spacing: 0.05em;">ACTIVE REPORT INGESTED</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">{display_label}</div>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #94a3b8; background: rgba(0, 0, 0, 0.3); padding: 0.3rem 0.6rem; border-radius: 6px;">
                    Elapsed: {summary.get('elapsed_seconds', '—')}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(
                f"""
                <div class="med-kpi-card">
                    <div class="med-kpi-label">PAGES EXTRACTED</div>
                    <div class="med-kpi-value">{summary.get("pages_count", "—")}</div>
                    <div class="med-kpi-subtext">250 DPI Scans</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                f"""
                <div class="med-kpi-card">
                    <div class="med-kpi-label">REPORTS DETECTED</div>
                    <div class="med-kpi-value">{summary.get("reports_count", "—")}</div>
                    <div class="med-kpi-subtext">Clinical Anchors</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol3:
            st.markdown(
                f"""
                <div class="med-kpi-card">
                    <div class="med-kpi-label">EVIDENCE CHUNKS</div>
                    <div class="med-kpi-value">{summary.get("chunks_count", "—")}</div>
                    <div class="med-kpi-subtext">Verifiable Grounding</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mcol4:
            st.markdown(
                f"""
                <div class="med-kpi-card">
                    <div class="med-kpi-label">NEO4J TRIPLES</div>
                    <div class="med-kpi-value" style="color: #818cf8;">{summary.get("triples_written", "—")}</div>
                    <div class="med-kpi-subtext">Sandbox: af2857f2</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        # Session Cleanup Button
        col_clean, _ = st.columns([1.5, 2])
        with col_clean:
            if st.button("End Session & Purge Report", type="secondary", use_container_width=True, help="Permanently wipes all files and graph records for this report."):
                with st.spinner("Clearing temporary session data and purging isolated Neo4j database..."):
                    execute_cleanup_temp_session(TEMP_PATIENT_ID)
                    st.session_state.temp_report_active = False
                    st.session_state.temp_report_summary = None
                    st.session_state.temp_messages = []
                st.toast("Temporary session data and Neo4j records permanently cleared.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Clinical Q&A Over Uploaded Report")

        # Suggested Questions Chips
        sample_cols = st.columns(3)
        sample_q = None
        if sample_cols[0].button("Primary Diagnosis & Staging", use_container_width=True):
            sample_q = "What is my primary diagnosis, clinical condition, and staging?"
        if sample_cols[1].button("Abnormal Test Results & Labs", use_container_width=True):
            sample_q = "Are there any abnormal lab test results, biomarkers, or findings documented?"
        if sample_cols[2].button("Treatments & Prescribed Drugs", use_container_width=True):
            sample_q = "What treatments, procedures, surgical notes, or medications are documented?"

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
                        graph_res = execute_answer_question(
                            question=active_query,
                            mode="individual",
                            selected_patients=[TEMP_PATIENT_ID],
                            backend="graph",
                        )
                        pi_res = execute_answer_question(
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

                    except Exception as exc:
                        st.error(f"Error analyzing report: {exc}")

    else:
        # Pipeline Flow Visual
        st.markdown("#### Document Intelligence Pipeline")
        st.markdown(
            """
            <div class="pipeline-grid">
                <div class="pipeline-step">
                    <div class="pipeline-num">STEP 01</div>
                    <div class="pipeline-name">250 DPI Scan</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-num">STEP 02</div>
                    <div class="pipeline-name">Bilingual OCR</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-num">STEP 03</div>
                    <div class="pipeline-name">Boundary Split</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-num">STEP 04</div>
                    <div class="pipeline-name">Neo4j Sandbox</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-num">STEP 05</div>
                    <div class="pipeline-name">Reasoning Tree</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Upload Form
        uploaded_file = st.file_uploader(
            "Upload Clinical PDF Document",
            type=["pdf"],
            help="Select a scanned or digital medical record PDF.",
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
                    summary = execute_ingest_user_report(
                        pdf_bytes=uploaded_file.getvalue(),
                        original_filename=uploaded_file.name,
                        patient_id=TEMP_PATIENT_ID,
                        progress_callback=status_container.write,
                    )
                    status_container.update(
                        label=f"Ingestion Completed in {summary['elapsed_seconds']}s",
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
        page_title="Medical Records RAG Workstation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Custom Clinical Design System
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state (SRS FR-9.1.4)
    init_session_state()

    # Render sidebar controls (SRS FR-9.1.1)
    mode, selected_patients, backend, compare_both = render_sidebar()

    # Executive App Bar Header
    st.markdown(
        """
        <div class="med-app-bar">
            <div class="med-app-bar-brand">
                <div class="med-brand-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <div>
                    <div class="med-brand-title">
                        <span>MED-RAG</span>
                        <span style="font-weight: 300; opacity: 0.35;">//</span>
                        <span>Clinical Intelligence Workstation</span>
                    </div>
                    <div class="med-brand-subtitle">
                        Multi-Modal GraphRAG & Hierarchical Reasoning over Longitudinal Patient Records
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Scope Stat Grid
    patient_str = ", ".join(get_display_label(p) for p in selected_patients) if selected_patients else "None"
    dev_info = get_device_info()
    accel_text = f"GPU ({dev_info['device_name']})" if dev_info["is_gpu_available"] else "CPU Mode"
    accel_color = "#34d399" if dev_info["is_gpu_available"] else "#94a3b8"

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(
            f"""
            <div class="med-kpi-card">
                <div class="med-kpi-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                    OPERATIONAL MODE
                </div>
                <div class="med-kpi-value">{mode}</div>
                <div class="med-kpi-subtext">{'Single-Patient Focus' if mode == 'Individual' else 'Cross-Patient Review'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div class="med-kpi-card">
                <div class="med-kpi-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    TARGET COHORT
                </div>
                <div class="med-kpi-value" title="{patient_str}">{patient_str}</div>
                <div class="med-kpi-subtext">{f'{len(selected_patients)} record(s) loaded' if selected_patients else 'No selection'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            """
            <div class="med-kpi-card">
                <div class="med-kpi-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                    RAG ENGINES
                </div>
                <div class="med-kpi-value" style="color: #38bdf8;">GraphRAG + PageIndex</div>
                <div class="med-kpi-subtext">Dual Grounded Comparison</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            f"""
            <div class="med-kpi-card">
                <div class="med-kpi-label">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/></svg>
                    COMPUTE ACCELERATION
                </div>
                <div class="med-kpi-value" style="color: {accel_color};">{accel_text}</div>
                <div class="med-kpi-subtext">Hardware Inference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

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
