"""
Medical Records RAG — Remote API Client for Streamlit Cloud Frontend

Handles transparent communication with the Azure Docker FastAPI backend:
- Health checking and connectivity tests
- Patient list fetching
- RAG question querying (GraphRAG & PageIndex)
- Evidence retrieval & image decoding
- Temporary report upload and session teardown
"""

import os
from typing import Any, Dict, List, Optional
import requests

try:
    import streamlit as st
except ImportError:
    st = None


def get_backend_url() -> Optional[str]:
    """Retrieves backend API URL from Streamlit secrets or OS environment."""
    url = None
    if st is not None:
        try:
            url = st.secrets.get("BACKEND_API_URL")
        except Exception:
            url = None

    if not url:
        url = os.getenv("BACKEND_API_URL")

    if url:
        return url.rstrip("/")
    return None


def get_api_key() -> Optional[str]:
    """Retrieves optional backend API key from Streamlit secrets or OS environment."""
    key = None
    if st is not None:
        try:
            key = st.secrets.get("BACKEND_API_KEY")
        except Exception:
            key = None

    if not key:
        key = os.getenv("BACKEND_API_KEY")
    return key


def is_remote_mode() -> bool:
    """Returns True if a remote backend URL is configured."""
    return get_backend_url() is not None


def _get_headers() -> Dict[str, str]:
    headers = {}
    key = get_api_key()
    if key:
        headers["X-API-Key"] = key
    return headers


def check_health(timeout: int = 5) -> Dict[str, Any]:
    """Calls GET /health to check if Azure backend is online."""
    base_url = get_backend_url()
    if not base_url:
        return {"status": "unconfigured"}
    try:
        resp = requests.get(f"{base_url}/health", headers=_get_headers(), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "code": resp.status_code, "detail": resp.text}
    except Exception as exc:
        return {"status": "unreachable", "detail": str(exc)}


def fetch_patients(timeout: int = 10) -> List[Dict[str, Any]]:
    """Calls GET /patients to retrieve available patient metadata."""
    base_url = get_backend_url()
    if not base_url:
        return []
    resp = requests.get(f"{base_url}/patients", headers=_get_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("patients", [])


def send_query(
    question: str,
    mode: str,
    selected_patients: List[str],
    backend: str,
    timeout: int = 1000,
) -> Dict[str, Any]:
    """Calls POST /query to execute RAG inference on Azure backend."""
    base_url = get_backend_url()
    if not base_url:
        raise ValueError("Backend URL is not configured.")

    payload = {
        "question": question,
        "mode": mode,
        "selected_patients": selected_patients,
        "backend": backend,
    }
    resp = requests.post(
        f"{base_url}/query",
        json=payload,
        headers=_get_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_evidence(evidence_id: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Calls GET /evidence/{id} to retrieve evidence chunk and base64 preview."""
    base_url = get_backend_url()
    if not base_url:
        return None
    try:
        resp = requests.get(
            f"{base_url}/evidence/{evidence_id}",
            headers=_get_headers(),
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def upload_temp_report(
    pdf_bytes: bytes,
    filename: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Calls POST /upload_report with multipart/form-data to ingest PDF in Azure sandbox."""
    base_url = get_backend_url()
    if not base_url:
        raise ValueError("Backend URL is not configured.")

    files = {"file": (filename, pdf_bytes, "application/pdf")}
    resp = requests.post(
        f"{base_url}/upload_report",
        files=files,
        headers=_get_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def cleanup_temp_session(timeout: int = 15) -> Dict[str, Any]:
    """Calls POST /cleanup_session to wipe ephemeral sandbox session on Azure."""
    base_url = get_backend_url()
    if not base_url:
        return {"status": "unconfigured"}
    try:
        resp = requests.post(
            f"{base_url}/cleanup_session",
            headers=_get_headers(),
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "code": resp.status_code}
    except Exception as exc:
        return {"status": "failed", "detail": str(exc)}
