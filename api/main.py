"""
Medical Records RAG — FastAPI Cloud Backend Service

Exposes RESTful endpoints for:
- Health check & readiness inspection
- Patient registry lookup
- RAG question answering (GraphRAG & PageIndex)
- Evidence snippet & citation lookups
- Ephemeral single-report sandbox upload & ingestion
- Ephemeral session cleanup with guaranteed zero-trace purging
"""

import asyncio
import base64
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Header, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from evidence_store import EvidenceRecord, get_evidence_by_id, load_evidence
from orchestrate import answer_question
from patients import PATIENTS, all_patient_ids, get_display_label, is_temp_patient
from temp_session import (
    TEMP_PATIENT_ID,
    cleanup_temp_patient,
    ingest_user_report,
    purge_all_temporary_data,
)

# -----------------------------------------------------------------------------
# Ephemeral Session & Security State
# -----------------------------------------------------------------------------
_LAST_ACTIVITY_TIME: float = time.time()
_SESSION_INACTIVITY_TTL: int = int(os.getenv("TEMP_SESSION_TTL_SECONDS", "1800"))  # Default 30 min
_API_KEY: Optional[str] = os.getenv("BACKEND_API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Verifies optional shared API key if configured on Azure."""
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return api_key


async def _background_ttl_janitor():
    """Background task that sweeps and purges abandoned temporary sessions."""
    global _LAST_ACTIVITY_TIME
    while True:
        await asyncio.sleep(60)  # Check every minute
        if TEMP_PATIENT_ID in PATIENTS:
            idle_time = time.time() - _LAST_ACTIVITY_TIME
            if idle_time > _SESSION_INACTIVITY_TTL:
                print(f"[Janitor] Temporary session idle for {int(idle_time)}s (> {_SESSION_INACTIVITY_TTL}s). Purging data...")
                cleanup_temp_patient(TEMP_PATIENT_ID)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks: starts janitor and purges temp data on exit."""
    # Startup: ensure clean state
    purge_all_temporary_data()
    janitor_task = asyncio.create_task(_background_ttl_janitor())
    print("[FastAPI Backend] Medical Records RAG API started successfully.")
    try:
        yield
    finally:
        # Shutdown: guarantee complete zero-data wipe
        janitor_task.cancel()
        print("[FastAPI Backend] Shutting down. Executing zero-trace data purge...")
        purge_all_temporary_data()


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Medical Records RAG API",
    description="Backend clinical RAG engine (GraphRAG + PageIndex) running locally on Azure Docker.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Streamlit Community Cloud and local clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Clinical query string")
    mode: str = Field("individual", description="'individual' or 'group'")
    selected_patients: List[str] = Field(..., description="List of patient IDs")
    backend: str = Field("graph", description="'graph' or 'pageindex'")


class CleanupResponse(BaseModel):
    status: str
    patient_id: str
    deleted_files_count: int
    neo4j_nodes_deleted: int


# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """System health check & engine readiness probe."""
    return {
        "status": "healthy",
        "service": "medical-records-rag-backend",
        "timestamp": time.time(),
        "llm_provider": os.getenv("LLM_PROVIDER", "local"),
        "ocr_engine": os.getenv("OCR_ENGINE", "mineru"),
        "registered_patients_count": len(all_patient_ids()),
        "temp_session_active": TEMP_PATIENT_ID in PATIENTS,
    }


@app.get("/patients", tags=["Patients"])
def get_patients(api_key: Optional[str] = Security(verify_api_key)):
    """Returns list of all available patients and their display metadata."""
    results = []
    for pid in all_patient_ids():
        results.append({
            "id": pid,
            "label": get_display_label(pid),
            "is_temp": is_temp_patient(pid),
        })
    return {"patients": results}


@app.post("/query", tags=["Inference"])
def query_rag(req: QueryRequest, api_key: Optional[str] = Security(verify_api_key)):
    """Executes RAG retrieval and answer generation (GraphRAG or PageIndex)."""
    global _LAST_ACTIVITY_TIME
    _LAST_ACTIVITY_TIME = time.time()

    start_time = time.time()
    try:
        result = answer_question(
            question=req.question,
            mode=req.mode,
            selected_patients=req.selected_patients,
            backend=req.backend,
        )
        elapsed = round(time.time() - start_time, 2)
        result["latency_seconds"] = elapsed
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during answer generation: {str(exc)}",
        ) from exc


@app.get("/evidence/{evidence_id}", tags=["Evidence"])
def get_evidence(evidence_id: str, api_key: Optional[str] = Security(verify_api_key)):
    """Retrieves an evidence record by ID with base64 image preview if available."""
    patient_id = evidence_id.split("__")[0] if evidence_id else ""
    record = get_evidence_by_id(patient_id=patient_id, evidence_id=evidence_id) if patient_id else None

    # Fallback search across all known patients if not found
    if not record:
        for pid in all_patient_ids():
            rec = get_evidence_by_id(patient_id=pid, evidence_id=evidence_id)
            if rec:
                record = rec
                break

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence record '{evidence_id}' not found.",
        )

    res = record.to_dict()
    res["image_base64"] = None

    # If page image exists on disk, encode as base64 for remote Streamlit display
    if record.page_image_path:
        img_p = _ROOT_DIR / record.page_image_path
        if img_p.exists():
            try:
                raw_bytes = img_p.read_bytes()
                res["image_base64"] = base64.b64encode(raw_bytes).decode("utf-8")
            except Exception:
                pass

    return res


@app.post("/upload_report", tags=["Sandbox"])
async def upload_temp_report(
    file: UploadFile = File(...),
    api_key: Optional[str] = Security(verify_api_key),
):
    """
    Accepts an uploaded clinical report PDF, runs the end-to-end ingestion pipeline,
    and sets up the isolated ephemeral temporary patient session.
    """
    global _LAST_ACTIVITY_TIME
    _LAST_ACTIVITY_TIME = time.time()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF document.",
        )

    try:
        pdf_bytes = await file.read()
        summary = ingest_user_report(
            pdf_bytes=pdf_bytes,
            original_filename=file.filename,
            patient_id=TEMP_PATIENT_ID,
        )
        return {"status": "success", "summary": summary}
    except Exception as exc:
        # Guarantee cleanup on failure
        cleanup_temp_patient(TEMP_PATIENT_ID)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest temporary report: {str(exc)}",
        ) from exc


@app.post("/cleanup_session", tags=["Sandbox"])
def cleanup_session(api_key: Optional[str] = Security(verify_api_key)):
    """Immediately purges all temporary files, OCR extracts, and isolated Neo4j records."""
    res = cleanup_temp_patient(TEMP_PATIENT_ID)
    return {
        "status": "success",
        "patient_id": res.get("patient_id", TEMP_PATIENT_ID),
        "deleted_files_count": res.get("deleted_files_count", 0),
        "neo4j_nodes_deleted": res.get("neo4j_nodes_deleted", 0),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
