"""
Unit tests for the FastAPI Medical Records RAG Cloud Backend.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "medical-records-rag-backend"
    assert "registered_patients_count" in data


def test_patients_endpoint():
    response = client.get("/patients")
    assert response.status_code == 200
    data = response.json()
    assert "patients" in data
    assert len(data["patients"]) >= 2
    patient_ids = [p["id"] for p in data["patients"]]
    assert "patient_a" in patient_ids


def test_evidence_endpoint_invalid():
    response = client.get("/evidence/non_existent_evidence_id")
    assert response.status_code == 404


def test_cleanup_session_endpoint():
    response = client.post("/cleanup_session")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
