"""
Unit tests for the remote API client helper.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

import api_client


def test_api_client_unconfigured():
    with patch.dict(os.environ, {}, clear=True):
        assert api_client.is_remote_mode() is False
        assert api_client.get_backend_url() is None
        health = api_client.check_health()
        assert health["status"] == "unconfigured"


def test_api_client_configured():
    with patch.dict(os.environ, {"BACKEND_API_URL": "https://test-azure-app.azurecontainerapps.io"}):
        assert api_client.is_remote_mode() is True
        assert api_client.get_backend_url() == "https://test-azure-app.azurecontainerapps.io"


@patch("requests.get")
def test_api_client_fetch_patients(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "patients": [{"id": "patient_a", "label": "Patient A", "is_temp": False}]
    }

    with patch.dict(os.environ, {"BACKEND_API_URL": "https://test-azure-app.azurecontainerapps.io"}):
        patients = api_client.fetch_patients()
        assert len(patients) == 1
        assert patients[0]["id"] == "patient_a"
