"""
Unit tests for device_utils: automatic GPU detection, layer offloading, and CPU fallback.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

import device_utils


def test_device_info_structure():
    """Verify get_device_info returns expected keys and types."""
    info = device_utils.get_device_info()
    assert "is_gpu_available" in info
    assert "device_type" in info
    assert info["device_type"] in ("cuda", "cpu")
    assert "device_name" in info
    assert "gpu_name" in info
    assert "gpu_count" in info
    assert "llama_gpu_offload" in info


def test_force_cpu_override(monkeypatch):
    """Verify FORCE_CPU environment variable forces CPU mode."""
    monkeypatch.setenv("FORCE_CPU", "1")
    monkeypatch.delenv("FORCE_GPU", raising=False)
    assert device_utils.is_gpu_available() is False
    assert device_utils.get_llama_gpu_layers() == 0
    assert device_utils.get_mineru_device_mode() == "cpu"


def test_force_gpu_override(monkeypatch):
    """Verify FORCE_GPU environment variable forces GPU mode."""
    monkeypatch.setenv("FORCE_GPU", "1")
    monkeypatch.delenv("FORCE_CPU", raising=False)
    assert device_utils.is_gpu_available() is True
    assert device_utils.get_llama_gpu_layers() == -1
    assert device_utils.get_mineru_device_mode() == "cuda"


def test_local_llm_n_gpu_layers_explicit_override(monkeypatch):
    """Verify LOCAL_LLM_N_GPU_LAYERS environment variable takes precedence."""
    monkeypatch.setenv("LOCAL_LLM_N_GPU_LAYERS", "16")
    assert device_utils.get_llama_gpu_layers() == 16


@patch("device_utils.check_torch_gpu")
@patch("device_utils.check_llama_gpu")
@patch("device_utils.check_nvidia_smi")
def test_torch_gpu_detection(mock_smi, mock_llama, mock_torch, monkeypatch):
    """Verify PyTorch CUDA detection triggers GPU mode."""
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.delenv("FORCE_GPU", raising=False)
    monkeypatch.delenv("LOCAL_LLM_N_GPU_LAYERS", raising=False)

    mock_torch.return_value = (True, "NVIDIA GeForce RTX 4090", 1)
    mock_llama.return_value = False
    mock_smi.return_value = (False, None)

    assert device_utils.is_gpu_available() is True
    assert device_utils.get_llama_gpu_layers() == -1
    assert device_utils.get_mineru_device_mode() == "cuda"
    info = device_utils.get_device_info()
    assert info["device_type"] == "cuda"
    assert "RTX 4090" in info["device_name"]


@patch("device_utils.check_torch_gpu")
@patch("device_utils.check_llama_gpu")
@patch("device_utils.check_nvidia_smi")
def test_no_gpu_fallback_to_cpu(mock_smi, mock_llama, mock_torch, monkeypatch):
    """Verify all checks returning False results in CPU mode."""
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.delenv("FORCE_GPU", raising=False)
    monkeypatch.delenv("LOCAL_LLM_N_GPU_LAYERS", raising=False)

    mock_torch.return_value = (False, None, 0)
    mock_llama.return_value = False
    mock_smi.return_value = (False, None)

    assert device_utils.is_gpu_available() is False
    assert device_utils.get_llama_gpu_layers() == 0
    assert device_utils.get_mineru_device_mode() == "cpu"
    info = device_utils.get_device_info()
    assert info["device_type"] == "cpu"
    assert info["device_name"] == "CPU"


def test_sync_mineru_config(tmp_path, monkeypatch):
    """Verify sync_mineru_config writes correct device-mode."""
    cfg_file = tmp_path / "magic-pdf.json"
    cfg_file.write_text('{"device-mode": "dummy", "models-dir": "test"}', encoding="utf-8")

    monkeypatch.setenv("FORCE_CPU", "1")
    device_utils.sync_mineru_config(config_path=cfg_file)
    import json
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["device-mode"] == "cpu"

    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.setenv("FORCE_GPU", "1")
    device_utils.sync_mineru_config(config_path=cfg_file)
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["device-mode"] == "cuda"
