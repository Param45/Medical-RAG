"""
Medical Records RAG — Hardware & Device Utilities

Auto-detects GPU availability (CUDA, NVIDIA, llama.cpp GPU offload) and dynamically
routes heavy tasks (local LLM inference, MinerU OCR layout/recognition) to GPU
when available, with seamless, crash-proof fallback to CPU when GPU is absent or disabled.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

_ROOT_DIR = Path(__file__).resolve().parent


def check_torch_gpu() -> Tuple[bool, Optional[str], int]:
    """Checks if PyTorch is installed and has working CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "CUDA Device"
            return True, name, count
    except Exception:
        pass
    return False, None, 0


def check_llama_gpu() -> bool:
    """Checks if llama-cpp-python was compiled with GPU offload support."""
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    return False


def check_nvidia_smi() -> Tuple[bool, Optional[str]]:
    """Checks if nvidia-smi is available and reports an active GPU."""
    if not shutil.which("nvidia-smi"):
        return False, None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().splitlines()[0]
            return True, gpu_name
    except Exception:
        pass
    return False, None


def is_gpu_available() -> bool:
    """
    Returns True if an active GPU is detected and available for compute.
    Supports environment variable overrides:
    - FORCE_CPU=1 / true / yes -> always returns False
    - FORCE_GPU=1 / true / yes -> always returns True
    """
    force_cpu = os.getenv("FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
    if force_cpu:
        return False

    force_gpu = os.getenv("FORCE_GPU", "").strip().lower() in ("1", "true", "yes")
    if force_gpu:
        return True

    # 1. PyTorch CUDA check
    has_torch, _, _ = check_torch_gpu()
    if has_torch:
        return True

    # 2. llama-cpp GPU support check
    if check_llama_gpu():
        return True

    # 3. System NVIDIA GPU check
    has_smi, _ = check_nvidia_smi()
    if has_smi:
        return True

    return False


def get_device_info() -> Dict[str, Any]:
    """Returns a comprehensive diagnostic dictionary of compute device capabilities."""
    has_torch, torch_name, device_count = check_torch_gpu()
    has_llama_gpu = check_llama_gpu()
    has_smi, smi_name = check_nvidia_smi()

    gpu_active = is_gpu_available()
    gpu_name = torch_name or smi_name or ("GPU Device" if gpu_active else None)

    return {
        "is_gpu_available": gpu_active,
        "device_type": "cuda" if gpu_active else "cpu",
        "device_name": gpu_name or "CPU",
        "gpu_name": gpu_name,
        "gpu_count": device_count if has_torch else (1 if has_smi else 0),
        "torch_cuda": has_torch,
        "llama_gpu_offload": has_llama_gpu,
        "nvidia_smi": has_smi,
    }


def get_llama_gpu_layers() -> int:
    """
    Returns the number of layers to offload to GPU for llama-cpp-python:
    - -1: offload all layers to GPU if GPU is available
    -  0: CPU-only if no GPU is available or forced to CPU
    - Can be overridden explicitly via LOCAL_LLM_N_GPU_LAYERS.
    """
    env_layers = os.getenv("LOCAL_LLM_N_GPU_LAYERS")
    if env_layers is not None and env_layers.strip():
        try:
            return int(env_layers.strip())
        except ValueError:
            pass

    if is_gpu_available():
        return -1  # Offload all layers to GPU

    return 0  # CPU-only


def get_mineru_device_mode() -> str:
    """
    Returns 'cuda' if GPU is available, else 'cpu'.
    Used for MinerU / magic-pdf configuration.
    """
    return "cuda" if is_gpu_available() else "cpu"


def sync_mineru_config(config_path: Optional[Path] = None) -> Path:
    """
    Ensures magic-pdf.json has the appropriate device-mode ('cuda' or 'cpu')
    matching current hardware capabilities.
    Sets MINERU_TOOLS_CONFIG_JSON environment variable.
    """
    cfg_path = config_path or (_ROOT_DIR / "magic-pdf.json")
    desired_mode = get_mineru_device_mode()

    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            updated = False
            if data.get("device-mode") != desired_mode:
                data["device-mode"] = desired_mode
                updated = True

            current_models = data.get("models-dir", "")
            if not current_models or not Path(current_models).exists():
                local_models_dir = (_ROOT_DIR / "models").resolve()
                if local_models_dir.exists():
                    data["models-dir"] = str(local_models_dir)
                    updated = True

            if updated:
                cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print(f"[DeviceUtils] Synchronized {cfg_path.name} (mode: '{desired_mode}')")
        except Exception as exc:
            print(f"[DeviceUtils] Warning: could not update {cfg_path}: {exc}")

    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(cfg_path)
    return cfg_path
