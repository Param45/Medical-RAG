"""
Medical Records RAG Demo — LLM Client Wrapper (SRS §5.6)

Single thin wrapper function `chat(messages, system=None) -> str` used by:
- Terminology normalization fallback (SRS §5.4.2)
- GraphRAG relationship extraction (SRS §6.1)
- PageIndex summarization and tree traversal (SRS §7.1, §7.2)
- Answer generation (SRS §8.2)

Loads LLM_PROVIDER, LLM_API_KEY, and LLM_MODEL from .env.
Supports both Cloud APIs (Gemini, Anthropic, OpenAI) and Local LLMs (MedGemma GGUF via llama-cpp-python).
Includes a single retry on transient errors.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

# Singleton cache for local model instance
_LOCAL_LLAMA_MODEL: Any = None
_LOCAL_LLAMA_MODEL_PATH: Optional[str] = None


def _resolve_local_model_path(model_path_or_name: Optional[str] = None) -> str:
    """Resolve local GGUF model path, checking models directory or downloading if needed."""
    candidate_name = model_path_or_name or os.getenv("LOCAL_MODEL_PATH") or os.getenv("LLM_MODEL") or "medgemma-1.5-4b-it-Q4_K_M.gguf"
    
    # 1. Exact path as provided
    path_obj = Path(candidate_name)
    if path_obj.exists() and path_obj.is_file():
        return str(path_obj.resolve())

    # 2. Check within ./models directory
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    in_models = models_dir / path_obj.name
    if in_models.exists() and in_models.is_file():
        return str(in_models.resolve())

    # 3. If not found, attempt auto-download using download_models
    try:
        from download_models import download_llm_model
        print(f"[LOCAL LLM] Model '{path_obj.name}' not found locally. Auto-downloading...")
        downloaded = download_llm_model(filename=path_obj.name, target_dir=models_dir)
        return str(Path(downloaded).resolve())
    except Exception as exc:
        raise FileNotFoundError(
            f"Local GGUF model '{candidate_name}' could not be found or downloaded: {exc}"
        ) from exc


def _get_local_llama_client(model_path: Optional[str] = None) -> Any:
    """Load or return cached llama_cpp.Llama instance."""
    global _LOCAL_LLAMA_MODEL, _LOCAL_LLAMA_MODEL_PATH

    resolved_path = _resolve_local_model_path(model_path)
    if _LOCAL_LLAMA_MODEL is not None and _LOCAL_LLAMA_MODEL_PATH == resolved_path:
        return _LOCAL_LLAMA_MODEL

    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is required for local LLM inference. "
            "Install it via `pip install llama-cpp-python`."
        ) from e

    n_ctx = int(os.getenv("LOCAL_LLM_N_CTX", "4096"))
    n_threads_env = os.getenv("LOCAL_LLM_N_THREADS")
    n_threads = int(n_threads_env) if n_threads_env and n_threads_env.isdigit() else None

    print(f"[LOCAL LLM] Initializing Llama model from: {resolved_path} (n_ctx={n_ctx})")
    model_kwargs: Dict[str, Any] = {
        "model_path": resolved_path,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    if n_threads is not None:
        model_kwargs["n_threads"] = n_threads

    _LOCAL_LLAMA_MODEL = Llama(**model_kwargs)
    _LOCAL_LLAMA_MODEL_PATH = resolved_path
    return _LOCAL_LLAMA_MODEL


def _call_local(
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> str:
    """Run local inference using llama-cpp-python with chat format."""
    llm = _get_local_llama_client(model)

    formatted_messages: List[Dict[str, str]] = []
    if system:
        formatted_messages.append({"role": "system", "content": system})

    for m in messages:
        formatted_messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", ""),
        })

    response = llm.create_chat_completion(
        messages=formatted_messages,
        max_tokens=2048,
        temperature=0.1,
    )

    choices = response.get("choices", [])
    if choices and "message" in choices[0]:
        return choices[0]["message"].get("content", "") or ""
    return ""


def _call_gemini(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> str:
    """Call Google Gemini API using google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents: List[types.Content] = []
    for msg in messages:
        role = "user" if msg.get("role") in ("user", "human") else "model"
        text = msg.get("content", "")
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )

    config_kwargs: Dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    if response.text is None:
        return ""
    return response.text


def _call_anthropic(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> str:
    """Call Anthropic Claude API using official anthropic SDK."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package is required for LLM_PROVIDER=anthropic. Run `pip install anthropic`."
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    formatted_messages = [
        {
            "role": "user" if m.get("role") in ("user", "human") else "assistant",
            "content": m.get("content", ""),
        }
        for m in messages
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": formatted_messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    # Extract text content from blocks
    text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "".join(text_blocks)


def _call_openai(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
) -> str:
    """Call OpenAI or OpenAI-compatible API using openai SDK."""
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "openai package is required for LLM_PROVIDER=openai. Run `pip install openai`."
        ) from e

    client = openai.OpenAI(api_key=api_key)
    formatted_messages: List[Dict[str, str]] = []
    if system:
        formatted_messages.append({"role": "system", "content": system})

    for m in messages:
        formatted_messages.append(
            {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
    )
    choice = response.choices[0]
    return choice.message.content or ""


def chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    max_retries: int = 1,
    retry_delay_seconds: float = 1.0,
) -> str:
    """
    Unified LLM chat wrapper (SRS FR-5.6.1).

    Args:
        messages: List of message dicts with keys 'role' ('user'|'assistant'|'model') and 'content'.
        system: Optional system instruction prompt.
        max_retries: Number of retry attempts on transient failures (default: 1).
        retry_delay_seconds: Delay before retry in seconds.

    Returns:
        Plain-text completion from the LLM.

    Raises:
        ValueError: If configuration (.env) is invalid or missing required keys.
        RuntimeError: If the LLM call fails after retries.
    """
    provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    is_local = provider in ("local", "llama_cpp", "llama-cpp", "gguf", "medgemma")

    # API key only required for non-local providers
    if not is_local and not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is not set. Please set it in .env."
        )

    if not model:
        # Default fallback models per provider
        defaults = {
            "local": "medgemma-1.5-4b-it-Q4_K_M.gguf",
            "llama_cpp": "medgemma-1.5-4b-it-Q4_K_M.gguf",
            "gguf": "medgemma-1.5-4b-it-Q4_K_M.gguf",
            "medgemma": "medgemma-1.5-4b-it-Q4_K_M.gguf",
            "gemini": "gemini-2.5-flash",
            "google": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
        }
        model = defaults.get(provider, "medgemma-1.5-4b-it-Q4_K_M.gguf" if is_local else "gemini-2.5-flash")

    last_error: Optional[Exception] = None
    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        try:
            if is_local:
                return _call_local(model=model, messages=messages, system=system)
            elif provider in ("gemini", "google"):
                return _call_gemini(api_key=api_key, model=model, messages=messages, system=system)
            elif provider == "anthropic":
                return _call_anthropic(api_key=api_key, model=model, messages=messages, system=system)
            elif provider in ("openai", "groq"):
                return _call_openai(api_key=api_key, model=model, messages=messages, system=system)
            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: local, gemini, anthropic, openai."
                )
        except Exception as exc:
            last_error = exc
            if attempt < total_attempts:
                time.sleep(retry_delay_seconds)
            else:
                raise RuntimeError(
                    f"LLM call to provider '{provider}' (model: '{model}') failed after {attempt} attempt(s): {exc}"
                ) from exc

    raise RuntimeError(f"LLM call failed: {last_error}")
