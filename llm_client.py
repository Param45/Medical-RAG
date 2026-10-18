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

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

    from device_utils import get_llama_gpu_layers

    # 8192 matches .env.example; the 4096 default used to truncate or reject the
    # answer prompt whenever .env was absent, as it is inside the backend image.
    n_ctx = int(os.getenv("LOCAL_LLM_N_CTX", "8192"))
    n_threads_env = os.getenv("LOCAL_LLM_N_THREADS")
    n_threads = int(n_threads_env) if n_threads_env and n_threads_env.isdigit() else None
    n_gpu_layers = get_llama_gpu_layers()

    model_kwargs: Dict[str, Any] = {
        "model_path": resolved_path,
        "n_ctx": n_ctx,
        "verbose": False,
    }
    if n_threads is not None:
        model_kwargs["n_threads"] = n_threads

    if n_gpu_layers != 0:
        model_kwargs["n_gpu_layers"] = n_gpu_layers
        print(f"[LOCAL LLM] Attempting GPU acceleration with n_gpu_layers={n_gpu_layers} on: {resolved_path}")
        try:
            _LOCAL_LLAMA_MODEL = Llama(**model_kwargs)
            _LOCAL_LLAMA_MODEL_PATH = resolved_path
            print(f"[LOCAL LLM] Successfully initialized Llama model with GPU acceleration.")
            return _LOCAL_LLAMA_MODEL
        except Exception as exc:
            print(f"[LOCAL LLM] GPU initialization failed ({exc}). Gracefully falling back to CPU mode...")
            model_kwargs["n_gpu_layers"] = 0

    print(f"[LOCAL LLM] Initializing Llama model in CPU mode from: {resolved_path} (n_ctx={n_ctx})")
    model_kwargs["n_gpu_layers"] = 0
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

    # Leave the reply whatever context the prompt did not use, so an unexpectedly long
    # evidence block yields a shorter answer instead of "requested tokens exceed context".
    # The ceiling must clear a full extraction reply: this model prefaces its JSON
    # with several hundred words of reasoning, and at 2048 the JSON was cut off
    # mid-object. Generation still stops at EOS, so a higher ceiling costs nothing.
    max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS") or 4096)
    try:
        prompt_tokens = sum(len(llm.tokenize(m["content"].encode("utf-8"))) for m in formatted_messages)
        max_tokens = max(256, min(max_tokens, int(llm.n_ctx()) - prompt_tokens - 64))
    except Exception:
        pass  # Sizing is an optimisation; let llama.cpp apply its own limits.

    response = llm.create_chat_completion(
        messages=formatted_messages,
        max_tokens=max_tokens,
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


# ==============================================================================
# TOLERANT JSON PARSING OF MODEL OUTPUT
#
# The local GGUF model does not honour "reply with JSON only". It emits a bare
# "thought" channel marker followed by several hundred words of reasoning, and the
# JSON that follows is frequently cut off mid-object when the reply hits its token
# ceiling. Both callers used a greedy regex plus a single json.loads, so any such
# reply raised and the whole chunk was dropped — which is how a rebuild on the
# local model produced a graph with no clinical facts at all.
# ==============================================================================

# MedGemma/Gemma emit a channel marker on its own line before reasoning.
_REASONING_PREFIX = re.compile(
    r"^\s*(?:<?thought>?|<?think>?|model|assistant)\s*\n", re.IGNORECASE
)
_REASONING_TAGS = re.compile(r"<(thought|think|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def strip_model_preamble(text: str) -> str:
    """Remove reasoning channels and markdown fences from a model reply."""
    cleaned = (text or "").strip()
    cleaned = _REASONING_TAGS.sub("", cleaned).strip()
    cleaned = _REASONING_PREFIX.sub("", cleaned).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _salvage_objects(fragment: str) -> List[Any]:
    """
    Recover every complete top-level {...} object from a possibly truncated array,
    by scanning braces outside of string literals. A reply cut off mid-object keeps
    the facts that were already complete instead of losing the whole chunk.
    """
    objects: List[Any] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    objects.append(json.loads(fragment[start:i + 1]))
                except json.JSONDecodeError:
                    pass  # incomplete or malformed object; keep the rest
                start = -1
            elif depth < 0:
                depth = 0
    return objects


def parse_json_list(text: str) -> List[Any]:
    """
    Parse a list of JSON objects out of a model reply, tolerating reasoning
    preambles, markdown fences, trailing prose and truncation.

    Returns [] when nothing parseable is present.
    """
    cleaned = strip_model_preamble(text)
    if not cleaned:
        return []

    # Whole reply is already a JSON array.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Otherwise locate a real array start ('[' followed by '{') and salvage from
    # there. Anchoring matters: reasoning prose carries both '[' (from Cypher-style
    # sketches) and an odd number of quotes, which would desynchronise the scanner.
    # The answer follows the reasoning, so later candidates are tried first.
    for match in reversed(list(re.finditer(r"\[\s*\{", cleaned))):
        objects = _salvage_objects(cleaned[match.start():])
        if objects:
            return objects

    return _salvage_objects(cleaned)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON object out of a model reply. None when absent."""
    cleaned = strip_model_preamble(text)
    if not cleaned or cleaned.strip().lower() == "null":
        return None
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for obj in _salvage_objects(cleaned):
        if isinstance(obj, dict):
            return obj
    return None


# ==============================================================================
# CLOUD PROVIDER RATE LIMITING
#
# Free-tier Gemini quotas are enforced per minute and per day. A rebuild issues
# ~110 calls back to back, so without throttling the provider starts returning 429
# a few seconds in and the build dies partway through, leaving a half-populated
# graph. Cloud calls therefore pass through a sliding window sized from .env.
# Local calls are never throttled.
# ==============================================================================
_CLOUD_MINUTE_LOG: List[Tuple[float, int]] = []  # (timestamp, estimated tokens)
_CLOUD_DAY_LOG: List[float] = []
_CLOUD_LOCK = threading.Lock()

_RATE_LIMIT_ERROR = re.compile(r"429|RESOURCE_EXHAUSTED|quota|rate.?limit", re.IGNORECASE)


def _estimate_tokens(messages: List[Dict[str, str]], system: Optional[str]) -> int:
    """Rough token count for quota accounting (~4 characters per token)."""
    chars = len(system or "") + sum(len(m.get("content") or "") for m in messages)
    return max(1, chars // 4)


def throttle_cloud_call(est_tokens: int) -> None:
    """
    Block until another cloud request fits inside the configured quotas.

    Raises RuntimeError when the daily request quota is spent, so a long build
    stops with a clear reason instead of retrying against an exhausted key.
    """
    rpm = int(os.getenv("LLM_CLOUD_RPM") or 15)
    tpm = int(os.getenv("LLM_CLOUD_TPM") or 250000)
    rpd = int(os.getenv("LLM_CLOUD_RPD") or 500)

    while True:
        with _CLOUD_LOCK:
            now = time.time()
            _CLOUD_MINUTE_LOG[:] = [e for e in _CLOUD_MINUTE_LOG if now - e[0] < 60.0]
            _CLOUD_DAY_LOG[:] = [t for t in _CLOUD_DAY_LOG if now - t < 86400.0]

            if len(_CLOUD_DAY_LOG) >= rpd:
                raise RuntimeError(
                    f"Daily cloud request quota reached ({rpd} requests/day). "
                    "Wait for the quota to reset before rebuilding again."
                )

            fits_rpm = len(_CLOUD_MINUTE_LOG) < rpm
            fits_tpm = sum(n for _, n in _CLOUD_MINUTE_LOG) + est_tokens <= tpm

            if fits_rpm and fits_tpm:
                _CLOUD_MINUTE_LOG.append((now, est_tokens))
                _CLOUD_DAY_LOG.append(now)
                return

            # Wait for the oldest call to age out of the 60s window.
            wait = max(0.5, 60.0 - (now - _CLOUD_MINUTE_LOG[0][0]) + 0.25)
            reason = "request" if not fits_rpm else "token"
            used = len(_CLOUD_DAY_LOG)

        print(f"[RATE LIMIT] {reason} quota window full; waiting {wait:.1f}s ({used}/{rpd} today)")
        time.sleep(wait)


# ==============================================================================
# GLOBAL LLM ROUTING TOGGLE (Code-Level Only)
#
# By default, the Local LLM (MedGemma GGUF via llama-cpp-python) is used EVERYWHERE
# across all pipeline stages (ingestion, graph build, PageIndex, normalization,
# query classification, and final clinical answer synthesis).
#
# If the Local LLM degrades results or makes things worse on heavy ingestion tasks,
# you can switch back to the External Cloud LLM (Gemini) by changing this single
# line in code:
#     USE_LOCAL_LLM_EVERYWHERE = False
#
# Note: This is an internal developer decision and is NOT exposed in the UI.
# ==============================================================================
USE_LOCAL_LLM_EVERYWHERE: bool = True


def chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_external: bool = False,
    max_retries: int = 1,
    retry_delay_seconds: float = 1.0,
) -> str:
    """
    Unified LLM chat wrapper (SRS FR-5.6.1).

    By default, routes ALL tasks to the Local LLM (MedGemma GGUF).
    If USE_LOCAL_LLM_EVERYWHERE is set to False (or use_external=True is passed),
    ingestion / build tasks route to the configured external cloud provider (Gemini).

    Args:
        messages: List of message dicts with keys 'role' ('user'|'assistant'|'model') and 'content'.
        system: Optional system instruction prompt.
        task: Optional task type ('build' / 'ingestion' vs 'inference' / 'retrieve').
        provider: Optional explicit provider override ('gemini', 'local', 'anthropic', 'openai').
        model: Optional explicit model override.
        api_key: Optional explicit API key override.
        use_external: Optional flag to route this specific call to the External Cloud LLM.
        max_retries: Number of retry attempts on transient failures (default: 1).
        retry_delay_seconds: Delay before retry in seconds.

    Returns:
        Plain-text completion from the LLM.

    Raises:
        ValueError: If configuration (.env) is invalid or missing required keys.
        RuntimeError: If the LLM call fails after retries.
    """
    task_norm = (task or os.getenv("LLM_TASK", "")).strip().lower()

    if not provider:
        if USE_LOCAL_LLM_EVERYWHERE and not use_external:
            # Default everywhere to Local LLM
            provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        elif use_external:
            # Explicit call-level override for external LLM
            provider = (
                os.getenv("LLM_PROVIDER_EXTERNAL")
                or os.getenv("LLM_BUILD_PROVIDER")
                or "gemini"
            ).strip().lower()
        elif task_norm in ("build", "ingestion", "rebuild"):
            # When USE_LOCAL_LLM_EVERYWHERE is False, route build tasks to external LLM
            provider = (
                os.getenv("LLM_PROVIDER_EXTERNAL")
                or os.getenv("LLM_BUILD_PROVIDER")
                or os.getenv("LLM_PROVIDER_INGESTION")
                or "gemini"
            ).strip().lower()
        elif task_norm in ("inference", "query", "retrieve"):
            provider = (
                os.getenv("LLM_INFERENCE_PROVIDER")
                or os.getenv("LLM_PROVIDER")
                or "local"
            ).strip().lower()
        else:
            provider = os.getenv("LLM_PROVIDER", "local").strip().lower()

    if not model:
        if USE_LOCAL_LLM_EVERYWHERE and not use_external and provider in ("local", "llama_cpp", "llama-cpp", "gguf", "medgemma"):
            model = (
                os.getenv("LLM_MODEL")
                or os.getenv("LOCAL_MODEL_PATH")
                or "medgemma-1.5-4b-it-Q4_K_M.gguf"
            ).strip()
        elif use_external or (not USE_LOCAL_LLM_EVERYWHERE and task_norm in ("build", "ingestion", "rebuild")):
            model = (
                os.getenv("LLM_MODEL_EXTERNAL")
                or os.getenv("LLM_BUILD_MODEL")
                or os.getenv("LLM_MODEL_INGESTION")
                or ("gemini-3.5-flash-lite" if provider in ("gemini", "google") else "")
            ).strip()
        elif task_norm in ("inference", "query", "retrieve"):
            model = (
                os.getenv("LLM_INFERENCE_MODEL")
                or os.getenv("LLM_MODEL")
                or ("medgemma-1.5-4b-it-Q4_K_M.gguf" if provider in ("local", "llama_cpp", "llama-cpp", "gguf", "medgemma") else "")
            ).strip()
        else:
            model = os.getenv("LLM_MODEL", "").strip()

    if not api_key:
        if use_external or (not USE_LOCAL_LLM_EVERYWHERE and task_norm in ("build", "ingestion", "rebuild")):
            api_key = (
                os.getenv("LLM_API_KEY")
                or os.getenv("LLM_BUILD_API_KEY")
                or ""
            ).strip()
        elif task_norm in ("inference", "query", "retrieve"):
            api_key = (
                os.getenv("LLM_INFERENCE_API_KEY")
                or os.getenv("LLM_API_KEY")
                or ""
            ).strip()
        else:
            api_key = os.getenv("LLM_API_KEY", "").strip()

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
            "gemini": os.getenv("LLM_MODEL_EXTERNAL", "gemini-3.1-flash-lite"),
            "google": os.getenv("LLM_MODEL_EXTERNAL", "gemini-3.1-flash-lite"),
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
        }
        model = defaults.get(provider, "medgemma-1.5-4b-it-Q4_K_M.gguf" if is_local else "gemini-3.1-flash-lite")

    last_error: Optional[Exception] = None
    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        try:
            if not is_local:
                throttle_cloud_call(_estimate_tokens(messages, system))

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
                delay = retry_delay_seconds
                if not is_local and _RATE_LIMIT_ERROR.search(str(exc)):
                    # A quota rejection only clears when the window rolls over.
                    delay = max(delay, 60.0)
                    print(f"[RATE LIMIT] provider refused on quota; backing off {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"LLM call to provider '{provider}' (model: '{model}') failed after {attempt} attempt(s): {exc}"
                ) from exc

    raise RuntimeError(f"LLM call failed: {last_error}")


def chat_build(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience chat wrapper for rebuilding/ingestion tasks (Local LLM by default, or External LLM if USE_LOCAL_LLM_EVERYWHERE=False)."""
    return chat(messages, system=system, task="build", **kwargs)


def chat_inference(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    **kwargs,
) -> str:
    """Convenience chat wrapper for query/inference tasks (defaults to Local LLM)."""
    return chat(messages, system=system, task="inference", **kwargs)

