"""
Medical Records RAG Demo — LLM Client Wrapper (SRS §5.6)

Single thin wrapper function `chat(messages, system=None) -> str` used by:
- Terminology normalization fallback (SRS §5.4.2)
- GraphRAG relationship extraction (SRS §6.1)
- PageIndex summarization and tree traversal (SRS §7.1, §7.2)
- Answer generation (SRS §8.2)

Loads LLM_PROVIDER, LLM_API_KEY, and LLM_MODEL from .env.
Includes a single retry on transient errors.
"""

import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()


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
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is not set. Please set it in .env."
        )

    if not model:
        # Default fallback models per provider
        defaults = {
            "gemini": "gemini-2.5-flash",
            "google": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o",
        }
        model = defaults.get(provider, "gemini-2.5-flash")

    last_error: Optional[Exception] = None
    total_attempts = 1 + max_retries

    for attempt in range(1, total_attempts + 1):
        try:
            if provider in ("gemini", "google"):
                return _call_gemini(api_key=api_key, model=model, messages=messages, system=system)
            elif provider == "anthropic":
                return _call_anthropic(api_key=api_key, model=model, messages=messages, system=system)
            elif provider in ("openai", "groq"):
                return _call_openai(api_key=api_key, model=model, messages=messages, system=system)
            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: gemini, anthropic, openai."
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
