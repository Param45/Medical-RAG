"""
Medical Records RAG Demo — LLM Smoke Test Script

Tests the live LLM API configuration from .env by sending a minimal prompt.
Maps to BUILD_GUIDE Task 2.1.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from llm_client import chat


def run_smoketest() -> None:
    provider = os.getenv("LLM_PROVIDER", "(not set)")
    model = os.getenv("LLM_MODEL", "(not set)")
    api_key = os.getenv("LLM_API_KEY", "")
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "(too short/empty)"

    print("==================================================")
    print("           LLM CLIENT SMOKE TEST                 ")
    print("==================================================")
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"API Key  : {masked_key}")
    print("--------------------------------------------------")

    test_messages = [
        {"role": "user", "content": "Say hello in one word."}
    ]

    print("Sending prompt: 'Say hello in one word.' ...")
    start_time = time.time()
    try:
        response = chat(test_messages)
        elapsed = time.time() - start_time
        print("\n=== SUCCESS ===")
        print(f"Response ({elapsed:.2f}s): {response.strip()}")
        print("==================================================")
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"\n=== FAILED ({elapsed:.2f}s) ===")
        print(f"Error: {exc}")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    run_smoketest()
