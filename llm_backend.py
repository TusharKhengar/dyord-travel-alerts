"""
Shared LLM backend used by both the local CLI pipeline and the hosted
Streamlit app. Switches between a local Ollama model (free, offline, used
for local development) and Groq's hosted API (free tier, used when deployed
since hosting platforms can't run a multi-GB local model) via the
LLM_BACKEND environment variable.

When deployed on Streamlit Community Cloud, secrets set in the dashboard are
available via st.secrets but NOT as os.environ entries. This module merges
both sources so deployment "just works" with the documented TOML secrets.
"""
import json
import os
import re

import requests


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read a config value from os.environ first, then st.secrets."""
    value = os.getenv(key)
    if value is not None:
        return value.strip().strip("'\"")
    try:
        import streamlit as st
        val = st.secrets.get(key, default)
        if isinstance(val, str):
            val = val.strip().strip("'\"")
        return val
    except Exception:
        return default


LLM_BACKEND = _get_secret("LLM_BACKEND", "ollama")  # "ollama" or "groq"
OLLAMA_HOST = _get_secret("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _get_secret("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = _get_secret("GROQ_MODEL", "")  # Empty by default to allow auto-detection

_cached_groq_model = None


def _extract_json(text: str) -> dict:
    """Extract and parse a JSON object from text (handling markdown fences and whitespace)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the outermost { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse valid JSON from: {text[:200]}")


def _resolve_groq_model(client) -> str:
    """Find the best available chat model for the current Groq API key."""
    global _cached_groq_model
    if _cached_groq_model:
        return _cached_groq_model

    if GROQ_MODEL:
        _cached_groq_model = GROQ_MODEL
        return _cached_groq_model

    try:
        models = client.models.list()
        available_ids = [m.id for m in models.data if "whisper" not in m.id.lower() and "guard" not in m.id.lower()]

        # Priority preferences
        preferred = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "groq/compound-mini",
            "qwen/qwen3.6-27b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]
        for pref in preferred:
            if pref in available_ids:
                _cached_groq_model = pref
                return _cached_groq_model

        if available_ids:
            _cached_groq_model = available_ids[0]
            return _cached_groq_model
    except Exception:
        pass

    _cached_groq_model = "qwen/qwen3.8-27b"
    return _cached_groq_model


def chat_json(system_prompt: str, user_content: str) -> dict:
    """Send a chat request and parse a JSON object out of the reply."""
    if LLM_BACKEND == "groq":
        return _chat_json_groq(system_prompt, user_content)
    return _chat_json_ollama(system_prompt, user_content)


def _chat_json_ollama(system_prompt: str, user_content: str) -> dict:
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return _extract_json(content)


def _chat_json_groq(system_prompt: str, user_content: str) -> dict:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it as an environment variable or "
            "in Streamlit secrets (Advanced settings → Secrets)."
        )

    from groq import Groq

    client = Groq(api_key=api_key)
    model_to_use = _resolve_groq_model(client)

    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        return _extract_json(completion.choices[0].message.content)
    except Exception:
        # Retry without response_format if the model doesn't support json_object mode
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return _extract_json(completion.choices[0].message.content)

