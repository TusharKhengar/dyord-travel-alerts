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

import requests


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read a config value from os.environ first, then st.secrets."""
    value = os.getenv(key)
    if value is not None:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


LLM_BACKEND = _get_secret("LLM_BACKEND", "ollama")  # "ollama" or "groq"
OLLAMA_HOST = _get_secret("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _get_secret("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = _get_secret("GROQ_MODEL", "llama-3.1-8b-instant")


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
    return json.loads(content)


def _chat_json_groq(system_prompt: str, user_content: str) -> dict:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it as an environment variable or "
            "in Streamlit secrets (Advanced settings → Secrets)."
        )
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    except ImportError:
        # Fallback to raw requests if groq SDK is not installed
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
