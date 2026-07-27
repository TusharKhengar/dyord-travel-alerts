"""
Shared LLM backend used by both the local CLI pipeline and the hosted
Streamlit app. Switches between a local Ollama model (free, offline, used
for local development) and Groq's hosted API (free tier, used when deployed
since hosting platforms can't run a multi-GB local model) via the
LLM_BACKEND environment variable.
"""
import json
import os

import requests

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "groq"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


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
    api_key = os.environ["GROQ_API_KEY"]
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
