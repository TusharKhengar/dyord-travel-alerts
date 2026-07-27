# DYORD — Traveler News Impact Analyzer

Fetches recent news for a location, resolves the real article behind each
Google News link, and uses an LLM to judge whether the article is actually
relevant to a traveler's safety or plans — filtering out the noise (celebrity
gossip, local politics, etc.) that a simple keyword search can't distinguish
from real risk.

## How it works

1. **Fetch** — [`gnews`](https://github.com/ranahaani/gnews) pulls recent
   headlines for the given location.
2. **Resolve** — Google News wraps every link in a redirect token;
   [`googlenewsdecoder`](https://pypi.org/project/googlenewsdecoder/)
   decodes it back to the original publisher URL.
3. **Extract** — the article page is fetched and its main text pulled out
   with BeautifulSoup.
4. **Classify** — an LLM (see below) reads the title + article text and
   returns a structured judgment: is this relevant to travelers, and if so,
   how severe (`low` / `medium` / `high`)?
5. **Present** — results are ranked by severity and shown as traveler alerts,
   both via a CLI script and a Streamlit web app.

## Two LLM backends, one codebase

`llm_backend.py` abstracts the classification call behind a single
`chat_json()` function, switched via the `LLM_BACKEND` env var:

- **`ollama`** (default) — runs a small local model (`llama3.2:3b`) via
  [Ollama](https://ollama.com), fully offline and free. Used for local
  development.
- **`groq`** — calls Groq's hosted OpenAI-compatible API (`GROQ_API_KEY`
  required), used for the public deployment since free hosting tiers can't
  run a multi-GB local model.

## Run locally

```bash
pip install -r requirements.txt

# install Ollama (https://ollama.com) and pull a model once:
ollama pull llama3.2:3b

# CLI:
python dyord_pipeline.py "Mumbai" --max-articles 15

# Web UI:
streamlit run app.py
```

## A note on model quality

`llama3.2:3b` (the default local Ollama model) is small enough to run on a
laptop CPU, but it sometimes contradicts its own stated reasoning — e.g.
writing "unlikely to affect travelers" but still assigning `low` severity.
That's a real limitation of small models, not a bug in the classification
logic. The hosted deployment uses Groq's `llama-3.3-70b-versatile`, which is
far more consistent; if you want better local results, swap `OLLAMA_MODEL` to
a larger model (e.g. `llama3.1:8b`) at the cost of slower inference.

## Deploy

Deployed on **Streamlit Community Cloud** (share.streamlit.io), pointed at
this repo's `main` branch and `app.py`. Set these in the app's Secrets
(Advanced settings → Secrets, TOML format):

```toml
LLM_BACKEND = "groq"
GROQ_API_KEY = "your_key_here"
```

A `Dockerfile` is also included in case you'd rather self-host (Render,
Fly.io, Railway, or a Hugging Face Docker Space if you have a PRO plan — as
of this writing HF requires a paid plan for any Space that runs compute,
only static Spaces are free).

## Why not a custom-trained model?

An earlier version of this project trained a from-scratch BiLSTM classifier
on scraped news. That approach had a fundamental data leakage bug (train/test
split happened after `.repeat()`, so both splits drew from the same pool) and
no real labeled dataset to train on. Rather than patch a model that couldn't
validate itself honestly, this version uses an instruction-following LLM,
which does this kind of contextual judgment call far more reliably than a
small classifier trained on ad-hoc labels.
