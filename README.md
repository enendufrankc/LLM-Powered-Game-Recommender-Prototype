# Slot Game Recommender — POC

This repository is a small proof-of-concept for an LLM-powered slot-game recommender.

Features
- Programmatic dataset generation (LLM-backed with local fallback) — `generate_data.py` produces `games.json` with >=120 entries.
- Embedding-based similarity engine with cached embeddings and optional LLM reranker — `similarity.py`.
- Minimal Streamlit UI — `app_streamlit.py` to select a game and view recommendations with LLM explanations.

Quickstart (Windows / Powershell)
1. Create a virtual environment and activate it (optional but recommended):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Set your OpenAI API key as an environment variable to use real LLM and embeddings:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

3. Generate the dataset (will use LLM if key present, otherwise local fallback):

```powershell
python generate_data.py
```

4. Run the Streamlit app:

```powershell
streamlit run app_streamlit.py
```

Design notes and trade-offs
- Embeddings: by default the code will call OpenAI embeddings if `OPENAI_API_KEY` is set. Embeddings are cached in `embeddings.json` to avoid repeated calls.
- Reranker: an optional LLM reranker is used to produce human-readable one-line reasons for top recommendations. If no key is present, the system falls back to embedding-only similarity.
- Schema: the dataset follows the exact schema requested (see `generate_data.py`). The LLM generator is asked to return strict JSON; a local generator provides a deterministic fallback.
- Shortcuts: for speed and to keep this a POC I used a simple deterministic hash-based fallback embedding when OpenAI is unavailable. This is not production-grade but sufficient for demonstrating the pipeline.

Tests
- A simple pytest is included to validate dataset length and presence of required fields: `tests/test_dataset.py`.

Security
- API keys must be provided via the `OPENAI_API_KEY` environment variable. Never commit secrets.

Developer note
- Timebox: implemented as a focused POC. Prioritized a complete pipeline: generator, similarity engine, UI, and basic tests. Did not implement a web-hosting deployment or advanced UI polish.
