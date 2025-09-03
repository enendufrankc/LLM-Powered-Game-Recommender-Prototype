# Slot Game Recommender — Proof of Concept (POC)

A compact prototype that demonstrates an LLM-assisted slot-game recommender pipeline implemented in this repository.

This README describes exactly what is present in the project, how to set it up locally, and how to run the Streamlit demo.

## Contents and purpose

- Programmatic dataset generator: `game/generate_data.py` (generates `game/database/games.json`).
- Embedding & similarity logic: `game/similarity.py` (manages embeddings and returns a similarity engine).
- Minimal Streamlit UI: `main.py` (launches the interactive demo).
- Local data cache: `game/database/games.json` (dataset) and `game/database/embeddings.json` (embedding cache; ignored by git).
- An exploratory notebook used during development: `experiment/trial.ipynb`.

## Quick checklist (what I'll cover below)

- Clone the repo and create a virtual environment
- Install dependencies
- Provide your OpenAI key via `.env` (this project stores the key there)
- Generate the dataset (if needed)
- Run the Streamlit app
- Notes about how similarity is computed, files, and troubleshooting

## Prerequisites

- Python 3.10+
- Git
- `streamlit` and the packages listed in `requirements.txt`

## Setup — clone, venv, install

1. Clone the repository and change into it:

```powershell
git clone https://github.com/enendufrankc/LLM-Powered-Game-Recommender-Prototype.git
cd "LLM-Powered-Game-Recommender-Prototype"
```

2. Create and activate a virtual environment (PowerShell example):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the project's Python dependencies:

```powershell
pip install -r requirements.txt
```

## OpenAI key and `.env`

- Create or verify a `.env` file at the repository root with the single line:

```
OPENAI_API_KEY=sk-...
```

- The code in this repository will read the key (the project was developed assuming the key is kept in `.env`). If you prefer not to use a `.env` file, setting the `OPENAI_API_KEY` environment variable in your shell is also an option.

## Generate the dataset

The generator script `game/generate_data.py` produces the JSON dataset at `game/database/games.json`.

Run the generator if you need to (the repo already contains a `game/database/games.json` file that the demo can use):

```powershell
python -m game.generate_data
# or
python game\generate_data.py
```

The generator will use the OpenAI key if available; otherwise it falls back to a deterministic local generator so the pipeline can run offline for POC purposes.

## Run the Streamlit app

Start the UI with:

```powershell
streamlit run main.py
```

What the UI does (implemented in `main.py`):

- Loads games from `game/database/games.json`.
- Initializes the similarity engine from `game/similarity.py` (it reads or creates `game/database/embeddings.json`).
- Allows selecting a source game and viewing top-K similar games. There is an experimental option to rerank or annotate results with LLM-generated reasons when the OpenAI key is available.

## How recommendations are computed (brief)

- Each game's text is converted to an embedding vector.
- If an OpenAI key is present the code uses OpenAI to compute embeddings; otherwise a deterministic fallback embedding generator is used so recommendations still work for the POC.
- Embeddings are cached in `game/database/embeddings.json` to avoid recomputing or re-requesting them.
- The similarity engine performs vector similarity (top-K retrieval). The code for this is in `game/similarity.py`.

## Files & entry points (what's actually in this project)

- `main.py` — Streamlit UI entry point.
- `game/generate_data.py` — dataset generator.
- `game/similarity.py` — embedding management and similarity engine.
- `game/database/games.json` — dataset used by the demo (version checked into the repo).
- `game/database/embeddings.json` — local embedding cache (ignored by git).
- `experiment/trial.ipynb` — exploratory notebook used during development.

## Troubleshooting

- "No games found" in the UI: ensure `game/database/games.json` exists; run `python -m game.generate_data` to create it.
- If LLM features do not work: ensure your OpenAI key is present in `.env` as `OPENAI_API_KEY` or set in your environment.
- Slow runs or repeated embedding calls: confirm `game/database/embeddings.json` contains cached vectors; the project caches embeddings to speed repeated runs.
- Streamlit errors on start: ensure the virtual environment is activated and dependencies from `requirements.txt` are installed.

## Developer notes

- This is a focused POC. The project includes deterministic fallbacks for offline development so you can run the demo without an OpenAI key.
- Embeddings are cached locally in `game/database/embeddings.json` and that file is ignored by git.
- There are no automated tests included in this repository at the moment.

## Extending the project

- Change dataset generation: modify `game/generate_data.py`.
- Change embedding or similarity behavior: modify `game/similarity.py`.
- Modify UI: edit `main.py`.

## License & attribution

Follow licensing and usage terms for any third-party services you use (e.g., OpenAI). The code in this repo is a proof-of-concept.

---

If you'd like, I can also (a) add a short example `.env.example`, (b) add a small helper script to initialize the embedding cache, or (c) add a brief developer checklist for making changes to the similarity logic — tell me which you'd prefer.
