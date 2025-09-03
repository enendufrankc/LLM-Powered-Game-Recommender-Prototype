
Task: Build an LLM-powered slot-game recommender prototype
Goal: Deliver a working proof-of-concept repo that (1) programmatically generates a 100+ fictional slot-game dataset using an LLM, (2) builds a similarity engine that returns the top 3–5 similar games for a selected game, and (3) exposes a small UI (Streamlit or Gradio) that shows recommendations and LLM-generated explanation text for each recommended game.

High-level acceptance criteria
- generate_data.py produces games.json with >=100 diverse game objects following a defined schema.
- A similarity engine (embedding-based cosine similarity + optional LLM reranker) returns top 3–5 matches for any selected game.
- A Streamlit or Gradio app loads games.json and shows the selected game, recommended games, and a short LLM explanation for each recommendation.
- README.md describes design choices, how to run, and any trade-offs.
- requirements.txt included. API keys used from environment variables.

Project structure (suggested)
- generate_data.py
- similarity.py
- app_streamlit.py
- games.json (generated)
- embeddings.json (optional cached embeddings)
- requirements.txt
- README.md

Operational constraints
- Language: Python
- UI: Streamlit (preferred) or Gradio
- LLM: any provider (OpenAI example shown). Use your own API key via environment variable (OPENAI_API_KEY).
- Timebox: aim for a working POC in ~4 hours. Prioritize a clear, functional pipeline over complex refinements.

Schema suggestion for a "game" (use exact fields in dataset)
- id: string (uuid or short unique slug)
- name: string
- theme: string (e.g., Ancient Egypt, Space Odyssey)
- volatility: string (Low/Medium/High)
- rtp: float (e.g., 96.2)
- reels: int
- paylines: int | "cluster"
- max_win_x: int (max win multiplier)
- special_features: list[string] (e.g., free spins, cascading reels)
- art_style: string (e.g., cartoon, photorealistic, neon)
- provider_style: string (e.g., classic 3-reel, modern cluster)
- target_audience: string (e.g., high-rollers, casual)
- description: string (1–2 sentences)
- tags: list[string]

Exact LLM prompt for dataset generation
- System prompt: "You are a data generator. Output a JSON array of N slot game objects following this exact JSON schema: {list schema fields with types}. Ensure values are realistic, diverse, plausible, and varied across themes, volatility, RTPs, art style, and features. Use unique names. Return only valid JSON — no commentary."
- User prompt: "Generate 120 games following the schema. Use varied themes and include rich descriptions and tag lists. Keep descriptions short (1–2 sentences)."

Suggested implementation plan / steps for the coding agent

1) generate_data.py
- Read OPENAI_API_KEY from env.
- Use the LLM chat endpoint and provide the schema prompt above. Ask for 120 games.
- Parse and validate the JSON output. If the model returns non-JSON, retry/repair (e.g., attempt to extract first/last balanced JSON array).
- Save to games.json.
- OPTIONAL: compute text embeddings per game and save embeddings.json (embedding for a concatenation of name + theme + features + description + tags).

Sample generate_data.py (OpenAI example)
```python
# ...existing code...
import os
import json
import uuid
import time
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a data generator. Output a single JSON array of 120 slot game objects following this exact schema:
{
  "id": "string",
  "name": "string",
  "theme": "string",
  "volatility": "Low|Medium|High",
  "rtp":  float,
  "reels": int,
  "paylines": "int|cluster",
  "max_win_x": int,
  "special_features": ["string"],
  "art_style": "string",
  "provider_style": "string",
  "target_audience": "string",
  "description": "string (1-2 sentences)",
  "tags": ["string"]
}
Return only the JSON array and nothing else.
"""

def generate_games():
    prompt = "Generate 120 unique slot games following the schema exactly."
    for attempt in range(3):
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace with your model
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":prompt}],
            max_tokens=4000
        )
        text = resp["choices"][0]["message"]["content"].strip()
        try:
            games = json.loads(text)
            if isinstance(games, list) and len(games) >= 100:
                return games
        except Exception:
            # naive cleanup: try to extract JSON block
            start = text.find('[')
            end = text.rfind(']') + 1
            try:
                games = json.loads(text[start:end])
                return games
            except Exception:
                time.sleep(1)
    raise RuntimeError("Failed to generate valid JSON games from LLM")

if __name__ == "__main__":
    games = generate_games()
    # ensure ids are set
    for g in games:
        if not g.get("id"):
            g["id"] = str(uuid.uuid4())
    with open("games.json","w",encoding="utf-8") as f:
        json.dump(games, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(games)} games to games.json")
# ...existing code...
```

2) similarity.py — embeddings + cosine similarity + optional LLM rerank
- Compute embeddings for each game using an embedding model (e.g., text-embedding-3).
- Represent each game by a "combined text" string: name + theme + description + ','.join(special_features) + ','.join(tags) + art_style + volatility + provider_style.
- Store embeddings in embeddings.json keyed by game id to avoid re-requesting.
- For input game id, compute cosine similarity and return top K (e.g., 20) candidates.
- Optionally call the LLM to rerank the top candidates to a final top 3–5 with explanation per result. Provide a focused rerank prompt: include candidate games and ask for the top 5 matches and 1-sentence reason each.

Sample similarity.py
```python
# ...existing code...
import os
import json
import math
from typing import List, Dict
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-large"

def combine_game_text(g):
    parts = [
        g.get("name",""),
        g.get("theme",""),
        g.get("description",""),
        "features:" + ",".join(g.get("special_features",[])),
        "tags:" + ",".join(g.get("tags",[])),
        g.get("art_style",""),
        g.get("volatility",""),
        g.get("provider_style",""),
    ]
    return " | ".join([p for p in parts if p])

def get_embedding(text):
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)
    return resp["data"][0]["embedding"]

def load_games(path="games.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_embeddings(games, out_path="embeddings.json"):
    data = {}
    for g in games:
        tid = g["id"]
        text = combine_game_text(g)
        data[tid] = {
            "embedding": get_embedding(text),
            "text": text,
            "name": g["name"]
        }
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(data, f)
    return data

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)

def recommend(game_id, games, embeddings, top_k=5, candidates=20, rerank_with_llm=True):
    if game_id not in embeddings:
        raise KeyError("No embedding for game id")
    source_emb = embeddings[game_id]["embedding"]
    scores = []
    for gid,v in embeddings.items():
        if gid == game_id: continue
        scores.append((gid, cosine(source_emb, v["embedding"])))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [gid for gid,_ in scores[:candidates]]
    if not rerank_with_llm:
        return top_candidates[:top_k]
    # Rerank via LLM: ask for top_k with explanation
    candidate_payload = []
    gid_to_game = {g["id"]: g for g in games}
    for gid in top_candidates:
        g = gid_to_game[gid]
        candidate_payload.append({
            "id": gid,
            "name": g["name"],
            "theme": g["theme"],
            "volatility": g["volatility"],
            "features": g["special_features"],
            "tags": g["tags"],
            "description": g["description"]
        })
    prompt = f"""You are a recommender. Given a source game and a list of candidate games, return the top {top_k} candidate IDs sorted by similarity with a one-sentence reason each.  
Source game ID: {game_id}
Source game: {json.dumps(gid_to_game[game_id], ensure_ascii=False)}
Candidates: {json.dumps(candidate_payload, ensure_ascii=False)}
Return JSON like: [{{"id":"...","reason":"..."}},...] and nothing else."""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=1000
    )
    text = resp["choices"][0]["message"]["content"].strip()
    try:
        out = json.loads(text)
        return out
    except Exception:
        # fallback: return top candidates w/o reasons
        return [{"id": gid, "reason": ""} for gid in top_candidates[:top_k]]
# ...existing code...
```

3) app_streamlit.py (UI)
- Load games.json and embeddings.json (if used).
- Selectbox of game names.
- On selection, call similarity.recommend and show recommended games with name, theme, a short description, and the LLM explanation (if provided).
- Keep UI minimal, readable, and responsive.

Sample Streamlit app
```python
# ...existing code...
import streamlit as st
import json
from similarity import load_games, recommend

games = load_games("games.json")
id_to_game = {g["id"]: g for g in games}
game_names = [g["name"] for g in games]
name_to_id = {g["name"]: g["id"] for g in games}

st.title("Because you played ... — Game Recommender")

sel = st.selectbox("Select a game you played", game_names)
if sel:
    gid = name_to_id[sel]
    recs = recommend(gid, games, __import__("json").load(open("embeddings.json")) if st.sidebar.checkbox("Use cached embeddings") else __import__("json").load(open("embeddings.json")))
    st.subheader("You might like:")
    for i, r in enumerate(recs):
        # r may be {"id": "...","reason":"..."} or id string
        if isinstance(r, dict):
            rid = r["id"]; reason = r.get("reason","")
        else:
            rid = r; reason = ""
        g = id_to_game[rid]
        st.markdown(f"**{i+1}. {g['name']}** — {g['theme']} ({g['volatility']}, RTP {g['rtp']}%)")
        st.write(g["description"])
        if reason:
            st.info(reason)
# ...existing code...
```

requirements.txt (start minimal)
```text
# ...existing code...
openai>=1.0.0
streamlit>=1.18
numpy
```

Windows run instructions (PowerShell)
- Set API key (PowerShell for current session):
  - $env:OPENAI_API_KEY="sk-xxxx"
- Or permanently:
  - setx OPENAI_API_KEY "sk-xxxx"
- Install:
  - python -m venv .venv
  - Activate.ps1
  - pip install -r requirements.txt
- Generate data:
  - python generate_data.py
- Build embeddings (if separate step):
  - python -c "import similarity, json; games=json.load(open('games.json')); similarity.build_embeddings(games, 'embeddings.json')"
- Run app:
  - streamlit run app_streamlit.py

Testing & validation
- Unit tests: check dataset length >= 100, fields present, numeric ranges (rtp between 85 and 99), volatility in expected set.
- Integration: pick one game from games.json and verify recommend() returns 3–5 results and each recommended id exists in dataset.

README.md content (required sections)
- Project overview and challenge description.
- Schema decision and why fields chosen.
- Dataset generation approach (LLM prompt + validation + retries).
- Similarity engine design: embedding step, cosine similarity, optional LLM reranker (trade-offs: speed vs quality).
- How to run on Windows (commands above).
- How to swap LLM provider or model (notes on environment variable).
- Limitations and next steps (e.g., add caching, better downstream evaluation, offline fallback).

Notes on design decisions / suggestions to the coding agent
- Use embeddings for scale/speed + an LLM reranker for more nuanced results.
- Cache embeddings to avoid expensive repeats; store embeddings.json.
- Keep the UI simple; focus time on dataset quality and clear README.
- Protect API keys: read from env only, do not commit keys.
- If LLM is unreliable in returning strict JSON, implement a cleanup routine (extract first balanced JSON array bracket block, or ask the model to return only base64 encoded JSON to reduce stray commentary).
- For evaluation, compute average similarity within recommended set and show it for debugging.

Deliverables (what to commit)
- generate_data.py
- games.json
- similarity.py
- app_streamlit.py
- embeddings.json (if computed)
- requirements.txt
- README.md
- small test script or tests/ folder (optional)

Example LLM rerank prompt (exact)
"You are a domain-aware recommender. Rank the following candidate games by similarity to the source game. For each top 5, return its id and a one-sentence reason referencing concrete matching attributes (theme, volatility, features, art style). Return JSON array only."

Final instruction to the agent (paste as single instruction)
1. Implement generate_data.py using the dataset generation prompt above. Produce games.json (>=120 entries).
2. Implement similarity.py that computes embeddings, caches them, and recommends top candidates using cosine similarity. Add optional LLM reranker that returns final top 3–5 with one-sentence reasons.
3. Implement a minimal Streamlit UI app_streamlit.py that allows selecting a game and viewing recommendations with explanations.
4. Add requirements.txt and README.md describing design choices and run instructions (Windows).
5. Ensure API key usage via env var and do not hardcode secrets.
6. Include basic tests: dataset length and schema checks.
7. Produce a short developer note (in README) indicating time used and any shortcuts/trade-offs.

If you want, I can also:
- produce ready-to-run implementations for the three files (generate_data.py, similarity.py, app_streamlit.py) based on OpenAI calls and the prompts above;
- or produce the README.md content template now.

Which of those should I generate next?