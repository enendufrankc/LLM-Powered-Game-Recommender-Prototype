import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from game.similarity import SimilarityEngine

load_dotenv()


@st.cache_data
def load_games():
    """Load local games data from game/database/games.json."""
    local_path = os.path.join(os.path.dirname(__file__), "game", "database", "games.json")
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            # support {"games": [...]} or plain list
            if isinstance(d, dict) and "games" in d:
                return d["games"]
            if isinstance(d, list):
                return d
    except Exception:
        return []


@st.cache_resource
def get_engine(games):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in your environment or .env and restart Streamlit.")
    client = OpenAI(api_key=api_key)
    engine = SimilarityEngine(games, openai_client=client, embedding_model="text-embedding-3-small")
    # This will create and persist embeddings for any missing games
    engine.ensure_embeddings()
    return engine


def main():
    st.set_page_config(page_title="Slot Game Recommender", layout="wide")
    st.title("Slot Game Recommender — OpenAI Embeddings")

    games = load_games()
    if not games:
        st.error("No games found in 'game/database/games.json'. Run `generate_data.py` or provide the dataset.")
        return

    try:
        engine = get_engine(games)
    except Exception as e:
        st.error(f"Failed to initialize similarity engine: {e}")
        return

    game_map = {g["name"]: g for g in games}
    names = sorted(game_map.keys())

    selected_name = st.selectbox("Select a game", names)
    if not selected_name:
        st.stop()
    source = game_map[selected_name]

    col_top = st.columns([3, 1])
    with col_top[1]:
        top_k = st.selectbox("Top K", [3, 5, 10], index=1)
        rerank = st.checkbox("Rerank with LLM (experimental)", value=False)

    with st.spinner("Computing recommendations..."):
        try:
            results = engine.recommend(source["id"], top_k=top_k, rerank_with_llm=rerank)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
            return

    col1, col2 = st.columns([2, 3])
    with col1:
        st.header("Selected Game")
        st.subheader(source["name"])
        st.markdown(
            f"**Theme:** {source.get('theme','')}  \n**Volatility:** {source.get('volatility','')}  \n**RTP:** {source.get('rtp','')}%  \n**Art style:** {source.get('art_style','')}"
        )
        st.write(source.get("description", ""))
        st.write("**Special features:**", ", ".join(source.get("special_features", [])))

    with col2:
        st.header("Recommendations")
        for r in results:
            g = r.get("game", {})
            reason = r.get("reason", "")
            score = r.get("score")
            st.subheader(g.get("name", ""))
            st.markdown(
                f"**Theme:** {g.get('theme','')} — **Volatility:** {g.get('volatility','')} — **RTP:** {g.get('rtp','')}%"
            )
            st.write(g.get("description", ""))
            meta = []
            if score is not None:
                meta.append(f"Similarity: {score:.3f}")
            if reason:
                meta.append(f"Reason: {reason}")
            if meta:
                st.caption(" — ".join(meta))


if __name__ == "__main__":
    main()
