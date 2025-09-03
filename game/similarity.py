import os
import json
import math
from typing import List, Dict, Tuple, Optional

EMBED_FILE = os.path.join(os.path.dirname(__file__), "database", "embeddings.json")


def combined_text(game: Dict) -> str:
    parts = [game.get("name", ""), game.get("theme", ""), game.get("description", "")]
    parts += game.get("special_features", []) or []
    parts += game.get("tags", []) or []
    parts.append(game.get("art_style", ""))
    parts.append(game.get("volatility", ""))
    parts.append(game.get("provider_style", ""))
    return " \n ".join([str(p) for p in parts if p])


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


class SimilarityEngine:
    """
    Uses OpenAI embeddings (strictly) to embed game data and perform similarity search.
    Requires an OpenAI client instance (openai.OpenAI) passed as openai_client.
    Embeddings are stored in game/database/embeddings.json as a mapping id -> vector.
    """

    def __init__(self, games: List[Dict], openai_client, embedding_model: str = "text-embedding-3-small"):
        if openai_client is None:
            raise RuntimeError("SimilarityEngine requires an OpenAI client. Pass openai_client=OpenAI(api_key=...)")
        self.openai = openai_client
        self.model = embedding_model
        self.games = {g["id"]: g for g in games}
        self.embeddings: Dict[str, List[float]] = self._load_embeddings()

    def _load_embeddings(self) -> Dict[str, List[float]]:
        if os.path.exists(EMBED_FILE):
            try:
                with open(EMBED_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {k: [float(x) for x in v] for k, v in data.items()}
            except Exception as e:
                print("Failed to load embeddings:", e)
        return {}

    def _save_embeddings(self) -> None:
        dirpath = os.path.dirname(EMBED_FILE)
        os.makedirs(dirpath, exist_ok=True)
        with open(EMBED_FILE, "w", encoding="utf-8") as f:
            json.dump(self.embeddings, f, indent=2)

    def _embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenAI embeddings API for a list of texts. Returns list of embeddings (aligned with texts).
        """
        if not texts:
            return []
        resp = self.openai.embeddings.create(model=self.model, input=texts)
        emb_list: List[List[float]] = []
        try:
            for item in resp["data"]:
                emb_list.append(item["embedding"])
            return emb_list
        except Exception:
            pass
        try:
            for item in resp.data:
                emb_list.append(item.embedding)
            return emb_list
        except Exception as e:
            raise RuntimeError(f"Unable to parse embeddings response: {e}")

    def ensure_embeddings(self, batch_size: int = 32) -> None:
        """
        Ensure every game in self.games has an embedding stored in embeddings.json.
        Uses OpenAI embeddings API and saves the results.
        """
        missing = [gid for gid in self.games.keys() if gid not in self.embeddings]
        if not missing:
            return

        print(f"Embedding {len(missing)} games with OpenAI ({self.model})...")
        for i in range(0, len(missing), batch_size):
            batch_ids = missing[i : i + batch_size]
            texts = [combined_text(self.games[gid]) for gid in batch_ids]
            emb_batch = self._embed_texts_batch(texts)
            if len(emb_batch) != len(batch_ids):
                raise RuntimeError("Embedding batch size mismatch")
            for gid, emb in zip(batch_ids, emb_batch):
                self.embeddings[gid] = [float(x) for x in emb]
        self._save_embeddings()
        print("Saved embeddings to", EMBED_FILE)

    def _embed_single_and_persist(self, gid: str) -> List[float]:
        txt = combined_text(self.games[gid])
        emb = self._embed_texts_batch([txt])[0]
        self.embeddings[gid] = [float(x) for x in emb]
        self._save_embeddings()
        return self.embeddings[gid]

    def _explain_similarity(self, source: Dict, candidate: Dict, score: Optional[float]) -> str:
        """
        Produce a concise, human-readable explanation why candidate is similar to source.
        Uses field comparisons: theme, tags, special_features, art_style, volatility, provider_style, name overlap.
        """
        parts: List[str] = []
        # theme
        s_theme = (source.get("theme") or "").strip().lower()
        c_theme = (candidate.get("theme") or "").strip().lower()
        if s_theme and c_theme and s_theme == c_theme:
            parts.append(f"shares the same theme ({source.get('theme')}).")

        # art style
        s_art = (source.get("art_style") or "").strip().lower()
        c_art = (candidate.get("art_style") or "").strip().lower()
        if s_art and c_art and s_art == c_art:
            parts.append(f"has a similar art style ({source.get('art_style')}).")

        # volatility
        s_vol = (source.get("volatility") or "").strip().lower()
        c_vol = (candidate.get("volatility") or "").strip().lower()
        if s_vol and c_vol and s_vol == c_vol:
            parts.append(f"matches volatility ({source.get('volatility')}).")

        # provider style
        s_prov = (source.get("provider_style") or "").strip().lower()
        c_prov = (candidate.get("provider_style") or "").strip().lower()
        if s_prov and c_prov and s_prov == c_prov:
            parts.append(f"follows a similar provider style ({source.get('provider_style')}).")

        # tags overlap
        s_tags = set([t.strip().lower() for t in (source.get("tags") or []) if t])
        c_tags = set([t.strip().lower() for t in (candidate.get("tags") or []) if t])
        tags_common = sorted(list(s_tags & c_tags))
        if tags_common:
            parts.append(f"shares tags: {', '.join(tags_common)}.")

        # special features overlap
        s_feats = set([f.strip().lower() for f in (source.get("special_features") or []) if f])
        c_feats = set([f.strip().lower() for f in (candidate.get("special_features") or []) if f])
        feats_common = sorted(list(s_feats & c_feats))
        if feats_common:
            parts.append(f"both include features like {', '.join(feats_common)}.")

        # name overlap (simple word intersection)
        s_name_words = set(w.lower() for w in (source.get("name") or "").split() if len(w) > 2)
        c_name_words = set(w.lower() for w in (candidate.get("name") or "").split() if len(w) > 2)
        name_common = sorted(list(s_name_words & c_name_words))
        if name_common:
            parts.append(f"name similarity ({', '.join(name_common)}).")

        # build explanation
        if not parts:
            explanation = "Similar based on overall semantic embedding similarity."
        else:
            explanation = " ".join(parts)

        if score is not None:
            explanation = f"{explanation} (Embedding similarity: {score:.3f})"

        return explanation

    def recommend(self, source_id: str, top_k: int = 5, rerank_with_llm: bool = False, rerank_model: str = "gpt-4o-mini") -> List[Dict]:
        """
        Return top_k similar games to source_id.
        - requires embeddings for dataset; will embed missing games using OpenAI.
        - attempts optional LLM rerank if requested and available.
        Result format: list of {"game": <game dict>, "score": float, "reason": str}
        """
        if source_id not in self.games:
            raise KeyError("source_id not found in games")

        # ensure dataset embeddings
        self.ensure_embeddings()

        # ensure source embedding
        if source_id not in self.embeddings:
            self._embed_single_and_persist(source_id)

        source_emb = self.embeddings[source_id]

        scores: List[Tuple[str, float]] = []
        for gid, emb in self.embeddings.items():
            if gid == source_id:
                continue
            sc = cosine(source_emb, emb)
            scores.append((gid, sc))
        scores.sort(key=lambda x: x[1], reverse=True)

        # candidate pool for optional rerank
        candidate_pool = [gid for gid, _ in scores[: max(20, top_k)]]
        candidates = [self.games[gid] for gid in candidate_pool]

        # try LLM rerank if requested
        if rerank_with_llm:
            try:
                reranked = self._llm_rerank(source_id, candidates, final_k=top_k, model=rerank_model)
                if reranked:
                    # ensure returned items include explanations; if LLM didn't include score, augment it
                    for r in reranked:
                        if r.get("score") is None and r.get("game") and r["game"].get("id") in self.embeddings:
                            r["score"] = float(cosine(self.embeddings[source_id], self.embeddings[r["game"]["id"]]))
                        if not r.get("reason"):
                            r["reason"] = self._explain_similarity(self.games[source_id], r.get("game", {}), r.get("score"))
                    return reranked
            except Exception as e:
                print("LLM rerank failed — falling back to embedding-only results:", e)

        # fallback: return top_k by embedding score with richer explanation
        results: List[Dict] = []
        for gid, sc in scores[:top_k]:
            g = self.games.get(gid)
            reason = self._explain_similarity(self.games[source_id], g, sc)
            results.append({"game": g, "score": float(sc), "reason": reason})
        return results

    def _llm_rerank(self, source_id: str, candidates: List[Dict], final_k: int = 5, model: str = "gpt-4o-mini") -> List[Dict]:
        """
        Use an LLM to rerank candidate games by similarity to the source game.
        The function asks the model to return a JSON array of objects: [{ "id": "<game id>", "reason": "<one-sentence reason>"}]
        """
        if not candidates:
            return []

        source = self.games[source_id]
        source_txt = combined_text(source)

        candidate_payload = []
        for c in candidates:
            candidate_payload.append({"id": c["id"], "text": combined_text(c)})

        system_msg = (
            "You are a domain-aware recommender assistant. "
            "Given a source game and candidate games, rank candidates by similarity to the source. "
            "Focus on theme, features, art style, volatility and tags. "
            "Return a JSON array (no extra text) of up to the requested number of items. "
            "Each item must be an object with keys: id, reason (one short sentence)."
        )

        user_msg = {
            "source_id": source_id,
            "source_text": source_txt,
            "candidates": candidate_payload,
            "instructions": f"Return up to {final_k} items. JSON array only."
        }

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False, indent=2)},
        ]

        resp = self.openai.chat.completions.create(model=model, messages=messages, temperature=0.0)
        try:
            # extract content
            try:
                content = resp.choices[0].message.content
            except Exception:
                content = resp["choices"][0]["message"]["content"]

            # extract first JSON array
            import re
            m = re.search(r"(\[.*\])", content, flags=re.S)
            json_text = m.group(1) if m else content.strip()
            parsed = json.loads(json_text)

            results: List[Dict] = []
            for item in parsed[:final_k]:
                gid = item.get("id")
                reason = item.get("reason", "")
                g = self.games.get(gid)
                score = None
                if gid in self.embeddings:
                    score = float(cosine(self.embeddings[source_id], self.embeddings[gid]))
                # augment reason if sparse
                if not reason:
                    reason = self._explain_similarity(source, g, score)
                results.append({"game": g, "score": score, "reason": reason})
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM rerank response: {e}")
