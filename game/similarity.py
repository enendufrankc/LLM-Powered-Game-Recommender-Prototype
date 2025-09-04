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
        Use the LLM to produce a short, conversational explanation why `candidate` is similar to `source`.
        If the LLM call fails, fall back to a deterministic, compact explanation.
        """
        # Prepare concise context for the LLM
        def _flatten(g: Dict) -> Dict:
            return {
                "name": g.get("name", ""),
                "theme": g.get("theme", ""),
                "art_style": g.get("art_style", ""),
                "volatility": g.get("volatility", ""),
                "provider_style": g.get("provider_style", ""),
                "special_features": ", ".join(g.get("special_features", []) or []),
                "tags": ", ".join(g.get("tags", []) or []),
                "description": g.get("description", "")[:600],  # keep prompt small
            }

        s = _flatten(source or {})
        c = _flatten(candidate or {})

        system = "You are a helpful assistant that writes short, clear, conversational reasons why one slot game is a good recommendation for another."
        user = (
            "Given the SOURCE and CANDIDATE games below, write 1 short conversational sentence (10-25 words) "
            "explaining why the candidate is a good match for the source. Focus on theme, features, art style, volatility, and tags. "
            "If nothing obvious, say 'Similar overall based on semantic similarity.' "
            "Include the embedding similarity score in parentheses if provided. Output plain text only.\n\n"
            f"SOURCE:\nName: {s['name']}\nTheme: {s['theme']}\nArt style: {s['art_style']}\nVolatility: {s['volatility']}\nProvider: {s['provider_style']}\nFeatures: {s['special_features']}\nTags: {s['tags']}\n\n"
            f"CANDIDATE:\nName: {c['name']}\nTheme: {c['theme']}\nArt style: {c['art_style']}\nVolatility: {c['volatility']}\nProvider: {c['provider_style']}\nFeatures: {c['special_features']}\nTags: {c['tags']}\n\n"
        )
        if score is not None:
            user += f"Embedding similarity: {score:.3f}\n"

        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=100,
            )
            try:
                content = resp.choices[0].message.content
            except Exception:
                content = resp["choices"][0]["message"]["content"]
            explanation = content.strip()
            # Guard: keep it short and single-line
            explanation = " ".join(explanation.splitlines()).strip()
            if explanation:
                return explanation
        except Exception:
            # fall through to deterministic fallback
            pass

        # Deterministic fallback (compact, clear)
        parts: List[str] = []
        if s["theme"] and c["theme"] and s["theme"].strip().lower() == c["theme"].strip().lower():
            parts.append(f"shares the same theme ({s['theme']})")
        if s["art_style"] and c["art_style"] and s["art_style"].strip().lower() == c["art_style"].strip().lower():
            parts.append(f"similar art style ({s['art_style']})")
        if s["volatility"] and c["volatility"] and s["volatility"].strip().lower() == c["volatility"].strip().lower():
            parts.append(f"matches volatility ({s['volatility']})")
        # common tags/features (short)
        s_tags = set(t.strip().lower() for t in (source.get("tags") or []) if t)
        c_tags = set(t.strip().lower() for t in (candidate.get("tags") or []) if t)
        common = sorted(list(s_tags & c_tags))
        if common:
            parts.append(f"shares tags: {', '.join(common[:3])}")
        s_feats = set(f.strip().lower() for f in (source.get("special_features") or []) if f)
        c_feats = set(f.strip().lower() for f in (candidate.get("special_features") or []) if f)
        common_feats = sorted(list(s_feats & c_feats))
        if common_feats:
            parts.append(f"both have {', '.join(common_feats[:3])}")

        if parts:
            base = "; ".join(parts) + "."
        else:
            base = "Similar overall based on semantic embedding similarity."

        if score is not None:
            return f"{base} (Embedding similarity: {score:.3f})"
        return base

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
