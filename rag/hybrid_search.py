"""
Hybrid Search + Reranker for Insurance Knowledge Base.

Pipeline:
  User Query
      │
      ├── BM25 (keyword match, fast, exact terms)
      │
      ├── TF-IDF Cosine Similarity (semantic overlap, sklearn)
      │
      ├── Hybrid Score = α*BM25_norm + (1-α)*TFIDF_cosine
      │
      └── Rerank → Top-K filtered chunks for context injection
"""

import math
import re
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.knowledge_base import KNOWLEDGE_BASE
from config import cfg

ALPHA = cfg.rag.alpha
TOP_K = cfg.rag.top_k
MIN_SCORE = cfg.rag.min_score


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\b[a-z]+\b", text.lower())


class InsuranceRAG:
    """
    Retrieval-Augmented Generation index for insurance knowledge base.
    Initialized once at startup and reused for all queries.
    """

    def __init__(self):
        self.docs = KNOWLEDGE_BASE
        self.texts = [f"{d['title']} {d['content']}" for d in self.docs]
        self.tokenized = [_tokenize(t) for t in self.texts]

        # BM25 index
        self.bm25 = BM25Okapi(self.tokenized)

        # TF-IDF vectorizer (fit on KB at startup)
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        scores = np.array(self.bm25.get_scores(tokens))
        max_s = scores.max()
        return scores / max_s if max_s > 0 else scores

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q_vec = self.tfidf.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        return scores

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        alpha: float = ALPHA,
        min_score: float = MIN_SCORE,
    ) -> list[dict]:
        """
        Run hybrid search and return top-k reranked results.
        Each result includes the KB entry + its scores.
        """
        bm25_scores = self._bm25_scores(query)
        tfidf_scores = self._tfidf_scores(query)

        hybrid_scores = alpha * bm25_scores + (1 - alpha) * tfidf_scores

        ranked_indices = np.argsort(hybrid_scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k * 2]:  # Fetch 2x, filter below threshold
            score = hybrid_scores[idx]
            if score < min_score:
                break
            doc = self.docs[idx]
            results.append({
                "id": doc["id"],
                "category": doc["category"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(float(score), 4),
                "bm25_score": round(float(bm25_scores[idx]), 4),
                "tfidf_score": round(float(tfidf_scores[idx]), 4),
            })

        # ── Reranker: Boost results whose keywords overlap with the query ──────
        query_words = set(_tokenize(query))
        for r in results:
            doc = next(d for d in self.docs if d["id"] == r["id"])
            keyword_overlap = len(set(doc.get("keywords", [])) & query_words)
            r["score"] += keyword_overlap * 0.05  # Boost per matching keyword

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def format_context(self, results: list[dict]) -> str:
        """
        Formats retrieved chunks into a context block for LLM injection.
        """
        if not results:
            return ""
        lines = ["POLICY REFERENCE INFORMATION (use this to improve your response):"]
        lines.append("-" * 60)
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(r["content"])
            lines.append("")
        lines.append("-" * 60)
        return "\n".join(lines)


# Global singleton — initialized once at server startup
rag_index = InsuranceRAG()
