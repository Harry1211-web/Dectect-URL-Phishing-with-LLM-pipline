"""RAG tier 2 — semantic search over verified scam-warning articles (ChromaDB, cosine distance)."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed_one  # noqa: E402
from store import get_collection  # noqa: E402

# Keep chunks whose cosine distance <= threshold (calibrated by threshold_relevance*.py)
MAX_DISTANCE_VI = 0.60
MAX_DISTANCE_OTHER = 0.65

# Vietnamese diacritics after NFD decomposition — also catches letters with a tone mark such
# as "ừ", "ố", "ậ" (their precomposed forms differ from "ư", "ô", "â")
_VI_MARKS = {
    "̀", "́", "̃", "̉", "̣",  # huyền, sắc, ngã, hỏi, nặng
    "̂", "̆", "̛",                      # circumflex, breve, horn
}
AUTO_BY_LANGUAGE = "auto_by_language"


def is_vietnamese(text: str) -> bool:
    if "đ" in text or "Đ" in text:
        return True
    return any(c in _VI_MARKS for c in unicodedata.normalize("NFD", text))


def query_tier2(query: str, k: int = 3, max_distance=AUTO_BY_LANGUAGE) -> list[dict]:
    """Up to `k` most relevant chunks with distance <= `max_distance` (None = no filter)."""
    if max_distance is AUTO_BY_LANGUAGE:
        max_distance = MAX_DISTANCE_VI if is_vietnamese(query) else MAX_DISTANCE_OTHER
    col = get_collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_embeddings=[embed_one(query)], n_results=min(k, n))
    out = []
    for i, cid in enumerate(res["ids"][0]):
        dist = round(float(res["distances"][0][i]), 4)
        if max_distance is not None and dist > max_distance:
            continue
        meta = res["metadatas"][0][i]
        out.append({"id": cid, "text": res["documents"][0][i], "distance": dist,
                    "source": meta.get("source"), "file": meta.get("file")})
    return out
