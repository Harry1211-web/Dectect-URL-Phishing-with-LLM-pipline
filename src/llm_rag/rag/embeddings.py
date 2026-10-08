"""OpenAI embeddings for RAG tier 2 (ingest and query)."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, find_dotenv
from openai import OpenAI

_env_path = find_dotenv() or str(Path(__file__).resolve().parents[3] / ".env")
OPENAI_API_KEY = (dotenv_values(_env_path).get("OPENAI_API_KEY") or "").strip()

EMBED_MODEL = "text-embedding-3-small"
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing in .env (needed for embeddings).")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embedding vectors in the same order as `texts`."""
    if not texts:
        return []
    resp = _get_client().embeddings.create(input=texts, model=model)
    try:  # embeddings are cheap but still counted in the shared spend ledger
        from llm_client import record_spend
        record_spend(model, getattr(resp.usage, "prompt_tokens", 0) or 0, 0)
    except ImportError:
        pass
    return [item.embedding for item in resp.data]


def embed_one(text: str, model: str = EMBED_MODEL) -> list[float]:
    return embed([text], model)[0]
