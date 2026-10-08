"""Two-tier RAG that supports the Layer D LLM.

- Tier 1 (`tier1.py`): structured lookup of official brand domains (from `typosquatting.brands`).
- Tier 2 (`tier2.py`): semantic search over verified scam-warning articles
  (`dataset/rag_corpus/`), stored in ChromaDB (`data/chroma/`).

`pipeline.get_rag_context()` merges both tiers into a clearly framed reference block for the
prompt (framing = prompt-injection defence).
"""
