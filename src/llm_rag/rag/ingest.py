"""Load `dataset/rag_corpus/*.md` into ChromaDB (RAG tier 2).

Run: python src/llm_rag/rag/ingest.py [--reset]
- Chunks that disappeared from the corpus are deleted automatically (no --reset needed).
- Calls the OpenAI embedding API (small cost per run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import embed  # noqa: E402
from store import COLLECTION_NAME, client, read_corpus  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="drop the collection before loading")
    args = ap.parse_args()

    cli = client()
    if args.reset:
        try:
            cli.delete_collection(COLLECTION_NAME)
            print("Old collection deleted.", file=sys.stderr)
        except Exception:
            pass
    col = cli.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    chunks = read_corpus()
    if not chunks:
        raise SystemExit("No chunk found in dataset/rag_corpus/.")
    print(f"Loading {len(chunks)} chunks from {len({c['file'] for c in chunks})} files...",
          file=sys.stderr)
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{"section": c["section"], "source": c["source"], "file": c["file"]}
                   for c in chunks],
    )
    # upsert only adds/overwrites: delete chunks removed from the corpus (incl. poisoned docs)
    current = {c["id"] for c in chunks}
    stale = [i for i in col.get(include=[])["ids"] if i not in current]
    if stale:
        col.delete(ids=stale)
        print(f"Deleted {len(stale)} stale chunks.", file=sys.stderr)
    print(f"Collection '{COLLECTION_NAME}' now holds {col.count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
