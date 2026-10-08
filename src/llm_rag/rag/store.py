"""Chunking and ChromaDB storage for RAG tier 2.

- Source documents: `dataset/rag_corpus/tier2_warnings/*.md` (YAML front matter + Markdown).
- One chunk per H2 section (`## `) so each chunk stays semantically complete.
- Persistent store: `data/chroma/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
import yaml

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "dataset" / "rag_corpus" / "tier2_warnings"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "rag_tier2_warnings"

_RE_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Get or create the tier-2 collection (cosine distance)."""
    return client().get_or_create_collection(name=COLLECTION_NAME,
                                             metadata={"hnsw:space": "cosine"})


def _front_matter(raw: str) -> tuple[dict, str]:
    m = _RE_FRONT_MATTER.match(raw)
    if not m:
        raise ValueError("File has no YAML front matter (--- ... ---).")
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def chunk_file(path: Path) -> list[dict]:
    """Split one document into chunks [{id, text, section, source, file}].

    Front-matter keys used: id, title, publisher, source_url.
    """
    meta, body = _front_matter(path.read_text(encoding="utf-8"))
    doc_id = meta.get("id", path.stem)
    title = meta.get("title", path.stem)
    publisher = meta.get("publisher", "Unknown publisher")
    source_url = meta.get("source_url", "")
    source = f"{publisher} ({source_url})" if source_url else publisher

    h1 = re.search(r"^#\s+.+$", body, re.M)
    content = body[h1.end():].lstrip("\n") if h1 else body

    parts = re.split(r"^##\s+(.+)$", content, flags=re.M)
    chunks: list[dict] = []
    intro = parts[0].strip()
    if intro:
        chunks.append({"id": f"{doc_id}::0", "text": f"[{title}]\n{intro}",
                       "section": "(intro)", "source": source, "file": path.name})
    for i in range(1, len(parts), 2):
        section = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            chunks.append({"id": f"{doc_id}::{i // 2 + 1}",
                           "text": f"[{title} — {section}]\n{text}",
                           "section": section, "source": source, "file": path.name})
    return chunks


def read_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Chunk every Markdown document of the corpus (README excluded)."""
    chunks = []
    for p in sorted(corpus_dir.glob("*.md")):
        if not p.name.upper().startswith("README"):
            chunks.extend(chunk_file(p))
    return chunks
