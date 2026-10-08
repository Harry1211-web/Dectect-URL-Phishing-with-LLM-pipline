# RAG tier-2 corpus — scam warnings

- 22 verified articles about online scams in Vietnam, grouped by topic: impersonation of big
  brands, banks / e-wallets, e-commerce, public services, telecom carriers and common online
  fraud schemes. The articles themselves stay in Vietnamese (they are the retrieval data).
- Used by RAG tier 2 to support the LLM layer (`src/llm_rag/`).
- Storage: ChromaDB (`data/chroma/`, collection `rag_tier2_warnings`), loaded with
  `python src/llm_rag/rag/ingest.py [--reset]`.

## Loading rule (mandatory — defence against knowledge-base poisoning)

> Only load manually verified data; never crawl-and-load automatically.

Every `.md` file here was **read and verified by a person** (not crawled in bulk) before being
added, as recorded in its front matter:
- `source_url` — the original URL that was read;
- `verified_date` / `verified_by` — who verified it and when;
- preferred publishers: the police / Ministry of Public Security, Chongluadao.vn (community
  anti-fraud project), mainstream press quoting the police.

**Never load**: content from unidentified sources, unverified forums / social networks, or any text
that may carry hidden instructions (indirect prompt injection). Every file here has been read by
a person, not piped straight from the web into the store.

## File format

- `NNN_short-english-name.md` — one document per file: YAML front matter + Vietnamese content.
- Front-matter keys: `id`, `title`, `publisher`, `source_url`, `published_date`, `verified_date`,
  `verified_by`, `topics` (English tags), `related_vn_brands`, and optionally `original_source` /
  `cross_check_source`.
- `manifest.csv` — one row per document (id, title, publisher, dates, topics, related brands,
  file), to see what the store contains without opening every file.
- `ingest.py` chunks each document by H2 heading (`## `) for semantic retrieval; `rag_hits`
  returned by the LLM layer use `<id>::<chunk>` as identifiers.

## NCSC source status (khonggianmang.vn)

At collection time (2026-09-15) `khonggianmang.vn` and its known subdomains
(`blog.`, `canhbao.`, `congcu.`) **did not resolve in DNS** (checked with both the school resolver
and Google DNS `8.8.8.8`). Articles once published by NCSC were replaced by mainstream-press copies
(CafeF) when available. Re-checked 2026-09-17: still NXDOMAIN, so the infrastructure most likely
changed for good; retry monthly rather than every session.

## Still missing (add later)

- Retry `khonggianmang.vn` periodically — the most official channel (NCSC); high priority if it
  comes back online.
- A dedicated article on the risks of sharing personal data on social networks.
- One article per topic group so far (e.g. Shopee / Viettel stand for e-commerce / telecom); add
  separate articles for Lazada, Tiki, VNPT, MobiFone when time allows.
- New content: **read it directly in a browser, verify it, then write the `.md` file following the
  front-matter template above** — no script that downloads articles in bulk.
