# Scam-warning articles (archive of the removed RAG)

Layer D no longer uses RAG: on the Layer D dev set, adding these articles (tier 2) or the official
brand domains (tier 1) to the prompt did not change verdicts beyond run-to-run noise, because the
LLM already knows this general advice and the domains of big brands
(`reports/llm/rag_pipeline/`). The code, the ChromaDB store and the ingestion scripts were
removed; the articles are kept as verified reference material.

- 33 articles: 25 in Vietnamese (police, Chongluadao.vn, banks), 8 in English (FTC, Microsoft,
  NCSC UK, CFTC, Ledger, Google). `manifest.csv` lists them.
- One document per file: YAML front matter (`id`, `title`, `publisher`, `source_url`,
  `published_date`, `verified_date`, `verified_by`, `topics`, `related_vn_brands`) + the text in
  the source's language.
- Articles 023–033 were read through WebFetch, paraphrased, then approved by the project owner.
