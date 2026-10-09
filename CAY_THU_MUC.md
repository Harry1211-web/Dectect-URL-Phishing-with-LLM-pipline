# Directory tree

```
.
├── CLAUDE.md                      Instructions for Claude Code
├── CAY_THU_MUC.md                 This file: what each folder/file is for
├── requirements.txt               Python dependencies (tested versions)
├── .env                           API keys (OPENAI_API_KEY, optional OPENAI_MODEL) — not shared
│
├── src/
│   ├── contracts.py               Shared result types (RuleResult, Verdict)
│   ├── url_utils.py               URL helpers: host, registrable domain, normalization
│   ├── pipeline.py                Full A -> B -> C -> D pipeline for one URL (CLI)
│   ├── evaluate_pipeline.py       A+B+C evaluation on the test split (miss rate, LLM load, latency;
│   │                              also on trusted labels, without doubtful phishing labels)
│   │
│   ├── blocklist/                 Layer A — instant block from public phishing feeds
│   │   ├── updater.py             Monthly refresh of OpenPhish / PhishTank / Phishing.Database + Tranco
│   │   ├── lookup.py              URL / path-prefix / host / campaign-domain lookup (+ redirect targets)
│   │   └── platforms.py           Derives abused platforms (site builders, link-in-bio, QR...) from feeds
│   │
│   ├── typosquatting/             Layer B — brand impersonation
│   │   ├── typosquatting.py       Damerau-Levenshtein + Jaro-Winkler on keywords of the whole URL
│   │   ├── brands.py              Loads the brand catalogue from dataset/brands/
│   │   ├── paas_domains.py        Free hosting, URL shorteners, user-content hosts
│   │   └── evaluate.py            False-positive / detection evaluation of Layer B
│   │
│   ├── url_classifier/            Layer C — URL-only classifier (clean / suspicious)
│   │   ├── features.py            Hand-crafted lexical features (incl. redirect features)
│   │   ├── host_features.py       DNS / TLS certificate / RDAP domain age / Tranco rank (+ cache)
│   │   ├── model.py               Char CNN + BiLSTM feature extractor
│   │   ├── clean_labels.py        Out-of-fold detection of mislabelled training rows
│   │   ├── collect_host_features.py  Looks up host features for the dataset (into the cache)
│   │   ├── train.py               Stacking: 2 cross-fitted CNN/BiLSTM + XGBoost lexical/host branches, segment thresholds
│   │   │                          (fewest val misses with >= 50 % clean recognised; --thresholds-only refits them)
│   │   └── predict.py             Inference used by the pipeline (host branch / lexical fallback)
│   │
│   ├── llm_rag/                   Layer D — LLM + 2-tier RAG for suspicious URLs
│   │   ├── llm_client.py          Prompt building, page fetch, OpenAI call, verdict parsing
│   │   ├── html_sanitize.py       Removes hidden text / fake instructions (prompt injection)
│   │   ├── rag/                   RAG: tier1.py (brand table), tier2.py (Chroma semantic search),
│   │   │                          pipeline.py (framed context), ingest.py, store.py, embeddings.py,
│   │   │                          eval_retrieval.py / threshold_relevance*.py (RAG evaluation)
│   │   ├── llm_benchmark.py       LLM accuracy/latency on live Tranco vs OpenPhish URLs (paid API)
│   │   ├── vn_benchmark.py        LLM accuracy on Vietnamese brand scenarios (paid API)
│   │   ├── rag_ablation.py        Same pages with vs without RAG (paid API)
│   │   ├── redteam_prompt_injection.py     Hidden-instruction + RAG-poisoning red team (paid API)
│   │   ├── redteam_fake_trusted_source.py  Fake "verified by authority" red team (paid API)
│   │   └── live_traffic_sources.py         Live Tranco / OpenPhish samplers for the benchmarks
│   │
│   └── data_collection/           Builds the labelled URL dataset for Layer C
│       ├── sources.py             Downloaders: OpenPhish, PhishTank, Phishing.Database, PhiUSIIL, Tranco
│       ├── crawl_clean.py         Crawls clean URLs (with paths) from Tranco-ranked homepages
│       ├── build_dataset.py       Merge, clean, de-duplicate, cap per site, split by site, apply label fixes
│       └── audit_labels.py        Finds mislabelled URLs with model-independent evidence -> label_fixes.csv
│
├── tests/                         pytest suite (no paid API calls)
├── notebooks/
│   └── phishing_pipeline.ipynb    Runs every layer section by section, keeps outputs + metrics
│
├── dataset/
│   ├── brands/                    Brand catalogue: VN brands, international brands, secondary domains
│   ├── rag_corpus/tier2_warnings/ Verified scam-warning articles for RAG tier 2 (+ manifest.csv)
│   └── urls/
│       ├── raw/                   Cached downloads + crawled_clean.csv
│       ├── urls_labeled.csv       Final dataset (url, label, source, group, split)
│       ├── noisy_labels.csv       Training rows flagged as mislabelled (dropped from train)
│       ├── label_fixes.csv        Audited label corrections (url, old/new label, rule, evidence)
│       └── stats.json             Dataset statistics
│
├── data/
│   ├── chroma/                    ChromaDB store of RAG tier 2 (collection rag_tier2_warnings)
│   ├── blocklist/                 Layer A snapshots (*.txt.gz), blockable hosts, abused_platforms.txt, meta.json
│   ├── llm_spend.json             Ledger of OpenAI spend (Layer D budget guard)
│   ├── host_cache.sqlite          Cache of DNS / TLS / RDAP lookups (Layer C host features)
│   ├── tranco_top1m.csv           Tranco top-1M (popular-domain checks, common-token filter)
│   ├── allbrands.txt              Brand name list (not used by the code)
│   └── agency_domains.json        Authority -> domain map (not used by the code)
│
├── models/url_classifier/         Trained Layer C: cnn.pt + cnn_b.pt (cross-fitted CNNs), xgb_lexical.json, xgb_host.json, config.json
└── reports/                       Evaluation outputs (JSON) of every layer
```
