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
│   ├── evaluate_external.py       Layers A, B, C alone and A+B+C on the external PhreshPhish set
│   ├── evaluate_live.py           Full pipeline on live URLs: Vietnamese set, fresh pages of top sites (paid API)
│   ├── monitoring.py              Real-traffic verdict log + user feedback -> reported false-alarm / miss rates
│   │
│   ├── blocklist/                 Layer A — instant block from public phishing feeds
│   │   ├── updater.py             Hourly refresh of OpenPhish / PhishTank / Phishing.Database (per-feed age), Tranco monthly
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
│   ├── llm/                       Layer D — LLM verdict (phishing / suspicious / safe) for suspicious URLs
│   │   ├── llm_client.py          Prompt building, page fetch, OpenAI call, verdict parsing
│   │   ├── html_sanitize.py       Removes hidden text / fake instructions (patterns from data/injection_patterns/)
│   │   ├── llm_benchmark.py       LLM accuracy/latency on live Tranco vs OpenPhish URLs (paid API)
│   │   ├── vn_benchmark.py        LLM accuracy on Vietnamese brand scenarios (paid API)
│   │   ├── layer_d_eval.py        Layer D on the stored dev / test pages, per stratum (paid API)
│   │   ├── phreshphish_sets.py    Builds the Layer D dev / held-out test sets (PhreshPhish pages that reach D)
│   │   ├── redteam_prompt_injection.py     Hidden-instruction red team: no defence vs production (paid API)
│   │   ├── redteam_fake_trusted_source.py  Fake "verified by authority" red team (paid API)
│   │   └── live_traffic_sources.py         Live Tranco / OpenPhish samplers for the benchmarks
│   │
│   └── data_collection/           Builds the labelled URL dataset for Layer C
│       ├── sources.py             Downloaders: OpenPhish, PhishTank, Phishing.Database, PhiUSIIL, Tranco
│       ├── crawl_clean.py         Crawls clean URLs (with paths) from Tranco-ranked homepages
│       ├── vi_testset.py          Builds dataset/urls/vi/ (Chong Lua Dao phishing, .vn clean; zero overlap with training)
│       ├── build_dataset.py       Merge, clean, de-duplicate, cap per site, split by site, apply label fixes
│       └── audit_labels.py        Finds mislabelled URLs with model-independent evidence -> label_fixes.csv
│
├── tests/                         pytest suite (no paid API calls)
├── notebooks/
│   └── phishing_pipeline.ipynb    Runs every layer section by section, keeps outputs + metrics
│
├── dataset/
│   ├── brands/                    Brand catalogue: VN brands, international brands, secondary domains
│   ├── rag_corpus/tier2_warnings/ Archive: verified scam-warning articles of the removed RAG (not used by the code)
│   ├── external/
│   │   ├── phreshphish_urls.csv   External test set (PhreshPhish 2025, never used in training)
│   │   └── layer_d_dev.csv / layer_d_test.csv  Layer D tuning set / held-out test set (stratum, A/B/C)
│   └── urls/
│       ├── raw/                   Cached downloads + crawled_clean.csv
│       ├── vi/                    Vietnamese Layer C test set (phishing.csv, clean.csv), never used in training
│       ├── urls_labeled.csv       Final dataset (url, label, source, group, split)
│       ├── noisy_labels.csv       Training rows flagged as mislabelled (dropped from train)
│       ├── label_fixes.csv        Audited label corrections (url, old/new label, rule, evidence)
│       └── stats.json             Dataset statistics
│
├── data/
│   ├── blocklist/                 Layer A snapshots (*.txt.gz), blockable hosts, abused_platforms.txt, meta.json
│   ├── injection_patterns/        Prompt-injection patterns per language (vi, en, es, fr, de, zh...), merged by html_sanitize.py
│   ├── llm_spend.json             Ledger of OpenAI spend (Layer D budget guard)
│   ├── monitoring/                Verdict log, user feedback, hashing secret (no URLs stored; not in git)
│   ├── phreshphish_html.parquet   HTML of the Layer D dev / test pages (sha256 -> html, re-downloadable)
│   ├── host_cache.sqlite          Cache of DNS / TLS / RDAP lookups (Layer C host features)
│   └── tranco_top1m.csv           Tranco top-1M (popular-domain checks, common-token filter)
│
├── models/url_classifier/         Trained Layer C: cnn.pt + cnn_b.pt (cross-fitted CNNs), xgb_lexical.json, xgb_host.json, config.json
└── reports/                       Evaluation outputs (JSON) of every layer (live/ = evaluate_live.py)
```
