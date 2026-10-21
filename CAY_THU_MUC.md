# Directory tree

```
.
├── README.md                      Overview: architecture diagram, principles, structure, quick start
├── IMPROVEMENTS_NOTE.md           Optimizations kept / discarded, benchmark table, research contributions
├── CLAUDE.md                      Instructions for Claude Code
├── CAY_THU_MUC.md                 This file: what each folder/file is for
├── DE_TAI.md                     Thesis topic (Vietnamese): title, description, three main contributions
├── DEPLOY.txt                     Deployment guide in Vietnamese: Docker, BFF, HTTPS proxy, Turnstile, FE call example
├── requirements.txt               Python dependencies (tested versions)
├── requirements-api.txt           Runtime-only subset for the API image (torch CPU installed separately)
├── Dockerfile                     API image: src/, models/url_classifier/, dataset/brands/ (data/ and .env mounted)
├── docker-compose.yml             api (private network only) + bff (public entry, 127.0.0.1:8080) + updater (hourly)
├── .dockerignore                  Keeps data, datasets, notebooks, reports and .env out of the image
├── .env                           API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, VT_API_KEY, optional OPENAI_MODEL) and budgets — not shared
│
├── bff/                           Backend-for-frontend: the browser calls it without a key
│   ├── app.py                     Adds API_KEY server-side, forwards /api/analyze, /api/feedback to src/api.py;
│   │                              origin check, per-IP minute / day limits, optional Cloudflare Turnstile
│   ├── Dockerfile                 Small image (no model, no project code)
│   └── requirements.txt           fastapi, uvicorn, httpx
│
├── src/
│   ├── contracts.py               Shared result types (RuleResult, Verdict)
│   ├── url_utils.py               URL helpers: host, registrable domain, normalization
│   ├── api.py                     HTTP API for the UI (FastAPI): POST /analyze, POST /feedback, GET /health
│   │                              (X-API-Key, per-IP rate limit, http(s) URLs only, scores hidden by default)
│   ├── pipeline.py                Full A -> B -> C -> D pipeline for one URL (CLI); trusted official
│   │                              domains (verified VN brands + international brands without user content)
│   ├── evaluate_pipeline.py       A+B+C evaluation on the test split (miss rate, LLM load, latency;
│   │                              also on trusted labels, without doubtful phishing labels)
│   ├── evaluate_external.py       Layers A, B, C alone and A+B+C on the external PhreshPhish set
│   ├── evaluate_live.py           Full pipeline on live URLs: Vietnamese set, fresh pages of top sites (paid API)
│   ├── evaluate_trusted_shortcut.py  Trusted-domain shortcut before / after the international brand domains
│   ├── relabel_subset.py          ~300-page relabelling subset (export CSV) and error rates vs its VirusTotal labels (score)
│   ├── relabel_vt.py              Fills that subset with VirusTotal labels (>= 2 engines phishing, 0 clean, else unsure)
│   ├── layer_c_selection.py       Chooses Layer C on val: best of 5 DL URL models, then best of RF / XGBoost / LightGBM
│   │                              (+ ablation step: feature groups removed one at a time, 3 seeds)
│   ├── monitoring.py              Real-traffic verdict log + user feedback -> reported false-alarm / miss rates
│   │
│   ├── blocklist/                 Layer A — instant block from public phishing feeds and VirusTotal
│   │   ├── virustotal.py          VirusTotal URL report lookup (cached, never submits URLs), verdict rule
│   │   ├── updater.py             Hourly refresh of OpenPhish / PhishTank / Phishing.Database (per-feed age), Tranco monthly
│   │   ├── lookup.py              URL / path-prefix / host / campaign-domain lookup (+ redirect targets)
│   │   └── platforms.py           Derives abused platforms (site builders, link-in-bio, QR...) from feeds
│   │
│   ├── typosquatting/             Layer B — brand impersonation
│   │   ├── typosquatting.py       Damerau-Levenshtein + Jaro-Winkler on keywords of the whole URL
│   │   ├── brands.py              Loads the brand catalogue from dataset/brands/
│   │   ├── paas_domains.py        Free hosting, URL shorteners, user-content hosts
│   │   └── evaluate.py            False-positive / detection evaluation of Layer B (+ rejected variants)
│   │
│   ├── url_classifier/            Layer C — URL classifier + DOM model (clean / suspicious)
│   │   ├── features.py            Hand-crafted lexical features (incl. redirect features)
│   │   ├── host_features.py       DNS / TLS certificate / RDAP domain age / Tranco rank (+ cache)
│   │   ├── model.py               Deep URL models: CNN/BiLSTM, char-CNN, BiLSTM, URLNet, Transformer
│   │   ├── dom_features.py        DOM features of a page, DOM model (build / train / combine), combination rule
│   │   ├── clean_labels.py        Out-of-fold detection of mislabelled training rows
│   │   ├── collect_host_features.py  Looks up host features for the dataset (into the cache)
│   │   ├── train.py               Stacking: 2 cross-fitted DL models (--dl) + ML lexical/host branches (--ml), segment thresholds
│   │   │                          (fewest val misses with >= 50 % clean recognised; --thresholds-only refits them)
│   │   └── predict.py             Inference used by the pipeline (host branch / lexical fallback)
│   │
│   ├── llm/                       Layer D — LLM verdict (phishing / suspicious / safe) for suspicious URLs
│   │   ├── llm_client.py          Prompt building, page fetch, OpenAI / Anthropic call, verdict parsing,
│   │   │                          spend ledger with a budget per provider
│   │   ├── html_sanitize.py       Removes hidden text / fake instructions (patterns from data/injection_patterns/)
│   │   ├── llm_benchmark.py       LLM accuracy/latency on live Tranco vs OpenPhish URLs (paid API)
│   │   ├── vn_benchmark.py        LLM accuracy on Vietnamese brand scenarios (paid API)
│   │   ├── layer_d_eval.py        Layer D on the stored dev / test pages, per stratum (paid API; --model claude-*)
│   │   ├── compare_models.py      Layer D runs side by side: outcomes, C override, latency, cost, Wilson CI, McNemar
│   │   ├── phreshphish_sets.py    Builds the Layer D dev / held-out test sets (PhreshPhish pages that reach D)
│   │   ├── redteam_prompt_injection.py     Prompt-injection red team (hidden, advanced, hard visible / multi-step /
│   │   │                                   multilingual attacks): no defence vs production (paid API)
│   │   ├── redteam_fake_trusted_source.py  Fake "verified by authority" red team (paid API)
│   │   └── live_traffic_sources.py         Live Tranco / OpenPhish samplers for the benchmarks
│   │
│   ├── baselines/                 Layer C baselines (each model alone), same splits and rule (>= 50 % val clean recognised)
│   │   ├── common.py              Data splits, threshold rule, metrics (alone / with Layer B, ROC-AUC, PR-AUC)
│   │   ├── classic.py             Logistic regression, random forest, LightGBM (lexical / + host), Layer C itself
│   │   └── deep.py                char-CNN, BiLSTM, URLNet (reduced), small Transformer trained alone (URL only)
│   │
│   └── data_collection/           Builds the labelled URL dataset for Layer C
│       ├── sources.py             Downloaders: OpenPhish, PhishTank, Phishing.Database, PhiUSIIL, Tranco
│       ├── crawl_clean.py         Crawls clean URLs (with paths) from Tranco-ranked homepages
│       ├── vi_testset.py          Builds data/test_sets/vi/ (Chong Lua Dao phishing, .vn clean; zero overlap with training)
│       ├── build_dataset.py       Merge, clean, de-duplicate, cap per site, split by site, apply label fixes
│       └── audit_labels.py        Finds mislabelled URLs with model-independent evidence -> label_fixes.csv
│
├── tests/                         pytest suite (no paid API calls; test_api.py = HTTP API, test_bff.py = BFF)
├── notebooks/
│   ├── pipeline_experiment.ipynb  Every experiment end to end: setup, sanitizer, features, Layer C selection,
│                                  val / test / VN evaluation, Layer D, benchmark (keeps outputs)
│   └── pipeline_demo.ipynb        Demo: URLs set in the first cell -> full A -> D pipeline, verdict and per-layer detail
├── docs/
│   └── pipeline.svg               Pipeline diagram (Layers A -> D, verdicts)
│
├── dataset/
│   ├── brands/                    Brand catalogue: VN brands, international brands, secondary domains
│   ├── rag_corpus/tier2_warnings/ Archive: verified scam-warning articles of the removed RAG (not used by the code)
│   ├── external/
│   │   ├── phreshphish_urls.csv   External test set (PhreshPhish 2025, never used in training)
│   │   ├── layer_d_dev.csv / layer_d_test.csv  Layer D tuning set / held-out test set (stratum, A/B/C)
│   │   └── dom_pages.csv          DOM model pages (PhreshPhish outside every evaluation set and its domains; train / val)
│   └── urls/
│       ├── raw/                   Cached downloads + crawled_clean.csv
│       ├── urls_labeled.csv       Final dataset (url, label, source, group, split)
│       ├── noisy_labels.csv       Training rows flagged as mislabelled (dropped from train)
│       ├── label_fixes.csv        Audited label corrections (url, old/new label, rule, evidence)
│       └── stats.json             Dataset statistics
│
├── data/
│   ├── blocklist/                 Layer A snapshots (*.txt.gz), blockable hosts, abused_platforms.txt, meta.json
│   ├── injection_patterns/        Language data, one JSON per language (injection regexes, authorities, verdict words,
│   │                              bait words...), merged by html_sanitize.PatternLoader
│   ├── test_sets/vi/              Vietnamese real-world test set (Chong Lua Dao phishing, .vn clean), never trained on
│   ├── vt_cache.sqlite            VirusTotal lookups (not in git)
│   ├── llm_spend.json             Ledger of LLM spend per model and provider (budget guard: LLM_BUDGET_USD,
│   │                              ANTHROPIC_BUDGET_USD in .env)
│   ├── llm_verdict_cache.sqlite   Layer D verdicts per URL, reused for 6 h (not in git)
│   ├── monitoring/                Monthly verdict / feedback logs (kept MONITORING_KEEP_MONTHS), hashing secret (no URLs; not in git)
│   ├── phreshphish_html.parquet   HTML of the Layer D dev / test pages and DOM pages (sha256 -> html, re-downloadable)
│   ├── host_cache.sqlite          Cache of DNS / TLS / RDAP lookups (Layer C host features)
│   └── tranco_top1m.csv           Tranco top-1M (popular-domain checks, common-token filter)
│
├── models/url_classifier/         Production Layer C: dl_a.pt + dl_b.pt (cross-fitted DL), ml_lexical.*, ml_host.*, config.json;
│                                  dom_model.* + dom_config.json (DOM model and combination rule)
├── models/layer_c_selection/      Candidate DL weights, scores and feature cache of layer_c_selection.py (not in git)
└── reports/                       Evaluation outputs (JSON) of every layer (live/ = evaluate_live.py,
                                   baselines/ = src/baselines, relabel/ = relabel_subset.py, layer_c_selection/ =
                                   DL / ML / DOM candidates, feature-group ablation, benchmark.json = notebook summary,
                                   summary.md = before / after tables of every experiment and spend)
```
