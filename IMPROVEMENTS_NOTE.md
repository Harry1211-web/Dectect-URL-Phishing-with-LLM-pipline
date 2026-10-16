# Improvements note

What was changed to optimize the system, what was measured and dropped, the benchmark numbers,
and the contributions they support. Every number comes from `notebooks/pipeline_experiment.ipynb`
or the report files it cites; choices were made on validation / dev data and each test set was
scored once per chosen configuration.

## 1. Methods applied

| Method | What it is | Evidence |
|---|---|---|
| **Dynamic pattern sanitizer** | `html_sanitize.PatternLoader` merges one JSON file per language (`data/injection_patterns/<lang>.json`, 16 languages): injection regexes, quoted authorities and their reserved domains, verdict words, negations, bait words. No language-specific pattern is left in the sanitizer or the verdict parser. | Same detections as before (test suite unchanged); a new language is a new file. Hard red team: ≤ 1 flip to *safe* in 324 attacked real pages (`reports/llm/redteam_injection/`). |
| **VirusTotal in Layer A** | After the blocklists and the trusted-domain shortcut: VirusTotal URL report, ≥ 2 malicious engines → phishing; cached, never submits a URL, quota / network errors are skipped. | On a fixed sample of the Vietnamese set (95 Chong Lua Dao phishing + 95 clean): public feeds catch 1.1 % of the phishing, VirusTotal 70.5 % (precision 0.971, 2 clean pages flagged). |
| **Chong Lua Dao as Vietnamese ground truth** | The Vietnamese phishing labels come from the Chong Lua Dao blacklist (pinned public mirror, live hosts registered before the snapshot); it is *not* a Layer A feed, so the Vietnamese evaluation of Layer A is not circular. | 2,710 phishing URLs (1,604 domains). |
| **Zero-leakage Vietnamese test set** | Moved to `data/test_sets/vi/`, away from the training data; registrable-domain overlap checked against every training and evaluation file and the DOM pages. | 0 shared domains with training files and DOM pages; 2 phishing domains shared with the Layer D page sets (evaluation sets, never trained on). |
| **Single-language prompt, dynamic output language** | English prompt; `lang` (default `vi`) sets the language of the explanation; the VERDICT keyword stays English. | Dev, 651 pages: explanations in Vietnamese 100 % (gpt) / 99.7 % (Haiku); verdict agreement with the verdict-only run 97.2 % / 95.9 %; phishing → safe 15.7 → 15.4 % (gpt, McNemar p = 1.0), 16.5 → 14.8 % (Haiku, p = 0.07). Kept: no significant change. |
| **Layer C re-selection** | 5 deep URL models cross-fitted as in production, then RF / XGBoost / LightGBM on the winner's score + URL + host features, both chosen by "fewest val misses at ≥ 50 % val clean recognised". | BiLSTM (424 val misses vs 428 CNN/BiLSTM, 478 char-CNN, 667 URLNet, 1,128 Transformer) + XGBoost (18 val misses vs 24 LightGBM, 32 RF). Production test misses 12 → 10 at 51.1 % clean. |
| **VirusTotal labels for the relabelling subset** | 300 pages labelled by VirusTotal (≥ 2 engines phishing, 0 clean, else unsure) instead of hand labels. | 149 labelled: 4.7 % of the dataset labels disagree [95 % CI 2.3–9.4 %]; 124 URLs unknown to VirusTotal. |

## 2. Discarded methods and reasons

| Method | Status | Reason |
|---|---|---|
| Regex patterns hard-coded in Python | Used until commit f82f78b (Vietnamese + English regexes in `html_sanitize.py`); moved to data in d733ab8; the last lists (authority domains, verdict words, negations) moved in this round | Adding a language needed a code change; an earlier "translate everything" pass translated Vietnamese detection patterns and would have broken detection on Vietnamese pages. |
| Dual-language system prompts | **Never implemented.** The project used a Vietnamese-only prompt (f82f78b), then one English prompt | Two prompts would double the prompt-tuning and evaluation work; the dynamic output language gives Vietnamese explanations from one prompt without a significant verdict change (table above). |
| Synthetic / fabricated phishing URLs as labels | **Never used for training or testing.** Synthetic pages exist only in the red team and the Vietnamese brand scenarios, reported as such | A fabricated URL measures the generator, not real attackers; real feeds (PhishTank, OpenPhish, Chong Lua Dao) are used instead. |
| Monolithic, un-isolated datasets | The Vietnamese set used to sit in `dataset/urls/vi/` next to the training data (already domain-isolated); now in `data/test_sets/vi/` with overlap checks in the notebook and tests | Physical and checked separation makes leakage visible. |
| DOM model allowed to clear a URL flag ("and" rule) | Measured on DOM val: 1 miss, 52.6 % clean (URL only: 1 miss, 36.3 %). **Rejected** | The page is attacker-controlled: a cloaked clean-looking page would turn a flagged URL into "clean". |
| DOM model adding suspicion only ("or" rule) | Allowed, but cannot reach ≥ 50 % val clean (URL-only is already below it on these pages). **Not enabled** | Production Layer C stays URL-only; the DOM model (val PR-AUC 0.974, held-out ROC-AUC 0.935) is kept and measured. |
| VirusTotal detections as an ML feature | Not built | The number of engines that flag a URL is close to its label (leakage); the free quota (500 / day) cannot cover the training set. |
| Other Layer C candidates | Measured, lost on val | CNN/BiLSTM, char-CNN, URLNet, Transformer; LightGBM, random forest (section 3). |
| Earlier rounds | Measured, dropped | RAG for Layer D (no verdict gain, `reports/llm/rag_pipeline/`); Layer B compound-domain variants (more false alarms, no fewer pipeline misses). |

## 3. Benchmark summary

**Reading guide.** Phishing is the positive class. For Layers A–C a URL is *flagged* when it is
blocked or sent on to Layer D; Layer C is a high-recall gate that by design sends about half of
the clean URLs to the LLM, so its precision and F1 are low on purpose and the miss rate is the
number that matters. For Layer D and the full pipeline, *strict* = verdict *phishing*, *warn* =
*phishing* or *suspicious* (the user is warned).

### 3.1 Layer C and baselines — dataset test split (81,633 URLs), same rule for every model

| Model | Test miss % | Precision | Recall | F1 | ROC-AUC | VN miss % / clean % |
|---|---|---|---|---|---|---|
| Logistic regression (URL) | 5.37 | — | 0.946 | 0.762 | 0.918 | 30.6 / 97.5 |
| Random forest (URL) | 3.57 | — | 0.964 | 0.772 | 0.940 | 50.1 / 92.7 |
| LightGBM (URL) | 2.85 | — | 0.972 | 0.776 | 0.946 | 44.5 / 98.0 |
| Transformer (URL) | 2.80 | — | 0.972 | 0.777 | 0.946 | 27.3 / 78.5 |
| URLNet (URL) | 1.63 | — | 0.984 | 0.785 | 0.963 | 13.5 / 98.4 |
| char-CNN (URL) | 1.10 | — | 0.989 | 0.787 | 0.971 | 18.2 / 99.0 |
| BiLSTM (URL) | 1.02 | — | 0.990 | 0.787 | 0.972 | 17.4 / 98.0 |
| LightGBM (URL + host) | 0.033 | — | 0.9997 | 0.791 | 0.995 | 0.78 / 16.3 |
| Previous Layer C: CNN/BiLSTM + XGBoost | 0.033 | — | 0.9997 | 0.791 | 0.994 | 0.66 / 51.1 |
| **Final Layer C: BiLSTM + XGBoost** | **0.033** | — | **0.9997** | **0.790** | **0.994** | **0.59 / 50.1** |

Precision is ≈ 0.65 for every row (≈ 50 % of clean URLs are sent on by the rule); recall = 1 − miss.
Selection candidates on val (`reports/layer_c_selection/`): DL 424–1,128 misses; ML on BiLSTM:
XGBoost 18, LightGBM 24, random forest 32. With the same features, LightGBM and random forest
recognise only 31 % / 21 % of Vietnamese clean URLs at the val threshold (XGBoost 50 %).

### 3.2 Pipeline layers — production thresholds, host data from the cache, LLM not called

| Set | Layers | Missed | Clean recognised | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Dataset test (39,266 phishing / 42,367 clean) | A | 10,065 | 100 % | 1.000 | 0.744 | 0.853 |
| | B | 36,206 | 99.7 % | 0.964 | 0.078 | 0.144 |
| | C (previous) | 12 | 51.1 % | — | 0.9997 | — |
| | **C (final)** | **10** | **51.1 %** | 0.654 | 0.9997 | 0.791 |
| | **A + B + C** | **5** | **50.9 %** | 0.654 | 0.9999 | 0.791 |
| Vietnamese set (2,710 / 1,580) | A (feeds) | 2,700 | 99.9 % | 0.909 | 0.004 | 0.007 |
| | B | 2,653 | 99.8 % | 0.950 | 0.021 | 0.041 |
| | C (previous) | 28 (1.03 %) | 48.4 % | — | 0.990 | — |
| | **C (final)** | **29 (1.07 %)** | **52.7 %** | 0.782 | 0.989 | 0.874 |
| | **A + B + C** | **29** | **54.4 %** | 0.788 | 0.989 | 0.877 |
| VN sample, 95 + 95 | Blocklist only | 94 | 100 % | 1.000 | 0.011 | 0.021 |
| | **Blocklist + VirusTotal** | **28** | **97.9 %** | **0.971** | **0.705** | **0.817** |

The blocklist feeds are also sources of the dataset's phishing labels, so A on the dataset test
split is optimistic; "B + C (no A)" misses 10.

### 3.3 Layer D and full pipeline

| Setting | Model | Phishing → safe | Clean → phishing | Strict P / R / F1 | Warn P / R / F1 | p50 / p95 |
|---|---|---|---|---|---|---|
| Layer D, held-out test (2,004 pages), with C override | gpt-6-luna | 5.08 % | 2.90 % | 0.959 / 0.678 / 0.795 | 0.929 / 0.949 / 0.939 | 2.0 / 4.8 s |
| | claude-haiku-5-5 | 3.98 % | 4.80 % | 0.940 / 0.745 / 0.831 | 0.888 / 0.960 / 0.923 | 1.5 / 3.2 s |
| Full pipeline, live VN (300 + 300), previous Layer C | gpt-6-luna | — | — | 0.954 / 0.417 / 0.580 | 0.899 / 0.890 / 0.895 | 2.9 / 8.7 s |
| Full pipeline, live VN, **final** | gpt-6-luna | — | — | 0.957 / 0.517 / 0.671 | 0.900 / 0.900 / **0.900** | 2.7 / 8.0 s |
| | claude-haiku-5-5 | — | — | 0.969 / 0.523 / 0.680 | 0.910 / 0.907 / **0.908** | 2.3 / 7.4 s |

Run-to-run (same pages, second run): verdict agreement 95.4–97.4 %, phishing → safe 7.57 / 7.67 %
(gpt test) and 5.98 / 5.58 % (Haiku test), no McNemar p < 0.12. Cost per page: gpt $0.000129,
Haiku $0.000224. Live VN runs skip VirusTotal (quota); many blacklisted Vietnamese pages are
offline, so their verdict comes from the fallback rules (hence the low strict recall).

### 3.4 Latency (single URL, no LLM; `reports/pipeline/metrics.json`)

| Stage | p50 | p95 |
|---|---|---|
| A blocklist | 0.11 ms | 0.15 ms |
| B typosquatting | 0.50 ms | 2.4 ms |
| C (host data cached) | 31.7 ms | 34.6 ms |
| A + B + C | 32.5 ms | 35.2 ms |
| VirusTotal (cached / live) | ~1 ms / up to 2 s timeout | — |
| D (LLM) | 1.5–2.7 s | 3.2–8.0 s |

## 4. Contributions for a research paper

Each claim is followed by its evidence; claims without enough evidence are listed as limits.

1. **A recall-first cascade that keeps the LLM for the hard half.** Layers A + B + C miss 5 of
   39,266 test phishing URLs (0.013 %) while clearing 50.9 % of clean URLs in ~35 ms without any
   LLM call; on the held-out pages that reach the LLM, the C override cuts phishing → safe to
   5.1 % (gpt) / 4.0 % (Haiku).
2. **Host evidence plus a deep URL score, selected under one miss-first rule.** Across 5 deep
   models × 3 classifiers and 8 baselines, URL-only models miss 30–160× more test phishing than
   host-based ones. Adding the BiLSTM score to XGBoost is what lets the validation threshold
   transfer to unseen Vietnamese URLs: 50 % clean recognised at 0.59 % misses, vs 16–31 % for
   LightGBM / random forest with the same host features.
3. **A zero-leakage, real-world Vietnamese benchmark that exposes feed coverage gaps.** 4,290
   URLs (Chong Lua Dao phishing, Tranco `.vn` and verified brand domains) with checked
   domain isolation. The global feeds catch 0.4 % of this phishing and VirusTotal 70.5 %
   (sample); the ML gate misses 1.07 %; the full live pipeline reaches warn-F1 0.90–0.91.
4. **A model-agnostic, injection-hardened LLM layer with measured stability.** One sanitizer
   (data-driven, 16 languages) and one English prompt serve two LLM families: hard
   prompt-injection attacks flip ≤ 1 of 324 real pages to *safe*; two identical runs agree on
   95–97 % of verdicts with no significant difference; the model swap trades misses for false
   alarms (Haiku: fewer misses p = 0.004, more clean pages blocked p = 0.005).
5. **A security-constrained use of page structure.** A DOM model trained on PhreshPhish pages
   disjoint from every evaluation set ranks held-out pages well (ROC-AUC 0.935), but letting it
   clear URL flags is rejected because the page is attacker-controlled (cloaking); under the
   "add suspicion only" rule it does not meet the clean-traffic target, a negative result that
   argues for URL / host evidence as the gate.

**Limits.** Labels of public feeds are noisy (4.7 % disagreement with VirusTotal on the
relabelling subset, half of which VirusTotal does not know); the VirusTotal measurement covers a
190-URL sample; Layer C needs live host lookups at run time; Layer D still lets 4–5 % of the
phishing pages it sees pass as *safe*; each deep model was trained once (one seed), and the
BiLSTM / CNN-BiLSTM difference on val (424 vs 428 misses) is within noise.
