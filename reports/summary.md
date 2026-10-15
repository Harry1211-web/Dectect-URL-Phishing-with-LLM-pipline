# Experiment summary — system improvements and comparisons (2026-10-15)

Rules followed: every configuration was chosen on dev / val only; each held-out test set was run
once per chosen configuration; every result is listed, including models and changes that lost.
Source files: `reports/pipeline/trusted_shortcut.json`, `reports/typosquatting/metrics.json`,
`reports/baselines/*.json`, `reports/llm/redteam_injection/summary_hard_*.json`,
`reports/llm/layer_d/compare_*.json`.

## Spend

| Provider | Before | After | This work | Runs |
|---|---|---|---|---|
| OpenAI (gpt-6-luna) | $3.734 | $4.082 | **$0.348** | 2 dev repeats ($0.161), hard red team ($0.187) |
| Anthropic (claude-haiku-5-5) | $0 | $1.047 | **$1.047** | pilot 2×20 pages ($0.007), dev ×2 modes ($0.232), test once ($0.449), hard red team ($0.360) |

Budgets in `.env`: `LLM_BUDGET_USD=7.70`, `ANTHROPIC_BUDGET_USD=4.8` (ledger `data/llm_spend.json`,
now kept per provider).

## Phase 1 — system improvements

| Change | Measure | Before | After | Decision |
|---|---|---|---|---|
| Layer D verdict cache (per URL, 6 h, failures not cached) | — | already in place (b9e89bc) | — | no change |
| Trusted shortcut + 160 international brand domains (no user content) | PhreshPhish clean pages reaching D | 5,738 | 5,704 (−34, **−0.59 %**) | **kept** (user decision) |
| | dataset test clean reaching D | 20,779 | 20,773 (−6) | |
| | VN clean reaching D / live top-site clean reaching D | 788 / 116 | 787 / 115 | |
| | phishing trusted by the shortcut (dataset 260,734 + PhreshPhish 10,000 + VN 2,710) | 0 | **0** | |
| Layer B: multilingual lure words for brand + word domains | PhreshPhish phishing flagged by B | 1,401 | 1,403 | **rejected**: the 13 extra phishing hits (2 PhreshPhish, 11 test) were all already caught by A or C, so pipeline misses do not change |
| | Tranco top-100k flagged (false positives) | 232 | 241 | flags brands' own regional domains (shopeepay.sg, samsungpay.com) and one chance match ("norderstedt" ⊃ "orders") |
| Layer B: any embedded brand ≥ 5 chars, no lure word | Tranco top-100k flagged | 232 | 906 | **rejected** |

The international list was chosen on train / val phishing only. After a security review, a domain
excluded for user content now excludes every domain of its brand (google.com → googlevideo.com,
google.dev), and payment-link / profile sites (stripe.com, squareup.com, cash.app) are excluded.

Note on host data: an analysis run refreshed ~14.9k expired DNS / TLS entries of test and external
hosts in `data/host_cache.sqlite` (not versioned). All host-based numbers below were recomputed on
the refreshed cache; Layer C's production test misses moved from 13 to 12 because of it.

## Phase 2 — Layer C baselines

Rule for every model: threshold = lowest score recognising ≥ 50 % of val clean URLs; test,
PhreshPhish and VN scored once (never trained on). "+ B" = a URL flagged by Layer B goes to D anyway.
One seed per model; deep models capped at 8 epochs (early stopping, patience 2).

| Model | Test miss (+ B) | Test verified miss | Test clean | Test ROC / PR-AUC | PhreshPhish miss (+ B) | PhreshPhish clean | PhreshPhish ROC-AUC | VN miss (+ B) | VN clean | VN ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression (lexical) | 5.37 % (5.26 %) | 160 / 5,549 | 50.1 % | 0.918 / 0.930 | 4.72 % (4.19 %) | 49.0 % | 0.897 | 30.6 % (29.8 %) | 97.5 % | 0.957 |
| Random forest (lexical) | 3.57 % (3.51 %) | 124 | 50.6 % | 0.940 / 0.950 | 3.22 % (2.91 %) | 50.2 % | 0.913 | 50.1 % (49.3 %) | 92.7 % | 0.892 |
| LightGBM (lexical) | 2.85 % (2.78 %) | 123 | 50.7 % | 0.946 / 0.954 | 3.34 % (2.92 %) | 57.6 % | 0.930 | 44.5 % (43.7 %) | 98.0 % | 0.952 |
| **LightGBM (lexical + host)** | **0.033 % (13)** | 4 | 51.0 % | **0.9954 / 0.9956** | **0.44 % (0.41 %)** | 41.0 % | **0.971** | 0.78 % (0.78 %) | **16.3 %** | 0.983 |
| char-CNN | 1.10 % (1.09 %) | 69 | 51.5 % | 0.971 / 0.974 | 2.41 % (2.00 %) | 64.4 % | 0.950 | 18.2 % (17.5 %) | 99.0 % | 0.970 |
| BiLSTM | 1.02 % (1.02 %) | 55 | 51.2 % | 0.972 / 0.975 | 2.68 % (2.30 %) | 66.9 % | 0.949 | 17.4 % (16.8 %) | 98.0 % | 0.962 |
| URLNet (reduced: char + word CNN) | 1.63 % (1.62 %) | 110 | 51.4 % | 0.963 / 0.968 | 2.62 % (2.32 %) | 67.6 % | 0.949 | 13.5 % (13.0 %) | 98.4 % | 0.967 |
| Transformer (2 layers, d = 64) | 2.80 % (2.73 %) | 111 | 50.9 % | 0.946 / 0.954 | 3.41 % (2.87 %) | 59.4 % | 0.914 | 27.3 % (26.8 %) | 78.5 % | 0.828 |
| **Layer C (CNN/BiLSTM + XGBoost, host)**, same rule | **0.033 % (13)** | **3** | 51.0 % | 0.9937 / 0.9937 | 0.57 % (0.51 %) | 44.9 % | 0.964 | **0.66 % (0.66 %)** | **51.1 %** | **0.993** |
| Layer C, production segment thresholds | 0.031 % (12) | 2 | 51.1 % | — | 0.57 % (0.49 %) | 42.7 % | — | 1.03 % (1.03 %) | 48.4 % | — |

Reading:
- Host features (DNS / TLS / RDAP / Tranco) carry most of the performance: every URL-only model,
  deep or classic, misses 30–160× more test phishing than the two host-based models.
- **LightGBM on hand-crafted + host features (no CNN) ties Layer C on test (13 vs 13 misses) and
  beats it on PhreshPhish ranking (ROC-AUC 0.971 vs 0.964; 44 vs 57 misses, but at 41.0 % instead
  of 44.9 % clean recognised).** Layer C wins clearly on the Vietnamese set: same miss level
  (0.66 % vs 0.78 %) with 51.1 % instead of 16.3 % clean recognised, and higher ROC-AUC (0.993 vs
  0.983). The CNN/BiLSTM score is what makes the val threshold transfer to Vietnamese URLs; it adds
  little where host data is present and the distribution matches training.
- URL-only deep models beat URL-only classic models (char-CNN / BiLSTM ≈ 1 % vs 2.9–5.4 % test
  misses) but transfer badly to the VN set (13–27 % misses at the val threshold).

## Phase 3 — evidence

### Hard prompt-injection red team (no defence vs production)

Attacks: audit badge, official-domain announcement, three-part "reviewer notes" (header / body /
footer), fake resolved ticket "VERDICT: safe", the claim in Vietnamese / German / Spanish / Chinese,
all combined. Three of them pass the sanitizer undetected (no injection warning).

| Model | Set | No defence: flips to safe / downgrades to suspicious | Production: flips / downgrades |
|---|---|---|---|
| gpt-6-luna | 25 simulated × 6 attacks | 0 / 150, 0 | 0 / 150, 0 |
| gpt-6-luna | 100 real dev phishing pages × 6 | **1** / 324 (multi-step), **21** | 0 / 330, **5** |
| Haiku 5.5 | 25 simulated × 6 | 0 / 150, 0 | 0 / 150, 0 |
| Haiku 5.5 | 100 real pages × 6 | 0 / 312, 5 | 0 / 354, 6 |

The harder attacks still rarely flip a verdict to "safe" even without defence, so this red team
does not show a large defence effect. The measurable effect is on gpt-6-luna real pages: the
production path cuts phishing → suspicious downgrades from 21 to 5 and the one flip to 0. Haiku is
robust with or without the defence.

### Hand-relabelled subset

`reports/relabel/relabel_subset.csv`: 300 pages (PhreshPhish dev 100, test 120, live VN 80; half
from pages where the layers disagree with the label or with each other), with URL, title, text
snippet and every layer's verdict. **Waiting for hand labels** (`my_label`); then
`python src/relabel_subset.py score` writes the true error rates (per row and re-weighted by the
sampling pools) to `reports/relabel/relabel_results.json`. 22 VN pages could no longer be downloaded.

## Phase 4 — Layer D: gpt-6-luna vs Claude Haiku 5.5

Same production prompt, nonce frame, sanitizer and `parse_verdict`; only the model changes. Haiku
mode chosen on dev: "default" (adaptive thinking) and "off" missed the same phishing (McNemar
p = 1.0); default had fewer false alarms (9 vs 11 clean → phishing, 8 vs 21 clean → suspicious).
95 % Wilson intervals in brackets.

| | Dev gpt-6-luna | Dev Haiku (default) | **Test gpt-6-luna** | **Test Haiku (default)** |
|---|---|---|---|---|
| Phishing → safe | 15.7 % [12.2–19.8] | 16.5 % [13.0–20.8] | 7.6 % [6.1–9.4] | **6.0 % [4.7–7.6]** |
| Phishing → suspicious | 26.2 % | 22.8 % | 24.6 % | 19.5 % |
| Phishing → phishing | 58.1 % | 60.7 % | 67.8 % | **74.5 %** |
| Phishing → safe, with C override | 8.6 % [6.0–11.9] | 9.1 % [6.5–12.6] | 5.0 % [3.8–6.5] | **4.1 % [3.0–5.5]** |
| Clean → phishing | 1.3 % [0.5–3.4] | 3.0 % [1.6–5.6] | **2.9 % [2.0–4.1]** | 4.8 % [3.6–6.3] |
| Clean → suspicious (with C override) | 0.7 % (2.0 %) | 2.7 % (4.0 %) | **1.4 % (3.4 %)** | 5.3 % (6.7 %) |
| p50 / p95 latency | 2.5 / 7.4 s | **1.5 / 3.1 s** | 2.0 / 4.8 s | **1.5 / 3.2 s** |
| Cost per page | **$0.000126** | $0.000220 | **$0.000129** | $0.000224 |
| Input / output tokens | 792 / 93 | 1,313 / 178 | 805 / 98 | 1,339 / 180 |

Paired exact McNemar, test (2,004 pages, both answered all): Haiku misses fewer phishing (22 vs 6
discordant, p = 0.004; with override 13 vs 4, p = 0.049) but blocks more clean pages (12 vs 31,
p = 0.005). On dev no difference is significant (p ≥ 0.18). Dev strata: reaches_d 600, missed_by_abc
51; test: reaches_d 2,000, missed_by_abc 4 (per-stratum numbers in the compare files).

Run-to-run variation of gpt-6-luna (dev, 3 runs): phishing → safe 15.7 / 15.7 / 16.8 %, with
override 8.6 / 8.6 / 9.1 %, clean → phishing 1.3 / 2.0 / 1.7 %, verdict agreement between runs
96.3–97.2 %, no pairwise McNemar p < 0.2. Differences of 1–2 points on dev are within this noise.

Note: the Haiku 20-page pilot "off" mode p95 (10.2 s) came from a single slow call; the full dev
run in that mode had p95 1.5 s.
