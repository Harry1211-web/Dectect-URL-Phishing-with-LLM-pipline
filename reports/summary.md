# Experiment summary — system improvements and comparisons

Rules followed: every configuration was chosen on dev / val only; each held-out test set was run
once per chosen configuration; every result is listed, including models and changes that lost.
One production choice is a design trade-off fixed before training (no network lookups, Layer C below).
All numbers use the audited labels (`dataset/urls/label_fixes.csv`, `src/data_collection/audit_labels.py`).
Source files: `reports/benchmark.json`, `reports/layer_c_selection/*.json`,
`reports/pipeline/*.json`, `reports/typosquatting/metrics.json`, `reports/baselines/*.json`,
`reports/llm/redteam_injection/summary_hard_*.json`, `reports/llm/layer_d/compare_*.json`.

## Label audit

| Rule | Set | Change | Rows |
|---|---|---|---|
| PhiUSIIL domain registered after the dataset was collected (new owner) | dataset | drop | 2,703 |
| Homepage of a Tranco-ranked domain, not in current feeds, not reported by VirusTotal | dataset | phishing → clean | 188 |
| Hand labels (`reports/relabel/hand_review.csv`, 234 candidates) | dataset | phishing → clean / drop | 58 / 2 |
| | VN | clean → phishing / phishing → clean / drop (gambling, illegal investment) | 2 / 6 / 25 |
| | PhreshPhish | phishing → clean / drop | 5 / 6 |

Dataset after the audit: 534,908 URLs (train 374,699, val 78,984, test 81,225); VN set 2,681
phishing / 1,584 clean; PhreshPhish sample 9,991 / 10,003. 190 popular homepages could not be
checked on VirusTotal (free quota) and keep their phishing label.

## Layer C

Selection on val (threshold recognising 50 % of val clean URLs): DL step BiLSTM 414 misses
(CNN/BiLSTM 427, char-CNN 461, URLNet 594, Transformer 1,095); ML step on the BiLSTM score +
URL features + Tranco (production setting): **XGBoost 19**, LightGBM 19 (tie, XGBoost has the higher
val PR-AUC 0.98925 vs 0.98920), random forest 31. With network host features (`ml_network.json`):
XGBoost 16, LightGBM 20, random forest 28.

Production choice:
- **No network lookups** (`train.py --host-lookups none`): host features reduced to the Tranco rank
  (local file). A subdomain or user path of a shared platform does not inherit the platform's rank;
  the platform's own bare homepage (google.com/) does. Rule fixed before training: accept if val
  misses rise by at most 30 %.

| Layer C (production thresholds) | Val miss / clean | Test miss / clean | Test A+B+C miss | PhreshPhish A+B+C miss / clean | C latency p50 / p95 |
|---|---|---|---|---|---|
| DNS + TLS + RDAP + Tranco | 15 / 54.1 % | 10 / 54.5 % | 4 | 32 / 38.4 % | 1.4 / 2.5 s (new host) |
| **Tranco only (production)** | **13 / 52.0 %** | 12 / 52.6 % | **4** | 43 / 44.1 % | **32 / 36 ms** |

Ablation of the production setting (3 seeds, val misses mean, `ablation.json`): full 18.0; without
Tranco 392.7; without the BiLSTM score 29.3; without every hand-crafted feature 27.0; without URL
length / structure 21.0; other URL groups 17.3–19.0.
Ablation with network host features (`ablation_network.json`): full 17.0; without Tranco 71.0;
without DNS 18.0, TLS 17.3, RDAP 19.0; without DNS + TLS + RDAP 23.7 (PR-AUC lower on all four
sets); without every host feature 298.

## Layer C baselines

Rule for every model: threshold = lowest score recognising ≥ 50 % of val clean URLs; test,
PhreshPhish and VN scored once (never trained on). "+ B" = a URL flagged by Layer B goes to D anyway.
One seed per model.

| Model | Test miss (+ B misses) | Test verified miss | Test clean | Test ROC / PR-AUC | PhreshPhish miss (+ B) | PhreshPhish clean | PhreshPhish ROC-AUC | VN miss | VN clean | VN ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression (lexical) | 5.14 % (1,952) | 154 / 5,549 | 50.0 % | 0.9208 / 0.9313 | 4.46 % (3.95 %) | 48.2 % | 0.897 | 30.66 % | 97.2 % | 0.954 |
| Random forest (lexical) | 3.61 % (1,378) | 116 | 50.6 % | 0.9411 / 0.9506 | 3.06 % (2.77 %) | 49.5 % | 0.912 | 50.32 % | 92.4 % | 0.882 |
| LightGBM (lexical) | 2.76 % (1,043) | 130 | 50.7 % | 0.9474 / 0.9551 | 3.32 % (2.90 %) | 57.6 % | 0.930 | 46.66 % | 98.0 % | 0.948 |
| Transformer (2 layers, d = 64) | 2.77 % (1,050) | 134 | 50.5 % | 0.9472 / 0.9542 | 3.68 % (3.17 %) | 61.5 % | 0.916 | 40.17 % | 84.9 % | 0.825 |
| URLNet (reduced: char + word CNN) | 1.49 % (576) | 93 | 51.1 % | 0.9649 / 0.9693 | 2.23 % (1.97 %) | 63.0 % | 0.948 | 12.46 % | 96.9 % | 0.958 |
| char-CNN | 1.03 % (396) | 69 | 50.9 % | 0.9713 / 0.9745 | 2.22 % (2.05 %) | 63.3 % | 0.948 | 20.40 % | 98.1 % | 0.964 |
| BiLSTM | 1.02 % (394) | 69 | 51.1 % | 0.9731 / 0.9763 | 2.83 % (2.53 %) | 71.9 % | 0.953 | 22.79 % | 98.2 % | 0.964 |
| LightGBM (lexical + network host) | 0.026 % (10) | 3 | 50.9 % | 0.9959 / 0.9959 | 0.200 % (0.180 %) | 37.5 % | 0.972 | 0.075 % | 13.5 % | 0.992 |
| **Layer C (BiLSTM + XGBoost, URL + Tranco)**, same rule | **0.021 % (8)** | 4 | 51.1 % | 0.9894 / 0.9899 | 0.460 % (0.400 %) | 47.4 % | 0.966 | 0.522 % | **55.1 %** | **0.994** |

Reading:
- Every URL-only model, deep or classic, misses 1–5 % of test phishing; Layer C 0.02 %.
- LightGBM with network host features misses as few on test and fewer on PhreshPhish, but at the
  val threshold it recognises only 13.5 % of the Vietnamese clean URLs (Layer C: 55.1 %) and needs
  1–2.5 s of lookups per new host.
- URL-only deep models beat URL-only classic models on test but transfer badly to the VN set
  (12–23 % misses at the val threshold).

## Pipeline without the LLM (production thresholds)

| Set | A + B + C miss | Clean recognised | B + C (no A) miss |
|---|---|---|---|
| Dataset test (38,800 phishing / 42,425 clean) | 4 (0.010 %) | 52.5 % | 12 |
| VN (2,681 / 1,584) | 15 (0.56 %) | 58.3 % | 15 |
| PhreshPhish (9,991 / 10,003) | 43 (0.43 %) | 44.5 % | 43 |

Trusted shortcut (official domains of 160 international brands, no user-content domains): clean
URLs no longer sent to D: dataset test 7 of 20,173, PhreshPhish 38 of 5,589, VN 1 of 661; phishing
URLs trusted: **0** on every set.

Layer B on Tranco top-100k (false positives): 232 flagged (0.23 %), p50 0.21 ms.

## Hard prompt-injection red team (no defence vs production)

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

## Relabelled subset (VirusTotal labels)

`reports/relabel/relabel_subset.csv`: 300 pages (PhreshPhish dev 100, test 120, live VN 80; half
from pages where the layers disagree with the label or with each other). Reference labels are
VirusTotal's (`vt_label`, filled by `src/relabel_vt.py`: >= 2 engines phishing, 0 clean, else
unsure). `python src/relabel_subset.py score` writes the error rates against them to
`reports/relabel/relabel_results.json`: audited dataset labels disagree on 6 of 148 decided rows
(4.1 %, 95 % CI 1.9-8.6 %); 3 rows left the sets in the label audit (`relabel_subset.py refresh`).

## Layer D: gpt-6-luna vs Claude Haiku 5.5

Same production prompt, nonce frame, sanitizer and `parse_verdict`; only the model changes. Stored
verdicts rescored on the audited labels and the production Layer C (the prompt reads neither).
95 % Wilson intervals in brackets.

| | Dev gpt-6-luna | Dev Haiku (default) | **Test gpt-6-luna** | **Test Haiku (default)** |
|---|---|---|---|---|
| Pages | 648 | 648 | 2,004 | 2,004 |
| Phishing → safe | 13.7 % [10.5–17.8] | 14.6 % [11.2–18.7] | 7.6 % [6.1–9.4] | **6.0 % [4.7–7.6]** |
| Phishing → suspicious | 26.8 % | 23.3 % | 24.6 % | 19.5 % |
| Phishing → phishing | 59.5 % | 62.1 % | 67.8 % | **74.5 %** |
| Phishing → safe, with C override | 6.4 % [4.3–9.5] | 6.7 % [4.5–9.9] | 5.1 % [3.9–6.6] | **4.4 % [3.3–5.8]** |
| Clean → phishing | 1.3 % [0.5–3.3] | 3.0 % [1.6–5.5] | **2.9 % [2.0–4.1]** | 4.8 % [3.6–6.3] |
| Clean → suspicious (with C override) | 0.7 % (3.0 %) | 2.6 % (4.9 %) | **1.4 % (3.5 %)** | 5.3 % (6.8 %) |
| p50 / p95 latency | 2.5 / 7.3 s | **1.5 / 3.1 s** | 2.0 / 4.8 s | **1.5 / 3.2 s** |
| Cost per page | **$0.000126** | $0.000220 | **$0.000129** | $0.000224 |
| Input / output tokens | 793 / 92 | 1,315 / 177 | 805 / 98 | 1,339 / 180 |

Paired exact McNemar, test (2,004 pages): Haiku misses fewer phishing (22 vs 6 discordant,
p = 0.004; with override 12 vs 5, p = 0.14) but blocks more clean pages (12 vs 31, p = 0.005). On
dev no difference is significant (p ≥ 0.12). Strata: dev reaches_d 570, missed_by_abc 25; test
reaches_d 1,846, missed_by_abc 9.

Run-to-run variation of gpt-6-luna (dev, 3 runs): phishing → safe 13.7 / 13.7 / 14.9 %, with
override 6.4 / 6.4 / 6.7 %, clean → phishing 1.3 / 2.0 / 1.6 %, verdict agreement 96.3–97.2 %,
no pairwise McNemar p < 0.2.

## Full pipeline on live pages

| Run | Model | Phishing warned (phishing or suspicious) | Phishing → safe | Clean → safe | Warn F1 | p50 / p95 |
|---|---|---|---|---|---|---|
| VN, 300 + 300 | gpt-6-luna | 90.7 % | 9.3 % | 90.0 % | 0.904 | 0.2 / 7.7 s |
| | claude-haiku-5-5 | 92.7 % | 7.3 % | 89.3 % | 0.911 | 0.1 / 7.9 s |
| Deep links of top sites, 400 clean | gpt-6-luna | — | — | 93.0 % (0.75 % phishing) | — | 0.1 / 6.0 s |

About half of the Vietnamese phishing pages are offline: their verdict comes from the fallback
rules (suspicious).
