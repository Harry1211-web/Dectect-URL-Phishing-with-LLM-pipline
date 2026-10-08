"""Find and fix mislabelled URLs in the dataset with evidence independent of Layer C.

Rules (each fix is written with its reason and evidence to dataset/urls/label_fixes.csv):
  1. phishing -> clean, automatic: homepage of a Tranco top-100k registrable domain itself (not a
     subdomain: tenants of free hosting / dynamic DNS are excluded), not a shared platform and no
     longer in any phishing feed (feed noise / long-cleaned compromise).
  2. clean -> phishing, verified: clean URL listed in the current phishing feeds AND judged
     phishing by the LLM (Layer D, verdict only) on the fetched page.
Rejected after manual review of samples (kept out on purpose):
  - "page looks safe today" -> clean: taken-down phishing pages (Zoho forms, link-in-bio,
    shorteners) look harmless now, so today's content says nothing about the reported URL;
  - "young domain + LLM says phishing" -> phishing: mostly gambling sites or expired domains
    re-registered by someone else, which are not phishing.
The Layer C model is NOT used, so evaluating Layer C on the fixed labels is not circular.
LLM calls are budget-guarded (data/llm_spend.json).

Run:    python src/data_collection/audit_labels.py [--max-llm 2500]
Then:   python src/data_collection/build_dataset.py  (applies label_fixes.csv)
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
for p in (SRC, SRC / "llm_rag"):
    sys.path.insert(0, str(p))
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.paas_domains import is_untrusted_hosting, is_user_content_host  # noqa: E402
from url_utils import host_of, registrable_domain  # noqa: E402

ROOT = SRC.parent
DATA = ROOT / "dataset" / "urls" / "urls_labeled.csv"
FIXES = ROOT / "dataset" / "urls" / "label_fixes.csv"
TRANCO = ROOT / "data" / "tranco_top1m.csv"
POPULAR_TOP = 100_000


def _llm_verdict(url: str) -> tuple[str, str]:
    """(verdict or '', note) from Layer D on the live page; '' when the page cannot be used."""
    from llm_client import analyze_suspicious, fetch
    page = fetch(url)
    if not page[0] and not page[1]:
        return "", "page not fetchable"
    r = analyze_suspicious(url, page=page, use_rag=True)
    return (r["verdict"] or ""), ("llm " + (r["verdict"] or r["explanation"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-llm", type=int, default=2500, help="cap on LLM-verified candidates")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    with TRANCO.open(encoding="utf-8") as f:
        rank = {ln.rstrip().split(",", 1)[1]: i + 1 for i, ln in enumerate(f) if i < POPULAR_TOP}
    df["rd"] = df.url.map(registrable_domain)
    df["host"] = df.url.map(host_of)
    df["popular"] = df.rd.isin(rank)
    df["shared"] = [is_untrusted_hosting(r) or is_user_content_host(h) for r, h in zip(df.rd, df.host)]
    rest = df.url.str.replace(r"^https?://", "", regex=True).str.rstrip("/")
    df["bare"] = ~rest.str.contains("/") & ~rest.str.contains(r"\?")
    df["apex"] = [h in (r, "www." + r) for h, r in zip(df.host, df.rd)]

    fixes: list[dict] = []

    def fix(row, new: str, rule: str, evidence: str) -> None:
        fixes.append({"url": row.url, "old_label": row.label, "new_label": new, "rule": rule,
                      "evidence": evidence})

    # Rule 1 / 2 candidates
    cand = df[(df.label == "phishing") & df.popular & ~df.shared].copy()
    cand["listed"] = [check_blocklist(u)["flag"] is True for u in cand.url]
    cand = cand[~cand.listed]
    for row in cand[cand.bare & cand.apex].itertuples():
        fix(row, "clean", "1_popular_bare_homepage",
            f"Tranco rank {rank[row.rd]}, not in current feeds, homepage only")

    # Rule 2 candidates: clean URLs listed in the current feeds
    clean = df[df.label == "clean"]
    listed = clean[[check_blocklist(u)["flag"] is True for u in clean.url]]
    print(f"rule1 fixes: {len(fixes)} | rule2 candidates: {len(listed)}", flush=True)
    jobs = list(listed.itertuples())[:args.max_llm]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda r: _llm_verdict(r.url), jobs))
    for row, (verdict, note) in zip(jobs, results):
        if verdict == "phishing":
            fix(row, "phishing", "2_clean_in_feeds_llm_phishing", "listed in current feeds, " + note)

    with FIXES.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["url", "old_label", "new_label", "rule", "evidence"])
        w.writeheader()
        w.writerows(fixes)
    print(pd.DataFrame(fixes).groupby(["rule", "old_label", "new_label"]).size() if fixes else "no fixes")
    print(f"-> {FIXES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
