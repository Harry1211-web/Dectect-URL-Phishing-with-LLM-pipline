"""Find and fix mislabelled URLs of every dataset with evidence independent of Layer C.

Fixes go to dataset/urls/label_fixes.csv (dataset, url, old_label, new_label, rule, evidence);
new_label "drop" removes the row. Datasets: urls (urls_labeled.csv: train / val / test),
vi (data/test_sets/vi/), phreshphish (dataset/external/phreshphish_urls.csv).

Rules
  1. urls, phishing -> clean: homepage of a Tranco-ranked registrable domain itself (not a
     subdomain: tenants of free hosting / dynamic DNS are excluded), not a shared platform, not in
     the current feeds, and VirusTotal has looked it up without reporting it (>= 2 engines; a
     failed lookup keeps the label). Hand review
     (reports/relabel/hand_review.csv) found 24 / 24 such homepages clean.
  2. urls, clean -> phishing: clean URL listed in the current feeds AND judged phishing by the LLM
     (Layer D, verdict only) on the fetched page (paid; --max-llm 0 skips it).
  3. urls, drop: PhiUSIIL phishing domain registered after the dataset was collected: the site
     belongs to a new owner, so its host features describe another site.
  4. every dataset: hand labels of reports/relabel/hand_review.csv (my_label). phishing / clean
     replace the label; unsure and not_phishing_but_illegal drop the row. In vi a homepage label
     holds for its apex / www twin, and an illegal-but-not-phishing domain drops all its rows.
Rejected after manual review of samples (kept out on purpose):
  - "page looks safe today" -> clean: taken-down phishing pages (Zoho forms, link-in-bio,
    shorteners) look harmless now, so today's content says nothing about the reported URL;
  - "young domain + LLM says phishing" -> phishing: mostly gambling sites or expired domains
    re-registered by someone else, which are not phishing.
The Layer C model is NOT used, so evaluating Layer C on the fixed labels is not circular.
Rules work on the labels before any fix (old_label of earlier fixes), so a re-run is stable.

Run:    python src/data_collection/audit_labels.py [--max-llm 0] [--vt-requests 0]
        (writes label_fixes.csv, applies it to the vi and PhreshPhish files)
Then:   python src/data_collection/build_dataset.py --apply-fixes-only  (urls_labeled.csv)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
for p in (SRC, SRC / "llm"):
    sys.path.insert(0, str(p))
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.paas_domains import is_untrusted_hosting, is_user_content_host  # noqa: E402
from url_utils import host_of, registrable_domain  # noqa: E402

ROOT = SRC.parent
DATA = ROOT / "dataset" / "urls" / "urls_labeled.csv"
FIXES = ROOT / "dataset" / "urls" / "label_fixes.csv"
TRANCO = ROOT / "data" / "tranco_top1m.csv"
HAND_REVIEW = ROOT / "reports" / "relabel" / "hand_review.csv"
VI_DIR = ROOT / "data" / "test_sets" / "vi"
PHRESHPHISH = ROOT / "dataset" / "external" / "phreshphish_urls.csv"
FIELDS = ["dataset", "url", "old_label", "new_label", "rule", "evidence"]
# PhiUSIIL was collected before this date: a domain registered later has a new owner
PHIUSIIL_COLLECTED = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc).timestamp()
VT_MAX_AGE_S = 30 * 86400  # reuse VirusTotal lookups of the last 30 days
HAND_SETS = {"test": "urls", "vi": "vi", "phreshphish": "phreshphish"}


def load_fixes(dataset: str | None = None) -> pd.DataFrame:
    """label_fixes.csv (rows of one dataset when given)."""
    if not FIXES.exists():
        return pd.DataFrame(columns=FIELDS)
    f = pd.read_csv(FIXES, dtype=str).fillna("")
    if "dataset" not in f:
        f["dataset"] = "urls"
    return f[f.dataset == dataset] if dataset else f


def apply_fixes(df: pd.DataFrame, dataset: str, label_col: str = "label") -> tuple[pd.DataFrame, int]:
    """Apply the fixes of one dataset to a frame with a label column; returns (frame, rows changed)."""
    fx = load_fixes(dataset).drop_duplicates("url", keep="last").set_index("url").new_label
    new = df.url.map(fx)
    drop = new.eq("drop")
    change = new.notna() & ~drop & new.ne(df[label_col])
    out = df.copy()
    out.loc[change, label_col] = new[change]
    return out[~drop].reset_index(drop=True), int(change.sum() + drop.sum())


def apply_eval_set_fixes(which: tuple[str, ...] = ("vi", "phreshphish")) -> None:
    """Rewrite the vi and / or PhreshPhish files with their fixes (idempotent)."""
    if "vi" in which:
        _apply_vi()
    if "phreshphish" in which and PHRESHPHISH.exists():
        ph, n = apply_fixes(pd.read_csv(PHRESHPHISH), "phreshphish")
        ph.sort_values(["label", "date"]).to_csv(PHRESHPHISH, index=False)
        print(f"phreshphish: {n} rows changed or dropped", flush=True)


def _vi_frame() -> pd.DataFrame:
    return pd.concat([pd.read_csv(VI_DIR / f"{lab}.csv").assign(label=lab) for lab in ("phishing", "clean")],
                     ignore_index=True)


def _apply_vi() -> None:
    vi, n = apply_fixes(_vi_frame(), "vi")
    for lab in ("phishing", "clean"):
        part = vi[vi.label == lab].drop(columns="label").dropna(axis=1, how="all")
        part.sort_values("url").to_csv(VI_DIR / f"{lab}.csv", index=False)
    print(f"vi: {n} rows changed or dropped", flush=True)


def _llm_verdict(url: str) -> tuple[str, str]:
    """(verdict or '', note) from Layer D on the live page; '' when the page cannot be used."""
    from llm_client import analyze_suspicious, fetch
    page = fetch(url)
    if not page[0] and not page[1]:
        return "", "page not fetchable"
    r = analyze_suspicious(url, page=page)
    return (r["verdict"] or ""), ("llm " + (r["verdict"] or r["explanation"]))


def _original(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Labels before earlier fixes of this dataset (dropped rows are not in df any more)."""
    old = load_fixes(dataset).drop_duplicates("url").set_index("url").old_label
    df = df.copy()
    prev = df.url.map(old)
    df.loc[prev.notna(), "label"] = prev[prev.notna()]
    return df


def urls_rules(max_llm: int, vt_requests: int) -> list[dict]:
    """Rules 1-3 on urls_labeled.csv."""
    from url_classifier import host_features as hf
    from virustotal import lookup_batch, verdict_of

    df = _original(pd.read_csv(DATA), "urls")
    with TRANCO.open(encoding="utf-8") as f:
        rank = {ln.rstrip().split(",", 1)[1]: i + 1 for i, ln in enumerate(f)}
    df["rd"] = df.url.map(registrable_domain)
    df["host"] = df.url.map(host_of)
    df["shared"] = [is_untrusted_hosting(r) or is_user_content_host(h) for r, h in zip(df.rd, df.host)]
    rest = df.url.str.replace(r"^https?://", "", regex=True).str.rstrip("/")
    df["bare"] = ~rest.str.contains("/") & ~rest.str.contains(r"\?")
    df["apex"] = [h in (r, "www." + r) for h, r in zip(df.host, df.rd)]
    fixes: list[dict] = []

    def fix(row, new: str, rule: str, evidence: str) -> None:
        fixes.append({"dataset": "urls", "url": row.url, "old_label": row.label, "new_label": new,
                      "rule": rule, "evidence": evidence})

    # Rule 1
    cand = df[(df.label == "phishing") & df.rd.isin(rank) & ~df.shared & df.bare & df.apex].copy()
    cand = cand[[check_blocklist(u)["flag"] is not True for u in cand.url]]
    vt = lookup_batch(cand.url.tolist(), daily_limit=vt_requests, max_age_s=VT_MAX_AGE_S,
                      progress=lambda m: None)
    for row in cand.itertuples():
        v = verdict_of(vt[row.url])
        if vt[row.url]["status"] != "error" and v != "phishing":
            fix(row, "clean", "1_popular_bare_homepage",
                f"Tranco rank {rank[row.rd]}, not in current feeds, homepage only, VirusTotal {v}")

    # Rule 3
    phi = df[(df.label == "phishing") & (df.source == "phiusiil")]
    raw = hf.raw_lookups(phi.url.tolist(), offline=True)
    for row in phi.itertuples():
        reg = ((raw.get(row.host) or {}).get("rdap") or {}).get("registered") or 0
        if reg > PHIUSIIL_COLLECTED:
            day = dt.datetime.fromtimestamp(reg, dt.timezone.utc).date().isoformat()
            fix(row, "drop", "3_re_registered_domain", f"domain registered {day}, after PhiUSIIL collection")

    # Rule 2
    clean = df[df.label == "clean"]
    listed = clean[[check_blocklist(u)["flag"] is True for u in clean.url]]
    print(f"rule1 + rule3 fixes: {len(fixes)} | rule2 candidates: {len(listed)} (LLM on {min(len(listed), max_llm)})",
          flush=True)
    jobs = list(listed.itertuples())[:max_llm]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda r: _llm_verdict(r.url), jobs))
    for row, (verdict, note) in zip(jobs, results):
        if verdict == "phishing":
            fix(row, "phishing", "2_clean_in_feeds_llm_phishing", "listed in current feeds, " + note)
    return fixes


def hand_rules() -> list[dict]:
    """Rule 4: hand labels of hand_review.csv for every dataset."""
    rev = pd.read_csv(HAND_REVIEW, encoding="utf-8-sig", dtype=str).fillna("")
    rev["my_label"] = rev.my_label.str.strip()
    rev = rev[rev.my_label != ""]
    new_of = {"phishing": "phishing", "clean": "clean", "unsure": "drop", "not_phishing_but_illegal": "drop"}
    fixes: list[dict] = []

    def add(dataset, url, old, label, note):
        new = new_of[label]
        if new != old:
            fixes.append({"dataset": dataset, "url": url, "old_label": old, "new_label": new,
                          "rule": f"4_hand_review_{label}", "evidence": note.strip()[:200]})

    for r in rev.itertuples():
        add(HAND_SETS[r.set], r.url, r.dataset_label, r.my_label, r.my_note)

    # vi: same-site twins of hand-labelled homepages, every row of illegal-not-phishing domains
    vi = _original(_vi_frame(), "vi")
    vi["rd"] = vi.url.map(registrable_domain)
    hv = rev[rev.set == "vi"].copy()
    hv["rd"] = hv.url.map(registrable_domain)
    done = set(hv.url)
    illegal = set(hv[hv.my_label == "not_phishing_but_illegal"].rd)
    by_dom = dict(zip(hv[hv.action == "hand"].rd, hv[hv.action == "hand"].my_label))
    for r in vi[~vi.url.isin(done)].itertuples():
        if r.rd in illegal:
            add("vi", r.url, r.label, "not_phishing_but_illegal", "domain listed for illegal business, not phishing")
        elif r.rd in by_dom:
            add("vi", r.url, r.label, by_dom[r.rd], "same site as a hand-labelled homepage")
    return fixes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-llm", type=int, default=2500, help="cap on LLM-verified candidates (0 = none)")
    ap.add_argument("--vt-requests", type=int, default=480,
                    help="VirusTotal requests allowed (0 = cache only; free quota 500 / day)")
    args = ap.parse_args()

    present = {"urls": set(pd.read_csv(DATA, usecols=["url"]).url), "vi": set(_vi_frame().url),
               "phreshphish": set(pd.read_csv(PHRESHPHISH, usecols=["url"]).url)}
    # Earlier "drop" fixes of rows already removed from their file stay
    kept = load_fixes()
    gone = pd.Series([u not in present[d] for d, u in zip(kept.dataset, kept.url)],
                     index=kept.index, dtype=bool)
    kept = kept[(kept.new_label == "drop") & gone]
    fixes = kept.to_dict("records") + urls_rules(args.max_llm, args.vt_requests) + hand_rules()
    out = pd.DataFrame(fixes, columns=FIELDS).drop_duplicates(["dataset", "url"], keep="last")
    out.to_csv(FIXES, index=False, quoting=csv.QUOTE_MINIMAL)
    print(out.groupby(["dataset", "rule", "old_label", "new_label"]).size().to_string())
    print(f"-> {FIXES}")
    apply_eval_set_fixes()
    return 0


if __name__ == "__main__":
    sys.exit(main())
