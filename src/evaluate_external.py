"""Evaluate Layers A, B, C alone and A+B+C together on an external dataset never used in training.

Dataset: PhreshPhish (OpenText, arXiv 2507.10854, CC BY 4.0) — phishing and benign pages captured
by a real browser, cleaned with heuristics + human annotation. Only the test split, only URLs
collected from SINCE on, without any URL already in dataset/urls/urls_labeled.csv; at most
MAX_PER_HOST URLs per host so a few big sites do not dominate. HTML is not downloaded.
Results are also given on "unseen domains": registrable domains absent from our whole dataset.

Layer C looks hosts up live (DNS / TLS / RDAP, stored in the host cache); a host whose lookup
still fails after a retry is suspicious, as in production. Phishing pages of 2025
are mostly taken down by now, so they are scored by the lexical branch, while at click time a
live phishing host would be scored by the host branch.

Run:    python src/evaluate_external.py --build   (download + sample, once)
        python src/evaluate_external.py
Output: dataset/external/phreshphish_urls.csv, reports/external/metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import host_features as hf, predict  # noqa: E402
from url_utils import host_of, normalize_url, registrable_domain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "external" / "phreshphish_urls.csv"
OUT = ROOT / "reports" / "external" / "metrics.json"
OUR_DATASET = ROOT / "dataset" / "urls" / "urls_labeled.csv"
HF_FILES = "datasets/phreshphish/phreshphish/data/test-*.parquet"
SINCE = dt.date(2025, 7, 1)
PER_LABEL = 10_000
MAX_PER_HOST = 3
SEED = 0


def build() -> int:
    """Download the light columns of the PhreshPhish test split and write the sample."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    parts = []
    for path in sorted(fs.glob(HF_FILES)):
        t = pq.ParquetFile(path, filesystem=fs).read(columns=["url", "label", "target", "date"])
        d = t.to_pandas()
        parts.append(d[d.date >= SINCE])
        print(f"{path.rsplit('/', 1)[-1]}: {len(d)} rows, {len(parts[-1])} since {SINCE}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df.label.map({"phish": "phishing", "benign": "clean"})
    df["norm"] = df.url.map(normalize_url)
    known = set(pd.read_csv(OUR_DATASET, usecols=["url"]).url.map(normalize_url))
    n0 = len(df)
    df = df[~df.norm.isin(known)].drop_duplicates("norm")
    print(f"{n0} rows since {SINCE}; {len(df)} after removing URLs of our dataset and duplicates",
          flush=True)
    df["host"] = df.url.map(host_of)
    df = df.sample(frac=1, random_state=SEED).groupby(["label", "host"]).head(MAX_PER_HOST)
    out = pd.concat([g.head(PER_LABEL) for _, g in df.groupby("label")])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    out[["url", "label", "target", "date"]].sort_values(["label", "date"]).to_csv(DATA, index=False)
    print(out.label.value_counts().to_dict(), "->", DATA)
    from data_collection.audit_labels import apply_eval_set_fixes
    apply_eval_set_fixes(("phreshphish",))  # audited labels (label_fixes.csv)
    return 0


def collect_hosts(hosts: list[str], workers: int = 16) -> set[str]:
    """Live DNS / TLS / RDAP lookups into the host cache; returns the hosts still failing after
    one retry (resolver errors such as SERVFAIL are never cached as 'dead')."""
    from concurrent.futures import ThreadPoolExecutor

    def ok(h: str) -> bool:
        try:
            hf.lookup_host(h)
            return True
        except Exception:  # resolver / network error: retried, then treated as a failed lookup
            return False

    todo = hosts
    for attempt in (1, 2):
        with ThreadPoolExecutor(workers) as pool:
            todo = [h for h, good in zip(todo, pool.map(ok, todo)) if not good]
        print(f"  host lookups pass {attempt}: {len(hosts)} hosts, {len(todo)} failing", flush=True)
    return set(todo)


def score_abc(d: pd.DataFrame) -> pd.DataFrame:
    """Copy of `d` with the decisions of Layers A, B and C for every URL (+ C's score / threshold).
    Hosts are looked up live once, then C reads the host cache."""
    d = d.copy()
    urls = d.url.tolist()
    d["A"] = [check_blocklist(u)["flag"] is True for u in urls]
    d["B"] = [check_typosquatting_deep(u)["flag"] is True for u in urls]
    # A model without network lookups (config host_lookups = "none") has nothing to look up
    failed = collect_hosts(list(dict.fromkeys(map(host_of, urls)))) if predict.uses_network() else set()
    score, thr, used = zip(*[predict.score_urls(urls[i:i + 4096], host_mode="cache")
                             for i in range(0, len(urls), 4096)])
    d["score"], d["thr"], d["host_branch"] = np.concatenate(score), np.concatenate(thr), np.concatenate(used)
    # As in production (classify_url): a host that cannot be looked up is suspicious, never clean
    d["lookup_failed"] = d.url.map(host_of).isin(failed)
    d["C"] = (d.score >= d.thr) | d.lookup_failed
    return d


def layer_rates(flag: np.ndarray, ph: np.ndarray) -> dict:
    """One layer alone: share of phishing it flags and share of clean URLs it flags wrongly."""
    return {"phishing_flagged": int((flag & ph).sum()),
            "phishing_flag_rate": round(float((flag & ph).sum() / max(1, ph.sum())), 4),
            "clean_flagged": int((flag & ~ph).sum()),
            "clean_flag_rate": round(float((flag & ~ph).sum() / max(1, (~ph).sum())), 4)}


def pipeline(a: np.ndarray, b: np.ndarray, c: np.ndarray, ph: np.ndarray) -> dict:
    """A blocks, B or C send to Layer D, clean only when all three let the URL pass."""
    clean = ~a & ~b & ~c
    return {"phishing_blocked_by_A": int((a & ph).sum()),
            "phishing_sent_to_llm": int((~a & ~clean & ph).sum()),
            "phishing_labelled_clean": int((clean & ph).sum()),
            "phishing_miss_rate": round(float((clean & ph).sum() / max(1, ph.sum())), 5),
            "clean_blocked_by_A": int((a & ~ph).sum()),
            "clean_recognized_rate": round(float((clean & ~ph).sum() / max(1, (~ph).sum())), 4),
            "clean_sent_to_llm_rate": round(float((~a & ~clean & ~ph).sum() / max(1, (~ph).sum())), 4)}


def report_for(d: pd.DataFrame) -> dict:
    ph = (d.label == "phishing").to_numpy()
    a, b, c = d.A.to_numpy(), d.B.to_numpy(), d.C.to_numpy()
    return {"n_phishing": int(ph.sum()), "n_clean": int((~ph).sum()),
            "layers_alone": {"A": layer_rates(a, ph), "B": layer_rates(b, ph), "C": layer_rates(c, ph)},
            "A+B+C": pipeline(a, b, c, ph),
            "B+C (without A)": pipeline(np.zeros(len(d), bool), b, c, ph)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="download and sample the dataset")
    if ap.parse_args(argv).build:
        return build()

    d = pd.read_csv(DATA)
    print(f"external URLs: {len(d)}", d.label.value_counts().to_dict(), flush=True)
    d = score_abc(d)

    ours = set(pd.read_csv(OUR_DATASET, usecols=["url"]).url.map(registrable_domain))
    d["unseen_domain"] = ~d.url.map(registrable_domain).isin(ours)
    report = {"dataset": "PhreshPhish test split (huggingface.co/datasets/phreshphish/phreshphish)",
              "collected_since": SINCE.isoformat(), "max_urls_per_host": MAX_PER_HOST,
              "host_branch_used": {lab: int(g.host_branch.sum()) for lab, g in d.groupby("label")},
              "host_lookup_failed": {lab: int(g.lookup_failed.sum()) for lab, g in d.groupby("label")},
              "all": report_for(d), "unseen_domains": report_for(d[d.unseen_domain])}

    # Phishing labelled clean, with the segment that decided them
    miss = d[(d.label == "phishing") & ~d.B & ~d.C].copy()
    raw = hf.raw_lookups(miss.url.tolist(), offline=True)
    miss["segment"] = [hf.segment_of(u, raw[host_of(u)], bool(h))
                       for u, h in zip(miss.url, miss.host_branch)]
    report["missed_phishing"] = miss.sort_values("score")[
        ["url", "target", "date", "A", "unseen_domain", "segment", "score", "thr"]].to_dict("records")
    for k in ("all", "unseen_domains"):
        print(k, json.dumps(report[k]["A+B+C"]), "| without A:",
              json.dumps(report[k]["B+C (without A)"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
