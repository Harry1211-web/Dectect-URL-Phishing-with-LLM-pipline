"""Build the Layer D dev and held-out test sets from PhreshPhish pages (with their HTML).

Only pages that reach Layer D are kept: not blocked by A, flagged by B or C ("reaches_d").
Phishing pages that A+B+C label clean are kept too ("missed_by_abc"): they never reach D today,
but they show whether D would catch them if they were routed to it. Report the two apart.

- dev  (prompt tuning, run many times): drawn from dataset/external/phreshphish_urls.csv,
  whose hosts are already in the host cache; holds every A+B+C miss of that sample.
- test (held out, run once at the end): PhreshPhish URLs outside that sample and outside every
  registrable domain of dev, in natural proportions, so its rates estimate production.

Fewer but more varied pages: at most MAX_PER_DOMAIN URLs per registrable domain, at most
MAX_SAME_TEXT pages with the same visible text (phishing kits repeat) and at most
MAX_TARGET_SHARE of the phishing pages per impersonated brand.

Each source parquet has a single row group, so the HTML of a page means downloading the whole file:
the files are streamed one by one (~9 GB in all) and only the selected pages are kept.
Layer C looks the test hosts up live (DNS / TLS / RDAP), as in evaluate_external.py.

Run:    python src/llm_rag/phreshphish_sets.py [--dev-per-label 300] [--test-per-label 1000]
Output: dataset/external/layer_d_dev.csv, dataset/external/layer_d_test.csv (one row per page),
        data/phreshphish_html.parquet (sha256 -> raw HTML, re-downloadable, not committed)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_external import DATA as SAMPLE_CSV, HF_FILES, OUR_DATASET, SINCE, score_abc  # noqa: E402
from html_sanitize import sanitize_html  # noqa: E402
from url_utils import normalize_url, registrable_domain  # noqa: E402

ROOT = _HERE.parents[1]
OUT_DIR = ROOT / "dataset" / "external"
HTML_STORE = ROOT / "data" / "phreshphish_html.parquet"
SEED = 0
MAX_PER_DOMAIN = 2
MAX_SAME_TEXT = 2
MAX_TARGET_SHARE = 0.10
# Candidates drawn per page wanted: covers pages lost to dedup / empty HTML and, for the test
# set, URLs that never reach D (~57 % of clean and ~99 % of phishing URLs reach it)
OVERSAMPLE_DEV = 1.5
OVERSAMPLE_TEST = {"phishing": 1.6, "clean": 2.8}
OUT_COLUMNS = ["url", "label", "target", "date", "lang", "sha256", "stratum",
               "A", "B", "C", "score", "thr", "host_branch", "lookup_failed"]


def load_index() -> pd.DataFrame:
    """Light columns of the PhreshPhish test split (+ source file), like evaluate_external.build:
    collected since SINCE, without our training URLs, de-duplicated on the normalized URL."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    parts = []
    for path in sorted(fs.glob(HF_FILES)):
        t = pq.ParquetFile(path, filesystem=fs).read(
            columns=["sha256", "url", "label", "target", "date", "lang"])
        d = t.to_pandas()
        d["file"] = path
        parts.append(d[d.date >= SINCE])
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df.label.map({"phish": "phishing", "benign": "clean"})
    df["date"] = df.date.astype(str)
    df["norm"] = df.url.map(normalize_url)
    known = set(pd.read_csv(OUR_DATASET, usecols=["url"]).url.map(normalize_url))
    df = df[~df.norm.isin(known)].drop_duplicates("norm")
    df["domain"] = df.url.map(registrable_domain)
    print(f"PhreshPhish index: {len(df)} URLs", df.label.value_counts().to_dict(), flush=True)
    return df


def cap_per(d: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    """At most k rows per value of `col` (rows are taken in their current order)."""
    return d.groupby(col, sort=False, dropna=False).head(k)


def cap_targets(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """At most MAX_TARGET_SHARE of n phishing rows per impersonated brand (unknown target is free)."""
    known = d.target.notna() & (d.target.astype(str).str.strip() != "")
    rank = d[known].groupby("target", sort=False).cumcount()
    keep = ~known
    keep[rank.index] = rank < max(1, int(n * MAX_TARGET_SHARE))
    return d[keep]


def draw(d: pd.DataFrame, per_label: dict[str, int]) -> pd.DataFrame:
    """Shuffled draw of per_label[label] rows, domain- and brand-capped (phishing only)."""
    d = cap_per(d.sample(frac=1, random_state=SEED), "domain", MAX_PER_DOMAIN)
    out = []
    for label, n in per_label.items():
        g = d[d.label == label]
        if label == "phishing":
            g = cap_targets(g, n)
        out.append(g.head(n))
    return pd.concat(out)


def add_strata(d: pd.DataFrame) -> pd.DataFrame:
    """stratum = reaches_d | missed_by_abc; other rows (blocked by A, or clean pages that C passes)
    are dropped because Layer D never sees them."""
    reach = ~d.A & (d.B | d.C)
    missed = (d.label == "phishing") & ~d.A & ~d.B & ~d.C
    d = d[reach | missed].copy()
    d["stratum"] = (reach[reach | missed]).map({True: "reaches_d", False: "missed_by_abc"})
    return d


def dev_candidates(index: pd.DataFrame, per_label: int) -> pd.DataFrame:
    """Dev: the existing 20k external sample (hosts cached), every A+B+C miss + reaches_d draw."""
    sample = pd.read_csv(SAMPLE_CSV)
    sample["norm"] = sample.url.map(normalize_url)
    sample = sample[["norm"]].merge(index, on="norm")  # sha256, lang and source file of each URL
    d = add_strata(score_abc(sample))
    n = int(per_label * OVERSAMPLE_DEV)
    reach = draw(d[d.stratum == "reaches_d"], {"phishing": n, "clean": n})
    return pd.concat([d[d.stratum == "missed_by_abc"], reach])


def test_candidates(index: pd.DataFrame, dev: pd.DataFrame, per_label: int) -> pd.DataFrame:
    """Test: URLs never looked at (outside the 20k sample) on domains absent from dev."""
    seen = set(pd.read_csv(SAMPLE_CSV, usecols=["url"]).url.map(normalize_url))
    pool = index[~index.norm.isin(seen) & ~index.domain.isin(set(dev.domain))]
    d = draw(pool, {lab: int(per_label * f) for lab, f in OVERSAMPLE_TEST.items()})
    print(f"test: scoring {len(d)} candidate URLs with A / B / C (live host lookups)", flush=True)
    return add_strata(score_abc(d))


def fetch_html(rows: pd.DataFrame) -> dict[str, str]:
    """sha256 -> HTML for `rows`, streaming only the source files that hold missing pages.
    Pages already in HTML_STORE are not downloaded again."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    store = pd.read_parquet(HTML_STORE) if HTML_STORE.exists() else pd.DataFrame(
        columns=["sha256", "html"])
    have = dict(zip(store.sha256, store.html))
    todo = rows[~rows.sha256.isin(have)]
    fs = HfFileSystem()
    for i, (path, g) in enumerate(todo.groupby("file"), 1):
        want = set(g.sha256)
        t = pq.ParquetFile(path, filesystem=fs).read(columns=["sha256", "html"]).to_pandas()
        have.update(t[t.sha256.isin(want)].set_index("sha256").html.to_dict())
        print(f"  HTML {i}/{todo.file.nunique()}: {path.rsplit('/', 1)[-1]} -> {len(want)} pages",
              flush=True)
        del t
    if len(todo):
        HTML_STORE.parent.mkdir(parents=True, exist_ok=True)
        keep = set(rows.sha256) | set(store.sha256)
        pd.DataFrame({"sha256": list(have), "html": list(have.values())}).loc[
            lambda x: x.sha256.isin(keep)].to_parquet(HTML_STORE, compression="zstd", index=False)
    return have


def text_key(html: str | None, url: str) -> str | None:
    """Fingerprint of the visible text Layer D would read (digits and case ignored, so a kit
    deployed with another ticket number counts as the same page); None for an empty page."""
    if not html:
        return None
    clean = sanitize_html(html, url=url)
    text = re.sub(r"[\d\s]+", " ", f"{clean['title']} {clean['text'][:2500]}".lower()).strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest() if text else None


def finalize(d: pd.DataFrame, html: dict[str, str], per_label: int) -> pd.DataFrame:
    """Drop empty / duplicate pages, then keep per_label reaches_d pages per label and every
    missed_by_abc page."""
    d = d.copy()
    d["text_key"] = [text_key(html.get(s), u) for s, u in zip(d.sha256, d.url)]
    n0 = len(d)
    d = cap_per(d[d.text_key.notna()], ["label", "text_key"], MAX_SAME_TEXT)
    print(f"  {n0 - len(d)} pages dropped (empty HTML or repeated text)", flush=True)
    reach = d[d.stratum == "reaches_d"]
    parts = [d[d.stratum == "missed_by_abc"]]
    for lab in ("phishing", "clean"):
        parts.append(reach[reach.label == lab].head(per_label))
        if len(parts[-1]) < per_label:
            print(f"  warning: only {len(parts[-1])}/{per_label} {lab} pages reach D; "
                  "raise OVERSAMPLE", flush=True)
    return pd.concat(parts)[OUT_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-per-label", type=int, default=300,
                    help="reaches_d pages per label in dev (default 300)")
    ap.add_argument("--test-per-label", type=int, default=1000,
                    help="reaches_d pages per label in test (default 1000)")
    args = ap.parse_args(argv)

    index = load_index()
    dev = dev_candidates(index, args.dev_per_label)
    test = test_candidates(index, dev, args.test_per_label)
    print(f"HTML for {len(dev) + len(test)} candidate pages", flush=True)
    html = fetch_html(pd.concat([dev, test]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d, n in (("dev", dev, args.dev_per_label), ("test", test, args.test_per_label)):
        print(f"{name}:", flush=True)
        out = finalize(d, html, n)
        out.to_csv(OUT_DIR / f"layer_d_{name}.csv", index=False)
        print(f"  {len(out)} pages -> {OUT_DIR / f'layer_d_{name}.csv'}",
              out.groupby(["stratum", "label"]).size().to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
