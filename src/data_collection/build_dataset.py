"""Merge, clean and split the labelled URL dataset used to train/evaluate Layer C.

Sources
  phishing: Phishing.Database (active links, sampled), OpenPhish, PhishTank (when reachable),
            PhiUSIIL phishing URLs
  clean:    URLs crawled from Tranco-ranked homepages (crawl_clean.py), PhiUSIIL legit URLs
Cleaning
  - drop unparsable / over-long URLs, de-duplicate on the normalized URL;
  - drop URLs found with both labels;
  - drop "phishing" rows that are bare homepages of official brand domains or Tranco top-10k
    sites (feed noise; phishing on user-content hosts such as Google Sites is kept), and
    "clean" rows whose host is in the phishing set;
  - cap URLs per site (registrable domain, or full host on free hosting) so a few huge phishing
    campaigns or sites do not dominate.
Split: 70/15/15 train/val/test grouped by site, so one site never appears in two splits.
Audited label corrections (audit_labels.py -> label_fixes.csv) are applied at the end.

Run:    python src/data_collection/build_dataset.py
Output: dataset/urls/urls_labeled.csv (url,label,source,group,split) + dataset/urls/stats.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection import sources  # noqa: E402
from data_collection.crawl_clean import OUT_PATH as CRAWLED_PATH  # noqa: E402
from typosquatting.brands import all_official_domains  # noqa: E402
from typosquatting.paas_domains import is_untrusted_hosting, is_user_content_host  # noqa: E402
from url_utils import host_of, normalize_url, registrable_domain  # noqa: E402

OUT_DIR = sources.ROOT / "dataset" / "urls"
MAX_URL_LEN = 2000


def site_group(url: str) -> str:
    """Grouping key = one "site": eTLD+1, the full host on shared hosting (each subdomain is a
    tenant), host + first two path segments on user-content hosts (sites.google.com/view/<x>)."""
    host = host_of(url)
    if is_user_content_host(host):
        segs = [s for s in normalize_url(url).split("?", 1)[0].split("/")[1:] if s][:2]
        return "/".join([host, *segs])
    rd = registrable_domain(url)
    return host if is_untrusted_hosting(rd) else rd


def _split_of(group: str, seed: int) -> str:
    h = int(hashlib.md5(f"{seed}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def load_raw(phishing_database_limit: int) -> list[dict]:
    rows = []
    for name, fn in [("openphish", sources.openphish), ("phishtank", sources.phishtank),
                     ("phiusiil", sources.phiusiil)]:
        got = fn()
        print(f"  {name}: {len(got)}", file=sys.stderr)
        rows += got
    pdb = sources.phishing_database(limit=phishing_database_limit)
    print(f"  phishing_database (sampled): {len(pdb)}", file=sys.stderr)
    rows += pdb
    if CRAWLED_PATH.exists():
        with CRAWLED_PATH.open(encoding="utf-8", newline="") as f:
            crawled = [{"url": r["url"], "label": "clean", "source": "tranco_crawl"}
                       for r in csv.DictReader(f) if r["url"]]
        print(f"  tranco_crawl: {len(crawled)}", file=sys.stderr)
        rows += crawled
    return rows


def clean(rows: list[dict], max_per_site: int, seed: int) -> tuple[list[dict], Counter]:
    drops: Counter = Counter()
    top10k = set(sources.tranco_domains(10_000))
    official = all_official_domains()

    by_norm: dict[str, dict] = {}
    labels_seen: dict[str, set] = defaultdict(set)
    for r in rows:
        url = r["url"].strip()
        if not url or len(url) > MAX_URL_LEN or not host_of(url):
            drops["unparsable_or_too_long"] += 1
            continue
        norm = normalize_url(url)
        labels_seen[norm].add(r["label"])
        if norm in by_norm:
            drops["duplicate"] += 1
            continue
        by_norm[norm] = {**r, "url": url, "norm": norm}

    out = []
    phishing_hosts = {host_of(v["url"]) for v in by_norm.values() if v["label"] == "phishing"}
    for norm, r in by_norm.items():
        if len(labels_seen[norm]) > 1:
            drops["label_conflict"] += 1
            continue
        rd = registrable_domain(r["url"])
        if r["label"] == "phishing":
            # Phishing on user-content hosts of official domains (Google Sites/Docs, S3...) is
            # real and kept; a bare official or top-10k homepage is feed noise
            if is_user_content_host(host_of(r["url"])):
                pass
            elif rd in official and not norm.split("?", 1)[0].count("/"):
                drops["phishing_bare_official_homepage"] += 1
                continue
            elif rd in top10k and not is_untrusted_hosting(rd) and "/" not in norm.split("?", 1)[0]:
                drops["phishing_bare_top10k_homepage"] += 1
                continue
        elif host_of(r["url"]) in phishing_hosts:
            drops["clean_host_in_phishing_set"] += 1
            continue
        out.append({"url": r["url"], "label": r["label"], "source": r["source"],
                    "group": site_group(r["url"])})

    # Cap rows per (site, label)
    rng = random.Random(seed)
    rng.shuffle(out)
    per_site: Counter = Counter()
    capped = []
    for r in out:
        key = (r["group"], r["label"])
        if per_site[key] >= max_per_site:
            drops["over_site_cap"] += 1
            continue
        per_site[key] += 1
        capped.append(r)
    return capped, drops


FIXES_PATH = OUT_DIR / "label_fixes.csv"


def apply_label_fixes(rows: list[dict]) -> int:
    """Apply audited label corrections of this dataset (audit_labels.py) in place: new label, or
    "drop" removes the row. Returns how many rows changed or were dropped."""
    if not FIXES_PATH.exists():
        return 0
    with FIXES_PATH.open(encoding="utf-8", newline="") as f:
        fixes = {r["url"]: r["new_label"] for r in csv.DictReader(f) if r.get("dataset", "urls") == "urls"}
    n, kept = 0, []
    for r in rows:
        new = fixes.get(r["url"])
        if new == "drop":
            n += 1
            continue
        if new and new != r["label"]:
            r["label"], n = new, n + 1
        kept.append(r)
    rows[:] = kept
    return n


def _write(rows: list[dict]) -> None:
    with (OUT_DIR / "urls_labeled.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["url", "label", "source", "group", "split"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["label"], r["url"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phishing-database-limit", type=int, default=200_000)
    ap.add_argument("--max-per-site", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-fixes-only", action="store_true",
                    help="only apply label_fixes.csv to the existing urls_labeled.csv")
    args = ap.parse_args()
    if args.apply_fixes_only:
        with (OUT_DIR / "urls_labeled.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        n = apply_label_fixes(rows)
        _write(rows)
        print(f"{n} labels corrected or rows dropped")
        return 0

    print("Loading sources...", file=sys.stderr)
    raw = load_raw(args.phishing_database_limit)
    rows, drops = clean(raw, args.max_per_site, args.seed)
    for r in rows:
        r["split"] = _split_of(r["group"], args.seed)
    drops["labels_corrected_by_audit"] = apply_label_fixes(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(rows)

    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_raw": len(raw), "n_final": len(rows),
        "by_label": Counter(r["label"] for r in rows),
        "by_source": Counter(f"{r['source']}:{r['label']}" for r in rows),
        "by_split": {s: Counter(r["label"] for r in rows if r["split"] == s)
                     for s in ("train", "val", "test")},
        "n_sites": len({r["group"] for r in rows}),
        "dropped": drops,
        "params": vars(args),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
