"""Crawl clean URLs (with real paths/queries) from the homepages of Tranco-ranked domains.

Why: Tranco only lists bare domains. Training on bare clean domains vs. full phishing URLs would
teach the classifier "has a path => phishing". Here each homepage is fetched once and a few of its
same-site links are kept, giving realistic clean URLs.

Run:  python src/data_collection/crawl_clean.py --sites 40000 --workers 64
Output (append + resumable): dataset/urls/raw/crawled_clean.csv  (url, site, rank)
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection.sources import RAW_DIR, UA, tranco_domains  # noqa: E402
from url_utils import registrable_domain  # noqa: E402

OUT_PATH = RAW_DIR / "crawled_clean.csv"
MAX_BYTES = 1_500_000
_RE_HREF = re.compile(r"""<a\s[^>]*?href\s*=\s*["']([^"'<>\s]+)["']""", re.I)
# Skip static assets and non-page links; they are not what users click
_RE_SKIP = re.compile(r"\.(?:jpg|jpeg|png|gif|svg|webp|ico|css|js|pdf|zip|mp4|mp3|woff2?)(?:$|\?)", re.I)


def sample_sites(n: int, seed: int = 42) -> list[tuple[int, str]]:
    """Top 2k + every '.vn' domain in the top 500k (the system targets Vietnamese users) + a
    rank-stratified random sample of the rest, so famous and long-tail sites are both represented."""
    doms = tranco_domains(500_000)
    keep = set(range(2000)) | {i for i, d in enumerate(doms) if d.endswith(".vn")}
    rest = [i for i in range(2000, len(doms)) if i not in keep]
    keep |= set(random.Random(seed).sample(rest, max(0, min(n - len(keep), len(rest)))))
    return [(i + 1, doms[i]) for i in sorted(keep)]


def harvest(site: str, per_site: int, rng: random.Random) -> list[str]:
    """Fetch the homepage and return up to `per_site` distinct same-site page links (plus the homepage)."""
    for scheme in ("https", "http"):
        try:
            with requests.get(f"{scheme}://{site}/", timeout=(5, 8), stream=True,
                              headers={"User-Agent": UA, "Accept-Language": "en,vi;q=0.8"}) as r:
                if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                    continue
                body = r.raw.read(MAX_BYTES, decode_content=True).decode(r.encoding or "utf-8",
                                                                          errors="ignore")
                base = r.url
            break
        except (requests.RequestException, urllib3.exceptions.HTTPError, ValueError):
            continue
    else:
        return []

    site_rd = registrable_domain(base)
    links: dict[str, None] = {}
    for href in _RE_HREF.findall(body):
        if href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
            continue
        u = urljoin(base, href.replace("&amp;", "&")).split("#", 1)[0]
        parts = urlsplit(u)
        if parts.scheme not in ("http", "https") or _RE_SKIP.search(parts.path):
            continue
        if registrable_domain(u) != site_rd or len(u) > 500:
            continue
        if parts.path.strip("/") or parts.query:
            links[u] = None
    picked = rng.sample(list(links), min(per_site, len(links)))
    return [base] + picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=40000)
    ap.add_argument("--per-site", type=int, default=6)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    done: set[str] = set()
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8", newline="") as f:
            done = {r["site"] for r in csv.DictReader(f)}
    todo = [(rank, s) for rank, s in sample_sites(args.sites) if s not in done]
    print(f"{len(done)} sites already crawled, {len(todo)} to go", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_PATH.exists()
    lock = threading.Lock()
    n_urls = 0
    with OUT_PATH.open("a", encoding="utf-8", newline="") as f, \
            ThreadPoolExecutor(args.workers) as pool:
        w = csv.writer(f)
        if new_file:
            w.writerow(["url", "site", "rank"])
        futs = {pool.submit(harvest, s, args.per_site, random.Random(s)): (rank, s)
                for rank, s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rank, s = futs[fut]
            try:
                urls = fut.result()
            except Exception:
                urls = []
            with lock:
                # A row with empty url marks the site as visited so a resume skips it
                for u in urls or [""]:
                    w.writerow([u, s, rank])
                n_urls += len(urls)
            if i % 1000 == 0:
                f.flush()
                print(f"  {i}/{len(todo)} sites, {n_urls} urls", flush=True)
    print(f"done: {n_urls} new urls -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
