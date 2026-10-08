"""Derive "abused platforms" from the phishing feeds.

A popular registrable domain (in the Tranco list) on which phishing was reported for many
different tenants — distinct subdomains (x.webflow.io, y.webflow.io) or distinct first path
segments (flowcode.com/page/a, linkpop.com/b) — is a platform where anyone can publish pages.
A page there must not inherit the platform's popularity or domain age, and the platform must
never be blocked as a whole. The hand-made lists in typosquatting/paas_domains.py cannot keep up
with new site builders, link-in-bio and QR services; this list is rebuilt with every update.

Output: data/blocklist/abused_platforms.txt (one registrable domain per line)
"""

from __future__ import annotations

import gzip
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blocklist.updater import BLOCKLIST_DIR, FEEDS, TRANCO_PATH  # noqa: E402
from url_utils import registrable_domain  # noqa: E402

PLATFORMS_PATH = BLOCKLIST_DIR / "abused_platforms.txt"
MIN_TENANTS = 5


def build_abused_platforms(exclude_urls: set[str] | None = None,
                           min_tenants: int = MIN_TENANTS) -> int:
    """Write the platform list; `exclude_urls` (normalized URLs) keeps evaluation URLs out of it."""
    from typosquatting.paas_domains import USER_CONTENT_HOSTS
    exclude_urls = exclude_urls or set()
    popular = set()
    if TRANCO_PATH.exists():
        with TRANCO_PATH.open(encoding="utf-8") as f:
            popular = {ln.rstrip().split(",", 1)[-1] for ln in f}
    tenants: dict[str, set[str]] = defaultdict(set)
    for name in FEEDS:
        p = BLOCKLIST_DIR / f"{name}.txt.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for u in f:
                u = u.strip()
                if not u or u in exclude_urls:
                    continue
                host = u.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0]
                rd = registrable_domain(host)
                if rd not in popular or host in USER_CONTENT_HOSTS:
                    continue
                if host not in (rd, "www." + rd):
                    tenants[rd].add(host)                   # subdomain tenant
                else:
                    segs = u.split("?", 1)[0].split("/")
                    tenants[rd].add("/" + (segs[1] if len(segs) > 1 else ""))  # path tenant
    platforms = sorted(d for d, t in tenants.items() if len(t) >= min_tenants)
    PLATFORMS_PATH.write_text("\n".join(platforms), encoding="utf-8")
    return len(platforms)


if __name__ == "__main__":
    print(build_abused_platforms(), "platforms ->", PLATFORMS_PATH)
