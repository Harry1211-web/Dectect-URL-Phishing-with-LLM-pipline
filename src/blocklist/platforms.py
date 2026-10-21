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
from url_utils import file_version, registrable_domain  # noqa: E402

PLATFORMS_PATH = BLOCKLIST_DIR / "abused_platforms.txt"
MIN_TENANTS = 5
# A first path segment this short is a container, not a tenant: flowcode.com/p/<user>,
# shoutout.wix.com/so/<id>, linktr.ee/u/<user> -> the tenant is the second segment
MAX_CONTAINER_LEN = 4


def _path_tenant(url: str) -> str:
    """Tenant key of a URL on the platform's own host: '/user' or '/p/user'."""
    segs = [s for s in url.split("?", 1)[0].split("#", 1)[0].split("/")[1:] if s]
    if len(segs) >= 2 and len(segs[0]) <= MAX_CONTAINER_LEN:
        return f"/{segs[0]}/{segs[1]}"
    return "/" + (segs[0] if segs else "")


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
                    tenants[rd].add(_path_tenant(u))
    platforms = sorted(d for d, t in tenants.items() if len(t) >= min_tenants)
    tmp = PLATFORMS_PATH.with_name(PLATFORMS_PATH.name + ".tmp")  # readers never see a half file
    tmp.write_text("\n".join(platforms), encoding="utf-8")
    tmp.replace(PLATFORMS_PATH)
    file_version(PLATFORMS_PATH, check_every_s=0)  # this process reads the new list right away
    return len(platforms)


if __name__ == "__main__":
    print(build_abused_platforms(), "platforms ->", PLATFORMS_PATH)
