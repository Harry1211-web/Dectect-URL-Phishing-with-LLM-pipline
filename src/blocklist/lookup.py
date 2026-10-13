"""Layer A — instant block: look a URL up in the phishing blocklist.

Matching rules (normalized URLs, see url_utils.normalize_url):
  1. exact URL, or the URL without its query string;
  2. a listed URL is a path prefix of the URL ("evil.com/kit" blocks "evil.com/kit/login.php");
  3. whole host — only when the host is NOT shared or popular (free hosting, shorteners, Tranco
     top domains, official brand domains), so one phishing page on a compromised popular site or
     on vercel.app never blocks the whole site;
  4. whole registrable domain when CAMPAIGN_MIN_HOSTS..CAMPAIGN_MAX_HOSTS of its subdomains are
     listed (campaign domains rotating subdomains);
  5. rules 1-4 are also applied to destinations hidden in redirect parameters.
"""

from __future__ import annotations

import gzip
from collections import Counter
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blocklist.updater import BLOCKLIST_DIR, FEEDS, META_PATH, TRANCO_PATH  # noqa: E402
from contracts import RuleResult  # noqa: E402
from typosquatting.brands import all_official_domains  # noqa: E402
from typosquatting.paas_domains import is_untrusted_hosting  # noqa: E402
from url_utils import embedded_urls, host_of, normalize_url, registrable_domain  # noqa: E402

POPULAR_TOP_N = 100_000  # host-level blocking is disabled for these registrable domains
# A non-shared registrable domain with this many distinct listed hosts is a campaign domain
# rotating subdomains (a1.evil.com, a2.evil.com...): every subdomain of it is blocked
CAMPAIGN_MIN_HOSTS = 3
# More listed hosts than this usually means a hosting platform with many unrelated tenants,
# so only the listed hosts themselves are blocked
CAMPAIGN_MAX_HOSTS = 50


HOSTS_FILE = "blockable_hosts.txt.gz"


def _read_gz(path: Path) -> list[str]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load() -> tuple[frozenset[str], frozenset[str]]:
    """(listed normalized URLs, hosts eligible for host-level blocking), reloaded when the hourly
    updater rewrites meta.json (long-running pipelines pick up new feeds without a restart)."""
    try:
        stamp = META_PATH.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return _load_files(stamp)


@lru_cache(maxsize=1)
def _load_files(stamp: int) -> tuple[frozenset[str], frozenset[str]]:
    urls: set[str] = set()
    for name in FEEDS:
        urls.update(_read_gz(BLOCKLIST_DIR / f"{name}.txt.gz"))
    return frozenset(urls), frozenset(_read_gz(BLOCKLIST_DIR / HOSTS_FILE))


def build_blockable_hosts() -> int:
    """Precompute (at update time) the hosts that are safe to block as a whole."""
    _load_files.cache_clear()
    urls, _ = _load()
    hosts = {u.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0] for u in urls}
    keep = [h for h in hosts if h and not _is_shared_or_popular(h)]
    per_domain = Counter(registrable_domain(h) for h in keep)
    # "*.domain" entries = campaign domains blocked with all their subdomains
    keep = sorted(keep) + sorted(f"*.{d}" for d, n in per_domain.items() if CAMPAIGN_MIN_HOSTS <= n <= CAMPAIGN_MAX_HOSTS)
    with gzip.open(BLOCKLIST_DIR / HOSTS_FILE, "wt", encoding="utf-8") as f:
        f.write("\n".join(keep))
    _load_files.cache_clear()
    return len(keep)


@lru_cache(maxsize=1)
def _popular_domains() -> frozenset[str]:
    if not TRANCO_PATH.exists():
        return frozenset()
    out = []
    with TRANCO_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= POPULAR_TOP_N:
                break
            out.append(line.rstrip().split(",", 1)[-1].lower())
    return frozenset(out)


def _is_shared_or_popular(host: str) -> bool:
    rd = registrable_domain(host)
    return (is_untrusted_hosting(rd) or rd in _popular_domains()
            or rd in all_official_domains())


def reload() -> None:
    """Drop the in-memory cache after the updater refreshed the files."""
    _load_files.cache_clear()


def size() -> int:
    return len(_load()[0])


def check_blocklist(url: str) -> RuleResult:
    """Layer A entry point: flag=True when the URL is a known phishing URL."""
    t0 = time.perf_counter()
    urls, hosts = _load()

    def result(flag, reason) -> RuleResult:
        return RuleResult(name="blocklist", flag=flag, reason=reason,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 3))

    if not urls:
        return result("unknown", "No blocklist data yet (run src/blocklist/updater.py).")
    # The URL itself, then any destination hidden in it (open redirect / click tracker)
    for i, u in enumerate([url, *embedded_urls(url)[:3]]):
        why = _match(u, urls, hosts)
        if why:
            prefix = f"URL redirects to {u[:80]} — " if i else ""
            return result(True, prefix + why)
    return result(False, "Not in the blocklist.")


def _match(url: str, urls: frozenset[str], hosts: frozenset[str]) -> str | None:
    norm = normalize_url(url)
    candidates = {norm, norm.split("?", 1)[0]}
    path = norm.split("?", 1)[0]
    while "/" in path:  # every path prefix: a/b/c -> a/b -> a
        path = path.rsplit("/", 1)[0]
        candidates.add(path)
    if candidates & urls:
        return ("URL is in a known public phishing list "
                "(OpenPhish/PhishTank/Phishing.Database).")
    host = host_of(url)
    if host in hosts:
        return f"Domain {host} is in a known phishing list."
    if f"*.{registrable_domain(host)}" in hosts:
        return (f"Domain {registrable_domain(host)} belongs to a known phishing campaign "
                "(many of its subdomains are listed).")
    return None
