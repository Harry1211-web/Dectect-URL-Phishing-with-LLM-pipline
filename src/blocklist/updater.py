"""Layer A — refresh the phishing blocklist from public feeds (meant to run every hour).

Phishing pages live hours to days, so a monthly list misses most live attacks. Each feed has its
own minimum age (FEED_MAX_AGE_HOURS): an hourly run only downloads the feeds that are due, and the
large Tranco list is refreshed monthly.

Feeds: OpenPhish (community feed), PhishTank (anonymous dump; new API keys are no longer issued,
so it is often rate-limited and then skipped), Phishing.Database (GitHub, active links) as a
complement when PhishTank is unavailable. The Tranco top list is refreshed too (used to decide
when a host-level block is safe).

Run:   python src/blocklist/updater.py [--force]
Schedule hourly on Windows (example):
  schtasks /Create /SC HOURLY /TN PhishingBlocklist /TR "python <repo>\\src\\blocklist\\updater.py"
Output: data/blocklist/<source>.txt.gz (one normalized URL per line), blockable_hosts.txt.gz,
        meta.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_utils import normalize_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLOCKLIST_DIR = ROOT / "data" / "blocklist"
META_PATH = BLOCKLIST_DIR / "meta.json"
TRANCO_PATH = ROOT / "data" / "tranco_top1m.csv"
# A feed is downloaded again once its snapshot is older than this. PhishTank's anonymous dump is
# rate-limited (a failed download keeps the previous snapshot); Phishing.Database is a large file
FEED_MAX_AGE_HOURS = {"openphish": 1, "phishtank": 6, "phishing_database": 6}
TRANCO_MAX_AGE_DAYS = 30
UA = "Mozilla/5.0 (compatible; dacntt-phishing-blocklist/1.0)"

FEEDS = {
    "openphish": "https://openphish.com/feed.txt",
    "phishtank": "http://data.phishtank.com/data/online-valid.csv",
    "phishing_database": ("https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
                          "master/phishing-links-ACTIVE.txt"),
}
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def _get(url: str, retries: int = 3, timeout: float = 120) -> bytes | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
            if r.status_code == 200 and r.content:
                return r.content
            print(f"  {url} -> HTTP {r.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {url} -> {type(e).__name__}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return None


def _parse(source: str, body: bytes) -> list[str]:
    text = body.decode("utf-8", errors="ignore")
    if source == "phishtank":
        rows = csv.DictReader(io.StringIO(text))
        if not rows.fieldnames or "url" not in rows.fieldnames:
            return []  # rate-limit page instead of CSV
        return [r["url"] for r in rows]
    return [ln.strip() for ln in text.splitlines() if "://" in ln and not ln.startswith("#")]


def _age_hours(iso_time: str | None) -> float:
    if not iso_time:
        return float("inf")
    return (time.time() - datetime.fromisoformat(iso_time).timestamp()) / 3600


def due_feeds(sources: dict, force: bool = False) -> list[str]:
    """Feeds whose last download attempt is older than their FEED_MAX_AGE_HOURS."""
    return [name for name in FEEDS
            if force or _age_hours(sources.get(name, {}).get("attempted_at")
                                   or sources.get(name, {}).get("fetched_at"))
            >= FEED_MAX_AGE_HOURS[name]]


def update(force: bool = False) -> dict:
    """Download the feeds that are due; a failed feed keeps its previous snapshot."""
    old = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    sources = dict(old.get("sources", {}))
    due = due_feeds(sources, force)
    tranco_due = force or not TRANCO_PATH.exists() or \
        (time.time() - TRANCO_PATH.stat().st_mtime) / 86400 >= TRANCO_MAX_AGE_DAYS
    if not due and not tranco_due:
        print("Blocklist is fresh, nothing to do (use --force to refresh).", file=sys.stderr)
        return old
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)
    changed = False
    for name in due:
        body = _get(FEEDS[name])
        urls = sorted({normalize_url(u) for u in _parse(name, body)}) if body else []
        if not urls:
            prev = sources.get(name, {})
            sources[name] = {**prev, "attempted_at": datetime.now(timezone.utc).isoformat(),
                             "last_status": "failed, kept previous snapshot"
                             if prev.get("count") else "failed, no snapshot"}
            print(f"{name}: download failed", file=sys.stderr)
            continue
        with gzip.open(BLOCKLIST_DIR / f"{name}.txt.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(urls))
        now = datetime.now(timezone.utc).isoformat()
        sources[name] = {"count": len(urls), "fetched_at": now, "attempted_at": now, "last_status": "ok"}
        changed = True
        print(f"{name}: {len(urls)} URLs", file=sys.stderr)

    body = _get(TRANCO_URL) if tranco_due else None
    if body:
        changed = True
        with zipfile.ZipFile(io.BytesIO(body)) as z, z.open(z.namelist()[0]) as src, \
                TRANCO_PATH.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print("tranco: refreshed", file=sys.stderr)

    n_hosts, n_platforms = old.get("blockable_hosts"), old.get("abused_platforms")
    if changed:
        from blocklist.lookup import build_blockable_hosts
        from blocklist.platforms import build_abused_platforms
        n_platforms = build_abused_platforms()  # before hosts: platforms are never blocked whole
        print(f"abused platforms: {n_platforms}", file=sys.stderr)
        n_hosts = build_blockable_hosts()
        print(f"blockable hosts: {n_hosts}", file=sys.stderr)
    meta = {"updated_at": datetime.now(timezone.utc).isoformat() if changed else old.get("updated_at"),
            "sources": sources, "blockable_hosts": n_hosts, "abused_platforms": n_platforms}
    # Written last: running pipelines reload the lists when this file changes
    tmp = META_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(META_PATH)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download every feed and Tranco now")
    print(json.dumps(update(ap.parse_args().force), indent=2))
