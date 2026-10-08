"""Layer A — refresh the phishing blocklist from public feeds (meant to run once a month).

Feeds: OpenPhish (community feed), PhishTank (anonymous dump; new API keys are no longer issued,
so it is often rate-limited and then skipped), Phishing.Database (GitHub, active links) as a
complement when PhishTank is unavailable. The Tranco top list is refreshed too (used to decide
when a host-level block is safe).

Run:   python src/blocklist/updater.py [--force]
Schedule monthly on Windows (example):
  schtasks /Create /SC MONTHLY /D 1 /TN PhishingBlocklist /TR "python <repo>\\src\\blocklist\\updater.py"
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
REFRESH_DAYS = 30
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


def needs_refresh(max_age_days: float = REFRESH_DAYS) -> bool:
    """True when the blocklist was never built or is older than `max_age_days`."""
    if not META_PATH.exists():
        return True
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    age = time.time() - datetime.fromisoformat(meta["updated_at"]).timestamp()
    return age > max_age_days * 86400


def update(force: bool = False) -> dict:
    """Download every feed; a failed feed keeps its previous snapshot."""
    if not force and not needs_refresh():
        print("Blocklist is fresh, nothing to do (use --force to refresh).", file=sys.stderr)
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    sources = dict(old.get("sources", {}))
    for name, url in FEEDS.items():
        body = _get(url)
        urls = sorted({normalize_url(u) for u in _parse(name, body)}) if body else []
        if not urls:
            prev = sources.get(name, {})
            sources[name] = {**prev, "last_status": "failed, kept previous snapshot"
                             if prev.get("count") else "failed, no snapshot"}
            print(f"{name}: download failed", file=sys.stderr)
            continue
        with gzip.open(BLOCKLIST_DIR / f"{name}.txt.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(urls))
        sources[name] = {"count": len(urls), "fetched_at": datetime.now(timezone.utc).isoformat(),
                         "last_status": "ok"}
        print(f"{name}: {len(urls)} URLs", file=sys.stderr)

    body = _get(TRANCO_URL)
    if body:
        with zipfile.ZipFile(io.BytesIO(body)) as z, z.open(z.namelist()[0]) as src, \
                TRANCO_PATH.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print("tranco: refreshed", file=sys.stderr)

    from blocklist.lookup import build_blockable_hosts
    from blocklist.platforms import build_abused_platforms
    n_platforms = build_abused_platforms()  # before hosts: platforms are never blocked whole
    print(f"abused platforms: {n_platforms}", file=sys.stderr)
    n_hosts = build_blockable_hosts()
    print(f"blockable hosts: {n_hosts}", file=sys.stderr)
    meta = {"updated_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
            "blockable_hosts": n_hosts, "abused_platforms": n_platforms}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="refresh even if younger than 30 days")
    print(json.dumps(update(ap.parse_args().force), indent=2))
