"""Downloaders for public phishing feeds and clean-domain lists.

Each function returns a list of dicts {url, label, source} where label is 'phishing' or 'clean'.
Raw downloads are cached under dataset/urls/raw/ so a rebuild does not hit the network again.
"""

from __future__ import annotations

import csv
import io
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "urls" / "raw"
UA = "Mozilla/5.0 (compatible; dacntt-phishing-research/1.0)"

OPENPHISH_URL = "https://openphish.com/feed.txt"
# Anonymous PhishTank dump (new API keys are no longer issued); heavily rate-limited, so retried
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
PHISHING_DATABASE_URL = ("https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
                         "master/phishing-links-ACTIVE.txt")
PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def _download(url: str, dest: Path, max_age_days: float = 7, retries: int = 3,
              timeout: float = 120) -> Path | None:
    """Download `url` to `dest` unless a fresh copy exists; returns None when every retry fails."""
    if dest.exists() and time.time() - dest.stat().st_mtime < max_age_days * 86400:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return dest
            print(f"  {url} -> HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"  {url} -> {type(e).__name__}")
        time.sleep(5 * (attempt + 1))
    return dest if dest.exists() else None  # stale cache is better than nothing


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip() and not ln.startswith("#")]


def openphish() -> list[dict]:
    p = _download(OPENPHISH_URL, RAW_DIR / "openphish_feed.txt", max_age_days=1)
    return [{"url": u, "label": "phishing", "source": "openphish"} for u in _lines(p)] if p else []


def phishtank() -> list[dict]:
    p = _download(PHISHTANK_URL, RAW_DIR / "phishtank_online_valid.csv", max_age_days=1)
    if not p:
        return []
    with p.open(encoding="utf-8", errors="ignore", newline="") as f:
        rows = csv.DictReader(f)
        if not rows.fieldnames or "url" not in rows.fieldnames:
            return []  # rate-limit page saved instead of the CSV
        return [{"url": r["url"], "label": "phishing", "source": "phishtank"} for r in rows]


def phishing_database(limit: int | None = None, seed: int = 42) -> list[dict]:
    p = _download(PHISHING_DATABASE_URL, RAW_DIR / "phishing_database_active.txt", max_age_days=1)
    if not p:
        return []
    urls = [u for u in _lines(p) if "://" in u]
    if limit and len(urls) > limit:
        import random
        urls = random.Random(seed).sample(urls, limit)
    return [{"url": u, "label": "phishing", "source": "phishing_database"} for u in urls]


def phiusiil() -> list[dict]:
    """PhiUSIIL (UCI, 2024): label 1 = legitimate, 0 = phishing."""
    p = _download(PHIUSIIL_URL, RAW_DIR / "phiusiil.zip", max_age_days=3650)
    if not p:
        return []
    out = []
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="ignore")):
                out.append({"url": r["URL"],
                            "label": "clean" if r["label"].strip() == "1" else "phishing",
                            "source": "phiusiil"})
    return out


def tranco_domains(top_n: int = 1_000_000) -> list[str]:
    """Tranco top-1M domains in rank order (cached; refreshed monthly)."""
    p = _download(TRANCO_URL, RAW_DIR / "tranco_top1m.csv.zip", max_age_days=30)
    if not p:
        return []
    with zipfile.ZipFile(p) as z:
        with z.open(z.namelist()[0]) as f:
            rows = io.TextIOWrapper(f, encoding="utf-8").read().splitlines()
    return [r.split(",", 1)[1].strip().lower() for r in rows[:top_n] if "," in r]
