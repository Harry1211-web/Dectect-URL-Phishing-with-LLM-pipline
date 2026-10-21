"""Layer A — VirusTotal URL reputation (API v3, key VT_API_KEY in .env).

Only looks up URLs VirusTotal already knows; never submits a URL for scanning (a submission makes
the URL public on VirusTotal). Results are cached in data/vt_cache.sqlite.
Free API quota: 4 requests / minute, 500 / day. Live lookups use a short timeout and treat
quota errors as "unknown", so Layer A never blocks or waits on VirusTotal.

Verdict from last_analysis_stats: malicious engines >= VT_PHISHING_MIN -> phishing,
0 malicious and 0 suspicious -> clean, otherwise unsure.
"""

from __future__ import annotations

import base64
import json
import random
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import RuleResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "vt_cache.sqlite"
API = "https://www.virustotal.com/api/v3/urls/"
VT_PHISHING_MIN = 2           # malicious engines needed to call a URL phishing
FOUND_TTL_S = 24 * 3600       # cached verdict of a known URL
NOT_FOUND_TTL_S = 6 * 3600    # VirusTotal may learn the URL later
LIVE_TIMEOUT_S = 2.0
FREE_INTERVAL_S = 15.5        # 4 requests / minute

_lock = threading.Lock()


def api_key() -> str | None:
    return dotenv_values(ROOT / ".env").get("VT_API_KEY") or None


def url_id(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


@contextmanager
def _db():
    """Connection that commits on success and is always closed (the API opens one per lookup)."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, timeout=10)
    try:
        with con:
            con.execute("CREATE TABLE IF NOT EXISTS vt (url TEXT PRIMARY KEY, t REAL, status TEXT, stats TEXT)")
            yield con
    finally:
        con.close()


def _cached(url: str) -> dict | None:
    with _lock, _db() as con:
        row = con.execute("SELECT t, status, stats FROM vt WHERE url = ?", (url,)).fetchone()
    if not row:
        return None
    t, status, stats = row
    if time.time() - t > (FOUND_TTL_S if status == "found" else NOT_FOUND_TTL_S):
        return None
    return {"status": status, "stats": json.loads(stats) if stats else None, "cached": True}


def _store(url: str, status: str, stats: dict | None) -> None:
    with _lock, _db() as con:
        con.execute("INSERT OR REPLACE INTO vt VALUES (?, ?, ?, ?)",
                    (url, time.time(), status, json.dumps(stats) if stats else None))
        if random.randrange(200) == 0:  # now and then: expired rows are never read again
            con.execute("DELETE FROM vt WHERE t < ?", (time.time() - max(FOUND_TTL_S, NOT_FOUND_TTL_S),))


def lookup(url: str, timeout_s: float = LIVE_TIMEOUT_S, use_cache: bool = True) -> dict:
    """{status: found | not_found | error, stats, cached}. Errors (quota, network) are not cached."""
    if use_cache and (hit := _cached(url)):
        return hit
    key = api_key()
    if not key:
        return {"status": "error", "stats": None, "error": "VT_API_KEY missing", "cached": False}
    try:
        r = requests.get(API + url_id(url), headers={"x-apikey": key}, timeout=timeout_s)
    except requests.RequestException as e:
        return {"status": "error", "stats": None, "error": type(e).__name__, "cached": False}
    if r.status_code == 404:
        _store(url, "not_found", None)
        return {"status": "not_found", "stats": None, "cached": False}
    if r.status_code != 200:
        return {"status": "error", "stats": None, "error": f"HTTP {r.status_code}", "cached": False}
    try:
        attrs = r.json().get("data", {}).get("attributes", {})
    except (ValueError, AttributeError):  # proxy / captive-portal page instead of JSON
        return {"status": "error", "stats": None, "error": "invalid JSON", "cached": False}
    stats = dict(attrs.get("last_analysis_stats") or {})
    stats["last_analysis_date"] = attrs.get("last_analysis_date")
    _store(url, "found", stats)
    return {"status": "found", "stats": stats, "cached": False}


def verdict_of(res: dict) -> str:
    """phishing / clean / unsure from a lookup result."""
    s = res.get("stats") or {}
    if res.get("status") != "found":
        return "unsure"
    if s.get("malicious", 0) >= VT_PHISHING_MIN:
        return "phishing"
    if s.get("malicious", 0) == 0 and s.get("suspicious", 0) == 0:
        return "clean"
    return "unsure"


def check_virustotal(url: str, timeout_s: float = LIVE_TIMEOUT_S) -> RuleResult:
    """Layer A step after the blocklists: flag=True only when VirusTotal engines agree it is malicious."""
    t0 = time.perf_counter()
    res = lookup(url, timeout_s=timeout_s)
    v = verdict_of(res)
    s = res.get("stats") or {}
    if res["status"] == "error":
        flag, reason = "unknown", f"VirusTotal unavailable ({res.get('error')})."
    elif res["status"] == "not_found":
        flag, reason = False, "VirusTotal has no report for this URL."
    else:
        flag = v == "phishing"
        reason = (f"VirusTotal: {s.get('malicious', 0)} engines malicious, "
                  f"{s.get('suspicious', 0)} suspicious.")
    return RuleResult(name="virustotal", flag=flag, reason=reason,
                      latency_ms=round((time.perf_counter() - t0) * 1000, 3))


def lookup_batch(urls: list[str], interval_s: float = FREE_INTERVAL_S, daily_limit: int = 500,
                 progress=print) -> dict[str, dict]:
    """Offline lookups within the free quota (cache hits cost no request)."""
    out, sent = {}, 0
    for i, u in enumerate(urls):
        hit = _cached(u)
        if hit:
            out[u] = hit
            continue
        if sent >= daily_limit:
            out[u] = {"status": "error", "stats": None, "error": "daily limit", "cached": False}
            continue
        for attempt in range(3):
            res = lookup(u, timeout_s=20, use_cache=False)
            sent += 1
            if res.get("error") != "HTTP 429":
                break
            time.sleep(60)
        out[u] = res
        progress(f"{i + 1}/{len(urls)} {res['status']} {verdict_of(res)}")
        time.sleep(interval_s)
    return out
