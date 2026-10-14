"""Layer A updater: per-feed refresh ages, failed downloads, reload of running lookups (no network)."""

import gzip
import json
import os
from datetime import datetime, timedelta, timezone

from blocklist import lookup, updater


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def test_due_feeds_follow_each_feed_age():
    sources = {"openphish": {"attempted_at": _ago(2)},        # max age 1 h -> due
               "phishtank": {"attempted_at": _ago(2)},        # max age 6 h -> not due
               "phishing_database": {"fetched_at": _ago(7)}}  # no attempt recorded -> fetched_at
    assert updater.due_feeds(sources) == ["openphish", "phishing_database"]
    assert updater.due_feeds({}) == list(updater.FEEDS)       # never downloaded
    assert updater.due_feeds({k: {"attempted_at": _ago(0)} for k in updater.FEEDS}) == []
    assert updater.due_feeds({k: {"attempted_at": _ago(0)} for k in updater.FEEDS},
                             force=True) == list(updater.FEEDS)


def test_failed_download_keeps_previous_snapshot(tmp_path, monkeypatch):
    meta_path = tmp_path / "meta.json"
    tranco = tmp_path / "tranco.csv"
    tranco.write_text("1,example.com\n")                       # fresh: not re-downloaded
    for name, value in (("BLOCKLIST_DIR", tmp_path), ("META_PATH", meta_path), ("TRANCO_PATH", tranco)):
        monkeypatch.setattr(updater, name, value)
    meta_path.write_text(json.dumps({"sources": {
        "openphish": {"count": 1, "attempted_at": _ago(3)},
        "phishtank": {"count": 5, "attempted_at": _ago(7)},
        "phishing_database": {"count": 9, "attempted_at": _ago(0)}}}))
    downloads = []

    def fake_get(url, *a, **k):
        downloads.append(url)
        return b"https://evil.example/login\n" if url == updater.FEEDS["openphish"] else None

    monkeypatch.setattr(updater, "_get", fake_get)
    monkeypatch.setattr(lookup, "build_blockable_hosts", lambda: 1)
    monkeypatch.setattr("blocklist.platforms.build_abused_platforms", lambda: 0)
    meta = updater.update()
    assert downloads == [updater.FEEDS["openphish"], updater.FEEDS["phishtank"]]  # only due feeds
    assert meta["sources"]["openphish"]["count"] == 1 and meta["sources"]["openphish"]["last_status"] == "ok"
    pt = meta["sources"]["phishtank"]
    assert pt["count"] == 5 and pt["last_status"] == "failed, kept previous snapshot"
    with gzip.open(tmp_path / "openphish.txt.gz", "rt", encoding="utf-8") as f:
        assert f.read() == "evil.example/login"
    downloads.clear()
    updater.update()        # right after: nothing due, the failed feed waits its own age too
    assert downloads == []


def test_lookup_reloads_when_the_updater_rewrites_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "BLOCKLIST_DIR", tmp_path)
    monkeypatch.setattr(lookup, "META_PATH", tmp_path / "meta.json")
    lookup._load_files.cache_clear()

    def write(urls: list[str], stamp: int) -> None:
        with gzip.open(tmp_path / "openphish.txt.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(urls))
        (tmp_path / "meta.json").write_text("{}")
        os.utime(tmp_path / "meta.json", ns=(stamp, stamp))

    write(["a.example/x"], 1_000_000_000)
    assert lookup._load()[0] == {"a.example/x"}
    write(["a.example/x", "b.example/y"], 2_000_000_000)
    assert lookup._load()[0] == {"a.example/x", "b.example/y"}
    lookup._load_files.cache_clear()
