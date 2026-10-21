import gzip

import pytest

from blocklist import lookup
from blocklist.platforms import _path_tenant
from blocklist.updater import BLOCKLIST_DIR

pytestmark = pytest.mark.skipif(not (BLOCKLIST_DIR / "openphish.txt.gz").exists(),
                                reason="blocklist not built (run src/blocklist/updater.py)")


def _first_listed() -> str:
    with gzip.open(BLOCKLIST_DIR / "openphish.txt.gz", "rt", encoding="utf-8") as f:
        return f.readline().strip()


def test_exact_and_prefix_match():
    u = _first_listed()
    assert lookup.check_blocklist("https://" + u)["flag"] is True
    assert lookup.check_blocklist("http://www." + u + "/deeper/page.php")["flag"] is True


def test_popular_and_shared_hosts_are_not_blocked_whole():
    for u in ("https://www.google.com/", "https://vercel.app/", "https://github.io/"):
        assert lookup.check_blocklist(u)["flag"] is False


def test_redirect_to_listed_url_is_blocked():
    u = _first_listed()
    assert lookup.check_blocklist("https://www.google.com/url?q=https://" + u)["flag"] is True


def test_campaign_domain_blocks_new_subdomains():
    with gzip.open(BLOCKLIST_DIR / lookup.HOSTS_FILE, "rt", encoding="utf-8") as f:
        dom = next(ln.strip()[2:] for ln in f if ln.startswith("*."))
    assert lookup.check_blocklist(f"https://never-seen-before-123.{dom}/x")["flag"] is True


def test_bare_host_entry_only_blocks_through_host_rule(monkeypatch):
    # Bare-host entries: popular site, platform root, platform tenant, ordinary domain
    urls = frozenset({"amazon.com", "vercel.app", "bad.vercel.app", "x.jimdosite.com",
                      "ec2-1-2-3-4.compute-1.amazonaws.com", "evil.com", "kit.net/login"})
    monkeypatch.setattr(lookup, "_load", lambda: (urls, frozenset({"evil.com"})))
    monkeypatch.setattr(lookup, "_popular_domains", lambda: frozenset({"amazon.com", "jimdosite.com"}))
    for u in ("https://amazon.com/gp/cart", "https://www.amazon.com/", "https://amazon.com",
              "https://vercel.app/docs", "https://good.vercel.app/"):
        assert lookup.check_blocklist(u)["flag"] is False
    for u in ("https://evil.com/", "https://www.evil.com/x", "https://kit.net/login/a.php",
              "https://bad.vercel.app/login", "https://x.jimdosite.com/a",
              "http://ec2-1-2-3-4.compute-1.amazonaws.com/login"):
        assert lookup.check_blocklist(u)["flag"] is True
    assert lookup.check_blocklist("https://kit.net/other")["flag"] is False


def test_redirect_target_behind_decoys_or_nested_is_blocked(monkeypatch):
    """Decoy parameters before the real target, or a tracker inside a tracker, used to hide it
    (only the first three embedded URLs were checked, one level deep)."""
    monkeypatch.setattr(lookup, "_load", lambda: (frozenset({"phish.top/login"}), frozenset()))
    decoys = "&".join(f"d{i}=https://news{i}.example.org/a" for i in range(10))
    assert lookup.check_blocklist(f"https://www.google.com/url?{decoys}&q=https://phish.top/login")["flag"] is True
    inner = "https%3A%2F%2Ft.co%2Fr%3Fu%3Dhttps%253A%252F%252Fphish.top%252Flogin"
    assert lookup.check_blocklist(f"https://tracker.example/r?u={inner}")["flag"] is True
    assert lookup.check_blocklist(f"https://www.google.com/url?{decoys}")["flag"] is False


def test_embedded_url_extraction_is_bounded():
    import time

    from url_utils import MAX_EMBEDDED, all_embedded_urls
    url = "http://a.com/?" + "&".join(f"u{i}=http://x{i}.com/a" for i in range(200))
    t0 = time.perf_counter()
    assert len(all_embedded_urls(url)) == MAX_EMBEDDED
    assert time.perf_counter() - t0 < 1.0


def test_lookup_is_fast():
    lookup.check_blocklist("https://example.org/")
    assert lookup.check_blocklist("https://example.org/a")["latency_ms"] < 5


def test_path_tenant_skips_short_container_segment():
    assert _path_tenant("flowcode.com/p/abc?x=1") == "/p/abc"
    assert _path_tenant("linkpop.com/maily") == "/maily"
    assert _path_tenant("example.com/") == "/"


def test_virustotal_verdict_rule():
    from blocklist.virustotal import url_id, verdict_of
    found = lambda m, s=0: {"status": "found", "stats": {"malicious": m, "suspicious": s}}
    assert verdict_of(found(2)) == "phishing" and verdict_of(found(0)) == "clean"
    assert verdict_of(found(1)) == "unsure" and verdict_of(found(0, 1)) == "unsure"
    assert verdict_of({"status": "not_found", "stats": None}) == "unsure"
    assert "=" not in url_id("http://a.com/") and "/" not in url_id("http://a.com/?x=~~~")
