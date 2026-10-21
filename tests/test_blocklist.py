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
