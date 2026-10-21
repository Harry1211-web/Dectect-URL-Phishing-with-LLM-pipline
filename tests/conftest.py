"""Make `src/` importable and expose shared paths to the tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "llm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _no_virustotal(monkeypatch):
    """Tests never spend the VirusTotal quota: the lookup reports "unavailable"."""
    import pipeline
    monkeypatch.setattr(pipeline, "check_virustotal", lambda url, timeout_s=2.0: {
        "name": "virustotal", "flag": "unknown", "reason": "disabled in tests", "latency_ms": 0.0})


@pytest.fixture(autouse=True)
def _no_page_download_before_layer_d(monkeypatch):
    """The pipeline downloads the page before Layer D: an unreadable page unless a test sets one.
    (Tests of fetch itself import the module as `llm_client`, a separate module object.)"""
    from llm import llm_client
    monkeypatch.setattr(llm_client, "fetch", lambda url: ("", "", False, url))


@pytest.fixture(autouse=True)
def _no_page_download_in_layer_c(monkeypatch):
    """Layer C's DOM step would download pages: off unless a test turns it on."""
    from url_classifier import dom_features
    monkeypatch.setattr(dom_features, "is_active", lambda: False)
