"""HTTP API (src/api.py): authentication, input validation, rate limit. The pipeline is stubbed,
so no network lookup and no paid call is made."""

import pytest
from fastapi.testclient import TestClient

import api
import monitoring

KEY = "test-key"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", KEY)
    monkeypatch.setattr(api, "ALLOW_NO_AUTH", False)
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 100)
    monkeypatch.setattr(api, "EXPOSE_LAYERS", False)
    monkeypatch.setattr(api, "_hits", {})
    monkeypatch.setattr(monitoring, "MONITORING_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "SECRET_PATH", tmp_path / "k")
    seen = []

    def fake_analyze(url, **kw):
        seen.append(url)
        return {"url": url, "verdict": "suspicious", "decided_by": "C_classifier",
                "explanation": "", "layers": {}, "latency_ms": 1.0}

    monkeypatch.setattr(api, "analyze_url", fake_analyze)
    c = TestClient(api.app)
    c.seen = seen
    return c


def post(c, path, body, key=KEY):
    return c.post(path, json=body, headers={"X-API-Key": key} if key else {})


def test_key_is_required(client, monkeypatch):
    assert post(client, "/analyze", {"url": "example.com"}, key=None).status_code == 401
    assert post(client, "/analyze", {"url": "example.com"}, key="wrong").status_code == 401
    assert post(client, "/analyze", {"url": "example.com"}).status_code == 200
    # Fail closed: a server without API_KEY refuses every request unless ALLOW_NO_AUTH=1
    monkeypatch.setattr(api, "API_KEY", "")
    assert post(client, "/analyze", {"url": "example.com"}, key=None).status_code == 503
    monkeypatch.setattr(api, "ALLOW_NO_AUTH", True)
    assert post(client, "/analyze", {"url": "example.com"}, key=None).status_code == 200


@pytest.mark.parametrize("url", ["   ", "javascript:alert(1)", "JavaScript:x", "data:text/html,hi",
                                 "file:///etc/passwd", "mailto:a@b.c", "http://", "http://x.com:99999/",
                                 "x" * 5000])
def test_non_web_urls_are_rejected(client, url):
    assert post(client, "/analyze", {"url": url}).status_code == 422
    assert client.seen == []  # never reached the pipeline (a blank URL used to come back 'clean')


@pytest.mark.parametrize("url", ["example.com", "example.com:8080/login", " https://Ex.com/a?b=1 ",
                                 "vietcombank.com.vn/login"])
def test_web_urls_reach_the_pipeline_stripped(client, url):
    r = post(client, "/analyze", {"url": url})
    assert r.status_code == 200 and r.json()["verdict"] == "suspicious"
    assert client.seen == [url.strip()]


@pytest.mark.parametrize("explanation, public", [
    ("URL model rates it safe (score 0.012 < threshold 0.034).", "URL model rates it safe."),
    ("The URL looks suspicious (score 0.512) but the page content could not be analysed.",
     "The URL looks suspicious but the page content could not be analysed."),
    ("Label 'vietcombamk' — typosquat: suspected impersonation of 'vietcombank' (Damerau-Levenshtein"
     " = 1, Jaro-Winkler = 0.96 against 'vietcombank'); domain 'x.com' is not in the official list.",
     "Label 'vietcombamk' — typosquat: suspected impersonation of 'vietcombank'; domain 'x.com' is"
     " not in the official list."),
])
def test_scores_and_thresholds_are_hidden_by_default(client, monkeypatch, explanation, public):
    """Returned scores would be an evasion oracle: tweak the URL until it lands under a threshold."""
    full = {"url": "x.com", "verdict": "clean", "decided_by": "C_classifier", "explanation": explanation,
            "layers": {"C_classifier": {"score": 0.012, "threshold": 0.034}}, "latency_ms": 3.0}
    monkeypatch.setattr(api, "analyze_url", lambda url, **kw: full)
    body = post(client, "/analyze", {"url": "x.com"}).json()
    assert body == {"url": "x.com", "verdict": "clean", "decided_by": "C_classifier",
                    "explanation": public, "latency_ms": 3.0}
    monkeypatch.setattr(api, "EXPOSE_LAYERS", True)
    assert post(client, "/analyze", {"url": "x.com"}).json()["layers"]["C_classifier"]["threshold"] == 0.034


def test_rate_limit_per_client(client, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 3)
    codes = [post(client, "/analyze", {"url": "example.com"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert post(client, "/feedback", {"url": "example.com", "label": "clean"}).status_code == 429


def test_feedback_is_logged_without_the_url(client, tmp_path):
    url = "https://shop.example.com/account?token=abc123"
    assert post(client, "/feedback", {"url": url, "label": "clean"}).json() == {"ok": True}
    assert post(client, "/feedback", {"url": url, "label": "maybe"}).status_code == 422
    log = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("feedback-*.jsonl"))
    assert "abc123" not in log and '"label": "clean"' in log


def test_oversized_body_is_refused_before_parsing(client):
    r = client.post("/analyze", content=b'{"url": "' + b"a" * 100_000 + b'"}',
                    headers={"X-API-Key": KEY, "Content-Type": "application/json"})
    assert r.status_code == 413 and client.seen == []


def test_chunked_body_without_length_is_cut_off(client):
    """No Content-Length (chunked): the bytes actually received are counted."""
    def chunks():
        yield b'{"url": "'
        for _ in range(100):
            yield b"a" * 1000
        yield b'"}'
    r = client.post("/analyze", content=chunks(),
                    headers={"X-API-Key": KEY, "Content-Type": "application/json"})
    assert r.status_code == 413 and client.seen == []
    ok = client.post("/analyze", content=iter([b'{"url": ', b'"example.com"}']),
                     headers={"X-API-Key": KEY, "Content-Type": "application/json"})
    assert ok.status_code == 200


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert {"blocklist_urls", "blocklist_stale", "classifier_ready", "llm_enabled"} <= set(r.json())
