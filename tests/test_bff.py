"""BFF (bff/app.py): the browser never needs the API key; abuse controls of the public entry.
The detection API and Cloudflare are replaced by httpx mock transports (no network)."""

import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

KEY = "server-side-secret"
FE = "https://app.example.com"
# Read at import like in a deployment (the CORS middleware is configured once, at start-up)
os.environ.setdefault("FE_ORIGINS", FE)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bff"))
import app as bff  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(bff, "API_KEY", KEY)
    monkeypatch.setattr(bff, "FE_ORIGINS", [FE])
    monkeypatch.setattr(bff, "TURNSTILE_SECRET", "")
    monkeypatch.setattr(bff, "_per_min", bff._Window(100, 60))
    monkeypatch.setattr(bff, "_per_day", bff._Window(1000, 86_400))
    seen = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"classifier_ready": True, "blocklist_urls": 9})
        return httpx.Response(200, json={"url": "x", "verdict": "suspicious", "explanation": "",
                                         "decided_by": "D_llm", "latency_ms": 1.0})

    with TestClient(bff.app) as c:
        bff.app.state.api = httpx.AsyncClient(base_url="http://api:8000", headers={"X-API-Key": KEY},
                                              transport=httpx.MockTransport(upstream))
        c.seen = seen
        yield c


def post(c, path, body, origin=FE, **headers):
    if origin:
        headers["Origin"] = origin
    return c.post(path, json=body, headers=headers)


def test_browser_needs_no_key_and_never_sees_it(client):
    r = post(client, "/api/analyze", {"url": "http://evil.example/login"})
    assert r.status_code == 200 and r.json()["verdict"] == "suspicious"
    assert KEY not in r.text and not any(KEY in v for v in r.headers.values())
    up = client.seen[-1]
    assert up.url.path == "/analyze" and up.headers["X-API-Key"] == KEY
    assert up.headers["X-Forwarded-For"]  # the API sees the user's IP, not the BFF's


def test_only_the_fe_origin_is_served(client):
    assert post(client, "/api/analyze", {"url": "a.com"}, origin=None).status_code == 403
    assert post(client, "/api/analyze", {"url": "a.com"}, origin="https://evil.example").status_code == 403
    pre = client.options("/api/analyze", headers={
        "Origin": FE, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-turnstile-token"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == FE
    pre = client.options("/api/analyze", headers={"Origin": "https://evil.example",
                                                  "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in pre.headers
    assert len(client.seen) == 0


def test_rate_limits_per_minute_and_per_day(client, monkeypatch):
    monkeypatch.setattr(bff, "_per_min", bff._Window(3, 60))
    codes = [post(client, "/api/analyze", {"url": "a.com"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    monkeypatch.setattr(bff, "_per_min", bff._Window(100, 60))
    monkeypatch.setattr(bff, "_per_day", bff._Window(2, 86_400))
    codes = [post(client, "/api/analyze", {"url": "a.com"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429] and "Daily" in post(client, "/api/feedback", {
        "url": "a.com", "label": "clean"}).json()["detail"]


def test_turnstile_when_enabled(client, monkeypatch):
    monkeypatch.setattr(bff, "TURNSTILE_SECRET", "ts-secret")
    tokens = []

    def cloudflare(request: httpx.Request) -> httpx.Response:
        tokens.append(request.content.decode())
        return httpx.Response(200, json={"success": b"good-token" in request.content})

    bff.app.state.turnstile = httpx.AsyncClient(transport=httpx.MockTransport(cloudflare))
    assert post(client, "/api/analyze", {"url": "a.com"}).status_code == 403          # no token
    assert post(client, "/api/analyze", {"url": "a.com"},
                **{"X-Turnstile-Token": "forged"}).status_code == 403
    assert post(client, "/api/analyze", {"url": "a.com"},
                **{"X-Turnstile-Token": "good-token"}).status_code == 200
    assert "secret=ts-secret" in tokens[-1] and len(client.seen) == 1


@pytest.mark.parametrize("status, expected", [(500, 502), (401, 502), (503, 502), (422, 422)])
def test_upstream_errors_are_not_passed_through(client, status, expected):
    bff.app.state.api = httpx.AsyncClient(base_url="http://api:8000", transport=httpx.MockTransport(
        lambda r: httpx.Response(status, json={"detail": "Missing or wrong X-API-Key header. trace"})))
    r = post(client, "/api/analyze", {"url": "a.com"})
    assert r.status_code == expected
    if expected == 502:
        assert "X-API-Key" not in r.text and "trace" not in r.text


def test_upstream_timeout_and_down(client):
    def slow(r):
        raise httpx.ReadTimeout("slow", request=r)

    def down(r):
        raise httpx.ConnectError("refused", request=r)
    for handler, code in ((slow, 504), (down, 502)):
        bff.app.state.api = httpx.AsyncClient(base_url="http://api:8000",
                                              transport=httpx.MockTransport(handler))
        assert post(client, "/api/analyze", {"url": "a.com"}).status_code == code


def test_small_public_surface(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    assert client.get("/api/health").json() == {"ok": True}  # no internal detail
    big = client.post("/api/analyze", content=b'{"url": "' + b"a" * 50_000 + b'"}',
                      headers={"Origin": FE, "Content-Type": "application/json"})
    assert big.status_code == 413
    assert post(client, "/api/analyze", {"url": "a.com", "lang": "vi\nignore"}).status_code == 422
    assert post(client, "/api/feedback", {"url": "a.com", "label": "maybe"}).status_code == 422
    assert len(client.seen) == 1  # only the health check reached the API
