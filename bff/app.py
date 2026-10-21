"""Backend-for-frontend (BFF): the only service the browser talks to.

The front end cannot keep a secret, so it calls this service without any key; the BFF adds the
X-API-Key header of the detection API (src/api.py) server-side and forwards the request over the
private Docker network. The API itself is never published.

Since anyone can reach the BFF, it carries its own abuse controls:
  - allowed origins (FE_ORIGINS): CORS for browsers + a server-side Origin check on every POST
  - rate limits per client IP: per minute and per day (each analysis may cost an LLM call)
  - optional Cloudflare Turnstile (TURNSTILE_SECRET): a token per request proves a real browser
  - body size limit, strict request schema, upstream errors never passed through verbatim

Endpoints (what the FE uses)
  POST /api/analyze   {"url": "...", "explain": false, "lang": "vi"}  -> verdict of the API
  POST /api/feedback  {"url": "...", "label": "clean" | "phishing"}  -> {"ok": true}
  GET  /api/health    {"ok": bool} (no detail of the API)

Settings (environment variables)
  API_URL               detection API base URL (default http://api:8000, Docker network)
  API_KEY               key of the detection API (required; never sent to the browser)
  FE_ORIGINS            comma-separated FE origins, e.g. https://app.example.com (required)
  BFF_RATE_PER_MIN      requests per client IP and minute (default 10)
  BFF_RATE_PER_DAY      requests per client IP and day (default 300)
  TURNSTILE_SECRET      if set, every POST needs the header X-Turnstile-Token (Cloudflare)
  UPSTREAM_TIMEOUT_S    wait for the API (default 40; Layer D may take ~15 s)

Run: uvicorn app:app --app-dir bff --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "").strip()
FE_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("FE_ORIGINS", "").split(",") if o.strip()]
RATE_PER_MIN = int(os.getenv("BFF_RATE_PER_MIN", "10"))
RATE_PER_DAY = int(os.getenv("BFF_RATE_PER_DAY", "300"))
TURNSTILE_SECRET = os.getenv("TURNSTILE_SECRET", "").strip()
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "40"))
MAX_URL_LEN = 4096
MAX_BODY_BYTES = 16_384


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the API, one for Turnstile; closed on shutdown
    app.state.api = httpx.AsyncClient(base_url=API_URL, timeout=UPSTREAM_TIMEOUT_S,
                                      headers={"X-API-Key": API_KEY})
    app.state.turnstile = httpx.AsyncClient(timeout=5)
    yield
    await app.state.api.aclose()
    await app.state.turnstile.aclose()


# No /docs, /redoc or /openapi.json: the public surface is the three routes below
app = FastAPI(title="Phishing check BFF", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=FE_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Turnstile-Token"], max_age=600)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    """Bodies are tiny (one URL): refuse large ones before they are read. Chunked bodies without
    a length are capped by the reverse proxy (DEPLOY.txt)."""
    try:
        too_big = int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES
    except ValueError:
        too_big = True
    if too_big:
        return JSONResponse({"detail": "Request body too large."}, status_code=413)
    return await call_next(request)


# ----------------------------------------------------------------------------- abuse controls
class _Window:
    """Sliding-window counter per client IP (in memory, per worker process)."""

    def __init__(self, limit: int, seconds: float):
        self.limit, self.seconds = limit, seconds
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, key: str, now: float) -> bool:
        if self.limit <= 0:
            return True
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.seconds:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 50_000:  # drop idle clients so the table stays bounded
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.seconds]:
                    del self.hits[k]
            return True


_per_min = _Window(RATE_PER_MIN, 60)
_per_day = _Window(RATE_PER_DAY, 86_400)


def client_ip(request: Request) -> str:
    # Behind the reverse proxy, uvicorn --proxy-headers already put the real client IP here
    return request.client.host if request.client else "?"


def check_origin(origin: str | None = Header(default=None)) -> None:
    """CORS only binds browsers that honour it: the Origin is checked on the server as well.
    Non-browser clients can forge it; the rate limits and Turnstile cover them."""
    if not FE_ORIGINS:
        raise HTTPException(status_code=503, detail="Server has no FE_ORIGINS configured.")
    if (origin or "").rstrip("/") not in FE_ORIGINS:
        raise HTTPException(status_code=403, detail="Origin not allowed.")


def check_rate(request: Request) -> None:
    ip, now = client_ip(request), time.monotonic()
    # Day window first: a refused minute must not consume the day quota
    if not _per_day.allow(ip, now):
        raise HTTPException(status_code=429, detail="Daily limit reached, try again tomorrow.")
    if not _per_min.allow(ip, now):
        raise HTTPException(status_code=429, detail="Too many requests, retry in a minute.")


async def check_turnstile(request: Request,
                          x_turnstile_token: str | None = Header(default=None)) -> None:
    """Cloudflare Turnstile, when TURNSTILE_SECRET is set: one token per request, verified
    server-side (tokens are single-use and expire after 5 minutes)."""
    if not TURNSTILE_SECRET:
        return
    if not x_turnstile_token or len(x_turnstile_token) > 2048:
        raise HTTPException(status_code=403, detail="Missing human verification.")
    try:
        r = await request.app.state.turnstile.post(TURNSTILE_VERIFY_URL, data={
            "secret": TURNSTILE_SECRET, "response": x_turnstile_token,
            "remoteip": client_ip(request)})
        ok = r.status_code == 200 and r.json().get("success") is True
    except (httpx.HTTPError, ValueError):
        ok = False  # fail closed: no verification, no paid analysis
    if not ok:
        raise HTTPException(status_code=403, detail="Human verification failed.")


GUARDS = [Depends(check_origin), Depends(check_rate), Depends(check_turnstile)]


# ----------------------------------------------------------------------------- forwarding
class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LEN)
    explain: bool = False
    lang: str = Field("vi", max_length=8, pattern=r"^[A-Za-z-]{2,8}$")


class FeedbackRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LEN)
    label: Literal["clean", "phishing"]


# API answers passed to the browser as they are; anything else becomes a generic error so
# internal details (server misconfiguration, stack traces) never reach the client
_PASS_THROUGH = {200, 422, 429}


async def forward(request: Request, path: str, body: dict) -> JSONResponse:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Server has no API_KEY configured.")
    try:
        # The real client IP, so the API's own per-IP limit and logs see the user, not the BFF
        r = await request.app.state.api.post(path, json=body,
                                             headers={"X-Forwarded-For": client_ip(request)})
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="The analysis took too long, try again.")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Analysis service unavailable.")
    if r.status_code not in _PASS_THROUGH:
        raise HTTPException(status_code=502, detail="Analysis service error.")
    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except ValueError:
        raise HTTPException(status_code=502, detail="Analysis service error.")


@app.post("/api/analyze", dependencies=GUARDS)
async def analyze(req: AnalyzeRequest, request: Request) -> JSONResponse:
    return await forward(request, "/analyze", req.model_dump())


@app.post("/api/feedback", dependencies=GUARDS)
async def feedback(req: FeedbackRequest, request: Request) -> JSONResponse:
    return await forward(request, "/feedback", req.model_dump())


@app.get("/api/health")
async def health(request: Request) -> dict:
    """Up / down only: the API's own health detail (blocklist age, LLM state) stays internal."""
    try:
        r = await request.app.state.api.get("/health", timeout=5)
        return {"ok": r.status_code == 200 and bool(r.json().get("classifier_ready"))}
    except (httpx.HTTPError, ValueError):
        return {"ok": False}
