"""HTTP API around the detection pipeline, for the UI (browser extension / web app).

Endpoints
  POST /analyze   {"url": "...", "explain": false, "lang": "vi"}
                  -> {url, verdict, explanation, decided_by, latency_ms} (see API_EXPOSE_LAYERS)
  POST /feedback  {"url": "...", "label": "clean" | "phishing"}   -> user report (monitoring.py)
  GET  /health    readiness of every layer (blocklist age, classifier, LLM key)
Interactive docs: /docs

Settings (environment variables)
  API_KEY        required: every request must send the header "X-API-Key: <API_KEY>"
  ALLOW_NO_AUTH  "1" accepts requests without a key when API_KEY is unset (local testing only)
  CORS_ORIGINS   comma-separated origins allowed to call the API from a browser (default "*")
  USE_LLM, USE_VT  "0" disables Layer D / the VirusTotal lookup (default on)
  RATE_LIMIT_PER_MIN  POST requests per client IP and minute, per worker process (default 60, 0 = off)
  API_EXPOSE_LAYERS  "1" adds per-layer details (scores, thresholds) to /analyze; debug / demo only
                   (default: verdict, explanation without scores, decided_by, latency_ms)
Only http(s) URLs (or a bare host such as "example.com") are accepted.
API keys of the layers (OPENAI_API_KEY, VT_API_KEY) are read from .env by the layers themselves.

Run: uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hmac
import json
import os
import re
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "llm"))
from blocklist import lookup  # noqa: E402
from blocklist.updater import META_PATH  # noqa: E402
from monitoring import record_feedback, record_verdict  # noqa: E402
from pipeline import analyze_url  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import predict as classifier  # noqa: E402
from url_utils import MAX_URL_LEN, ensure_scheme  # noqa: E402

API_KEY = os.getenv("API_KEY", "").strip()
ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "0") == "1"
USE_LLM = os.getenv("USE_LLM", "1") != "0"
USE_VT = os.getenv("USE_VT", "1") != "0"
EXPOSE_LAYERS = os.getenv("API_EXPOSE_LAYERS", "0") == "1"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load blocklist, brand / Tranco tables and models once, so the first user request is not the slow one
    lookup.size()
    check_typosquatting_deep("https://example.com/")
    if classifier.is_available():
        classifier.classify_url("https://example.com/", use_host_features=False)
    yield


app = FastAPI(title="Phishing detection API", version="1.0", lifespan=lifespan,
              description="Layers A (blocklists, VirusTotal) -> B (typosquatting) -> "
                          "C (BiLSTM + XGBoost) -> D (LLM). Verdict: phishing / suspicious / clean.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])

MAX_BODY_BYTES = 16_384  # a request is one URL (<= 4096 chars) plus two small fields


class BodyLimit:
    """Refuse oversized bodies before they are held in memory (the URL length check only runs
    after the whole JSON body is parsed). Counts the bytes actually received, so a chunked body
    without Content-Length or with a false one is cut off as well."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        try:
            declared_too_big = length is not None and int(length) > MAX_BODY_BYTES
        except ValueError:
            declared_too_big = True
        if declared_too_big:
            resp = JSONResponse({"detail": "Request body too large."}, status_code=413)
            return await resp(scope, receive, send)
        received = 0

        async def counted():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    # Raised while FastAPI reads the body: answered as a normal 413
                    raise HTTPException(status_code=413, detail="Request body too large.")
            return message

        await self.app(scope, counted, send)


app.add_middleware(BodyLimit)


def check_key(x_api_key: str | None = Header(default=None)) -> None:
    # Fail closed: without API_KEY every request is refused (each one may cost an LLM call),
    # unless ALLOW_NO_AUTH=1 is set explicitly for local testing
    if not API_KEY:
        if ALLOW_NO_AUTH:
            return
        raise HTTPException(status_code=503, detail="Server has no API_KEY configured.")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


_hits: dict[str, deque] = {}
_hits_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    """Sliding one-minute window per client IP: a leaked key cannot flood the paid layers
    (LLM, VirusTotal quota) or turn the host lookups into a request amplifier."""
    if RATE_LIMIT_PER_MIN <= 0:
        return
    ip = request.client.host if request.client else "?"
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Too many requests, retry in a minute.")
        q.append(now)
        if len(_hits) > 10_000:  # drop idle clients so the table cannot grow without bound
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                del _hits[k]


_RE_OTHER_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):(?!\d)")


def _web_url(v: str) -> str:
    """Stripped http(s) URL with a host; anything else (empty, javascript:, file:) -> 422.
    A blank URL used to reach the classifier and come back 'clean'."""
    v = v.strip()
    # 'javascript:x' / 'mailto:x' have no '//', so ensure_scheme would read them as a host;
    # 'example.com:8080' (a port: digit after ':') stays accepted
    scheme = _RE_OTHER_SCHEME.match(v)
    try:
        parts = urlsplit(ensure_scheme(v))
        ok = (not scheme or scheme.group(1).lower() in ("http", "https")) \
            and parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
        _ = parts.port  # ValueError on a malformed port
    except ValueError:
        ok = False
    if not ok:
        raise ValueError("url must be an http(s) URL or a host name")
    return v


WebUrl = Annotated[str, Field(min_length=1, max_length=MAX_URL_LEN), AfterValidator(_web_url)]


class AnalyzeRequest(BaseModel):
    url: WebUrl = Field(examples=["http://vietcombank-xacthuc.com/login"])
    explain: bool = Field(False, description="ask the LLM for an explanation (slower, extra cost)")
    lang: str = Field("vi", max_length=8, description="explanation language (ISO code)")


class FeedbackRequest(BaseModel):
    url: WebUrl
    label: Literal["clean", "phishing"]


@app.post("/analyze", dependencies=[Depends(check_key), Depends(rate_limit)])
def analyze(req: AnalyzeRequest) -> dict:
    """Plain `def`: FastAPI runs it in a worker thread, so slow lookups never block other requests."""
    result = analyze_url(req.url, use_llm=USE_LLM, use_vt=USE_VT, explain=req.explain, lang=req.lang)
    record_verdict(result)  # no URL stored, see monitoring.py
    if not EXPOSE_LAYERS:
        result = public_result(result)
    # default=str: numpy numbers in the layer details become plain JSON values
    return json.loads(json.dumps(result, ensure_ascii=False, default=str))


# "(score 0.012 < threshold 0.034)", "(score 0.995)", "(Damerau-Levenshtein = 1, ... against 'x')"
_RE_SCORE_DETAIL = re.compile(r"\s*\((?:score|Damerau-Levenshtein)\b[^)]*\)", re.I)


def public_result(result: dict) -> dict:
    """What a client sees by default: verdict and explanation without per-layer scores and
    thresholds. Returned scores would let an attacker probe URL variants until one lands just
    under a threshold (evasion oracle). API_EXPOSE_LAYERS=1 returns everything (debug / demo)."""
    return {"url": result["url"], "verdict": result["verdict"], "decided_by": result["decided_by"],
            "explanation": _RE_SCORE_DETAIL.sub("", result.get("explanation") or ""),
            "latency_ms": result.get("latency_ms")}


@app.post("/feedback", dependencies=[Depends(check_key), Depends(rate_limit)])
def feedback(req: FeedbackRequest) -> dict:
    record_feedback(req.url, req.label)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(meta["updated_at"])).total_seconds() / 3600
    except (OSError, KeyError, ValueError):
        age_h = None
    from llm.llm_client import OPENAI_API_KEY
    return {
        "blocklist_urls": lookup.size(),
        "blocklist_age_hours": None if age_h is None else round(age_h, 1),
        "blocklist_stale": age_h is None or age_h > 6,
        "classifier_ready": classifier.is_available(),
        "llm_enabled": USE_LLM and bool(OPENAI_API_KEY),
        "virustotal_enabled": USE_VT,
    }
