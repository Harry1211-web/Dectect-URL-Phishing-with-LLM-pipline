"""Layer C inference: URL -> 'clean' / 'suspicious' (stacked CNN/BiLSTM + XGBoost).

cnn_score = mean phishing probability of the two cross-fitted CNNs; it feeds the host branch
(URL + DNS/TLS/RDAP/Tranco features) when the host resolves, the lexical branch otherwise. Each
segment (branch x popularity/age bucket x bare/path) has its own threshold chosen on validation
data; 'suspicious' means "send to Layer D (LLM)".
Fail-closed: when the host lookup cannot be completed (resolver timeout / network error, which an
attacker could provoke), the URL is 'suspicious' instead of falling back to a weaker decision.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.features import feature_matrix  # noqa: E402
from url_classifier.model import CharCnnLstm, encode_urls  # noqa: E402
from url_utils import host_of  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "url_classifier"
DEVICE = "cpu"  # single-URL inference is faster on CPU than paying the GPU transfer overhead
HOST_BUDGET_S = 2.5  # max time spent on DNS/TLS/RDAP lookups for one URL


@lru_cache(maxsize=1)
def _load() -> tuple[list[CharCnnLstm], xgb.XGBClassifier, xgb.XGBClassifier | None, dict]:
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    cnns = []
    for name in ("cnn.pt", "cnn_b.pt"):
        if (MODEL_DIR / name).exists():
            m = CharCnnLstm()
            m.load_state_dict(torch.load(MODEL_DIR / name, map_location=DEVICE, weights_only=True))
            m.eval()
            cnns.append(m)

    def load(name: str) -> xgb.XGBClassifier | None:
        if not (MODEL_DIR / name).exists():
            return None
        m = xgb.XGBClassifier()
        m.load_model(MODEL_DIR / name)
        m.set_params(device="cpu")
        return m

    return cnns, load("xgb_lexical.json"), load("xgb_host.json"), cfg


def is_available() -> bool:
    return (MODEL_DIR / "config.json").exists() and (MODEL_DIR / "xgb_lexical.json").exists()


def _cnn_score(urls: list[str]) -> np.ndarray:
    cnns = _load()[0]
    x = torch.from_numpy(encode_urls(urls))
    with torch.no_grad():
        return np.mean([torch.sigmoid(m(x)).numpy() for m in cnns], axis=0).astype(np.float32)


def _matrices(urls: list[str], rows: list) -> tuple[np.ndarray, np.ndarray]:
    """(lexical matrix, host matrix) with the training column order."""
    hand = feature_matrix(urls)
    cs = _cnn_score(urls)[:, None]
    hx = np.array([hf.host_feature_vector(u, r or {}) for u, r in zip(urls, rows)], dtype=np.float32)
    return np.hstack([hand, cs]), np.hstack([hand, hx, cs])


def score_urls(urls: list[str], host_mode: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch scoring -> (score, threshold, used_host_branch) per URL.

    host_mode: "none" = lexical branch only, "cache" = host branch where cached host data says
    the host is live (no network), "live" = look hosts up (network).
    """
    _, clf_lex, clf_host, cfg = _load()
    rows: list = [None] * len(urls)
    if host_mode != "none" and clf_host is not None:
        raw = hf.raw_lookups(urls, offline=host_mode == "cache")
        rows = [raw[host_of(u)] for u in urls]
    lex, host = _matrices(urls, rows)
    score = clf_lex.predict_proba(lex)[:, 1]
    thr = np.full(len(urls), cfg["threshold_lexical"])
    used = np.array([hf.is_live(r) for r in rows]) if clf_host is not None else np.zeros(len(urls), bool)
    if used.any():
        score[used] = clf_host.predict_proba(host[used])[:, 1]
        thr[used] = cfg["threshold_host"]
    seg_thr = cfg.get("segment_thresholds")
    if seg_thr:
        thr = np.array([seg_thr.get(hf.segment_of(u, r, bool(h)), t)
                        for u, r, h, t in zip(urls, rows, used, thr)])
    return score, thr, used


def classify_url(url: str, use_host_features: bool = True, threshold: float | None = None) -> dict:
    """Layer C entry point -> {'label', 'score', 'threshold', 'branch', 'segment', ...}."""
    t0 = time.perf_counter()
    _, clf_lex, clf_host, cfg = _load()
    raw, host_ms = None, 0.0
    if use_host_features and clf_host is not None:
        t_h = time.perf_counter()
        raw = hf.lookup_host_live(host_of(url), HOST_BUDGET_S)
        host_ms = round((time.perf_counter() - t_h) * 1000, 1)
        if raw is None:  # lookup failed or timed out: fail closed, let the LLM layer decide
            return {"label": "suspicious", "score": None, "threshold": None,
                    "branch": "lookup_failed", "segment": None, "host_ms": host_ms,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    lex, host = _matrices([url], [raw])
    if hf.is_live(raw):
        branch, score, t = "host", float(clf_host.predict_proba(host)[0, 1]), cfg["threshold_host"]
    else:
        branch, score, t = "lexical", float(clf_lex.predict_proba(lex)[0, 1]), cfg["threshold_lexical"]
    segment = hf.segment_of(url, raw, branch == "host")
    t = cfg.get("segment_thresholds", {}).get(segment, t)
    if threshold is not None:
        t = threshold
    return {"label": "suspicious" if score >= t else "clean", "score": round(score, 6),
            "threshold": t, "branch": branch, "segment": segment, "host_ms": host_ms,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


if __name__ == "__main__":
    for u in sys.argv[1:]:
        print(u, classify_url(u))
