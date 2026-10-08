"""Layer C inference: URL -> 'clean' / 'suspicious' with CNN/BiLSTM + XGBoost.

Host branch (URL + DNS/TLS/RDAP/Tranco features) when the host resolves, lexical branch
(URL only) otherwise or when host features are disabled. Each segment (branch x popularity/age
bucket x bare/path) has its own threshold, chosen on validation data to recognise as many clean
URLs as possible within the phishing-miss budget; 'suspicious' means "send to Layer D (LLM)".
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
def _load() -> tuple[CharCnnLstm, xgb.XGBClassifier, xgb.XGBClassifier | None, dict]:
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    cnn = CharCnnLstm()
    cnn.load_state_dict(torch.load(MODEL_DIR / "cnn.pt", map_location=DEVICE, weights_only=True))
    cnn.eval()

    def load(name: str) -> xgb.XGBClassifier | None:
        if not (MODEL_DIR / name).exists():
            return None
        m = xgb.XGBClassifier()
        m.load_model(MODEL_DIR / name)
        m.set_params(device="cpu")
        return m

    return cnn, load("xgb_lexical.json"), load("xgb_host.json"), cfg


def is_available() -> bool:
    return (MODEL_DIR / "config.json").exists() and (MODEL_DIR / "xgb_lexical.json").exists()


def _lexical_matrix(urls: list[str]) -> np.ndarray:
    cnn = _load()[0]
    with torch.no_grad():
        emb = cnn.embed(torch.from_numpy(encode_urls(urls))).numpy()
    return np.hstack([emb, feature_matrix(urls)])


def score_urls(urls: list[str], host_mode: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch scoring -> (score, threshold, used_host_branch) per URL.

    host_mode: "none" = lexical branch only, "cache" = host branch where cached host data says
    the host is live (no network), "live" = look hosts up (network).
    """
    _, clf_lex, clf_host, cfg = _load()
    lex = _lexical_matrix(urls)
    score = clf_lex.predict_proba(lex)[:, 1]
    thr = np.full(len(urls), cfg["threshold_lexical"])
    used = np.zeros(len(urls), bool)
    rows: list = [None] * len(urls)
    if host_mode != "none" and clf_host is not None:
        raw = hf.raw_lookups(urls, offline=host_mode == "cache")
        rows = [raw[host_of(u)] for u in urls]
        used = np.array([hf.is_live(r) for r in rows])
        if used.any():
            hx = np.array([hf.host_feature_vector(u, r or {}) for u, r in zip(urls, rows)],
                          dtype=np.float32)
            p_host = clf_host.predict_proba(np.hstack([lex, hx])[used])[:, 1]
            score[used] = p_host
            thr[used] = cfg["threshold_host"]
    seg_thr = cfg.get("segment_thresholds")
    if seg_thr:
        thr = np.array([seg_thr.get(hf.segment_of(u, r, bool(h)), t)
                        for u, r, h, t in zip(urls, rows, used, thr)])
    return score, thr, used


def classify_url(url: str, use_host_features: bool = True, threshold: float | None = None) -> dict:
    """Layer C entry point -> {'label', 'score', 'threshold', 'branch', 'latency_ms', 'host_ms'}."""
    t0 = time.perf_counter()
    _, clf_lex, clf_host, cfg = _load()
    lex = _lexical_matrix([url])
    branch, host_ms = "lexical", 0.0
    if use_host_features and clf_host is not None:
        t_h = time.perf_counter()
        raw = hf.lookup_host_live(host_of(url), HOST_BUDGET_S)
        host_ms = round((time.perf_counter() - t_h) * 1000, 1)
        if hf.is_live(raw):
            hx = np.array([hf.host_feature_vector(url, raw)], dtype=np.float32)
            score = float(clf_host.predict_proba(np.hstack([lex, hx]))[0, 1])
            branch, t = "host", cfg["threshold_host"]
    if branch == "lexical":
        score = float(clf_lex.predict_proba(lex)[0, 1])
        t = cfg["threshold_lexical"]
        raw = None
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
