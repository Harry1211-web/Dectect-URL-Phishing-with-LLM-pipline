"""Layer C inference: URL -> 'clean' / 'suspicious' (stacked DL model + ML classifier).

The DL architecture and the ML model are those of models/url_classifier/config.json (chosen on
validation data by src/layer_c_selection.py).
dl_score = mean phishing probability of the two cross-fitted DL models; it feeds the host branch
(URL + DNS/TLS/RDAP/Tranco features) when the host resolves, the lexical branch otherwise. Each
segment (branch x popularity/age bucket x bare/path) has its own threshold chosen on validation
data; 'suspicious' means "send to Layer D (LLM)".
Fail-closed: when the host lookup cannot be completed (resolver timeout / network error, which an
attacker could provoke), the URL is 'suspicious' instead of falling back to a weaker decision.
A model trained with host_lookups = "none" looks nothing up: its host branch (URL + Tranco rank)
scores every URL, so Layer C needs no network access.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.features import feature_matrix  # noqa: E402
from url_classifier.model import DL_ARCHS, encode_inputs  # noqa: E402
from url_utils import host_of  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "url_classifier"
DEVICE = "cpu"  # single-URL inference is faster on CPU than paying the GPU transfer overhead
HOST_BUDGET_S = 2.5  # max time spent on DNS/TLS/RDAP lookups for one URL


class _LgbModel:
    """LightGBM booster with the predict_proba interface of the other models."""

    def __init__(self, path: Path):
        self.booster = lgb.Booster(model_file=str(path))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.booster.predict(x)
        return np.column_stack([1 - p, p])


def _load_ml(ml: str, path: Path):
    if not path.exists():
        return None
    if ml == "xgboost":
        m = xgb.XGBClassifier()
        m.load_model(path)
        m.set_params(device="cpu")
        return m
    if ml == "lightgbm":
        return _LgbModel(path)
    m = joblib.load(path)
    m.set_params(n_jobs=1)  # one URL at a time: no thread pool start-up
    return m


_ML_EXT = {"xgboost": "json", "lightgbm": "txt", "random_forest": "joblib"}


@lru_cache(maxsize=1)
def _load() -> tuple[list[torch.nn.Module], object, object | None, dict]:
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    arch, ml = cfg.get("dl_arch", "cnn_bilstm"), cfg.get("ml", "xgboost")
    dls = []
    for name in ("dl_a.pt", "dl_b.pt"):
        if (MODEL_DIR / name).exists():
            m = DL_ARCHS[arch]()
            m.load_state_dict(torch.load(MODEL_DIR / name, map_location=DEVICE, weights_only=True))
            m.eval()
            dls.append(m)
    ext = _ML_EXT[ml]
    return dls, _load_ml(ml, MODEL_DIR / f"ml_lexical.{ext}"), _load_ml(ml, MODEL_DIR / f"ml_host.{ext}"), cfg


def is_available() -> bool:
    if not (MODEL_DIR / "config.json").exists():
        return False
    ml = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8")).get("ml", "xgboost")
    return (MODEL_DIR / f"ml_lexical.{_ML_EXT[ml]}").exists()


def uses_network() -> bool:
    """True when the saved model needs DNS / TLS / RDAP lookups (config host_lookups)."""
    return _load()[3].get("host_lookups", "network") == "network"


def _host_cols(cfg: dict) -> list[int]:
    """Positions in HOST_FEATURE_NAMES of the host features the saved model was trained on."""
    names = set(cfg.get("host_feature_names", hf.HOST_FEATURE_NAMES))
    return [i for i, n in enumerate(hf.HOST_FEATURE_NAMES) if n in names]


def _dl_score(urls: list[str]) -> np.ndarray:
    dls, cfg = _load()[0], _load()[3]
    c, w = (torch.from_numpy(a) for a in encode_inputs(cfg.get("dl_arch", "cnn_bilstm"), urls))
    with torch.no_grad():
        return np.mean([torch.sigmoid(m(c, w)).numpy() for m in dls], axis=0).astype(np.float32)


def _matrices(urls: list[str], rows: list) -> tuple[np.ndarray, np.ndarray]:
    """(lexical matrix, host matrix) with the training column order."""
    hand = feature_matrix(urls)
    cs = _dl_score(urls)[:, None]
    hx = np.array([hf.host_feature_vector(u, r or {}) for u, r in zip(urls, rows)],
                  dtype=np.float32)[:, _host_cols(_load()[3])]
    return np.hstack([hand, cs]), np.hstack([hand, hx, cs])


def score_urls(urls: list[str], host_mode: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch scoring -> (score, threshold, used_host_branch) per URL.

    host_mode: "none" = lexical branch only, "cache" = host branch where cached host data says
    the host is live (no network), "live" = look hosts up (network).
    """
    _, clf_lex, clf_host, cfg = _load()
    rows: list = [None] * len(urls)
    offline = not uses_network()
    if host_mode != "none" and clf_host is not None:
        if offline:
            rows = [hf.NO_LOOKUP] * len(urls)
        else:
            raw = hf.raw_lookups(urls, offline=host_mode == "cache")
            rows = [raw[host_of(u)] for u in urls]
    lex, host = _matrices(urls, rows)
    score = clf_lex.predict_proba(lex)[:, 1]
    thr = np.full(len(urls), cfg["threshold_lexical"])
    used = np.array([r is not None and (offline or hf.is_live(r)) for r in rows])
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
    if use_host_features and clf_host is not None and not uses_network():
        raw = hf.NO_LOOKUP
    elif use_host_features and clf_host is not None:
        t_h = time.perf_counter()
        raw = hf.lookup_host_live(host_of(url), HOST_BUDGET_S)
        host_ms = round((time.perf_counter() - t_h) * 1000, 1)
        if raw is None:  # lookup failed or timed out: fail closed, let the LLM layer decide
            return {"label": "suspicious", "score": None, "threshold": None,
                    "branch": "lookup_failed", "segment": None, "host_ms": host_ms,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
    lex, host = _matrices([url], [raw])
    if raw is not None and (raw is hf.NO_LOOKUP or hf.is_live(raw)):
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
