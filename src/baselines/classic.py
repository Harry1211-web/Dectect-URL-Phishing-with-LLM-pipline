"""Classic baselines on hand-crafted features, and Layer C itself under the same rule.

  logreg        Logistic regression, lexical URL features (standardised, class-balanced)
  random_forest Random forest, lexical URL features
  lightgbm_lex  LightGBM, lexical URL features
  lightgbm_host LightGBM, lexical + host features (DNS / TLS / RDAP / Tranco from the host cache,
                NaN where unknown), i.e. the same inputs as Layer C without the CNN score
  layer_c       production Layer C scores (stacked CNN/BiLSTM + XGBoost), single val threshold;
                the production segment thresholds are reported next to it
Lexical features = url_classifier.features (the hand-crafted part of Layer C).
Run:    python src/baselines/classic.py [--models logreg random_forest ...]
Output: reports/baselines/<model>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EVAL_SETS, b_flags_for, evaluate_scores, load_splits, metrics_at, save,
)
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.features import feature_matrix  # noqa: E402
from url_utils import host_of  # noqa: E402

MODELS = ("logreg", "random_forest", "lightgbm_lex", "lightgbm_host", "layer_c")
SEED = 42


def host_matrix(urls: list[str], chunk: int = 20_000) -> np.ndarray:
    out = []
    for i in range(0, len(urls), chunk):
        part = urls[i:i + chunk]
        raw = hf.raw_lookups(part, offline=True)
        out.append(np.array([hf.host_feature_vector(u, raw[host_of(u)] or {}) for u in part],
                            dtype=np.float32))
    return np.vstack(out)


def fit_predict(name: str, x: dict, y_tr: np.ndarray, y_va: np.ndarray) -> dict[str, np.ndarray]:
    """Train one model on x['train'] (early stopping on val where the model has it)."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(x["train"], y_tr)
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                   class_weight="balanced", random_state=SEED)
        m.fit(x["train"], y_tr)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=127,
                               subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                               random_state=SEED, verbose=-1)
        m.fit(x["train"], y_tr, eval_set=[(x["val"], y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return {s: m.predict_proba(x[s])[:, 1] for s in x if s != "train"}


def layer_c_scores(splits: dict) -> tuple[dict, dict]:
    """Production scores and production (segment) thresholds of every set, host data from cache."""
    from url_classifier import predict
    scores, thr = {}, {}
    for s in ("val", *EVAL_SETS):
        urls = splits[s].url.tolist()
        parts = [predict.score_urls(urls[i:i + 4096], host_mode="cache")
                 for i in range(0, len(urls), 4096)]
        scores[s] = np.concatenate([p[0] for p in parts])
        thr[s] = np.concatenate([p[1] for p in parts])
    return scores, thr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = ap.parse_args()

    splits = load_splits()
    b_flags = b_flags_for(splits)
    y_tr, y_va = splits["train"].y.to_numpy(), splits["val"].y.to_numpy()
    lex = host = None
    for name in args.models:
        t0 = time.time()
        if name == "layer_c":
            scores, seg_thr = layer_c_scores(splits)
            extra = {"production_segment_thresholds": {
                s: {"alone": metrics_at(splits[s].y.to_numpy(), scores[s] - seg_thr[s], 0.0,
                                        verified=splits[s].verified.to_numpy()),
                    "with_B": metrics_at(splits[s].y.to_numpy(), scores[s] - seg_thr[s], 0.0,
                                         b_flags[s], verified=splits[s].verified.to_numpy())}
                for s in EVAL_SETS}}
        else:
            if lex is None:
                lex = {s: feature_matrix(d.url.tolist()) for s, d in splits.items()}
            x = lex
            if name == "lightgbm_host":
                if host is None:
                    host = {s: np.hstack([lex[s], host_matrix(d.url.tolist())])
                            for s, d in splits.items()}
                x = host
            if name in ("logreg",):  # no NaN in lexical features, but guard anyway
                x = {s: np.nan_to_num(v) for s, v in x.items()}
            scores = fit_predict(name, x, y_tr, y_va)
            extra = {}
        extra["train_seconds"] = round(time.time() - t0, 1)
        print(name, flush=True)
        save(evaluate_scores(name, scores, splits, b_flags, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
