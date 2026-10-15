"""Shared data, operating rule and metrics for the Layer C baselines.

Every baseline is trained on the Layer C train split (rows flagged by clean_labels.py dropped),
tuned / early-stopped on val, and scored once on test, PhreshPhish and the Vietnamese set
(never trained on). Operating rule, the same for every model and for Layer C itself:
  threshold = lowest score that still recognises >= MIN_CLEAN_RECOGNIZED of the val clean URLs;
  main number = share of phishing scored below it (phishing labelled clean).
Results are reported for the model alone and combined with Layer B (a URL B flags goes to
Layer D whatever the model says), plus threshold-free ROC-AUC and PR-AUC.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier.train import MIN_CLEAN_RECOGNIZED, VERIFIED_SOURCES  # noqa: E402

DATA_PATH = ROOT / "dataset" / "urls" / "urls_labeled.csv"
NOISY_PATH = DATA_PATH.parent / "noisy_labels.csv"
PHRESHPHISH = ROOT / "dataset" / "external" / "phreshphish_urls.csv"
VI_DIR = ROOT / "dataset" / "urls" / "vi"
OUT_DIR = ROOT / "reports" / "baselines"


def load_splits() -> dict[str, pd.DataFrame]:
    """train / val / test of Layer C plus the two external sets; y = 1 for phishing, verified =
    clean row or phishing from a human-verified feed (PhishTank / OpenPhish)."""
    df = pd.read_csv(DATA_PATH)
    df["y"] = (df.label == "phishing").astype(np.int64)
    df["verified"] = (df.y == 0) | df.source.isin(VERIFIED_SOURCES)
    out = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if NOISY_PATH.exists():
        noisy = set(pd.read_csv(NOISY_PATH).url)
        out["train"] = out["train"][~out["train"].url.isin(noisy)].reset_index(drop=True)
    ext = pd.read_csv(PHRESHPHISH)
    out["phreshphish"] = pd.DataFrame({"url": ext.url, "y": (ext.label == "phishing").astype(np.int64)})
    vi = [pd.read_csv(VI_DIR / f"{lab}.csv", usecols=["url"]).assign(y=int(lab == "phishing"))
          for lab in ("phishing", "clean")]
    out["vi"] = pd.concat(vi, ignore_index=True)
    for k in ("phreshphish", "vi"):
        out[k]["verified"] = True
    return out


EVAL_SETS = ("test", "phreshphish", "vi")


def layer_b_flags(urls: list[str]) -> np.ndarray:
    return np.array([check_typosquatting_deep(u)["flag"] is True for u in urls])


def threshold_for_clean(score_val: np.ndarray, y_val: np.ndarray,
                        min_clean: float = MIN_CLEAN_RECOGNIZED) -> float:
    """Lowest threshold t with share(val clean score < t) >= min_clean (fewest misses under the
    clean-recognised floor)."""
    clean = np.sort(score_val[y_val == 0])
    k = int(np.ceil(min_clean * len(clean)))
    return float(np.nextafter(clean[k - 1], np.inf))


def metrics_at(y: np.ndarray, p: np.ndarray, t: float, b_flag: np.ndarray | None = None,
               verified: np.ndarray | None = None) -> dict:
    """Phishing labelled clean (score < t and, with B, not flagged by B) and clean recognised."""
    sent = p >= t if b_flag is None else (p >= t) | b_flag
    ph, cl = y == 1, y == 0
    out = {"phishing": int(ph.sum()), "phishing_missed": int((ph & ~sent).sum()),
           "miss_rate": round(float((ph & ~sent).sum() / max(1, ph.sum())), 5),
           "clean": int(cl.sum()),
           "clean_recognized": round(float((cl & ~sent).sum() / max(1, cl.sum())), 5)}
    if verified is not None and (ph & ~verified).any():
        v = ph & verified
        out["verified_phishing"] = int(v.sum())
        out["verified_missed"] = int((v & ~sent).sum())
        out["verified_miss_rate"] = round(float((v & ~sent).sum() / max(1, v.sum())), 5)
    return out


def evaluate_scores(name: str, scores: dict[str, np.ndarray], splits: dict[str, pd.DataFrame],
                    b_flags: dict[str, np.ndarray], extra: dict | None = None) -> dict:
    """Threshold from val, then every evaluation set: alone, with Layer B, AUCs."""
    yv = splits["val"].y.to_numpy()
    t = threshold_for_clean(scores["val"], yv)
    res = {"model": name, "time": datetime.now(timezone.utc).isoformat(),
           "rule": f"threshold = lowest score recognising >= {MIN_CLEAN_RECOGNIZED:.0%} of val clean",
           "threshold": t, "val": metrics_at(yv, scores["val"], t), **(extra or {})}
    for s in EVAL_SETS:
        y, p = splits[s].y.to_numpy(), scores[s]
        ver = splits[s].verified.to_numpy()
        res[s] = {"roc_auc": round(float(roc_auc_score(y, p)), 5),
                  "pr_auc": round(float(average_precision_score(y, p)), 5),
                  "alone": metrics_at(y, p, t, verified=ver),
                  "with_B": metrics_at(y, p, t, b_flags[s], verified=ver)}
    return res


def save(res: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{res['model']}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({s: {"auc": res[s]["roc_auc"], **res[s]["alone"]} for s in EVAL_SETS}),
          flush=True)
    return path


def b_flags_for(splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {s: layer_b_flags(splits[s].url.tolist()) for s in EVAL_SETS}
