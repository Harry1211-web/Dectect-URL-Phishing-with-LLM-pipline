"""Choose the Layer C pair on validation data: best DL URL model, then best ML classifier.

Step dl : every architecture of url_classifier/model.py is cross-fitted exactly as in production
          (two models on 70 % / 30 % of the train sites, out-of-fold train scores, same epochs and
          early stopping). Winner = fewest val phishing missed when the val threshold recognises
          >= 50 % of val clean URLs (baselines/common.py rule); ties -> higher val PR-AUC.
Step ml : the winner's score + hand-crafted (+ host) features feed XGBoost, LightGBM and random
          forest in the production two-branch layout (lexical branch on every row, host branch on
          live hosts). Same rule on the routed val score.
Test, PhreshPhish and the Vietnamese set are scored for every candidate and reported, but never
used to choose. The winning pair is then trained into production with
  python src/url_classifier/train.py --dl <arch> --ml <ml> --dl-dir models/layer_c_selection
Run:    python src/layer_c_selection.py dl [--archs ...]
        python src/layer_c_selection.py ml [--dl <arch>]
Output: reports/layer_c_selection/{dl,ml}.json, models/layer_c_selection/ (DL weights, scores,
        feature cache; not versioned)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import common  # noqa: E402
from url_classifier import train as T  # noqa: E402
from url_classifier.model import DL_ARCHS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEL_DIR = ROOT / "models" / "layer_c_selection"
OUT_DIR = ROOT / "reports" / "layer_c_selection"
EVAL = ("val", *common.EVAL_SETS)
SEED = 42


def _splits() -> tuple[dict, object]:
    """Evaluation splits (baselines/common.py) and the train split with its DL halves."""
    tr, _, _, _ = T.load_splits()
    return common.load_splits(), tr


def _val_summary(y: np.ndarray, p: np.ndarray) -> dict:
    t = common.threshold_for_clean(p, y)
    return {**common.metrics_at(y, p, t), "threshold": t,
            "roc_auc": round(float(roc_auc_score(y, p)), 5),
            "pr_auc": round(float(average_precision_score(y, p)), 5)}


def _winner(cands: dict) -> str:
    return min(cands, key=lambda k: (cands[k]["val"]["phishing_missed"], -cands[k]["val"]["pr_auc"]))


def _save(name: str, out: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def step_dl(archs: list[str], epochs: int) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, tr = _splits()
    b_flags = common.b_flags_for(splits)
    path = OUT_DIR / "dl.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"candidates": {}}
    for arch in archs:
        t0 = time.time()
        models, oof, log = T.cross_fit_dl(arch, tr, splits["val"], device, epochs, SEED,
                                          dl_dir=SEL_DIR, save_dir=SEL_DIR)
        scores = {s: T.dl_scores(models, arch, splits[s].url.tolist(), device) for s in EVAL}
        np.savez_compressed(SEL_DIR / f"scores_{arch}.npz", train=oof, **scores)
        res = common.evaluate_scores(arch, scores, splits, b_flags, {
            "training": log, "seconds": round(time.time() - t0),
            "parameters": sum(p.numel() for p in models[0].parameters())})
        res["val"] = _val_summary(splits["val"].y.to_numpy(), scores["val"])
        out["candidates"][arch] = res
        print(arch, json.dumps(res["val"]), flush=True)
        del models
        torch.cuda.empty_cache()
    out.update({"time": datetime.now(timezone.utc).isoformat(), "rule": res["rule"] + "; ties: val PR-AUC",
                "winner": _winner(out["candidates"]), "epochs_max": epochs,
                "patience": T.DL_PATIENCE})
    _save("dl", out)
    print("DL winner:", out["winner"])
    return 0


def _base(name: str, d) -> dict:
    """Hand-crafted + host matrices of one split, cached (host data from the offline cache)."""
    f = SEL_DIR / f"base_{name}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    b = T.base_matrices(d)
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **b)
    return b


def step_ml(arch: str | None) -> int:
    arch = arch or json.loads((OUT_DIR / "dl.json").read_text(encoding="utf-8"))["winner"]
    splits, tr = _splits()
    b_flags = common.b_flags_for(splits)
    z = np.load(SEL_DIR / f"scores_{arch}.npz")
    sets = {"train": tr, **{s: splits[s] for s in EVAL}}
    M = {k: T.with_dl_score(_base(k, d), z[k]) for k, d in sets.items()}
    y = {k: d.y.to_numpy() for k, d in sets.items()}
    out = {"dl": arch, "candidates": {}}
    for ml in T.ML_MODELS:
        t0 = time.time()
        B = T.fit_branches(ml, M, y, SEED)
        res = common.evaluate_scores(ml, {s: B["score"][s] for s in EVAL}, splits, b_flags,
                                     {"train_seconds": B["train_seconds"],
                                      "seconds": round(time.time() - t0)})
        res["val"] = _val_summary(y["val"], B["score"]["val"])
        out["candidates"][ml] = res
        print(ml, json.dumps(res["val"]), flush=True)
    out.update({"time": datetime.now(timezone.utc).isoformat(),
                "rule": res["rule"] + " (routed host / lexical score); ties: val PR-AUC",
                "winner": _winner(out["candidates"])})
    _save("ml", out)
    print("ML winner:", out["winner"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dl", "ml"))
    ap.add_argument("--archs", nargs="+", choices=tuple(DL_ARCHS), default=list(DL_ARCHS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--dl", choices=tuple(DL_ARCHS), default=None,
                    help="ml step: DL architecture to use (default: winner of the dl step)")
    args = ap.parse_args()
    return step_dl(args.archs, args.epochs) if args.step == "dl" else step_ml(args.dl)


if __name__ == "__main__":
    sys.exit(main())
