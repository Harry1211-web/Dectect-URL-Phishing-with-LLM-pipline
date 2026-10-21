"""Choose the Layer C pair on validation data: best DL URL model, then best ML classifier.

Step dl : every architecture of url_classifier/model.py is cross-fitted exactly as in production
          (two models on 70 % / 30 % of the train sites, out-of-fold train scores, same epochs and
          early stopping). Winner = fewest val phishing missed when the val threshold recognises
          >= 50 % of val clean URLs (baselines/common.py rule); ties -> higher val PR-AUC.
Step ml : the winner's score + hand-crafted (+ host) features feed XGBoost, LightGBM and random
          forest in the production two-branch layout (lexical branch on every row, host branch on
          live hosts). Same rule on the routed val score.
Step ablation : the chosen pair refitted with one feature group removed at a time (URL groups,
          Tranco, DNS, TLS, RDAP, whole host block, all hand-crafted features, DL score), several
          seeds each, same rule. Routing (host branch on live hosts) stays fixed, so only the
          information in the features changes. Measures what each group contributes.
Test, PhreshPhish and the Vietnamese set are scored for every candidate and reported, but never
used to choose. The winning pair is then trained into production with
  python src/url_classifier/train.py --dl <arch> --ml <ml> --dl-dir models/layer_c_selection
Run:    python src/layer_c_selection.py dl [--archs ...]
        python src/layer_c_selection.py ml [--dl <arch>]
        python src/layer_c_selection.py ablation [--seeds 42 43 44]   (resumes a partial report)
Output: reports/layer_c_selection/{dl,ml,ablation}.json, models/layer_c_selection/ (DL weights,
        scores, feature cache; not versioned)
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


def ablation_groups() -> dict[str, tuple[list[str], list[str], bool]]:
    """{config: (dropped URL features, dropped host features, drop DL score)}; 'full' drops nothing."""
    from url_classifier.features import FEATURE_NAMES as lex
    from url_classifier.host_features import HOST_FEATURE_NAMES as host
    url_groups = {
        "url_length_structure": ["url_len", "host_len", "path_len", "query_len", "n_host_dots",
                                 "n_host_hyphens", "n_host_digits", "n_subdomain_labels",
                                 "domain_label_len", "is_ip", "has_port", "has_at",
                                 "has_double_slash_path", "n_path_segments", "n_query_params",
                                 "domain_has_digit"],
        "url_characters": ["digit_ratio", "upper_ratio", "n_hyphen", "n_underscore", "n_percent",
                           "n_equals", "n_ampersand", "n_tilde", "host_entropy", "url_entropy",
                           "longest_token_len", "has_hex_string"],
        "url_lure_words": ["n_sensitive_words", "has_punycode", "has_file_ext"],
        "url_platform_redirect": ["is_free_hosting", "is_shortener", "has_redirect_param",
                                  "n_embedded_urls", "embedded_on_untrusted_host"],
        "url_tld": [n for n in lex if n.startswith("tld_")],
    }
    covered = sum(url_groups.values(), [])
    assert sorted(covered) == sorted(lex), f"URL features without a group: {set(lex) - set(covered)}"
    host_groups = {
        "host_tranco": ["tranco_log_rank", "in_tranco"],
        "host_dns": [n for n in host if n.startswith("dns_")],
        "host_tls": [n for n in host if n.startswith("tls_")],
        "host_rdap": ["rdap_found", "domain_age_days", "days_to_expiry"],
    }
    assert sorted(sum(host_groups.values(), [])) == sorted(host)
    return {"full": ([], [], False),
            **{g: (n, [], False) for g, n in url_groups.items()},
            **{g: ([], n, False) for g, n in host_groups.items()},
            "all_host": ([], list(host), False),
            "all_handcrafted": (list(lex), [], False),
            "dl_score": ([], [], True)}


def _ablated(base: dict, dl: np.ndarray, drop_lex: list[str], drop_host: list[str],
             drop_dl: bool) -> dict:
    """Production matrices (train.with_dl_score layout) without the dropped columns."""
    from url_classifier.features import FEATURE_NAMES as lex
    from url_classifier.host_features import HOST_FEATURE_NAMES as host
    hand = base["hand"][:, [i for i, n in enumerate(lex) if n not in drop_lex]]
    hostx = base["hostx"][:, [i for i, n in enumerate(host) if n not in drop_host]]
    dl_col = [] if drop_dl else [dl.astype(np.float32)[:, None]]
    # A branch left with no column gets a constant one (the model then scores every row alike)
    lex_m = np.hstack([hand, *dl_col]) if hand.shape[1] or dl_col else np.zeros((len(dl), 1), np.float32)
    return {"lex": lex_m, "host": np.hstack([hand, hostx, *dl_col]),
            "live": base["live"], "complete": base["complete"]}


def step_ablation(arch: str | None, ml: str | None, seeds: list[int]) -> int:
    sel = json.loads((OUT_DIR / "ml.json").read_text(encoding="utf-8"))
    arch, ml = arch or sel["dl"], ml or sel["winner"]
    splits, tr = _splits()
    b_flags = common.b_flags_for(splits)
    z = np.load(SEL_DIR / f"scores_{arch}.npz")
    sets = {"train": tr, **{s: splits[s] for s in EVAL}}
    bases = {k: _base(k, d) for k, d in sets.items()}
    y = {k: d.y.to_numpy() for k, d in sets.items()}
    groups = ablation_groups()
    path = OUT_DIR / "ablation.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if (out.get("dl"), out.get("ml")) != (arch, ml):
        out = {}  # report of another pair: start over
    out.update({"dl": arch, "ml": ml, "seeds": seeds,
                "rule": "val threshold recognising >= 50% of val clean; routing (live hosts) fixed",
                "groups": {g: {"url": u, "host": h, "dl_score": d} for g, (u, h, d) in groups.items()}})
    runs = out.setdefault("runs", {})
    for name, (drop_lex, drop_host, drop_dl) in groups.items():
        for seed in seeds:
            key = f"{name}|{seed}"
            if key in runs:
                continue
            t0 = time.time()
            M = {k: _ablated(bases[k], z[k], drop_lex, drop_host, drop_dl) for k in sets}
            B = T.fit_branches(ml, M, y, seed)
            res = common.evaluate_scores(name, {s: B["score"][s] for s in EVAL}, splits, b_flags,
                                         {"seconds": round(time.time() - t0)})
            res["val"] = _val_summary(y["val"], B["score"]["val"])
            runs[key] = res
            _save("ablation", out)  # after every fit: an interrupted run resumes here
            print(key, "val missed", res["val"]["phishing_missed"], flush=True)
    out["summary"] = _ablation_summary(runs, list(groups), seeds)
    out["time"] = datetime.now(timezone.utc).isoformat()
    _save("ablation", out)
    print(json.dumps(out["summary"], indent=1))
    return 0


def _ablation_summary(runs: dict, names: list[str], seeds: list[int]) -> dict:
    """Per config: val misses of every seed and their mean, mean PR-AUC of every evaluation set
    (threshold-free: on the external sets the val threshold lands at a different clean rate per
    config, so their miss counts are not comparable), mean misses / clean rate of each set."""
    out = {}
    for name in names:
        rs = [runs[f"{name}|{s}"] for s in seeds]
        mean = lambda f: round(float(np.mean([f(r) for r in rs])), 5)  # noqa: E731
        out[name] = {"val_missed": [r["val"]["phishing_missed"] for r in rs],
                     "val_missed_mean": mean(lambda r: r["val"]["phishing_missed"]),
                     "val_pr_auc": mean(lambda r: r["val"]["pr_auc"])}
        for s in common.EVAL_SETS:
            out[name].update({f"{s}_pr_auc": mean(lambda r: r[s]["pr_auc"]),
                              f"{s}_missed_mean": mean(lambda r: r[s]["alone"]["phishing_missed"]),
                              f"{s}_clean_recognized": mean(lambda r: r[s]["alone"]["clean_recognized"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dl", "ml", "ablation"))
    ap.add_argument("--archs", nargs="+", choices=tuple(DL_ARCHS), default=list(DL_ARCHS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--dl", choices=tuple(DL_ARCHS), default=None,
                    help="ml / ablation step: DL architecture (default: winner of the dl step)")
    ap.add_argument("--ml", choices=T.ML_MODELS, default=None,
                    help="ablation step: ML classifier (default: winner of the ml step)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="ablation step: one refit per seed and configuration")
    args = ap.parse_args()
    if args.step == "dl":
        return step_dl(args.archs, args.epochs)
    if args.step == "ml":
        return step_ml(args.dl)
    return step_ablation(args.dl, args.ml, args.seeds)


if __name__ == "__main__":
    sys.exit(main())
