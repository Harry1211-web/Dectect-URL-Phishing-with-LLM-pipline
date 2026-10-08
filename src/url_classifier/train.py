"""Train Layer C: char CNN/BiLSTM feature extractor + two XGBoost branches.

  lexical branch : CNN/BiLSTM embedding + hand-crafted URL features (no network needed)
  host branch    : the same + host features (DNS, TLS certificate, RDAP domain age, Tranco rank)
At inference the host branch is used when the host resolves (the normal case for a URL a user
clicks); the lexical branch is the fallback when lookups fail or time out.

Data: dataset/urls/urls_labeled.csv (build_dataset.py), split by site.
  train -> 70% of its sites train the CNN/BiLSTM, the other 30% train XGBoost on embeddings
           the CNN never saw; rows flagged by clean_labels.py are dropped first;
  val   -> early stopping + thresholds (recall on human-verified phishing: PhishTank/OpenPhish);
  test  -> final metrics only.
The host branch is trained and evaluated ONLY on live hosts (DNS resolves): most old phishing
domains in the feeds are dead now, so "does not resolve" would be a fake shortcut.
Host features must be collected first (collect_host_features.py); they are read from the cache.

Run:    python src/url_classifier/train.py [--reuse-cnn]
Output: models/url_classifier/{cnn.pt, xgb_lexical.json, xgb_host.json, config.json},
        reports/url_classifier/metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.features import FEATURE_NAMES, feature_matrix  # noqa: E402
from url_classifier.model import EMBED_DIM, CharCnnLstm, encode_urls  # noqa: E402
from url_utils import host_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "urls" / "urls_labeled.csv"
MODEL_DIR = ROOT / "models" / "url_classifier"
REPORT_DIR = ROOT / "reports" / "url_classifier"
RECALL_TARGETS = (0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 1.0)
NOISY_PATH = DATA_PATH.parent / "noisy_labels.csv"
# Human-verified phishing feeds: thresholds are chosen and recall is reported on these, because
# aggregated feeds contain mislabelled normal pages that would dictate a near-zero threshold
VERIFIED_SOURCES = ("phishtank", "openphish")
TRANCO_FEATURES = ("tranco_log_rank", "in_tranco")


def _cnn_part(group: str) -> bool:
    return int(hashlib.md5(f"cnn:{group}".encode()).hexdigest(), 16) % 100 < 70


def predict_logits(model: CharCnnLstm, x: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def embeddings(model: CharCnnLstm, x: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model.embed(torch.from_numpy(x[i:i + bs]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def train_cnn(x_tr, y_tr, x_va, y_va, device: str, epochs: int, seed: int) -> CharCnnLstm:
    torch.manual_seed(seed)
    model = CharCnnLstm().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(x_tr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            xb = torch.from_numpy(x_tr[b]).to(device)
            yb = torch.from_numpy(y_tr[b].astype(np.float32)).to(device)
            # Plain fp32: mixed precision (AMP) crashes the CUDA driver on the GTX 1650 used here
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        auc = roc_auc_score(y_va, predict_logits(model, x_va, device))
        print(f"  epoch {ep}: loss={tot / len(idx):.4f} val_auc={auc:.5f} ({time.time() - t0:.0f}s)",
              flush=True)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold t such that recall(score >= t) >= target."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos) + 1e-9))  # phishing allowed below the threshold
    return float(pos[min(k, len(pos) - 1)])


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    return {"threshold": round(t, 6), "recall": round(tp / max(1, tp + fn), 5),
            "fpr_clean_sent_to_llm": round(fp / max(1, fp + tn), 5),
            "clean_recognized": round(tn / max(1, fp + tn), 5),
            "precision": round(tp / max(1, tp + fp), 5), "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def evaluate(name: str, y_va, p_va, y_te, p_te, ver_va, ver_te) -> dict:
    """Thresholds from verified-phishing recall on val; test metrics on all rows and on the
    verified subset (verified phishing + all clean)."""
    ops = {}
    for tgt in RECALL_TARGETS:
        t = threshold_for_recall(y_va[ver_va], p_va[ver_va], tgt)
        ops[f"val_recall_{tgt}"] = {"test_verified": metrics_at(y_te[ver_te], p_te[ver_te], t),
                                    "test": metrics_at(y_te, p_te, t)}
    res = {"n_test": int(len(y_te)), "test_roc_auc": round(roc_auc_score(y_te, p_te), 5),
           "test_pr_auc": round(average_precision_score(y_te, p_te), 5), "operating_points": ops}
    print(f"  {name}: n={len(y_te)} AUC={res['test_roc_auc']}  " + "  ".join(
        f"R{k.split('_')[-1]}: ver_recall={v['test_verified']['recall']} "
        f"clean_ok={v['test']['clean_recognized']}" for k, v in ops.items()), flush=True)
    return res


VERIFIED_MISS_WEIGHT = 3.0  # a missed human-verified phishing costs 3, any other phishing 1
# Per-segment thresholds may spend only this fraction of the single-threshold miss cost on val:
# at 0.5 they recognise ~8 pt more clean URLs AND miss ~30 % fewer phishing on the test split
SEGMENT_BUDGET_FRACTION = 0.5


def segment_thresholds(seg: np.ndarray, score: np.ndarray, y: np.ndarray, verified: np.ndarray,
                       budget: float) -> tuple[dict, float]:
    """Per-segment thresholds maximising clean URLs recognised (score < t) on validation data,
    subject to a weighted phishing-miss budget. Lagrangian: each segment maximises
    clean_below(t) - lam * miss_cost_below(t); lam is binary-searched to meet the budget."""
    w = np.where(verified, VERIFIED_MISS_WEIGHT, 1.0) * (y == 1)
    curves = {}
    for s in np.unique(seg):
        m = seg == s
        sc, ww = score[m], w[m]
        t = np.unique(np.concatenate([[0.0], sc, [1.01]]))
        clean = np.sort(sc[y[m] == 0])
        o = np.argsort(sc)
        cost = np.concatenate([[0.0], np.cumsum(ww[o])])
        curves[s] = (t, np.searchsorted(clean, t, "left"), cost[np.searchsorted(sc[o], t, "left")])

    def pick(lam: float) -> tuple[dict, float]:
        thr, total = {}, 0.0
        for s, (t, c, cost) in curves.items():
            k = int(np.argmax(c - lam * cost))
            thr[s], total = float(t[k]), total + float(cost[k])
        return thr, total

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pick(mid)[1] <= budget else (mid, hi)
    return pick(hi)


def fit_xgb(xtr, ytr, xva, yva, seed: int) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(n_estimators=2000, max_depth=8, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
                          device="cpu",  # the installed xgboost build has no CUDA
                          eval_metric="auc", early_stopping_rounds=80, random_state=seed)
    m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cnn", action="store_true", help="load models/url_classifier/cnn.pt")
    ap.add_argument("--target-recall", type=float, default=0.998,
                    help="recall on verified validation phishing used to pick the thresholds")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(DATA_PATH)
    df["y"] = (df["label"] == "phishing").astype(np.int64)
    tr = df[df.split == "train"]
    n_noisy = 0
    if NOISY_PATH.exists():  # drop likely mislabelled rows found by clean_labels.py
        noisy = set(pd.read_csv(NOISY_PATH).url)
        n_noisy = int(tr.url.isin(noisy).sum())
        tr = tr[~tr.url.isin(noisy)]
    parts = {"cnn": tr[tr.group.map(_cnn_part)], "xgb": tr[~tr.group.map(_cnn_part)],
             "val": df[df.split == "val"], "test": df[df.split == "test"]}
    print({k: len(v) for k, v in parts.items()}, f"device={device}", flush=True)

    # ---------------------------------------------------------------- CNN/BiLSTM extractor
    enc = {k: encode_urls(d.url.tolist()) for k, d in parts.items()}
    y = {k: d.y.to_numpy() for k, d in parts.items()}
    t0 = time.time()
    if args.reuse_cnn:
        cnn = CharCnnLstm().to(device)
        cnn.load_state_dict(torch.load(MODEL_DIR / "cnn.pt", map_location=device, weights_only=True))
    else:
        print("Training CNN/BiLSTM extractor...", flush=True)
        cnn = train_cnn(enc["cnn"], y["cnn"], enc["val"], y["val"], device, args.epochs, args.seed)
    cnn_secs = time.time() - t0

    print("Computing URL features...", flush=True)
    sets = ("xgb", "val", "test")
    lex = {k: np.hstack([embeddings(cnn, enc[k], device), feature_matrix(parts[k].url.tolist())])
           for k in sets}
    ver = {k: ((parts[k].y == 0) | parts[k].source.isin(VERIFIED_SOURCES)).to_numpy() for k in sets}
    lex_names = [f"cnn_emb_{i}" for i in range(EMBED_DIM)] + FEATURE_NAMES

    # ---------------------------------------------------------------- lexical branch
    print("Training XGBoost lexical branch...", flush=True)
    t0 = time.time()
    clf_lex = fit_xgb(lex["xgb"], y["xgb"], lex["val"], y["val"], args.seed)
    xgb_secs = time.time() - t0
    p_lex = {k: clf_lex.predict_proba(lex[k])[:, 1] for k in sets}
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    results = {
        "cnn_bilstm_only": evaluate("cnn_only", y["val"], sig(predict_logits(cnn, enc["val"], device)),
                                    y["test"], sig(predict_logits(cnn, enc["test"], device)),
                                    ver["val"], ver["test"]),
        "lexical_branch": evaluate("lexical", y["val"], p_lex["val"], y["test"], p_lex["test"],
                                   ver["val"], ver["test"]),
    }
    thr_lex = threshold_for_recall(y["val"][ver["val"]], p_lex["val"][ver["val"]], args.target_recall)

    # ---------------------------------------------------------------- host branch (live hosts)
    print("Reading host features from cache...", flush=True)
    raw = {k: hf.raw_lookups(parts[k].url.tolist(), offline=True) for k in sets}
    live = {k: np.array([hf.is_live(raw[k][host_of(u)]) for u in parts[k].url]) for k in sets}
    hostx = {k: np.array([hf.host_feature_vector(u, raw[k][host_of(u)] or {})
                          for u in parts[k].url], dtype=np.float32) for k in sets}
    host_names = lex_names + hf.HOST_FEATURE_NAMES
    full = {k: np.hstack([lex[k], hostx[k]]) for k in sets}
    live_stats = {k: {"rows_with_host_data": int(sum(raw[k][host_of(u)] is not None
                                                     for u in parts[k].url)),
                      "live": int(live[k].sum()),
                      "live_by_label": parts[k][live[k]].label.value_counts().to_dict()}
                  for k in sets}
    print(live_stats, flush=True)

    L = live

    def live_eval(name: str, cols: np.ndarray) -> tuple[xgb.XGBClassifier, dict, np.ndarray]:
        m = fit_xgb(full["xgb"][L["xgb"]][:, cols], y["xgb"][L["xgb"]],
                    full["val"][L["val"]][:, cols], y["val"][L["val"]], args.seed)
        pv = m.predict_proba(full["val"][L["val"]][:, cols])[:, 1]
        pt = m.predict_proba(full["test"][L["test"]][:, cols])[:, 1]
        res = evaluate(name, y["val"][L["val"]], pv, y["test"][L["test"]], pt,
                       ver["val"][L["val"]], ver["test"][L["test"]])
        return m, res, pv

    print("Training XGBoost host branch (live hosts)...", flush=True)
    all_cols = np.arange(len(host_names))
    no_rank = np.array([i for i, n in enumerate(host_names) if n not in TRANCO_FEATURES])
    clf_host, results["host_branch_live"], pv_host = live_eval("host", all_cols)
    _, results["host_branch_live_without_tranco"], _ = live_eval("host_no_tranco", no_rank)
    # Lexical branch on the same live rows, for a like-for-like comparison
    Lv, Lt = L["val"], L["test"]
    results["lexical_branch_live"] = evaluate("lexical_live", y["val"][Lv], p_lex["val"][Lv],
                                              y["test"][Lt], p_lex["test"][Lt],
                                              ver["val"][Lv], ver["test"][Lt])
    thr_host = threshold_for_recall(y["val"][Lv][ver["val"][Lv]], pv_host[ver["val"][Lv]],
                                    args.target_recall)

    # ---------------------------------------------------------------- routed system on test
    p_host_te = clf_host.predict_proba(full["test"])[:, 1]
    use_host = Lt
    score = np.where(use_host, p_host_te, p_lex["test"])
    flag = np.where(use_host, p_host_te >= thr_host, p_lex["test"] >= thr_lex)
    te = parts["test"].assign(score=score, flag=flag, branch=np.where(use_host, "host", "lexical"))
    yt = y["test"]

    def routed(mask: np.ndarray) -> dict:
        ph, cl = (yt == 1) & mask, (yt == 0) & mask
        return {"n": int(mask.sum()), "phishing_missed": int((ph & ~flag).sum()),
                "recall": round(float((ph & flag).sum() / max(1, ph.sum())), 5),
                "clean_recognized": round(float((cl & ~flag).sum() / max(1, cl.sum())), 5)}

    routed_report = {"all": routed(np.ones(len(te), bool)), "verified": routed(ver["test"]),
                     "live_hosts": routed(use_host), "dead_hosts": routed(~use_host),
                     "by_source": {s: routed((te.source == s).to_numpy()) for s in te.source.unique()}}
    print(json.dumps(routed_report["all"]), json.dumps(routed_report["verified"]), flush=True)

    # ---------------------------------------------------------------- per-segment thresholds
    # Same weighted miss budget as the single thresholds above, spent where it buys the most
    # clean URLs (e.g. Tranco-ranked domains) instead of uniformly
    def segments(k: str, use: np.ndarray) -> np.ndarray:
        return np.array([hf.segment_of(u, raw[k][host_of(u)], bool(h))
                         for u, h in zip(parts[k].url, use)])

    p_host_va = clf_host.predict_proba(full["val"])[:, 1]
    sv = np.where(Lv, p_host_va, p_lex["val"])
    flag_v = np.where(Lv, p_host_va >= thr_host, p_lex["val"] >= thr_lex)
    wv = np.where(ver["val"], VERIFIED_MISS_WEIGHT, 1.0) * (y["val"] == 1)
    seg_thr, seg_cost = segment_thresholds(segments("val", Lv), sv, y["val"], ver["val"],
                                           SEGMENT_BUDGET_FRACTION * float(wv[~flag_v].sum()))
    seg_te = segments("test", use_host)
    flag = score >= np.array([seg_thr.get(s, 0.0) for s in seg_te])
    te = te.assign(flag=flag, segment=seg_te)
    routed_seg = {"all": routed(np.ones(len(te), bool)), "verified": routed(ver["test"]),
                  "by_segment": {s: routed(seg_te == s) for s in sorted(set(seg_te))}}
    print("segment thresholds:", json.dumps(routed_seg["all"]), json.dumps(routed_seg["verified"]),
          flush=True)

    # ---------------------------------------------------------------- save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.reuse_cnn:
        torch.save(cnn.state_dict(), MODEL_DIR / "cnn.pt")
    clf_lex.save_model(MODEL_DIR / "xgb_lexical.json")
    clf_host.save_model(MODEL_DIR / "xgb_host.json")
    config = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_recall_on_verified_val": args.target_recall,
        "threshold_lexical": thr_lex, "threshold_host": thr_host,
        "thresholds_by_val_recall": {
            "lexical": {str(t): threshold_for_recall(y["val"][ver["val"]], p_lex["val"][ver["val"]], t)
                        for t in RECALL_TARGETS},
            "host": {str(t): threshold_for_recall(y["val"][Lv][ver["val"][Lv]],
                                                  pv_host[ver["val"][Lv]], t) for t in RECALL_TARGETS}},
        "segment_thresholds": seg_thr, "segment_miss_cost_val": seg_cost,
        "lexical_feature_names": lex_names, "host_feature_names": host_names,
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (MODEL_DIR / "xgb.json").unlink(missing_ok=True)  # single-branch model of the previous version
    imp = sorted(zip(host_names, clf_host.feature_importances_.tolist()), key=lambda x: -x[1])
    report = {
        **{k: v for k, v in config.items() if not k.endswith("feature_names")},
        "data": {**{k: len(v) for k, v in parts.items()}, "dropped_noisy": n_noisy,
                 "host_data": live_stats},
        "train_seconds": {"cnn": round(cnn_secs), "xgb_lexical": round(xgb_secs)},
        "results": results, "routed_system_test_single_threshold": routed_report,
        "routed_system_test": routed_seg,
        "test_missed_phishing_examples": te[(yt == 1) & ~flag].sort_values("score").head(30)[
            ["url", "source", "branch", "segment", "score"]].to_dict("records"),
        "top_features_host_branch": imp[:30],
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
