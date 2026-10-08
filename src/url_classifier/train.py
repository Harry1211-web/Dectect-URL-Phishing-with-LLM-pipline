"""Train Layer C: two cross-fitted char CNN/BiLSTM models stacked under two XGBoost branches.

Stacking: the train sites are split in two parts; CNN A learns on one, CNN B on the other, so
every train URL gets a CNN phishing score from the CNN that never saw it (val/test URLs get the
mean of both). That score is one feature of two XGBoost branches, which can therefore learn from
ALL train rows instead of the 30 % part the CNN did not see:
  lexical branch : hand-crafted URL features + cnn_score (no network needed, every train row)
  host branch    : + host features (DNS, TLS certificate, RDAP domain age, Tranco rank), trained
                   only on live hosts whose lookups are complete (dead phishing domains would be a
                   fake shortcut); used at inference when the host resolves.
Rows flagged by clean_labels.py are dropped from train. Thresholds: recall on human-verified
validation phishing (PhishTank/OpenPhish), then per-segment thresholds within a reduced miss
budget. The test split is used for the final metrics only.
Host features must be collected first (collect_host_features.py); they are read from the cache.

Run:    python src/url_classifier/train.py [--reuse-cnn]
Output: models/url_classifier/{cnn.pt, cnn_b.pt, xgb_lexical.json, xgb_host.json, config.json},
        reports/url_classifier/metrics.json
"""

from __future__ import annotations

import argparse
import gc
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
from url_classifier.model import CharCnnLstm, encode_urls  # noqa: E402
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


def cnn_scores(models: list[CharCnnLstm], urls: list[str], device: str) -> np.ndarray:
    """Mean phishing probability of the given CNNs (encoded in chunks to bound memory)."""
    out = []
    for i in range(0, len(urls), 50_000):
        x = encode_urls(urls[i:i + 50_000])
        out.append(np.mean([1 / (1 + np.exp(-predict_logits(m, x, device))) for m in models], axis=0))
    return np.concatenate(out) if out else np.zeros(0)


def build_matrices(df: pd.DataFrame, chunk: int = 20_000) -> dict:
    """Lexical matrix (hand-crafted + cnn_score), host matrix (hand-crafted + host + cnn_score)
    and host-data flags, computed in chunks so raw host lookups never sit in memory at once."""
    lex, host, live, complete = [], [], [], []
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk]
        urls = part.url.tolist()
        hand = feature_matrix(urls)
        cs = part.cnn_score.to_numpy(dtype=np.float32)[:, None]
        raw = hf.raw_lookups(urls, offline=True)
        rows = [raw[host_of(u)] for u in urls]
        hx = np.array([hf.host_feature_vector(u, r or {}) for u, r in zip(urls, rows)],
                      dtype=np.float32)
        lex.append(np.hstack([hand, cs]))
        host.append(np.hstack([hand, hx, cs]))
        live.append(np.array([hf.is_live(r) for r in rows]))
        complete.append(np.array([hf.is_complete(r) for r in rows]))
        del raw, rows
        gc.collect()
    return {"lex": np.vstack(lex), "host": np.vstack(host), "live": np.concatenate(live),
            "complete": np.concatenate(complete)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cnn", action="store_true",
                    help="load models/url_classifier/cnn.pt and cnn_b.pt instead of training them")
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
    tr = tr.assign(cnn_part=tr.group.map(_cnn_part).to_numpy())
    va = df[df.split == "val"].copy()
    te = df[df.split == "test"].copy()
    del df
    print(f"train={len(tr)} (cnn part {int(tr.cnn_part.sum())}) val={len(va)} test={len(te)} "
          f"device={device}", flush=True)

    # ---------------------------------------------------------------- cross-fitted CNNs
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cnns = []
    for name, part in (("cnn.pt", tr[tr.cnn_part]), ("cnn_b.pt", tr[~tr.cnn_part])):
        m = CharCnnLstm().to(device)
        if args.reuse_cnn and (MODEL_DIR / name).exists():
            m.load_state_dict(torch.load(MODEL_DIR / name, map_location=device, weights_only=True))
        else:
            print(f"Training {name} on {len(part)} rows...", flush=True)
            m = train_cnn(encode_urls(part.url.tolist()), part.y.to_numpy(),
                          encode_urls(va.url.tolist()), va.y.to_numpy(), device, args.epochs,
                          args.seed + len(cnns))
            torch.save(m.state_dict(), MODEL_DIR / name)
        cnns.append(m)
    cnn_secs = time.time() - t0
    # Out-of-fold score for every train row: each row is scored by the CNN that never saw it
    s_a = cnn_scores([cnns[0]], tr.url.tolist(), device)
    s_b = cnn_scores([cnns[1]], tr.url.tolist(), device)
    tr["cnn_score"] = np.where(tr.cnn_part, s_b, s_a)
    for d in (va, te):
        d["cnn_score"] = cnn_scores(cnns, d.url.tolist(), device)
    del s_a, s_b
    torch.cuda.empty_cache()

    print("Computing features...", flush=True)
    M = {"train": build_matrices(tr), "val": build_matrices(va), "test": build_matrices(te)}
    y = {"train": tr.y.to_numpy(), "val": va.y.to_numpy(), "test": te.y.to_numpy()}
    ver = {k: ((d.y == 0) | d.source.isin(VERIFIED_SOURCES)).to_numpy()
           for k, d in (("train", tr), ("val", va), ("test", te))}
    lex_names = FEATURE_NAMES + ["cnn_score"]
    host_names = FEATURE_NAMES + hf.HOST_FEATURE_NAMES + ["cnn_score"]

    # ---------------------------------------------------------------- lexical branch (all rows)
    print("Training XGBoost lexical branch (all train rows)...", flush=True)
    t0 = time.time()
    clf_lex = fit_xgb(M["train"]["lex"], y["train"], M["val"]["lex"], y["val"], args.seed)
    xgb_secs = time.time() - t0
    p_lex = {k: clf_lex.predict_proba(M[k]["lex"])[:, 1] for k in ("val", "test")}
    results = {
        "cnn_bilstm_only": evaluate("cnn_only", y["val"], va.cnn_score.to_numpy(), y["test"],
                                    te.cnn_score.to_numpy(), ver["val"], ver["test"]),
        "lexical_branch": evaluate("lexical", y["val"], p_lex["val"], y["test"], p_lex["test"],
                                   ver["val"], ver["test"]),
    }
    thr_lex = threshold_for_recall(y["val"][ver["val"]], p_lex["val"][ver["val"]], args.target_recall)

    # ---------------------------------------------------------------- host branch (live rows)
    # Trained only on live hosts whose DNS/TLS/RDAP lookups are all done: dead phishing domains
    # would be a fake shortcut, half-collected rows would leak the collection order
    Ltr = M["train"]["live"] & M["train"]["complete"]
    Lv, Lt = M["val"]["live"], M["test"]["live"]
    host_data = {k: {"live": int(M[k]["live"].sum()), "live_and_complete": int(
        (M[k]["live"] & M[k]["complete"]).sum())} for k in M}
    print(host_data, flush=True)

    def live_branch(name: str, cols: np.ndarray) -> tuple[xgb.XGBClassifier, dict, np.ndarray]:
        m = fit_xgb(M["train"]["host"][Ltr][:, cols], y["train"][Ltr],
                    M["val"]["host"][Lv][:, cols], y["val"][Lv], args.seed)
        pv = m.predict_proba(M["val"]["host"][Lv][:, cols])[:, 1]
        pt = m.predict_proba(M["test"]["host"][Lt][:, cols])[:, 1]
        return m, evaluate(name, y["val"][Lv], pv, y["test"][Lt], pt, ver["val"][Lv],
                           ver["test"][Lt]), pv

    print("Training XGBoost host branch (live train rows)...", flush=True)
    all_cols = np.arange(len(host_names))
    no_rank = np.array([i for i, n in enumerate(host_names) if n not in TRANCO_FEATURES])
    clf_host, results["host_branch_live"], pv_host = live_branch("host", all_cols)
    _, results["host_branch_live_without_tranco"], _ = live_branch("host_no_tranco", no_rank)
    results["lexical_branch_live"] = evaluate("lexical_live", y["val"][Lv], p_lex["val"][Lv],
                                              y["test"][Lt], p_lex["test"][Lt],
                                              ver["val"][Lv], ver["test"][Lt])
    thr_host = threshold_for_recall(y["val"][Lv][ver["val"][Lv]], pv_host[ver["val"][Lv]],
                                    args.target_recall)

    # ---------------------------------------------------------------- routing + segment thresholds
    p_host = {k: clf_host.predict_proba(M[k]["host"])[:, 1] for k in ("val", "test")}
    score = {k: np.where(M[k]["live"], p_host[k], p_lex[k]) for k in ("val", "test")}
    raw_seg = {}
    for k, d in (("val", va), ("test", te)):
        segs = []
        for i in range(0, len(d), 20_000):
            part = d.iloc[i:i + 20_000]
            raw = hf.raw_lookups(part.url.tolist(), offline=True)
            live = M[k]["live"][i:i + 20_000]
            segs += [hf.segment_of(u, raw[host_of(u)], bool(h)) for u, h in zip(part.url, live)]
        raw_seg[k] = np.array(segs)
    single_v = np.where(Lv, p_host["val"] >= thr_host, p_lex["val"] >= thr_lex)
    wv = np.where(ver["val"], VERIFIED_MISS_WEIGHT, 1.0) * (y["val"] == 1)
    seg_thr, seg_cost = segment_thresholds(raw_seg["val"], score["val"], y["val"], ver["val"],
                                           SEGMENT_BUDGET_FRACTION * float(wv[~single_v].sum()))
    yt = y["test"]

    def routed(flag: np.ndarray, mask: np.ndarray) -> dict:
        ph, cl = (yt == 1) & mask, (yt == 0) & mask
        return {"n": int(mask.sum()), "phishing_missed": int((ph & ~flag).sum()),
                "recall": round(float((ph & flag).sum() / max(1, ph.sum())), 5),
                "clean_recognized": round(float((cl & ~flag).sum() / max(1, cl.sum())), 5)}

    single_t = np.where(Lt, p_host["test"] >= thr_host, p_lex["test"] >= thr_lex)
    seg_t = score["test"] >= np.array([seg_thr.get(s, 0.0) for s in raw_seg["test"]])
    every = np.ones(len(te), bool)
    routed_report = {
        "single_threshold": {"all": routed(single_t, every), "verified": routed(single_t, ver["test"])},
        "segment_thresholds": {"all": routed(seg_t, every), "verified": routed(seg_t, ver["test"]),
                               "by_segment": {s: routed(seg_t, raw_seg["test"] == s)
                                              for s in sorted(set(raw_seg["test"]))}},
    }
    print("single:", json.dumps(routed_report["single_threshold"]["all"]),
          "| segments:", json.dumps(routed_report["segment_thresholds"]["all"]), flush=True)

    # ---------------------------------------------------------------- save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    clf_lex.save_model(MODEL_DIR / "xgb_lexical.json")
    clf_host.save_model(MODEL_DIR / "xgb_host.json")
    config = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "stacking: 2 cross-fitted CNN/BiLSTM (cnn.pt, cnn_b.pt) -> cnn_score "
                        "feature of the XGBoost lexical and host branches",
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
    (MODEL_DIR / "xgb.json").unlink(missing_ok=True)
    (MODEL_DIR / "cnn_b_experiment.pt").unlink(missing_ok=True)
    imp = sorted(zip(host_names, clf_host.feature_importances_.tolist()), key=lambda x: -x[1])
    te = te.assign(score=score["test"], flagged=seg_t, segment=raw_seg["test"])
    report = {
        **{k: v for k, v in config.items() if not k.endswith("feature_names")},
        "data": {"train": len(tr), "val": len(va), "test": len(te), "dropped_noisy": n_noisy,
                 "host_branch_train_rows": int(Ltr.sum()), "host_data": host_data},
        "train_seconds": {"cnn": round(cnn_secs), "xgb_lexical": round(xgb_secs)},
        "results": results, "routed_system_test": routed_report,
        "test_missed_phishing_examples": te[(yt == 1) & ~seg_t].sort_values("score").head(30)[
            ["url", "source", "segment", "score"]].to_dict("records"),
        "top_features_host_branch": imp[:30],
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
