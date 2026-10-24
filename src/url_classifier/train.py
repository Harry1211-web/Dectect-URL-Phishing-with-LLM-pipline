"""Train Layer C: two cross-fitted deep URL models (DL) stacked under two ML branches.

DL = one architecture of url_classifier/model.py (--dl), ML = XGBoost, LightGBM or random
forest (--ml); the pair is chosen on validation data by src/layer_c_selection.py.
Stacking: the train sites are split in two parts; DL model A learns on one, B on the other, so
every train URL gets a DL phishing score from the model that never saw it (val/test URLs get the
mean of both). That score is one feature of two ML branches, which can therefore learn from
ALL train rows instead of the 30 % part the DL model did not see:
  lexical branch : hand-crafted URL features + dl_score (no network needed, every train row)
  host branch    : + host features (DNS, TLS certificate, RDAP domain age, Tranco rank), trained
                   only on live hosts whose lookups are complete (dead phishing domains would be a
                   fake shortcut); used at inference when the host resolves.
--host-lookups none: the host branch only gets the Tranco features (local file, no network
lookup), is trained on every train row and scores every URL; segments then use no domain age.
Rows flagged by clean_labels.py are dropped from train. Thresholds: recall on human-verified
validation phishing (PhishTank/OpenPhish), then per-segment thresholds that miss the fewest
validation phishing while still recognising MIN_CLEAN_RECOGNIZED of the clean URLs. URLs that
Layer B flags go to Layer D anyway, so they are left out of that choice.
The test split is used for the final metrics only.
Host features must be collected first (collect_host_features.py); they are read from the cache.

Run:    python src/url_classifier/train.py --dl <arch> --ml <xgboost|lightgbm|random_forest>
                                         [--dl-dir models/layer_c_selection]  (reuse DL models)
                                         [--host-lookups none|network]
        python src/url_classifier/train.py --thresholds-only   (refit the segment thresholds of
                                                                 the saved models, no training)
Output: models/url_classifier/{dl_a.pt, dl_b.pt, ml_lexical.*, ml_host.*, config.json},
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
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.features import FEATURE_NAMES, feature_matrix  # noqa: E402
from url_classifier.model import DL_ARCHS, encode_inputs  # noqa: E402
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


def _cnn_part(group: str) -> bool:
    return int(hashlib.md5(f"cnn:{group}".encode()).hexdigest(), 16) % 100 < 70


def dl_split(tr: pd.DataFrame) -> np.ndarray:
    """True = train row of DL model A (70 % of the sites), False = model B."""
    return tr.group.map(_cnn_part).to_numpy()


def predict_logits(model: torch.nn.Module, x: tuple[np.ndarray, np.ndarray], device: str,
                   bs: int = 512) -> np.ndarray:
    model.eval()
    c, w = x
    out = []
    with torch.no_grad():
        for i in range(0, len(c), bs):
            out.append(model(torch.from_numpy(c[i:i + bs]).to(device),
                             torch.from_numpy(w[i:i + bs]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


DL_PATIENCE = 3  # epochs without val ROC-AUC gain before early stopping


def train_dl(arch: str, x_tr: tuple, y_tr: np.ndarray, x_va: tuple, y_va: np.ndarray, device: str,
             epochs: int, seed: int) -> tuple[torch.nn.Module, list]:
    """One DL model: class-weighted BCE, AdamW, early stopping on val ROC-AUC -> (model, log)."""
    torch.manual_seed(seed)
    model = DL_ARCHS[arch]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4 if arch == "transformer" else 2e-3,
                            weight_decay=1e-4)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc, best_state, bad, log = -1.0, None, 0, []
    rng = np.random.default_rng(seed)
    c_tr, w_tr = x_tr
    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(c_tr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            yb = torch.from_numpy(y_tr[b].astype(np.float32)).to(device)
            # Plain fp32: mixed precision (AMP) crashes the CUDA driver on the GTX 1650 used here
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(c_tr[b]).to(device),
                                 torch.from_numpy(w_tr[b]).to(device)), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        auc = roc_auc_score(y_va, predict_logits(model, x_va, device))
        log.append({"epoch": ep, "val_auc": round(float(auc), 5), "seconds": round(time.time() - t0)})
        print(f"  {arch} epoch {ep}: loss={tot / len(idx):.4f} val_auc={auc:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= DL_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, log


def dl_scores(models: list[torch.nn.Module], arch: str, urls: list[str], device: str) -> np.ndarray:
    """Mean phishing probability of the given DL models (encoded in chunks to bound memory)."""
    out = []
    for i in range(0, len(urls), 50_000):
        x = encode_inputs(arch, urls[i:i + 50_000])
        out.append(np.mean([1 / (1 + np.exp(-predict_logits(m, x, device))) for m in models], axis=0))
    return np.concatenate(out) if out else np.zeros(0)


ML_MODELS = ("xgboost", "lightgbm", "random_forest")
ML_EXT = {"xgboost": "json", "lightgbm": "txt", "random_forest": "joblib"}


def fit_ml(name: str, xtr, ytr, xva, yva, seed: int):
    """One ML branch; the boosted trees stop early on val ROC-AUC. Missing values stay NaN: all
    three models handle them natively."""
    if name == "xgboost":
        m = xgb.XGBClassifier(n_estimators=2000, max_depth=8, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
                              device="cpu",  # the installed xgboost build has no CUDA
                              eval_metric="auc", early_stopping_rounds=80, random_state=seed)
        m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    elif name == "lightgbm":
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=127,
                               subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, verbose=-1)
        m.fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                   random_state=seed)
        m.fit(xtr, ytr)
    else:
        raise ValueError(name)
    return m


def save_ml(name: str, m, path: Path) -> None:
    if name == "xgboost":
        m.save_model(path)
    elif name == "lightgbm":
        m.booster_.save_model(str(path))
    else:
        joblib.dump(m, path, compress=3)


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
# Missing phishing matters more than LLM load: segment thresholds minimise the weighted miss cost
# on val as long as at least this share of ALL clean val URLs is still recognised as clean
MIN_CLEAN_RECOGNIZED = 0.50
# No segment may be fail-open: a segment with no validation phishing would otherwise get a
# "never flag" threshold (1.01); a score this high always means suspicious
MAX_SEGMENT_THRESHOLD = 0.5


def segment_thresholds(seg: np.ndarray, score: np.ndarray, y: np.ndarray, verified: np.ndarray,
                       n_clean: int, min_clean: float = MIN_CLEAN_RECOGNIZED) -> tuple[dict, float]:
    """Per-segment thresholds with the lowest weighted phishing-miss cost on validation data
    such that >= min_clean * n_clean clean URLs are recognised (score < t). Lagrangian: each
    segment maximises clean_below(t) - lam * miss_cost_below(t); lam is binary-searched to the
    largest value that still meets the clean floor. Returns (thresholds, miss cost)."""
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

    def pick(lam: float) -> tuple[dict, float, int]:
        thr, total, clean = {}, 0.0, 0
        for s, (t, c, cost) in curves.items():
            k = int(np.argmax(c - lam * cost))
            thr[s], total, clean = float(t[k]), total + float(cost[k]), clean + int(c[k])
        return thr, total, clean

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pick(mid)[2] >= min_clean * n_clean else (lo, mid)
    thr, cost, _ = pick(lo)
    return {s: min(t, MAX_SEGMENT_THRESHOLD) for s, t in thr.items()}, cost


def flagged_by_layer_b(urls: list[str]) -> np.ndarray:
    """URLs Layer B flags: they go to Layer D whatever Layer C says."""
    return np.array([check_typosquatting_deep(u)["flag"] is True for u in urls])


def fit_segment_thresholds(urls: list[str], seg: np.ndarray, score: np.ndarray, y: np.ndarray,
                           verified: np.ndarray) -> tuple[dict, float]:
    """Segment thresholds on the validation URLs that only Layer C decides (not flagged by B)."""
    m = ~flagged_by_layer_b(urls)
    return segment_thresholds(seg[m], score[m], y[m], verified[m], int((y == 0).sum()))


def routed_counts(y: np.ndarray, flag: np.ndarray, mask: np.ndarray) -> dict:
    """Layer C alone on the rows in `mask`: flag = suspicious."""
    ph, cl = (y == 1) & mask, (y == 0) & mask
    return {"n": int(mask.sum()), "phishing_missed": int((ph & ~flag).sum()),
            "recall": round(float((ph & flag).sum() / max(1, ph.sum())), 5),
            "clean_recognized": round(float((cl & ~flag).sum() / max(1, cl.sum())), 5)}


def host_feature_names(network: bool) -> list[str]:
    return hf.HOST_FEATURE_NAMES if network else hf.OFFLINE_FEATURE_NAMES


def raw_rows(urls: list[str], network: bool) -> list:
    """Cached raw host lookups of `urls`, or empty stand-ins when the model uses no network data."""
    if not network:
        return [hf.NO_LOOKUP] * len(urls)
    raw = hf.raw_lookups(urls, offline=True)
    return [raw[host_of(u)] for u in urls]


def _score_split(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(score, segment) of every URL with the saved models and cached host data."""
    from url_classifier import predict
    urls = d.url.tolist()
    score, seg = [], []
    for i in range(0, len(urls), 4096):
        part = urls[i:i + 4096]
        s, _, used = predict.score_urls(part, host_mode="cache")
        rows = raw_rows(part, predict.uses_network())
        score.append(s)
        seg += [hf.segment_of(u, r, bool(h)) for u, r, h in zip(part, rows, used)]
    return np.concatenate(score), np.array(seg)


def thresholds_only() -> int:
    """Refit the segment thresholds of the saved models on validation data (no training) and
    update config.json plus the segment-threshold parts of the report."""
    df = pd.read_csv(DATA_PATH)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    y = {k: (d.label == "phishing").to_numpy().astype(np.int64) for k, d in (("val", va), ("test", te))}
    ver = {k: (y[k] == 0) | d.source.isin(VERIFIED_SOURCES).to_numpy()
           for k, d in (("val", va), ("test", te))}
    score_v, seg_v = _score_split(va)
    seg_thr, seg_cost = fit_segment_thresholds(va.url.tolist(), seg_v, score_v, y["val"], ver["val"])
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    cfg.update({"segment_thresholds": seg_thr, "segment_miss_cost_val": seg_cost,
                "segment_min_clean_recognized_val": MIN_CLEAN_RECOGNIZED})
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("segment thresholds:", json.dumps(seg_thr), "| val miss cost:", seg_cost, flush=True)

    report_path = REPORT_DIR / "metrics.json"
    if report_path.exists():
        score_t, seg_t = _score_split(te)
        flag = score_t >= np.array([seg_thr.get(s, 0.0) for s in seg_t])
        every = np.ones(len(te), bool)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update({k: cfg[k] for k in ("segment_thresholds", "segment_miss_cost_val",
                                            "segment_min_clean_recognized_val")})
        report["routed_system_test"]["segment_thresholds"] = {
            "all": routed_counts(y["test"], flag, every),
            "verified": routed_counts(y["test"], flag, ver["test"]),
            "by_segment": {s: routed_counts(y["test"], flag, seg_t == s) for s in sorted(set(seg_t))}}
        missed = te.assign(score=score_t, segment=seg_t)[(y["test"] == 1) & ~flag]
        report["test_missed_phishing_examples"] = missed.sort_values("score").head(30)[
            ["url", "source", "segment", "score"]].to_dict("records")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("test (Layer C alone):", json.dumps(report["routed_system_test"]["segment_thresholds"]["all"]))
    return 0


def base_matrices(df: pd.DataFrame, chunk: int = 20_000, network: bool = True) -> dict:
    """Hand-crafted matrix, host matrix and host-data flags (no DL score), computed in chunks so
    raw host lookups never sit in memory at once. network=False: Tranco columns only and every
    row counts as live (the host branch scores every URL)."""
    cols = [hf.HOST_FEATURE_NAMES.index(n) for n in host_feature_names(network)]
    hand, host, live, complete = [], [], [], []
    for i in range(0, len(df), chunk):
        urls = df.url.iloc[i:i + chunk].tolist()
        hand.append(feature_matrix(urls))
        rows = raw_rows(urls, network)
        host.append(np.array([hf.host_feature_vector(u, r or {}) for u, r in zip(urls, rows)],
                             dtype=np.float32)[:, cols])
        live.append(np.array([hf.is_live(r) for r in rows]) if network else np.ones(len(urls), bool))
        complete.append(np.array([hf.is_complete(r) for r in rows]))
        del rows
        gc.collect()
    return {"hand": np.vstack(hand), "hostx": np.vstack(host), "live": np.concatenate(live),
            "complete": np.concatenate(complete)}


def with_dl_score(base: dict, dl_score: np.ndarray) -> dict:
    """Lexical matrix (hand-crafted + dl_score) and host matrix (hand-crafted + host + dl_score)."""
    cs = dl_score.astype(np.float32)[:, None]
    return {"lex": np.hstack([base["hand"], cs]), "host": np.hstack([base["hand"], base["hostx"], cs]),
            "live": base["live"], "complete": base["complete"]}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """(train without noisy rows, val, test, number of dropped noisy rows); y = 1 for phishing."""
    df = pd.read_csv(DATA_PATH)
    df["y"] = (df["label"] == "phishing").astype(np.int64)
    tr = df[df.split == "train"]
    n_noisy = 0
    if NOISY_PATH.exists():  # drop likely mislabelled rows found by clean_labels.py
        noisy = set(pd.read_csv(NOISY_PATH).url)
        n_noisy = int(tr.url.isin(noisy).sum())
        tr = tr[~tr.url.isin(noisy)]
    tr = tr.assign(dl_part=dl_split(tr)).reset_index(drop=True)
    return (tr, df[df.split == "val"].reset_index(drop=True),
            df[df.split == "test"].reset_index(drop=True), n_noisy)


def cross_fit_dl(arch: str, tr: pd.DataFrame, va: pd.DataFrame, device: str, epochs: int,
                 seed: int, dl_dir: Path | None = None, save_dir: Path | None = None
                 ) -> tuple[list[torch.nn.Module], np.ndarray, dict]:
    """DL models A (on dl_part) and B (on the rest); returns (models, out-of-fold train score,
    training log). With dl_dir, saved <arch>_a.pt / <arch>_b.pt are loaded instead of trained."""
    models, log = [], {}
    x_va = encode_inputs(arch, va.url.tolist())
    for i, (tag, part) in enumerate((("a", tr[tr.dl_part]), ("b", tr[~tr.dl_part]))):
        path = (dl_dir / f"{arch}_{tag}.pt") if dl_dir else None
        if path is not None and path.exists():
            m = DL_ARCHS[arch]().to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            log[tag] = "loaded"
        else:
            print(f"Training {arch} {tag} on {len(part)} rows...", flush=True)
            t0 = time.time()
            m, ep = train_dl(arch, encode_inputs(arch, part.url.tolist()), part.y.to_numpy(),
                             x_va, va.y.to_numpy(), device, epochs, seed + i)
            log[tag] = {"epochs": ep, "seconds": round(time.time() - t0)}
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(m.state_dict(), save_dir / f"{arch}_{tag}.pt")
        models.append(m)
    # Out-of-fold score for every train row: each row is scored by the model that never saw it
    urls = tr.url.tolist()
    oof = np.where(tr.dl_part, dl_scores([models[1]], arch, urls, device),
                   dl_scores([models[0]], arch, urls, device))
    return models, oof, log


def fit_branches(ml: str, M: dict, y: dict, seed: int) -> dict:
    """Lexical branch on every train row, host branch on live hosts with complete lookups (dead
    phishing domains would be a fake shortcut, half-collected rows would leak the collection
    order). Returns the models and the routed score of every scored split."""
    t0 = time.time()
    lex = fit_ml(ml, M["train"]["lex"], y["train"], M["val"]["lex"], y["val"], seed)
    lex_secs = time.time() - t0
    Ltr, Lv = M["train"]["live"] & M["train"]["complete"], M["val"]["live"]
    t0 = time.time()
    host = fit_ml(ml, M["train"]["host"][Ltr], y["train"][Ltr], M["val"]["host"][Lv],
                  y["val"][Lv], seed)
    out = {"lex": lex, "host": host, "train_seconds": {"lexical": round(lex_secs),
                                                       "host": round(time.time() - t0)},
           "p_lex": {}, "p_host": {}, "score": {}}
    for k in M:
        if k == "train":
            continue
        out["p_lex"][k] = lex.predict_proba(M[k]["lex"])[:, 1]
        out["p_host"][k] = host.predict_proba(M[k]["host"])[:, 1]
        out["score"][k] = np.where(M[k]["live"], out["p_host"][k], out["p_lex"][k])
    return out


def segments(d: pd.DataFrame, live: np.ndarray, network: bool = True) -> np.ndarray:
    segs = []
    for i in range(0, len(d), 20_000):
        part = d.url.iloc[i:i + 20_000].tolist()
        segs += [hf.segment_of(u, r, bool(h))
                 for u, r, h in zip(part, raw_rows(part, network), live[i:i + 20_000])]
    return np.array(segs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dl", choices=tuple(DL_ARCHS), default="cnn_bilstm")
    ap.add_argument("--ml", choices=ML_MODELS, default="xgboost")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dl-dir", type=Path, default=None,
                    help="load <arch>_a.pt / <arch>_b.pt from this folder instead of training")
    ap.add_argument("--target-recall", type=float, default=0.998,
                    help="recall on verified validation phishing used to pick the thresholds")
    ap.add_argument("--host-lookups", choices=("none", "network"), default="none",
                    help="network: also DNS / TLS / RDAP features (lookups at inference, kept for experiments)")
    ap.add_argument("--thresholds-only", action="store_true",
                    help="only refit the segment thresholds of the saved models")
    args = ap.parse_args()
    if args.thresholds_only:
        return thresholds_only()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = args.host_lookups == "network"

    tr, va, te, n_noisy = load_splits()
    print(f"train={len(tr)} (DL part a {int(tr.dl_part.sum())}) val={len(va)} test={len(te)} "
          f"dl={args.dl} ml={args.ml} device={device}", flush=True)

    # ---------------------------------------------------------------- cross-fitted DL models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dls, oof, dl_log = cross_fit_dl(args.dl, tr, va, device, args.epochs, args.seed, args.dl_dir)
    dl_secs = time.time() - t0
    for m, name in zip(dls, ("dl_a.pt", "dl_b.pt")):
        torch.save(m.state_dict(), MODEL_DIR / name)
    tr["dl_score"] = oof
    for d in (va, te):
        d["dl_score"] = dl_scores(dls, args.dl, d.url.tolist(), device)
    torch.cuda.empty_cache()

    print("Computing features...", flush=True)
    M = {k: with_dl_score(base_matrices(d, network=network), d.dl_score.to_numpy())
         for k, d in (("train", tr), ("val", va), ("test", te))}
    y = {"train": tr.y.to_numpy(), "val": va.y.to_numpy(), "test": te.y.to_numpy()}
    ver = {k: ((d.y == 0) | d.source.isin(VERIFIED_SOURCES)).to_numpy()
           for k, d in (("train", tr), ("val", va), ("test", te))}
    lex_names = FEATURE_NAMES + ["dl_score"]
    host_names = FEATURE_NAMES + host_feature_names(network) + ["dl_score"]

    print(f"Training {args.ml} branches...", flush=True)
    B = fit_branches(args.ml, M, y, args.seed)
    Ltr, Lv, Lt = M["train"]["live"] & M["train"]["complete"], M["val"]["live"], M["test"]["live"]
    host_data = {k: {"live": int(M[k]["live"].sum()), "live_and_complete": int(
        (M[k]["live"] & M[k]["complete"]).sum())} for k in M}
    p_lex, p_host = B["p_lex"], B["p_host"]
    results = {
        "dl_only": evaluate("dl_only", y["val"], va.dl_score.to_numpy(), y["test"],
                            te.dl_score.to_numpy(), ver["val"], ver["test"]),
        "lexical_branch": evaluate("lexical", y["val"], p_lex["val"], y["test"], p_lex["test"],
                                   ver["val"], ver["test"]),
        "host_branch_live": evaluate("host", y["val"][Lv], p_host["val"][Lv], y["test"][Lt],
                                     p_host["test"][Lt], ver["val"][Lv], ver["test"][Lt]),
        "lexical_branch_live": evaluate("lexical_live", y["val"][Lv], p_lex["val"][Lv],
                                        y["test"][Lt], p_lex["test"][Lt], ver["val"][Lv],
                                        ver["test"][Lt]),
    }
    thr_lex = threshold_for_recall(y["val"][ver["val"]], p_lex["val"][ver["val"]], args.target_recall)
    thr_host = threshold_for_recall(y["val"][Lv][ver["val"][Lv]], p_host["val"][Lv][ver["val"][Lv]],
                                    args.target_recall)

    # ---------------------------------------------------------------- routing + segment thresholds
    score = B["score"]
    seg = {"val": segments(va, Lv, network), "test": segments(te, Lt, network)}
    seg_thr, seg_cost = fit_segment_thresholds(va.url.tolist(), seg["val"], score["val"],
                                               y["val"], ver["val"])
    yt = y["test"]
    single_t = np.where(Lt, p_host["test"] >= thr_host, p_lex["test"] >= thr_lex)
    seg_t = score["test"] >= np.array([seg_thr.get(s, 0.0) for s in seg["test"]])
    seg_v = score["val"] >= np.array([seg_thr.get(s, 0.0) for s in seg["val"]])
    every = np.ones(len(te), bool)
    routed_report = {
        "segment_thresholds_val": routed_counts(y["val"], seg_v, np.ones(len(va), bool)),
        "single_threshold": {"all": routed_counts(yt, single_t, every),
                             "verified": routed_counts(yt, single_t, ver["test"])},
        "segment_thresholds": {"all": routed_counts(yt, seg_t, every),
                               "verified": routed_counts(yt, seg_t, ver["test"]),
                               "by_segment": {s: routed_counts(yt, seg_t, seg["test"] == s)
                                              for s in sorted(set(seg["test"]))}},
    }
    print("single:", json.dumps(routed_report["single_threshold"]["all"]),
          "| segments:", json.dumps(routed_report["segment_thresholds"]["all"]), flush=True)

    # ---------------------------------------------------------------- save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ext = ML_EXT[args.ml]
    save_ml(args.ml, B["lex"], MODEL_DIR / f"ml_lexical.{ext}")
    save_ml(args.ml, B["host"], MODEL_DIR / f"ml_host.{ext}")
    for old in ("cnn.pt", "cnn_b.pt", "xgb_lexical.json", "xgb_host.json", "xgb.json",
                "cnn_b_experiment.pt", *(f"ml_{b}.{e}" for b in ("lexical", "host")
                                         for e in ML_EXT.values() if e != ext)):
        (MODEL_DIR / old).unlink(missing_ok=True)
    config = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dl_arch": args.dl, "ml": args.ml, "host_lookups": args.host_lookups,
        "architecture": f"stacking: 2 cross-fitted {args.dl} models (dl_a.pt, dl_b.pt) -> dl_score "
                        f"feature of the {args.ml} lexical and host branches",
        "target_recall_on_verified_val": args.target_recall,
        "threshold_lexical": thr_lex, "threshold_host": thr_host,
        "thresholds_by_val_recall": {
            "lexical": {str(t): threshold_for_recall(y["val"][ver["val"]], p_lex["val"][ver["val"]], t)
                        for t in RECALL_TARGETS},
            "host": {str(t): threshold_for_recall(y["val"][Lv][ver["val"][Lv]],
                                                  p_host["val"][Lv][ver["val"][Lv]], t)
                     for t in RECALL_TARGETS}},
        "segment_thresholds": seg_thr, "segment_miss_cost_val": seg_cost,
        "segment_min_clean_recognized_val": MIN_CLEAN_RECOGNIZED,
        "lexical_feature_names": lex_names, "host_feature_names": host_names,
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    imp = getattr(B["host"], "feature_importances_", None)
    te = te.assign(score=score["test"], flagged=seg_t, segment=seg["test"])
    report = {
        **{k: v for k, v in config.items() if not k.endswith("feature_names")},
        "data": {"train": len(tr), "val": len(va), "test": len(te), "dropped_noisy": n_noisy,
                 "host_branch_train_rows": int(Ltr.sum()), "host_data": host_data},
        "train_seconds": {"dl": round(dl_secs), **B["train_seconds"]}, "dl_training": dl_log,
        "results": results, "routed_system_test": routed_report,
        "test_missed_phishing_examples": te[(yt == 1) & ~seg_t].sort_values("score").head(30)[
            ["url", "source", "segment", "score"]].to_dict("records"),
        "top_features_host_branch": (sorted(zip(host_names, np.asarray(imp, float).tolist()),
                                            key=lambda x: -x[1])[:30] if imp is not None else []),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
