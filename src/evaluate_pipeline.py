"""Evaluate A -> B -> C on the dataset test split, without the paid LLM layer.

Two settings, because the blocklist (A) comes from the same feeds as the phishing test URLs:
  with_A    : realistic production flow (A catches known URLs)
  without_A : B + C only — how the pipeline handles phishing that is NOT yet in any feed
Key numbers: phishing URLs that end as "clean" (must be ~0) and the share of clean URLs the
classifier recognises as clean (the rest is sent to Layer D = LLM load / cost).
Layer C uses host features from the cache (collect_host_features.py), no network calls.
Every number is also reported on "trusted labels": test phishing rows whose label is doubtful by
model-independent evidence are left out (see suspect_labels). Also measures real single-URL
latency of every stage.

Run:    python src/evaluate_pipeline.py
Output: reports/pipeline/metrics.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import analyze_url  # noqa: E402
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from typosquatting.brands import all_official_domains  # noqa: E402
from typosquatting.paas_domains import is_untrusted_hosting, is_user_content_host  # noqa: E402
from url_classifier import host_features as hf, predict  # noqa: E402
from url_utils import host_of, registrable_domain  # noqa: E402
from data_collection.audit_labels import PHIUSIIL_COLLECTED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pipeline" / "metrics.json"
VERIFIED_SOURCES = ("phishtank", "openphish")


def _pct(v: list[float], q: float) -> float:
    return round(float(np.percentile(v, q)), 3) if v else 0.0


def suspect_labels(te: pd.DataFrame) -> pd.DataFrame:
    """Phishing rows whose label is doubtful, by evidence independent of Layers B and C.

    popular_homepage : homepage of a Tranco-ranked registrable domain, not a shared platform and
                       not in the current feeds (old compromise or feed noise)
    re_registered    : PhiUSIIL domain registered after the dataset was collected, so today's
                       host data describes a different owner
    official_domain  : URL on a brand's own official domain (reported pages of the real brand)
    """
    with (ROOT / "data" / "tranco_top1m.csv").open(encoding="utf-8") as f:
        ranked = {ln.rstrip().split(",", 1)[-1] for ln in f}
    ph = te.label == "phishing"
    host, rd = te.url.map(host_of), te.url.map(registrable_domain)
    shared = pd.Series([is_untrusted_hosting(r) or is_user_content_host(h)
                        for r, h in zip(rd, host)], index=te.index)
    rest = te.url.str.replace(r"^https?://", "", regex=True).str.rstrip("/")
    bare = ~rest.str.contains("/") & ~rest.str.contains(r"\?")
    apex = pd.Series([h in (r, "www." + r) for h, r in zip(host, rd)], index=te.index)
    out = pd.DataFrame(index=te.index)
    out["popular_homepage"] = ph & ~te.A & bare & apex & rd.isin(ranked) & ~shared
    phi = ph & (te.source == "phiusiil")
    raw = hf.raw_lookups(te.url[phi].tolist(), offline=True)
    reg = host[phi].map(lambda h: ((raw.get(h) or {}).get("rdap") or {}).get("registered") or 0)
    out["re_registered"] = False
    out.loc[phi, "re_registered"] = reg > PHIUSIIL_COLLECTED
    out["official_domain"] = ph & rd.isin(all_official_domains()) & ~shared
    return out


def outcome(te: pd.DataFrame, c_flag: np.ndarray) -> dict:
    """Final routing for both settings given the classifier decisions."""
    ph = (te.label == "phishing").to_numpy()
    ver = ph & te.source.isin(VERIFIED_SOURCES).to_numpy()
    b = te.B.to_numpy()
    out = {}
    for name, a in (("with_A", te.A.to_numpy()), ("without_A", np.zeros(len(te), bool))):
        clean_final = ~a & ~b & ~c_flag
        to_llm = ~a & ~clean_final
        out[name] = {
            "phishing_blocked_by_A": int((a & ph).sum()),
            "phishing_sent_to_llm": int((to_llm & ph).sum()),
            "phishing_labelled_clean": int((clean_final & ph).sum()),
            "verified_phishing_labelled_clean": int((clean_final & ver).sum()),
            "phishing_miss_rate": round(float((clean_final & ph).sum() / ph.sum()), 6),
            "clean_recognized_rate": round(float((clean_final & ~ph).sum() / (~ph).sum()), 4),
            "clean_sent_to_llm_rate": round(float((to_llm & ~ph).sum() / (~ph).sum()), 4),
        }
    return out


def main() -> int:
    df = pd.read_csv(ROOT / "dataset" / "urls" / "urls_labeled.csv")
    te = df[df.split == "test"].reset_index(drop=True)
    urls = te.url.tolist()
    print(f"test URLs: {len(te)}", flush=True)

    te["A"] = [check_blocklist(u)["flag"] is True for u in urls]
    te["B"] = [check_typosquatting_deep(u)["flag"] is True for u in urls]
    sus = suspect_labels(te)
    trusted = ~sus.any(axis=1).to_numpy()

    cfg = json.loads((predict.MODEL_DIR / "config.json").read_text())
    report: dict = {"n_test": len(te), "n_phishing": int((te.label == "phishing").sum()),
                    "n_verified_phishing": int(te.source.isin(VERIFIED_SOURCES).sum()),
                    "B_flags_on_clean": int((te.B & (te.label == "clean")).sum()),
                    "B_flags_on_phishing": int((te.B & (te.label == "phishing")).sum()),
                    "suspect_phishing_labels": {k: int(v.sum()) for k, v in sus.items()}
                    | {"total": int((~trusted).sum())}}
    for mode in ("none", "cache"):  # URL-only vs URL + host features
        chunks = [predict.score_urls(urls[i:i + 4096], host_mode=mode)
                  for i in range(0, len(urls), 4096)]
        score = np.concatenate([c[0] for c in chunks])
        prod = np.concatenate([c[1] for c in chunks])  # production (per-segment) thresholds
        used = np.concatenate([c[2] for c in chunks])
        key = "classifier_url_only" if mode == "none" else "classifier_with_host_features"
        report[key] = {"host_branch_used_rows": int(used.sum()), "by_threshold": {}}
        for target in cfg["thresholds_by_val_recall"]["lexical"]:
            t = np.where(used, cfg["thresholds_by_val_recall"]["host"][target],
                         cfg["thresholds_by_val_recall"]["lexical"][target])
            report[key]["by_threshold"][f"val_recall_{target}"] = outcome(te, score >= t)
        report[key]["production_threshold"] = outcome(te, score >= prod)
        report[key]["production_threshold_trusted_labels"] = outcome(te[trusted],
                                                                     (score >= prod)[trusted])
        print(key, json.dumps(report[key]["production_threshold"]), flush=True)
        print(key, "trusted labels",
              json.dumps(report[key]["production_threshold_trusted_labels"]), flush=True)

    # Real single-URL latency (network lookups + models), LLM disabled
    sample = random.Random(0).sample(urls, 200)
    lat = {"A": [], "B": [], "C": [], "C_host_lookup": [], "total": []}
    analyze_url(sample[0], use_llm=False)
    for u in sample:
        r = analyze_url(u, use_llm=False)
        lat["total"].append(r["latency_ms"])
        lat["A"].append(r["layers"]["A_blocklist"]["latency_ms"])
        if "B_typosquatting" in r["layers"]:
            lat["B"].append(r["layers"]["B_typosquatting"]["latency_ms"])
        c = r["layers"].get("C_classifier", {})
        if "latency_ms" in c:
            lat["C"].append(c["latency_ms"])
            lat["C_host_lookup"].append(c["host_ms"])
    report["latency_ms_single_url"] = {k: {"p50": _pct(v, 50), "p95": _pct(v, 95),
                                           "max": _pct(v, 100)} for k, v in lat.items()}
    print(report["latency_ms_single_url"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
