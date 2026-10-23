"""Compare Layer D runs (layer_d_eval.py outputs) of different models, or repeated runs of one
model, on the same pages.

Per run: phishing labelled safe / suspicious / phishing, clean labelled phishing / suspicious,
the same after the production C override (D "safe" and C score >= LLM_SAFE_OVERRIDE_SCORE on a
non-official domain -> suspicious), p50 / p95 latency and cost per page, with 95 % Wilson
intervals. Per pair of runs: exact McNemar test on the pages both runs answered, for
"phishing missed" (labelled safe) and for "clean page blocked" (labelled phishing).
Label, stratum and C score come from the current page sets (dataset/external/layer_d_*.csv), so
stored runs follow label fixes and a retrained Layer C; pages no longer in a set are left out.
Run:    python src/llm/compare_models.py reports/llm/layer_d/eval_dev_a.json eval_dev_b.json ... --out name
Output: reports/llm/layer_d/compare_<name>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from pipeline import LLM_SAFE_OVERRIDE_SCORE  # noqa: E402
from typosquatting.brands import all_official_domains  # noqa: E402
from url_utils import registrable_domain  # noqa: E402

OUT_DIR = _HERE.parents[1] / "reports" / "llm" / "layer_d"
SETS_DIR = _HERE.parents[1] / "dataset" / "external"


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95 % Wilson score interval of k / n."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def with_override(r: dict) -> str | None:
    """Verdict after the production rule of pipeline.analyze_url."""
    if (r["verdict"] == "safe" and r["c_score"] >= LLM_SAFE_OVERRIDE_SCORE
            and registrable_domain(r["url"]) not in all_official_domains()):
        return "suspicious"
    return r["verdict"]


def refresh(run: dict) -> dict:
    """Label / stratum / C score of every page from its current set file."""
    pages = pd.read_csv(SETS_DIR / f"layer_d_{run['set']}.csv").drop_duplicates("url").set_index("url")
    rows = []
    for r in run["details"]:
        if r["url"] in pages.index:
            p = pages.loc[r["url"]]
            rows.append({**r, "label": p.label, "stratum": p.stratum, "c_score": float(p.score)})
    return {**run, "details": rows}


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def outcome_table(rows: list[dict], key: str) -> dict:
    ph = [r for r in rows if r["label"] == "phishing" and r[key] is not None]
    cl = [r for r in rows if r["label"] != "phishing" and r[key] is not None]
    return {"phishing_labelled_safe": rate(sum(r[key] == "safe" for r in ph), len(ph)),
            "phishing_labelled_suspicious": rate(sum(r[key] == "suspicious" for r in ph), len(ph)),
            "phishing_labelled_phishing": rate(sum(r[key] == "phishing" for r in ph), len(ph)),
            "clean_labelled_phishing": rate(sum(r[key] == "phishing" for r in cl), len(cl)),
            "clean_labelled_suspicious": rate(sum(r[key] == "suspicious" for r in cl), len(cl))}


def run_summary(run: dict, strata: list[str] | None) -> dict:
    rows = [r for r in run["details"] if strata is None or r["stratum"] in strata]
    for r in rows:
        r["verdict_override"] = with_override(r) if r["verdict"] else None
    ok = [r for r in rows if r["verdict"]]
    lat = sorted(r["latency_s"] for r in ok if r.get("latency_s") is not None)
    return {"n": len(rows), "n_no_verdict": len(rows) - len(ok),
            "D_alone": outcome_table(rows, "verdict"),
            "D_with_C_override": outcome_table(rows, "verdict_override"),
            "latency_p50_s": float(np.percentile(lat, 50)) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "cost_per_page_usd": round(sum(r.get("cost_usd") or 0 for r in rows) / max(1, len(rows)), 7),
            "input_tokens_mean": round(float(np.mean([r.get("input_tokens") or 0 for r in ok])), 1),
            "output_tokens_mean": round(float(np.mean([r.get("output_tokens") or 0 for r in ok])), 1)}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (two-sided binomial on discordant pairs)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    return {"only_first": only_a, "only_second": only_b,
            "p_value": round(binomtest(only_a, n, 0.5).pvalue, 4) if n else 1.0}


def pair_tests(r1: dict, r2: dict, strata: list[str] | None, key: str) -> dict:
    d1 = {r["url"]: r for r in r1["details"] if strata is None or r["stratum"] in strata}
    d2 = {r["url"]: r for r in r2["details"]}
    both = [(d1[u], d2[u]) for u in d1 if u in d2 and d1[u]["verdict"] and d2[u]["verdict"]]
    ph = [(x, y) for x, y in both if x["label"] == "phishing"]
    cl = [(x, y) for x, y in both if x["label"] != "phishing"]
    return {"n_pages_both_answered": len(both),
            "phishing_missed (safe)": mcnemar([x[key] == "safe" for x, _ in ph],
                                              [y[key] == "safe" for _, y in ph]),
            "phishing_not_flagged_phishing": mcnemar([x[key] != "phishing" for x, _ in ph],
                                                     [y[key] != "phishing" for _, y in ph]),
            "clean_blocked (phishing)": mcnemar([x[key] == "phishing" for x, _ in cl],
                                                [y[key] == "phishing" for _, y in cl]),
            "verdict_agreement": round(sum(x[key] == y[key] for x, y in both) / max(1, len(both)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {p.stem: refresh(json.loads(p.read_text(encoding="utf-8"))) for p in args.runs}
    views = {"all": None, "reaches_d": ["reaches_d"], "missed_by_abc": ["missed_by_abc"]}
    out = {"runs": {k: {"model": r["model"], "thinking": r.get("thinking"), "set": r["set"],
                        "time": r["time"],
                        **{v: run_summary(r, s) for v, s in views.items()}}
                    for k, r in runs.items()},
           "pairs": {f"{a} vs {b}": {v: {"D_alone": pair_tests(runs[a], runs[b], s, "verdict"),
                                         "D_with_C_override": pair_tests(runs[a], runs[b], s,
                                                                         "verdict_override")}
                                     for v, s in views.items()}
                     for a, b in combinations(runs, 2)}}
    path = OUT_DIR / f"compare_{args.out}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in out["runs"].items():
        a = r["all"]
        print(k, "| miss", a["D_alone"]["phishing_labelled_safe"]["rate"],
              "| miss+override", a["D_with_C_override"]["phishing_labelled_safe"]["rate"],
              "| clean->phishing", a["D_alone"]["clean_labelled_phishing"]["rate"],
              "| p50/p95", a["latency_p50_s"], a["latency_p95_s"], "| $/page", a["cost_per_page_usd"])
    for k, p in out["pairs"].items():
        print(k, json.dumps(p["all"]["D_alone"]))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
