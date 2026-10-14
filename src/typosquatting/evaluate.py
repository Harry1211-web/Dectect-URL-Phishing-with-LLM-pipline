"""Evaluate Layer B (typosquatting) and compare it with the old Levenshtein-only rule.

Sets
  clean    : Tranco top-100k homepages + crawled clean URLs (dataset test split)
  phishing : phishing URLs of the dataset test split (only part of them impersonate a brand)
  vn_brand : simulated attacks on Vietnamese brands (llm scenarios) — all impersonate a brand
Variants
  dl_jw      : Damerau-Levenshtein + Jaro-Winkler gate (production)
  dl_only    : Damerau-Levenshtein without the Jaro-Winkler gate
Run:    python src/typosquatting/evaluate.py
Output: reports/typosquatting/metrics.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))
from typosquatting import typosquatting as ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "typosquatting" / "metrics.json"


def _load_sets() -> dict[str, list[str]]:
    with (ROOT / "data" / "tranco_top1m.csv").open(encoding="utf-8") as f:
        tranco = [f"https://{ln.rstrip().split(',', 1)[1]}/" for _, ln in zip(range(100_000), f)]
    clean_crawl, phishing = [], []
    with (ROOT / "dataset" / "urls" / "urls_labeled.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test":
                continue
            if r["label"] == "phishing":
                phishing.append(r["url"])
            elif r["source"] == "tranco_crawl":
                clean_crawl.append(r["url"])
    from redteam_prompt_injection import BASE_SCENARIOS
    from vn_benchmark import EXTRA_SCENARIOS
    vn = [s["url"] for s in BASE_SCENARIOS + EXTRA_SCENARIOS]
    return {"clean_tranco_top100k": tranco, "clean_crawled_test": clean_crawl,
            "phishing_test": phishing, "vn_brand_scenarios": vn}


def _run(urls: list[str]) -> dict:
    flagged, kinds, lat, examples = 0, Counter(), [], []
    for u in urls:
        t0 = time.perf_counter()
        r = ts.check_typosquatting(u)
        lat.append((time.perf_counter() - t0) * 1000)
        if r["flag"] is True:
            flagged += 1
            kinds[r.get("kind")] += 1
            if len(examples) < 40:
                examples.append({"url": u, "brand": r.get("brand"), "kind": r.get("kind")})
    return {"n": len(urls), "flagged": flagged, "flag_rate": round(flagged / max(1, len(urls)), 5),
            "by_kind": dict(kinds), "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 3), "examples": examples}


def main() -> int:
    sets = _load_sets()
    report = {}
    for variant, (short, long_) in {"dl_jw": (ts.JW_MIN_SHORT, ts.JW_MIN_LONG),
                                    "dl_only": (0.0, 0.0)}.items():
        ts.JW_MIN_SHORT, ts.JW_MIN_LONG = short, long_
        report[variant] = {name: _run(urls) for name, urls in sets.items()}
        print(variant, {k: (v["flagged"], v["n"]) for k, v in report[variant].items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
