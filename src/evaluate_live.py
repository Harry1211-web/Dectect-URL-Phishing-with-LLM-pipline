"""Run the full pipeline (A -> B -> C -> D, live page fetch) on live URLs.

Modes (never used for training or tuning — measurement only):
  --vi         data/test_sets/vi/: Vietnamese phishing (Chong Lua Dao) + clean .vn domains
  --top-sites  fresh deep links harvested from the homepages of Tranco top domains
               (URLs already in the training data are skipped): false alarms in normal browsing
Run:    python src/evaluate_live.py --vi --n 300 [--model claude-haiku-5-5]   (paid LLM API)
        python src/evaluate_live.py --top-sites --n 400
Output: reports/live/<mode>[_tag].json — verdicts, deciding layer, latency, spend
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_collection.crawl_clean import harvest  # noqa: E402
from data_collection.sources import tranco_domains  # noqa: E402
from llm.llm_client import DEFAULT_MODEL, provider_of, spent_usd  # noqa: E402
from pipeline import analyze_url  # noqa: E402
from url_utils import normalize_url  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VI_DIR = ROOT / "data" / "test_sets" / "vi"
TRAINING = ROOT / "dataset" / "urls" / "urls_labeled.csv"
OUT_DIR = ROOT / "reports" / "live"
TOP_SITES = 5000


def vi_urls(n: int, seed: int) -> list[tuple[str, str]]:
    ph = pd.read_csv(VI_DIR / "phishing.csv").url.sample(frac=1, random_state=seed).head(n)
    cl = pd.read_csv(VI_DIR / "clean.csv").url.sample(frac=1, random_state=seed).head(n)
    return [(u, "phishing") for u in ph] + [(u, "clean") for u in cl]


def top_site_urls(n: int, seed: int) -> list[tuple[str, str]]:
    """One fresh deep link per top site, skipping URLs present in the training data."""
    rng = random.Random(seed)
    sites = tranco_domains(TOP_SITES)
    rng.shuffle(sites)
    known = set(pd.read_csv(TRAINING, usecols=["url"]).url.map(normalize_url))

    def one(site: str) -> str | None:
        links = [u for u in harvest(site, 5, random.Random(rng.random()))
                 if urlsplit_path(u) and normalize_url(u) not in known]
        return links[0] if links else None

    out: list[str] = []
    with ThreadPoolExecutor(32) as pool:
        for u in pool.map(one, sites[:n * 3]):
            if u and len(out) < n:
                out.append(u)
    return [(u, "clean") for u in out]


def urlsplit_path(url: str) -> bool:
    """True for a deep link (not the bare homepage)."""
    rest = url.split("://", 1)[-1]
    return "/" in rest and rest.split("/", 1)[1].strip("/") != ""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vi", action="store_true")
    g.add_argument("--top-sites", action="store_true")
    ap.add_argument("--n", type=int, default=300, help="URLs per label")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Layer D model")
    args = ap.parse_args()

    jobs = vi_urls(args.n, args.seed) if args.vi else top_site_urls(args.n, args.seed)
    print(f"{len(jobs)} URLs", file=sys.stderr, flush=True)
    provider = provider_of(args.model)
    spent0 = spent_usd(provider)

    def run(job: tuple[str, str]) -> dict:
        url, label = job
        t0 = time.perf_counter()
        try:
            r = analyze_url(url, use_vt=False, llm_model=args.model)
        except Exception as e:  # one broken URL must not stop the run
            return {"url": url, "label": label, "verdict": None, "decided_by": f"error:{type(e).__name__}"}
        d = r["layers"].get("D_llm") or {}
        return {"url": url, "label": label, "verdict": r["verdict"], "decided_by": r["decided_by"],
                "llm_verdict": d.get("verdict"), "llm_unusable": d.get("timed_out"),
                "c_score": (r["layers"].get("C_classifier") or {}).get("score"),
                "dom_score": (r["layers"].get("C_classifier") or {}).get("dom_score"),
                "layer_ms": {k: v.get("latency_ms") for k, v in r["layers"].items()
                             if isinstance(v, dict)},
                "latency_s": round(time.perf_counter() - t0, 2)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(run, jobs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", file=sys.stderr, flush=True)

    def summary(rs: list[dict]) -> dict:
        n = len(rs)
        v = Counter(r["verdict"] for r in rs)
        lat = sorted(r["latency_s"] for r in rs if r.get("latency_s") is not None)
        return {"n": n, **{f"verdict_{k}": c for k, c in v.items()},
                **{f"rate_{k}": round(c / max(n, 1), 4) for k, c in v.items()},
                "decided_by": dict(Counter(r["decided_by"] for r in rs)),
                "llm_called": sum(r.get("llm_verdict") is not None or bool(r.get("llm_unusable"))
                                  for r in rs),
                "latency_p50_s": lat[len(lat) // 2] if lat else None,
                "latency_p95_s": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None}

    out = {"time": datetime.now(timezone.utc).isoformat(), "mode": "vi" if args.vi else "top_sites",
           "n_per_label": args.n, "seed": args.seed, "model": args.model,
           "llm_spend_usd": round(spent_usd(provider) - spent0, 4),
           **{label: summary([r for r in rows if r["label"] == label])
              for label in sorted({r["label"] for r in rows})},
           "details": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{out['mode']}{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
