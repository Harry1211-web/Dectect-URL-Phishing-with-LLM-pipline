"""Benchmark Layer D (LLM) accuracy and latency on live legit (Tranco) vs phishing (OpenPhish) URLs.

Reports accuracy, precision, recall and latency p50/p95.
Run: python src/llm/llm_benchmark.py --n 15 [--model ...]   (calls the OpenAI API)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_traffic_sources import fetch_openphish_urls, fetch_tranco_domains  # noqa: E402
from llm_client import (  # noqa: E402
    DEFAULT_MODEL, VALID_VERDICTS, VERDICT_PHISHING, VERDICT_SAFE, VERDICT_SUSPICIOUS,
    build_prompt, call_openai,
    check_typosquatting_with_redirect, fetch, openai_client,
)

ROOT = _HERE.parents[1]
OUT_DIR = ROOT / "reports" / "llm" / "llm_benchmark"


def prepare_sample(url: str, label: str) -> dict | None:
    """Fetch the page and build the prompt exactly like production (typosquatting rule on the
    original and the redirected URL). None when the page cannot be fetched."""
    title, content, injection, final_url = fetch(url)
    if not title and not content:
        return None
    typo_flag = check_typosquatting_with_redirect(url, final_url)["flag"]
    prompt = build_prompt(url, title, content, injection, typosquatting=typo_flag,
                          final_url=final_url)
    return {"url": url, "final_url": final_url, "label": label, "prompt": prompt,
            "typosquatting": typo_flag, "prompt_injection_suspected": injection}


def collect_samples(n: int, workers: int = 15, seed: int = 55) -> list[dict]:
    """`n` live legit + `n` live phishing samples, fetched in parallel."""
    jobs = ([(f"https://{d}", "legit") for d in fetch_tranco_domains(n * 3, seed=seed)]
            + [(u, "phishing") for u in fetch_openphish_urls(n * 3, seed=seed)])
    got: dict[str, list[dict]] = {"legit": [], "phishing": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(prepare_sample, u, lb): lb for u, lb in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None and len(got[futs[fut]]) < n:
                got[futs[fut]].append(r)
            if len(got["legit"]) >= n and len(got["phishing"]) >= n:
                ex.shutdown(wait=False, cancel_futures=True)  # break alone would wait for all jobs
                break
    return got["legit"][:n] + got["phishing"][:n]


def run_benchmark(samples: list[dict], model: str, temperature: float | None = None) -> list[dict]:
    rows = []
    for i, s in enumerate(samples, 1):
        print(f"  [OpenAI] {i}/{len(samples)}: {s['url'][:60]}", flush=True)
        res = call_openai(s["prompt"], model, temperature)
        rows.append({**{k: v for k, v in s.items() if k != "prompt"}, **res, "provider": "openai"})
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    """Accuracy / precision / recall on rows with a valid verdict; latency on successful calls."""
    valid = [r for r in rows if r.get("verdict") in VALID_VERDICTS]
    n_errors = sum(1 for r in rows if r.get("error"))
    if not valid:
        return {"n_valid": 0, "n_api_errors": n_errors}
    tp = sum(r["label"] == "phishing" and r["verdict"] == VERDICT_PHISHING for r in valid)
    fp = sum(r["label"] != "phishing" and r["verdict"] == VERDICT_PHISHING for r in valid)
    fn = sum(r["label"] == "phishing" and r["verdict"] != VERDICT_PHISHING for r in valid)
    tn = len(valid) - tp - fp - fn
    # Failed calls return almost instantly and would skew latency
    lat = sorted(r["latency_s"] for r in rows if "latency_s" in r and not r.get("error"))
    return {
        "n_valid": len(valid), "n_api_errors": n_errors,
        "n_bad_format": len(rows) - len(valid) - n_errors,
        "accuracy": round((tp + tn) / len(valid), 4),
        "precision_phishing": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall_phishing": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        # Three-way view: a phishing page is missed only when labelled "safe"; a clean page
        # labelled "suspicious" is a warning, not a block
        "phishing_labelled_safe": sum(r["label"] == "phishing" and r["verdict"] == VERDICT_SAFE
                                      for r in valid),
        "phishing_labelled_suspicious": sum(r["label"] == "phishing"
                                            and r["verdict"] == VERDICT_SUSPICIOUS for r in valid),
        "clean_labelled_suspicious": sum(r["label"] != "phishing"
                                         and r["verdict"] == VERDICT_SUSPICIOUS for r in valid),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p95_s": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15, help="samples per class")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=None,
                    help="not sent by default (some models reject it)")
    ap.add_argument("--seed", type=int, default=55)
    args = ap.parse_args()

    if openai_client() is None:
        print("Error: OPENAI_API_KEY is missing in .env.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Collecting {args.n * 2} live samples (Tranco + OpenPhish)...", file=sys.stderr)
    samples = collect_samples(args.n, workers=args.workers, seed=args.seed)
    print(f"Got {len(samples)} reachable samples.", file=sys.stderr)
    rows = run_benchmark(samples, args.model, temperature=args.temperature)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT_DIR / f"samples_n{args.n}_seed{args.seed}.csv",
                              index=False, encoding="utf-8-sig")
    summary = {
        "time": datetime.now(timezone.utc).isoformat(), "provider": "openai",
        "model": args.model,
        "temperature": args.temperature, "n_per_class": args.n, "n_samples": len(samples),
        "metrics": compute_metrics(rows),
    }
    out = OUT_DIR / f"summary_n{args.n}_seed{args.seed}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
