"""RAG ablation: the same page analysed with and without the RAG block.

Both branches use PROMPT_TEMPLATE_RAG and differ only in `rag_context`, so any verdict flip is
caused by RAG alone.
Run: python src/llm_rag/rag_ablation.py --n 15   (calls the OpenAI API)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_traffic_sources import fetch_openphish_urls, fetch_tranco_domains  # noqa: E402
from llm_benchmark import compute_metrics  # noqa: E402
from llm_client import (  # noqa: E402
    DEFAULT_MODEL, PROMPT_TEMPLATE_RAG, VALID_VERDICTS, build_prompt, call_openai,
    check_typosquatting_with_redirect, fetch,
)
from rag import tier2  # noqa: E402

ROOT = _HERE.parents[1]
OUT_DIR = ROOT / "reports" / "llm_rag" / "rag_pipeline"


def _sample(url: str, label: str, relevance_filter: bool = True) -> dict | None:
    title, content, injection, final_url = fetch(url)
    if not title and not content:
        return None
    typo_flag = check_typosquatting_with_redirect(url, final_url)["flag"]
    from rag.pipeline import get_rag_context
    try:
        max_d = tier2.AUTO_BY_LANGUAGE if relevance_filter else None
        rag_context = get_rag_context(final_url, title, content, max_distance=max_d)
    except Exception as e:
        rag_context = f"[RAG lỗi, bỏ qua: {type(e).__name__}]"

    def prompt(ctx: str) -> str:
        return build_prompt(url, title, content, injection, typosquatting=typo_flag,
                            template=PROMPT_TEMPLATE_RAG, rag_context=ctx, final_url=final_url)

    return {"url": url, "label": label, "rag_context_empty": not rag_context.strip(),
            "prompt_no_rag": prompt(""), "prompt_rag": prompt(rag_context)}


def collect(n: int, seed: int, relevance_filter: bool = True) -> list[dict]:
    jobs = ([(f"https://{d}", "legit") for d in fetch_tranco_domains(n * 3, seed=seed)]
            + [(u, "phishing") for u in fetch_openphish_urls(n * 3, seed=seed)])
    got: dict[str, list[dict]] = {"legit": [], "phishing": []}
    for u, lb in jobs:
        if len(got["legit"]) >= n and len(got["phishing"]) >= n:
            break
        if len(got[lb]) >= n:
            continue
        s = _sample(u, lb, relevance_filter)
        if s is not None:
            got[lb].append(s)
    return got["legit"][:n] + got["phishing"][:n]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-relevance-filter", action="store_true",
                    help="disable the tier-2 distance threshold")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    samples = collect(args.n, args.seed, relevance_filter=not args.no_relevance_filter)
    print(f"Got {len(samples)} reachable samples.", file=sys.stderr)
    rows = []
    for i, s in enumerate(samples, 1):
        print(f"  {i}/{len(samples)}: {s['url'][:60]} (empty_rag={s['rag_context_empty']})",
              flush=True)
        a = call_openai(s["prompt_no_rag"], args.model, temperature=args.temperature)
        b = call_openai(s["prompt_rag"], args.model, temperature=args.temperature)
        rows.append({"url": s["url"], "label": s["label"],
                     "rag_context_empty": s["rag_context_empty"],
                     **{f"{k}_no_rag": a.get(k) for k in ("verdict", "explanation", "latency_s", "error")},
                     **{f"{k}_rag": b.get(k) for k in ("verdict", "explanation", "latency_s", "error")}})

    def branch(suffix: str) -> list[dict]:
        return [{"label": r["label"], "verdict": r[f"verdict_{suffix}"],
                 "latency_s": r[f"latency_s_{suffix}"], "error": r[f"error_{suffix}"]} for r in rows]

    # Compare only pairs where both branches returned a valid verdict
    pairs = [r for r in rows if r["verdict_no_rag"] in VALID_VERDICTS
             and r["verdict_rag"] in VALID_VERDICTS]
    flips = [r for r in pairs if r["verdict_no_rag"] != r["verdict_rag"]]
    summary = {
        "time": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "temperature": args.temperature, "relevance_filter": not args.no_relevance_filter,
        "n_samples": len(rows), "n_valid_pairs": len(pairs), "n_flips": len(flips),
        "flip_rate": round(len(flips) / len(pairs), 4) if pairs else None,
        "metrics_no_rag": compute_metrics(branch("no_rag")),
        "metrics_rag": compute_metrics(branch("rag")),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"ablation_n{args.n}_seed{args.seed}_t{args.temperature}.json"
    out.write_text(json.dumps({**summary, "details": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
