"""Evaluate Layer D on the stored PhreshPhish pages (dataset/external/layer_d_{dev,test}.csv).

Each page is sanitized and prompted exactly as in production (llm_client.build_prompt); one run =
one configuration, results per stratum (reaches_d = what D sees today, missed_by_abc = what D
would catch if routed to it). Prompt changes are compared on dev first; --content-chars N sets
the page text kept in the prompt (production: llm_client.MAX_CONTENT_CHARS).
--model claude-haiku-5-5 runs the same prompt on Anthropic (model comparison, --thinking off|default).
--explain [--lang vi] also asks for the explanation in the user's language (does it change verdicts?).
Run:    python src/llm/layer_d_eval.py --dev [--tag x] [options]   (calls the paid LLM API)
        python src/llm/layer_d_eval.py --test --tag final           (held-out: run ONCE)
Output: reports/llm/layer_d/eval_{dev,test}_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from html_sanitize import sanitize_html  # noqa: E402
from llm_benchmark import compute_metrics  # noqa: E402
from llm_client import DEFAULT_MODEL, MAX_CONTENT_CHARS, build_prompt, call_llm  # noqa: E402

ROOT = _HERE.parents[1]
DEV_SET = ROOT / "dataset" / "external" / "layer_d_dev.csv"
TEST_SET = ROOT / "dataset" / "external" / "layer_d_test.csv"
HTML_STORE = ROOT / "data" / "phreshphish_html.parquet"
OUT_DIR = ROOT / "reports" / "llm" / "layer_d"


def load_pages(path: Path, content_chars: int) -> pd.DataFrame:
    """Stored pages as Layer D sees them (sanitized title / text, injection flag)."""
    d = pd.read_csv(path)
    html = pd.read_parquet(HTML_STORE, filters=[("sha256", "in", list(set(d.sha256)))]).set_index("sha256").html
    rows = []
    for r in d.itertuples():
        clean = sanitize_html(html.get(r.sha256, ""), url=r.url)
        rows.append({"title": clean["title"], "text": clean["text"][:content_chars],
                     "injection": clean["prompt_injection_suspected"]})
    d = pd.concat([d, pd.DataFrame(rows, index=d.index)], axis=1)
    return d[(d.title != "") | (d.text != "")].reset_index(drop=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--test", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--content-chars", type=int, default=MAX_CONTENT_CHARS)
    ap.add_argument("--thinking", choices=("default", "off"), default="default",
                    help="Anthropic models only: keep adaptive thinking or disable it")
    ap.add_argument("--limit", type=int, default=None, help="at most N pages per label")
    ap.add_argument("--seed", type=int, default=None,
                    help="with --limit: random sample of N pages per label instead of the first N")
    ap.add_argument("--explain", action="store_true", help="ask for the explanation as well")
    ap.add_argument("--lang", default="vi", help="with --explain: explanation language")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pages = load_pages(TEST_SET if args.test else DEV_SET, args.content_chars)
    if args.limit:
        pages = (pages.groupby("label").sample(args.limit, random_state=args.seed)
                 if args.seed is not None else pages.groupby("label").head(args.limit))
        pages = pages.reset_index(drop=True)
    print(f"{len(pages)} pages", file=sys.stderr)

    def run(i: int) -> dict:
        r = pages.iloc[i]
        prompt = build_prompt(r.url, r.title, r.text, bool(r.injection), typosquatting=bool(r.B),
                              explain=args.explain, lang=args.lang)
        res = call_llm(prompt, args.model, thinking=args.thinking)
        if i and i % 50 == 0:
            print(f"{i}/{len(pages)}", file=sys.stderr, flush=True)
        return {"url": r.url, "label": "phishing" if r.label == "phishing" else "legit",
                "stratum": r.stratum, "lang": r.lang, "target": r.target, "B": bool(r.B),
                "c_score": float(r.score),
                **{k: res.get(k) for k in ("verdict", "latency_s", "error", "cost_usd",
                                           "input_tokens", "output_tokens")},
                **({"explanation": res.get("explanation")} if args.explain else {})}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, range(len(pages))))

    def summary(rs: list[dict]) -> dict:
        return {"n": len(rs), **compute_metrics(rs),
                "input_tokens_mean": round(float(np.mean([r["input_tokens"] or 0 for r in rs])), 1)}

    out = {"time": datetime.now(timezone.utc).isoformat(), "model": args.model,
           "thinking": args.thinking if args.model.startswith("claude") else None,
           "set": "test" if args.test else "dev", "tag": args.tag,
           "content_chars": args.content_chars,
           "explain_lang": args.lang if args.explain else None,
           "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
           "all": summary(rows),
           "by_stratum": {s: summary([r for r in rows if r["stratum"] == s])
                          for s in sorted({r["stratum"] for r in rows})},
           "details": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"eval_{out['set']}{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
