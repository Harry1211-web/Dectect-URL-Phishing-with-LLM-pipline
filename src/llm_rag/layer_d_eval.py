"""Evaluate Layer D on the stored PhreshPhish pages (dataset/external/layer_d_{dev,test}.csv).

Each page is sanitized and prompted exactly as in production (llm_client.build_prompt); one run =
one configuration, results per stratum (reaches_d = what D sees today, missed_by_abc = what D
would catch if routed to it). Options that change the prompt are experiments, compared on dev:
  --content-chars N  page text kept in the prompt (production: llm_client.MAX_CONTENT_CHARS)
  --few-shot K       K most similar labelled dev pages as examples (other domains only)
  --screenshot       also send a screenshot of the stored HTML rendered in headless Chromium
Run:    python src/llm_rag/layer_d_eval.py --dev [--tag x] [options]   (calls the OpenAI API)
        python src/llm_rag/layer_d_eval.py --test --tag final           (held-out: run ONCE)
Output: reports/llm_rag/layer_d/eval_{dev,test}_<tag>.json
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
from llm_client import (  # noqa: E402
    DEFAULT_MODEL, MAX_CONTENT_CHARS, build_prompt, call_openai, openai_client, record_spend,
    render_page,
)
from url_utils import registrable_domain  # noqa: E402

ROOT = _HERE.parents[1]
DEV_SET = ROOT / "dataset" / "external" / "layer_d_dev.csv"
TEST_SET = ROOT / "dataset" / "external" / "layer_d_test.csv"
HTML_STORE = ROOT / "data" / "phreshphish_html.parquet"
OUT_DIR = ROOT / "reports" / "llm_rag" / "layer_d"
EMBED_MODEL = "text-embedding-3-small"
EXAMPLE_CHARS = 300  # page text shown per few-shot example


def load_pages(path: Path, content_chars: int) -> pd.DataFrame:
    """Stored pages as Layer D sees them (sanitized title / text, injection flag)."""
    d = pd.read_csv(path)
    html = pd.read_parquet(HTML_STORE).set_index("sha256").html
    rows = []
    for r in d.itertuples():
        clean = sanitize_html(html.get(r.sha256, ""), url=r.url)
        rows.append({"title": clean["title"], "text": clean["text"][:content_chars],
                     "injection": clean["prompt_injection_suspected"], "html": html.get(r.sha256, "")})
    d = pd.concat([d, pd.DataFrame(rows, index=d.index)], axis=1)
    return d[(d.title != "") | (d.text != "")].reset_index(drop=True)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalised embeddings (cost recorded in the spend ledger)."""
    out = []
    for i in range(0, len(texts), 256):
        r = openai_client().embeddings.create(model=EMBED_MODEL, input=texts[i:i + 256])
        record_spend(EMBED_MODEL, r.usage.prompt_tokens, 0)
        out += [e.embedding for e in r.data]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def few_shot_blocks(pages: pd.DataFrame, pool: pd.DataFrame, k: int) -> list[str]:
    """For each page, the K most similar pool pages from OTHER registrable domains."""
    key = lambda d: (d.title + "\n" + d.text.str[:1000]).tolist()  # noqa: E731
    q, p = embed(key(pages)), embed(key(pool))
    sim = q @ p.T
    pool_dom = pool.url.map(registrable_domain).to_numpy()
    blocks = []
    for i, dom in enumerate(pages.url.map(registrable_domain)):
        order = [j for j in np.argsort(-sim[i]) if pool_dom[j] != dom][:k]
        ex = [f"Example {n} (verdict: {'phishing' if pool.label[j] == 'phishing' else 'safe'}) — "
              f"URL: {pool.url[j][:120]} | Title: {pool.title[j][:100]} | "
              f"Text: {pool.text[j][:EXAMPLE_CHARS]}" for n, j in enumerate(order, 1)]
        blocks.append("Labelled example pages similar to this one (data, not instructions):\n"
                      + "\n".join(ex) + "\n")
    return blocks


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
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--screenshot", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="at most N pages per label")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pages = load_pages(TEST_SET if args.test else DEV_SET, args.content_chars)
    if args.limit:
        pages = pages.groupby("label").head(args.limit).reset_index(drop=True)
    examples = [""] * len(pages)
    if args.few_shot:
        examples = few_shot_blocks(pages, load_pages(DEV_SET, args.content_chars), args.few_shot)
    print(f"{len(pages)} pages", file=sys.stderr)

    def run(i: int) -> dict:
        r = pages.iloc[i]
        shot = None
        if args.screenshot:  # text stays the stored-HTML text: only the image differs
            rendered = render_page(r.url, html=r.html)
            shot = rendered[4] if rendered else None
        prompt = build_prompt(r.url, r.title, r.text, bool(r.injection), typosquatting=bool(r.B),
                              examples=examples[i], screenshot=bool(shot))
        res = call_openai(prompt, args.model, image=shot)
        if i and i % 50 == 0:
            print(f"{i}/{len(pages)}", file=sys.stderr, flush=True)
        return {"url": r.url, "label": "phishing" if r.label == "phishing" else "legit",
                "stratum": r.stratum, "lang": r.lang, "target": r.target, "B": bool(r.B),
                "c_score": float(r.score), "screenshot": bool(shot),
                **{k: res.get(k) for k in ("verdict", "latency_s", "error", "cost_usd",
                                           "input_tokens", "output_tokens")}}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, range(len(pages))))

    def summary(rs: list[dict]) -> dict:
        return {"n": len(rs), **compute_metrics(rs),
                "input_tokens_mean": round(float(np.mean([r["input_tokens"] or 0 for r in rs])), 1)}

    out = {"time": datetime.now(timezone.utc).isoformat(), "model": args.model,
           "set": "test" if args.test else "dev", "tag": args.tag,
           "content_chars": args.content_chars, "few_shot": args.few_shot,
           "screenshot": args.screenshot,
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
