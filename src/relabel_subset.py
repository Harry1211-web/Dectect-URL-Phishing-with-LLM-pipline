"""Relabelled subset: how much of the measured Layer D error is label noise?

export : draws ~300 pages from the Layer D dev / held-out test sets (PhreshPhish) and the live
         Vietnamese run (reports/live/vi.json), half from pages where the layers disagree (with the
         dataset label or with each other), half from the others, inside each source. Writes a CSV
         with URL, title, a text snippet and every layer's verdict. Vietnamese pages are not
         stored: their text is fetched now. Reference labels: VirusTotal (src/relabel_vt.py ->
         column vt_label: phishing / clean / unsure); there are no hand labels.
score  : recomputes the error rates against vt_label (unsure rows left out):
         per subset row (with 95 % Wilson interval) and re-weighted by the sampling pools
         (weight = pool size / rows drawn), so the oversampled disagreements do not inflate rates.
Run:    python src/relabel_subset.py export --haiku-dev <eval_dev.json> --haiku-test <eval_test.json>
        python src/relabel_subset.py score
Output: reports/relabel/relabel_subset.csv, reports/relabel/relabel_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "llm"))
from llm.compare_models import wilson, with_override  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "relabel"
CSV_PATH = OUT_DIR / "relabel_subset.csv"
LAYER_D = ROOT / "reports" / "llm" / "layer_d"
GPT_RUNS = {"dev": LAYER_D / "eval_dev_v7_nonce.json", "test": LAYER_D / "eval_test_final_v2.json"}
SETS = {"dev": ROOT / "dataset" / "external" / "layer_d_dev.csv",
        "test": ROOT / "dataset" / "external" / "layer_d_test.csv"}
HTML_STORE = ROOT / "data" / "phreshphish_html.parquet"
LIVE_VI = ROOT / "reports" / "live" / "vi.json"
QUOTA = {"dev": 100, "test": 120, "vi": 80}
SNIPPET_CHARS = 500
SEED = 2026


def _verdicts(path: Path) -> dict[str, dict]:
    return {r["url"]: r for r in json.loads(path.read_text(encoding="utf-8"))["details"]}


def _phreshphish_rows(name: str, haiku_path: Path) -> pd.DataFrame:
    from html_sanitize import sanitize_html
    d = pd.read_csv(SETS[name])
    gpt, haiku = _verdicts(GPT_RUNS[name]), _verdicts(haiku_path)
    html = pd.read_parquet(HTML_STORE, filters=[("sha256", "in", list(set(d.sha256)))]).set_index("sha256").html
    rows = []
    for r in d.itertuples():
        g, h = gpt.get(r.url), haiku.get(r.url)
        if not g or not h or not g["verdict"] or not h["verdict"]:
            continue  # page without text (not sent to the LLM) or no verdict
        clean = sanitize_html(html.get(r.sha256, ""), url=r.url)
        label = "phishing" if r.label == "phishing" else "clean"
        d_final = with_override(g)
        disagree = ((label == "phishing" and g["verdict"] == "safe")
                    or (label == "clean" and g["verdict"] == "phishing")
                    or g["verdict"] != h["verdict"]
                    or (not r.C and g["verdict"] == "phishing"))
        rows.append({"source": f"phreshphish_{name}", "url": r.url, "title": clean["title"][:200],
                     "snippet": clean["text"][:SNIPPET_CHARS], "dataset_label": label,
                     "stratum": r.stratum, "B": bool(r.B), "C_score": round(float(r.score), 5),
                     "C_decision": "suspicious" if r.C else "clean",
                     "D_gpt": g["verdict"], "D_haiku": h["verdict"], "D_gpt_with_C_override": d_final,
                     "pool": "disagree" if disagree else "agree"})
    return pd.DataFrame(rows)


def _vi_rows() -> pd.DataFrame:
    live = json.loads(LIVE_VI.read_text(encoding="utf-8"))["details"]
    rows = []
    for r in live:
        bad = ((r["label"] == "phishing" and r["verdict"] == "clean")
               or (r["label"] != "phishing" and r["verdict"] == "phishing")
               or (r.get("llm_verdict") and r["llm_verdict"] == "safe" and r["label"] == "phishing"))
        rows.append({"source": "vi_live", "url": r["url"], "dataset_label":
                     "phishing" if r["label"] == "phishing" else "clean",
                     "C_score": r.get("c_score"), "D_gpt": r.get("llm_verdict"),
                     "pipeline_verdict": r["verdict"], "decided_by": r.get("decided_by"),
                     "pool": "disagree" if bad else "agree"})
    return pd.DataFrame(rows)


def _draw(d: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Half of `n` from each pool (all of a pool when it is smaller), with pool weights."""
    out = []
    for pool in ("disagree", "agree"):
        g = d[d.pool == pool]
        k = min(len(g), n // 2 if pool == "disagree" else n - min(n // 2, (d.pool == "disagree").sum()))
        pick = g.iloc[rng.choice(len(g), size=k, replace=False)] if k else g.iloc[:0]
        out.append(pick.assign(weight=round(len(g) / max(1, k), 4)))
    return pd.concat(out)


def export(haiku_dev: Path, haiku_test: Path) -> int:
    rng = np.random.default_rng(SEED)
    parts = [_draw(_phreshphish_rows("dev", haiku_dev), QUOTA["dev"], rng),
             _draw(_phreshphish_rows("test", haiku_test), QUOTA["test"], rng)]
    vi = _draw(_vi_rows(), QUOTA["vi"], rng)
    from llm_client import fetch  # SSRF-guarded fetch + sanitizer, as Layer D
    titles, snippets = [], []
    for u in vi.url:
        t, text, _, _ = fetch(u)
        titles.append(t[:200])
        snippets.append(text[:SNIPPET_CHARS] or "(page could not be downloaded now)")
    parts.append(vi.assign(title=titles, snippet=snippets))
    d = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)  # mix sources / pools
    d.insert(0, "id", range(1, len(d) + 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(d.groupby(["source", "pool"]).size().to_dict(), "->", CSV_PATH)
    return 0


REF = "vt_label"  # reference label column (VirusTotal, filled by relabel_vt.py)


def _rates(d: pd.DataFrame, col: str) -> dict:
    """Errors of verdict column `col` against the reference label; 'clean' and 'safe' both pass."""
    d = d[d[col].notna() & (d[col] != "")]
    ph, cl = d[d[REF] == "phishing"], d[d[REF] == "clean"]
    miss = ph[col].isin(["safe", "clean"])
    block = cl[col] == "phishing"
    warn = cl[col] == "suspicious"

    def both(mask: pd.Series, group: pd.DataFrame) -> dict:
        k, n = int(mask.sum()), len(group)
        w = group.weight
        return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n),
                "weighted_rate": round(float((w * mask).sum() / w.sum()), 4) if n else None}

    return {"phishing_passed": both(miss, ph), "clean_blocked": both(block, cl),
            "clean_warned": both(warn, cl)}


def score() -> int:
    d = pd.read_csv(CSV_PATH)
    if REF not in d:
        print(f"no {REF} column: run python src/relabel_vt.py first")
        return 1
    d[REF] = d[REF].astype(str).str.strip().str.lower()
    labelled = d[d[REF].isin(["phishing", "clean"])]
    print(f"{len(labelled)} of {len(d)} rows labelled (unsure / empty left out)")
    noise = labelled.dataset_label != labelled[REF]
    out = {"n_rows": len(d), "n_labelled": len(labelled),
           "dataset_label_wrong": {"k": int(noise.sum()), "n": len(labelled),
                                   "rate": round(float(noise.mean()), 4) if len(labelled) else None,
                                   "ci95": wilson(int(noise.sum()), len(labelled)),
                                   "by_source": labelled.assign(w=noise).groupby("source").w.mean()
                                   .round(4).to_dict()}}
    for src, g in [("all", labelled), *labelled.groupby("source")]:
        out[src] = {col: _rates(g, col) for col in
                    ("D_gpt", "D_gpt_with_C_override", "D_haiku", "pipeline_verdict")
                    if col in g and g[col].notna().any()}
    (OUT_DIR / "relabel_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["dataset_label_wrong"], indent=2))
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--haiku-dev", type=Path, required=True)
    e.add_argument("--haiku-test", type=Path, required=True)
    sub.add_parser("score")
    args = ap.parse_args()
    return export(args.haiku_dev, args.haiku_test) if args.cmd == "export" else score()


if __name__ == "__main__":
    sys.exit(main())
