"""Before / after the international extension of the trusted-domain shortcut (pipeline.py).

before : verified official domains of Vietnamese brands only
after  : + official domains of international brands that host no third-party pages
Per set: clean URLs that reach Layer D today (A does not block, B or C flags them) and that the
shortcut now ends as clean (LLM calls saved), and phishing URLs the shortcut would trust (new
misses: they skip B, C and D). Sets: dataset test split, PhreshPhish, Vietnamese set (A/B/C from
the host cache, no network), every phishing URL of the dataset (train / val / test, shortcut
only), and the clean pages of the live top-site run that reached Layer D.
The international list was chosen on train / val phishing only; the other sets measure it.
Run:    python src/evaluate_trusted_shortcut.py
Output: reports/pipeline/trusted_shortcut.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocklist.lookup import check_blocklist  # noqa: E402
from pipeline import trusted_intl_domains, trusted_official_domain  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pipeline" / "trusted_shortcut.json"
DATASET = ROOT / "dataset" / "urls" / "urls_labeled.csv"


def load_sets() -> pd.DataFrame:
    d = pd.read_csv(DATASET, usecols=["url", "label", "split"])
    sets = [d[d.split == "test"].assign(set="dataset_test")[["url", "label", "set"]],
            pd.read_csv(ROOT / "dataset" / "external" / "phreshphish_urls.csv")
            .assign(set="phreshphish")[["url", "label", "set"]]]
    for lab in ("phishing", "clean"):
        v = pd.read_csv(ROOT / "data" / "test_sets" / "vi" / f"{lab}.csv", usecols=["url"])
        sets.append(v.assign(label=lab, set="vi"))
    return pd.concat(sets, ignore_index=True)


def route(d: pd.DataFrame) -> pd.DataFrame:
    """A, B, C decisions per URL (C from cached host data)."""
    urls = d.url.tolist()
    d = d.assign(A=[check_blocklist(u)["flag"] is True for u in urls],
                 B=[check_typosquatting_deep(u)["flag"] is True for u in urls])
    parts = [predict.score_urls(urls[i:i + 4096], host_mode="cache") for i in range(0, len(urls), 4096)]
    d["C"] = np.concatenate([p[0] for p in parts]) >= np.concatenate([p[1] for p in parts])
    return d


def counts(g: pd.DataFrame) -> dict:
    ph, cl = g[g.label == "phishing"], g[g.label == "clean"]
    reach = ~cl.A & (cl.B | cl.C)
    saved = reach & cl.after & ~cl.before
    caught = ph.A | ph.B | ph.C
    return {"clean": len(cl), "clean_reaching_D_before": int((reach & ~cl.before).sum()),
            "clean_saved_from_D": int(saved.sum()),
            "clean_saved_rate": round(float(saved.sum() / max(1, (reach & ~cl.before).sum())), 4),
            "clean_saved_examples": cl[saved].url.head(15).tolist(),
            "phishing": len(ph), "phishing_trusted_before": int(ph.before.sum()),
            "phishing_trusted_after": int(ph.after.sum()),
            "new_misses (caught before, trusted after)": int((ph.after & ~ph.before & caught).sum()),
            "phishing_trusted_examples": ph[ph.after & ~ph.before].url.head(15).tolist()}


def main() -> int:
    d = route(load_sets())
    d["before"] = [trusted_official_domain(u, include_intl=False) is not None for u in d.url]
    d["after"] = [trusted_official_domain(u) is not None for u in d.url]
    out = {"time": datetime.now(timezone.utc).isoformat(),
           "n_trusted_intl_domains": len(trusted_intl_domains()),
           "sets": {s: counts(g) for s, g in d.groupby("set")}}

    full = pd.read_csv(DATASET, usecols=["url", "label", "split"])
    ph = full[full.label == "phishing"]
    hits = ph[[trusted_official_domain(u) is not None and trusted_official_domain(u, False) is None
               for u in ph.url]]
    out["dataset_all_phishing"] = {"n": len(ph), "trusted_after_only": len(hits),
                                   "by_split": hits.split.value_counts().to_dict(),
                                   "examples": hits.url.head(20).tolist()}

    live = ROOT / "reports" / "live" / "top_sites.json"
    if live.exists():
        rows = [r for r in json.loads(live.read_text(encoding="utf-8"))["details"]
                if r["label"] != "phishing" and str(r.get("decided_by", "")).startswith(("D_", "fallback"))]
        saved = [r["url"] for r in rows if trusted_official_domain(r["url"]) is not None]
        out["live_top_sites_clean_reaching_D"] = {"n": len(rows), "saved": len(saved),
                                                  "examples": saved[:15]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "sets"}, indent=2))
    for s, c in out["sets"].items():
        print(s, {k: v for k, v in c.items() if "examples" not in k})
    return 0


if __name__ == "__main__":
    sys.exit(main())
