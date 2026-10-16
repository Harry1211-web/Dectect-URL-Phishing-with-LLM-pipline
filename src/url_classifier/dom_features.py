"""DOM model of Layer C: page HTML -> phishing score, used when the page can be downloaded.

Features (DOM_FEATURE_NAMES): forms and where they post, password / hidden inputs, links,
scripts, iframes, images and favicon pointing to other domains, meta refresh, obfuscated
JavaScript calls, right-click blocking, text size, brand names of the brand catalogue mentioned
off their official domains, and bait words (urgency / credential lures, every language of
data/injection_patterns/), plus the sanitizer's prompt-injection flag.

Data: PhreshPhish pages (test split, collected since evaluate_external.SINCE) that are NOT in the
20k external Layer C sample, the Layer D dev / test sets, or any of their registrable domains,
so every number already reported on those sets stays valid. Train / val split by registrable
domain. ML model chosen on val among XGBoost, LightGBM and random forest (highest val PR-AUC);
the way its score is combined with the URL score is chosen on val by `combine`. The DOM score can
only ADD suspicion ("or"): the page is attacker-controlled and can be cloaked (a clean-looking
page served to the fetcher), so it never clears a URL that Layer C flagged. The "and" rule (DOM
may clear a flag) is still measured on val and reported, never used.

Run:    python src/url_classifier/dom_features.py build [--per-label 3000]   (downloads HTML)
        python src/url_classifier/dom_features.py train
        python src/url_classifier/dom_features.py combine     (after the final URL model)
Output: dataset/external/dom_pages.csv, models/url_classifier/dom_model.*, dom_config.json,
        reports/layer_c_selection/dom.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

_SRC = Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "llm")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from html_sanitize import PATTERNS, sanitize_html  # noqa: E402
from typosquatting.brands import load_all_brands  # noqa: E402
from url_utils import registrable_domain  # noqa: E402

ROOT = _SRC.parent
PAGES_PATH = ROOT / "dataset" / "external" / "dom_pages.csv"
MODEL_DIR = ROOT / "models" / "url_classifier"
REPORT_PATH = ROOT / "reports" / "layer_c_selection" / "dom.json"
VAL_SHARE = 20  # % of registrable domains in val
MIN_BRAND_LEN = 4  # shorter brand labels match ordinary words
SEED = 42

DOM_FEATURE_NAMES = [
    "n_forms", "n_forms_external_action", "n_forms_empty_action", "n_inputs", "n_password_inputs",
    "n_hidden_inputs", "n_links", "external_link_ratio", "null_link_ratio", "n_external_domains",
    "n_scripts", "external_script_ratio", "n_iframes", "n_images", "external_image_ratio",
    "favicon_external", "has_meta_refresh", "n_obfuscation_calls", "blocks_right_click",
    "title_len", "text_len", "n_brands_off_domain", "brand_in_title_off_domain", "n_bait_words",
    "bait_words_per_kchar", "injection_suspected",
]
_RE_OBFUSCATION = re.compile(r"\beval\(|\batob\(|\bunescape\(|document\.write\(|fromCharCode", re.I)
_RE_RIGHT_CLICK = re.compile(r"contextmenu|event\.button\s*==\s*2", re.I)
_RE_BAIT = PATTERNS.regex("bait_words")


@lru_cache(maxsize=1)
def _brands() -> tuple[tuple[str, re.Pattern, frozenset[str]], ...]:
    out = []
    for b in load_all_brands():
        if len(b["label"]) >= MIN_BRAND_LEN:
            out.append((b["label"], re.compile(rf"\b{re.escape(b['label'])}\b", re.I),
                        frozenset(registrable_domain("http://" + d) for d in b["domains"])))
    return tuple(out)


def _domain(url: str, base: str) -> str:
    try:
        return registrable_domain(urljoin(base, url))
    except ValueError:
        return ""


def dom_features(html: str, url: str) -> list[float]:
    """Feature vector (DOM_FEATURE_NAMES order) of one page; zeros for an empty page."""
    if not html:
        return [0.0] * len(DOM_FEATURE_NAMES)
    soup = BeautifulSoup(html, "html.parser")
    own = registrable_domain(url)
    forms = soup.find_all("form")
    actions = [(f.get("action") or "").strip() for f in forms]
    inputs = soup.find_all("input")
    types = [(i.get("type") or "text").lower() for i in inputs]
    hrefs = [(a.get("href") or "").strip() for a in soup.find_all("a")]
    link_domains = [_domain(h, url) for h in hrefs if h.lower().startswith(("http", "//"))]
    scripts = [s.get("src") for s in soup.find_all("script")]
    ext_scripts = [s for s in scripts if s and _domain(s, url) not in ("", own)]
    imgs = [i.get("src") or "" for i in soup.find_all("img")]
    icon = next((l.get("href") for l in soup.find_all("link")
                 if "icon" in " ".join(l.get("rel") or []).lower()), None)
    clean = sanitize_html(html, url=url)
    text = f"{clean['title']} {clean['text']}"
    brands = [(lab, doms) for lab, rx, doms in _brands() if rx.search(text) and own not in doms]
    title_brand = any(re.search(rf"\b{re.escape(lab)}\b", clean["title"], re.I) for lab, _ in brands)
    n_bait = len(_RE_BAIT.findall(text))
    inline_js = " ".join(s.get_text(" ") for s in soup.find_all("script") if not s.get("src"))
    return [
        len(forms),
        sum(a.lower().startswith(("http", "//")) and _domain(a, url) != own for a in actions),
        sum(a in ("", "#") or a.lower().startswith("javascript") for a in actions),
        len(inputs), types.count("password"), types.count("hidden"),
        len(hrefs),
        sum(d != own for d in link_domains) / max(1, len(hrefs)),
        sum(h in ("", "#") or h.lower().startswith("javascript") for h in hrefs) / max(1, len(hrefs)),
        len({d for d in link_domains if d != own}),
        len(scripts), len(ext_scripts) / max(1, len(scripts)),
        len(soup.find_all("iframe")),
        len(imgs), sum(i.lower().startswith(("http", "//")) and _domain(i, url) != own
                       for i in imgs) / max(1, len(imgs)),
        float(bool(icon) and icon.lower().startswith(("http", "//")) and _domain(icon, url) != own),
        float(soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}) is not None),
        len(_RE_OBFUSCATION.findall(inline_js)),
        float(bool(_RE_RIGHT_CLICK.search(html[:200_000]))),
        len(clean["title"]), len(clean["text"]),
        len(brands), float(title_brand),
        n_bait, 1000 * n_bait / max(1, len(text)),
        float(clean["prompt_injection_suspected"]),
    ]


def dom_matrix(html: list[str], urls: list[str]) -> np.ndarray:
    return np.array([dom_features(h, u) for h, u in zip(html, urls)], dtype=np.float32)


# ------------------------------------------------------------------------------------- data
def _val_domain(domain: str) -> bool:
    return int(hashlib.md5(f"dom:{domain}".encode()).hexdigest(), 16) % 100 < VAL_SHARE


def build(per_label: int) -> int:
    """Draw PhreshPhish pages outside every evaluation set and its domains, download their HTML."""
    from llm import phreshphish_sets as P
    from url_utils import normalize_url
    index = P.load_index()
    used = pd.concat([pd.read_csv(P.SAMPLE_CSV, usecols=["url"]),
                      *(pd.read_csv(P.OUT_DIR / f"layer_d_{s}.csv", usecols=["url"])
                        for s in ("dev", "test"))])
    used_norm, used_dom = set(used.url.map(normalize_url)), set(used.url.map(registrable_domain))
    pool = index[~index.norm.isin(used_norm) & ~index.domain.isin(used_dom)]
    print(f"pool outside the evaluation sets: {len(pool)}", pool.label.value_counts().to_dict(), flush=True)
    d = P.draw(pool, {"phishing": int(per_label * 1.3), "clean": int(per_label * 1.3)})
    html = P.fetch_html(d)
    d = d[[bool(html.get(s)) for s in d.sha256]]
    d = d.groupby("label").head(per_label)
    d["split"] = np.where(d.domain.map(_val_domain), "val", "train")
    d[["url", "label", "target", "date", "lang", "sha256", "domain", "split"]].to_csv(PAGES_PATH, index=False)
    print(d.groupby(["split", "label"]).size().to_dict(), flush=True)
    return 0


def load_pages() -> pd.DataFrame:
    from llm.phreshphish_sets import HTML_STORE
    d = pd.read_csv(PAGES_PATH)
    html = pd.read_parquet(HTML_STORE, filters=[("sha256", "in", list(set(d.sha256)))]).set_index("sha256").html
    d["html"] = d.sha256.map(html).fillna("")
    d["y"] = (d.label == "phishing").astype(np.int64)
    return d


# ------------------------------------------------------------------------------------- model
def train() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from baselines.common import metrics_at, threshold_for_clean
    from url_classifier.train import ML_EXT, ML_MODELS, fit_ml, save_ml
    d = load_pages()
    x = dom_matrix(d.html.tolist(), d.url.tolist())
    tr, va = (d.split == "train").to_numpy(), (d.split == "val").to_numpy()
    y = d.y.to_numpy()
    out = {"n": {f"{s}_{lab}": int(n) for (s, lab), n in d.groupby(["split", "label"]).size().items()},
           "candidates": {}}
    models = {}
    for ml in ML_MODELS:
        m = fit_ml(ml, x[tr], y[tr], x[va], y[va], SEED)
        p = m.predict_proba(x[va])[:, 1]
        t = threshold_for_clean(p, y[va])
        out["candidates"][ml] = {"val_pr_auc": round(float(average_precision_score(y[va], p)), 5),
                                 "val_roc_auc": round(float(roc_auc_score(y[va], p)), 5),
                                 "val_at_50pct_clean": metrics_at(y[va], p, t)}
        models[ml] = m
        print(ml, json.dumps(out["candidates"][ml]), flush=True)
    best = max(out["candidates"], key=lambda k: out["candidates"][k]["val_pr_auc"])
    imp = getattr(models[best], "feature_importances_", None)
    out.update({"time": datetime.now(timezone.utc).isoformat(), "winner": best,
                "rule": "highest val PR-AUC",
                "feature_importance": sorted(zip(DOM_FEATURE_NAMES, np.asarray(imp, float).tolist()),
                                             key=lambda z: -z[1]) if imp is not None else []})
    for e in ML_EXT.values():
        (MODEL_DIR / f"dom_model.{e}").unlink(missing_ok=True)
    save_ml(best, models[best], MODEL_DIR / f"dom_model.{ML_EXT[best]}")
    (MODEL_DIR / "dom_config.json").write_text(json.dumps(
        {"ml": best, "feature_names": DOM_FEATURE_NAMES, "trained_at": out["time"]}, indent=2),
        encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


@lru_cache(maxsize=1)
def _load_dom():
    from url_classifier.predict import _ML_EXT, _load_ml
    cfg_path = MODEL_DIR / "dom_config.json"
    if not cfg_path.exists():
        return None, {}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _load_ml(cfg["ml"], MODEL_DIR / f"dom_model.{_ML_EXT[cfg['ml']]}"), cfg


def dom_score(html: str, url: str) -> float | None:
    """Phishing probability of the page from its DOM; None when no DOM model or empty page."""
    model, _ = _load_dom()
    if model is None or not html:
        return None
    return float(model.predict_proba(dom_matrix([html], [url]))[0, 1])


def combine() -> int:
    """Choose on DOM val how the DOM score joins the final URL decision (see pipeline)."""
    from url_classifier import predict
    d = load_pages()
    va = d[d.split == "val"].reset_index(drop=True)
    model, cfg = _load_dom()
    p_dom = model.predict_proba(dom_matrix(va.html.tolist(), va.url.tolist()))[:, 1]
    # Live host lookups first (stored in the host cache), as in production: these hosts were
    # never collected; a host whose lookup fails stays on the lexical branch
    from data_collection.vi_testset import lookup
    from url_utils import host_of
    lookup(sorted({host_of(u) for u in va.url}))
    s, thr, _ = predict.score_urls(va.url.tolist(), host_mode="cache")
    url_flag = s >= thr
    y = va.y.to_numpy()

    def outcome(flag: np.ndarray) -> dict:
        return {"phishing_missed": int(((y == 1) & ~flag).sum()), "phishing": int((y == 1).sum()),
                "clean_recognized": round(float(((y == 0) & ~flag).sum() / max(1, (y == 0).sum())), 5)}

    res = {"url_only": outcome(url_flag)}
    # "or": the DOM score can only add suspicion (a URL-clean page with a phishing-like DOM goes
    # to D); its threshold keeps the clean share at least that of "and" below
    grid = np.unique(np.quantile(p_dom, np.linspace(0, 1, 201)))
    floor = 0.5
    best_or = None
    for t in grid[::-1]:
        o = outcome(url_flag | (p_dom >= t))
        if o["clean_recognized"] >= floor:
            best_or = (float(t), o)
    # "and" (DOM may also clear a URL flag): measured for the report only, never selectable
    best_and = None
    for t in grid:
        o = outcome(url_flag & (p_dom >= t))
        if o["clean_recognized"] >= floor and (best_and is None or o["phishing_missed"] <= best_and[1]["phishing_missed"]):
            best_and = (float(t), o)
    for name, b in (("or", best_or), ("and", best_and)):
        res[name] = {"dom_threshold": b[0], **b[1]} if b else None
    feasible = {k: v for k, v in res.items()
                if k != "and" and v and v["clean_recognized"] >= floor}
    choice = (min(feasible, key=lambda k: (feasible[k]["phishing_missed"], -feasible[k]["clean_recognized"]))
              if feasible else "url_only")
    cfg.update({"combine": choice, "dom_threshold": (res[choice] or {}).get("dom_threshold"),
                "combine_rule": f"fewest val phishing missed with >= {floor:.0%} val clean recognised"})
    (MODEL_DIR / "dom_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rep["combine_val"] = {"options": res, "choice": choice, "n_val": len(va)}
    REPORT_PATH.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["combine_val"], indent=2))
    _load_dom.cache_clear()
    return 0


def is_active() -> bool:
    """True when a DOM model exists and val chose to combine it with the URL score."""
    model, cfg = _load_dom()
    return model is not None and cfg.get("combine", "url_only") != "url_only"


def combined_label(url_label: str, p_dom: float | None) -> str:
    """Final Layer C label once the DOM score is known (dom_config.json 'combine')."""
    _, cfg = _load_dom()
    mode, t = cfg.get("combine", "url_only"), cfg.get("dom_threshold")
    if p_dom is None or t is None or mode == "url_only":
        return url_label
    # Only "or" exists: the page can add suspicion, never remove it (cloaking)
    return "suspicious" if url_label == "suspicious" or p_dom >= t else "clean"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("build", "train", "combine"))
    ap.add_argument("--per-label", type=int, default=3000)
    args = ap.parse_args()
    return {"build": lambda: build(args.per_label), "train": train, "combine": combine}[args.step]()


if __name__ == "__main__":
    sys.exit(main())
