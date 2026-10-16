"""Evaluate Layer B (typosquatting) and compare it with the old Levenshtein-only rule.

Sets
  clean    : Tranco top-100k homepages + crawled clean URLs (dataset test split)
  phishing : phishing URLs of the dataset test split (only part of them impersonate a brand)
  vn_brand : simulated attacks on Vietnamese brands (llm scenarios) — all impersonate a brand
  phreshphish_clean / phreshphish_phishing, vi_clean : external sets (never used for tuning)
Variants
  dl_jw      : Damerau-Levenshtein + Jaro-Winkler gate (production)
  dl_only    : Damerau-Levenshtein without the Jaro-Winkler gate
  compound_multilingual_lure : brand + word domains ("kleinanzeigen-zahlung.de") also count as a
               lure with German / French / Spanish / Italian / Portuguese / Dutch / Polish words
               (REJECTED: its extra phishing hits were all caught by A or C already, and it flags
               brands' own regional domains such as shopeepay.sg)
  compound_no_lure : any embedded brand of >= 5 characters, no lure word needed (REJECTED:
               Tranco top-100k false positives 232 -> 906)
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
    import pandas as pd
    ext = pd.read_csv(ROOT / "dataset" / "external" / "phreshphish_urls.csv")
    vi_clean = pd.read_csv(ROOT / "data" / "test_sets" / "vi" / "clean.csv").url.tolist()
    return {"clean_tranco_top100k": tranco, "clean_crawled_test": clean_crawl,
            "phishing_test": phishing, "vn_brand_scenarios": vn,
            "phreshphish_clean": ext[ext.label == "clean"].url.tolist(),
            "phreshphish_phishing": ext[ext.label == "phishing"].url.tolist(),
            "vi_clean": vi_clean}


# Lure words of the rejected compound_multilingual_lure variant
_MULTILINGUAL_LURE = frozenset({
    "zahlung", "bezahlung", "bezahlen", "konto", "kundenkonto", "sicherheit", "kundenservice",
    "kunden", "paket", "sendung", "verifizierung", "bestaetigung", "anmelden", "anmeldung",
    "rueckerstattung", "erstattung", "rechnung", "versand", "paiement", "compte", "securite",
    "connexion", "livraison", "colis", "remboursement", "facture", "verification", "pago", "pagos",
    "cuenta", "seguridad", "verificacion", "envio", "paquete", "reembolso", "pagamento", "conta",
    "seguranca", "entrega", "encomenda", "conto", "sicurezza", "verifica", "spedizione", "rimborso",
    "betaling", "betalen", "rekening", "veiligheid", "pakket", "bezorging", "platnosc",
    "przesylka", "dostawa", "delivery", "parcel", "package", "tracking", "shipment", "invoice",
    "payments", "checkout", "pay", "order", "orders", "unlock", "help",
})
_PRODUCTION_LURE = ts._embedding_is_lure


def _lure_multilingual(brand: str, label: str, path_tokens: set[str], suffix: str) -> bool:
    """Words under 6 letters only as the exact rest of the label, longer ones as a substring."""
    if _PRODUCTION_LURE(brand, label, path_tokens, suffix):
        return True
    rest = label.replace(brand, "")
    return any(w == rest or (len(w) >= 6 and w in rest) for w in _MULTILINGUAL_LURE)


def _lure_none(brand: str, label: str, path_tokens: set[str], suffix: str) -> bool:
    return len(brand) >= 5 or _PRODUCTION_LURE(brand, label, path_tokens, suffix)


def _run(urls: list[str]) -> dict:
    flagged, kinds, lat, examples = 0, Counter(), [], []
    for u in urls:
        t0 = time.perf_counter()
        r = ts.check_typosquatting_deep(u)
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
    jw = (ts.JW_MIN_SHORT, ts.JW_MIN_LONG)
    for variant, (short, long_), lure in (("dl_jw", jw, _PRODUCTION_LURE),
                                          ("dl_only", (0.0, 0.0), _PRODUCTION_LURE),
                                          ("compound_multilingual_lure", jw, _lure_multilingual),
                                          ("compound_no_lure", jw, _lure_none)):
        ts.JW_MIN_SHORT, ts.JW_MIN_LONG = short, long_
        ts._embedding_is_lure = lure
        report[variant] = {name: _run(urls) for name, urls in sets.items()}
        print(variant, {k: (v["flagged"], v["n"]) for k, v in report[variant].items()}, flush=True)
    ts.JW_MIN_SHORT, ts.JW_MIN_LONG = jw
    ts._embedding_is_lure = _PRODUCTION_LURE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
