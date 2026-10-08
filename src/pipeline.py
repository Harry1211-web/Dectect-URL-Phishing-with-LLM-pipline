"""Full detection pipeline for one clicked URL: A -> B -> C -> D.

  A. blocklist      known phishing URL/host (also redirect targets)     -> phishing (stop)
  B. typosquatting  brand impersonation (also in redirect targets)      -> suspicious (needs D)
  C. URL classifier CNN/BiLSTM + XGBoost (+ host features), high-recall gate
                    -> clean (stop) / suspicious
  D. LLM + RAG      reads the page, final verdict for suspicious URLs

Fallback when D is disabled, times out or fails: a B hit or a high classifier score gives
"phishing", anything else that reached D stays "suspicious" (the user is warned, not blocked).

Run: python src/pipeline.py <url> [<url> ...] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import predict as classifier  # noqa: E402

# Classifier score above which the fallback labels a URL "phishing" without the LLM
FALLBACK_PHISHING_SCORE = 0.9


def analyze_url(url: str, use_llm: bool = True, llm_timeout_s: float | None = None,
                use_host_features: bool = True, explain: bool = False) -> dict:
    """Run the layers in order and return {verdict, decided_by, explanation, layers, latency_ms}.

    explain=True asks the LLM for a Vietnamese explanation (user's choice, extra cost); by
    default Layer D only classifies.
    """
    t0 = time.perf_counter()
    layers: dict = {}

    def done(verdict: str, decided_by: str, explanation: str) -> dict:
        return {"url": url, "verdict": verdict, "decided_by": decided_by,
                "explanation": explanation, "layers": layers,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    a = check_blocklist(url)
    layers["A_blocklist"] = a
    if a["flag"] is True:
        return done("phishing", "A_blocklist", a["reason"])

    b = check_typosquatting_deep(url)
    layers["B_typosquatting"] = b

    c = (classifier.classify_url(url, use_host_features=use_host_features)
         if classifier.is_available() else None)
    layers["C_classifier"] = c or {"label": "unknown", "reason": "model not trained"}
    if b["flag"] is not True and c is not None and c["label"] == "clean":
        return done("clean", "C_classifier",
                    f"Mô hình URL đánh giá an toàn (điểm {c['score']:.3f} < ngưỡng {c['threshold']:.3f}).")

    if use_llm:
        from llm_rag.llm_client import DEFAULT_LLM_TIMEOUT_S, analyze_suspicious
        t_d = time.perf_counter()
        d = analyze_suspicious(url, rule_results=[b], explain=explain,
                               timeout_s=llm_timeout_s or DEFAULT_LLM_TIMEOUT_S)
        d["latency_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        layers["D_llm_rag"] = d
        if not d["timed_out"]:
            verdict = "phishing" if d["verdict"] == "phishing" else "clean"
            return done(verdict, "D_llm_rag", d["explanation"] or (
                f"LLM phân loại trang là {'lừa đảo' if verdict == 'phishing' else 'an toàn'} "
                "(chọn xem giải thích chi tiết nếu cần)."))

    # Fallback: no usable LLM verdict
    score = c["score"] if c else 0.0
    if b["flag"] is True:
        return done("phishing", "fallback_B", b["reason"])
    if score >= FALLBACK_PHISHING_SCORE:
        return done("phishing", "fallback_C",
                    f"Mô hình URL đánh giá rủi ro rất cao (điểm {score:.3f}).")
    return done("suspicious", "fallback_C",
                f"URL có dấu hiệu đáng ngờ (điểm {score:.3f}) nhưng chưa phân tích được nội dung "
                "trang — hãy cẩn trọng.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--no-llm", action="store_true", help="skip Layer D (no API cost)")
    ap.add_argument("--no-host", action="store_true", help="URL-only classifier (no network lookups)")
    ap.add_argument("--explain", action="store_true", help="ask the LLM for an explanation")
    args = ap.parse_args()
    for u in args.urls:
        print(json.dumps(analyze_url(u, use_llm=not args.no_llm, use_host_features=not args.no_host,
                                     explain=args.explain),
                         ensure_ascii=False, indent=2))
