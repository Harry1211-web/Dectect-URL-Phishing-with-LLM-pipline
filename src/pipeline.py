"""Full detection pipeline for one clicked URL: A -> B -> C -> D.

  A. blocklist      known phishing URL/host (also redirect targets)     -> phishing (stop)
     verified official domain of a Vietnamese brand (no user content, no hidden redirect)
                                                                        -> clean (stop)
  B. typosquatting  brand impersonation (also in redirect targets)      -> suspicious (needs D)
  C. URL classifier CNN/BiLSTM + XGBoost (+ host features), high-recall gate
                    -> clean (stop) / suspicious
  D. LLM            reads the page, final verdict for suspicious URLs; a "safe" verdict on a
                    URL that C scores as near-certain phishing becomes "suspicious"

Fallback when D is disabled, times out or fails: a B hit or a high classifier score gives
"phishing", anything else that reached D stays "suspicious" (the user is warned, not blocked).

Run: python src/pipeline.py <url> [<url> ...] [--no-llm] [--log]   (--log: see monitoring.py)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocklist.lookup import check_blocklist  # noqa: E402
from typosquatting.brands import (  # noqa: E402
    all_official_domains, label_of_domain, load_vn_brands, verified_official_domains,
)
from typosquatting.paas_domains import is_user_content_host  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import predict as classifier  # noqa: E402
from url_utils import (  # noqa: E402
    embedded_urls, ensure_scheme, host_of, is_ip_host, registrable_domain, split_host,
)

# Classifier score above which the fallback labels a URL "phishing" without the LLM
FALLBACK_PHISHING_SCORE = 0.9
# An LLM "safe" verdict is overruled to "suspicious" when the URL classifier is this sure: chosen
# on the Layer D dev set (catches 26/40 of D's phishing misses, touches 4/291 clean pages).
# Not applied on official brand domains: there a high score is a classifier error (google.com
# login pages), and no dev / test phishing page fell in that case
LLM_SAFE_OVERRIDE_SCORE = 0.99

# Verified official domains that still go through B/C/D: marketplaces and classifieds host
# third-party content (fake shops, scam listings); zalo.me serves user profiles and group links
TRUST_EXCLUDED_CATEGORIES = frozenset({"ecommerce"})
TRUST_EXCLUDED_DOMAINS = frozenset({"zalo.me"})

# Allowlist, not a blocklist of redirect forms: the shortcut skips every check, so it only takes a
# bare path (no query, no fragment: redirect parameters and client-side '#' redirects have too many
# encodings and parsers to rule out). The decoded path may only hold letters, digits and '-._~/',
# with no '//' and no segment that is a host or an IP ('/go/evil.com'); anything else goes through
# B/C/D. Costs about 8 % of the shortcut hits on training clean URLs (utm_ links), none on the VN set.
_RE_ALLOWED_PATH = re.compile(r"[\w\-.~/]*")
_DOTS = str.maketrans({"。": ".", "．": ".", "｡": "."})


def _decode(text: str, rounds: int = 3) -> str:
    # NFKC: full-width forms ('％2E', 'ｅｖｉｌ'); browsers also read ideographic dots in hosts as '.'
    for _ in range(rounds):
        text, before = unquote(unicodedata.normalize("NFKC", text).translate(_DOTS)), text
        if text == before:
            break
    return unicodedata.normalize("NFKC", text).translate(_DOTS)


def _names_a_host(token: str) -> bool:
    token = token.strip(".-_~").lower()
    _, dom, suf = split_host(token)
    return bool(dom and suf) or is_ip_host(token)


def may_redirect(url: str) -> bool:
    """True unless the URL is a bare path that cannot name another destination (open redirect on
    an official domain): any query or fragment, '/go/evil.com', '//evil.com', encoded forms..."""
    if "?" in url or "#" in url:
        return True
    try:
        parts = urlsplit(ensure_scheme(url))
    except ValueError:
        return True
    path = _decode(parts.path)
    if not _RE_ALLOWED_PATH.fullmatch(path) or "//" in path:
        return True
    return any(_names_a_host(seg) for seg in path.split("/") if "." in seg)


def trusted_official_domain(url: str) -> str | None:
    """Registrable domain when the URL is a page of a verified official Vietnamese brand domain
    that can be trusted without further checks, else None."""
    host = host_of(url)
    domain = registrable_domain(host)
    if domain not in verified_official_domains() or domain in TRUST_EXCLUDED_DOMAINS:
        return None
    label = label_of_domain(domain)
    if any(b["label"] == label and b["category"] in TRUST_EXCLUDED_CATEGORIES
           for b in load_vn_brands()):
        return None
    # User-published content or a hidden destination (open redirect) is not the brand's own page
    if "\\" in url or is_user_content_host(host) or embedded_urls(url) or may_redirect(url):
        return None
    return domain


def analyze_url(url: str, use_llm: bool = True, llm_timeout_s: float | None = None,
                use_host_features: bool = True, explain: bool = False) -> dict:
    """Run the layers in order and return {verdict, decided_by, explanation, layers, latency_ms}.

    explain=True asks the LLM for an explanation in the page's language (user's choice, extra
    cost); by default Layer D only classifies.
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

    official = trusted_official_domain(url)
    if official:
        return done("clean", "verified_official_domain",
                    f"{official} is a verified official domain of '{label_of_domain(official)}'.")

    b = check_typosquatting_deep(url)
    layers["B_typosquatting"] = b

    c = (classifier.classify_url(url, use_host_features=use_host_features)
         if classifier.is_available() else None)
    layers["C_classifier"] = c or {"label": "unknown", "reason": "model not trained"}
    if b["flag"] is not True and c is not None and c["label"] == "clean":
        return done("clean", "C_classifier",
                    f"URL model rates it safe (score {c['score']:.3f} < threshold {c['threshold']:.3f}).")

    if use_llm:
        from llm.llm_client import DEFAULT_LLM_TIMEOUT_S, analyze_suspicious
        t_d = time.perf_counter()
        d = analyze_suspicious(url, rule_results=[b], explain=explain,
                               timeout_s=llm_timeout_s or DEFAULT_LLM_TIMEOUT_S)
        d["latency_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        layers["D_llm"] = d
        if not d["timed_out"]:
            # LLM 'suspicious' = not enough evidence either way: warn the user, never 'clean'
            verdict = {"phishing": "phishing", "suspicious": "suspicious"}.get(d["verdict"], "clean")
            score = (c or {}).get("score")
            if (verdict == "clean" and score is not None and score >= LLM_SAFE_OVERRIDE_SCORE
                    and registrable_domain(url) not in all_official_domains()):
                return done("suspicious", "D_llm+C_override",
                            f"The page content looks safe to the LLM, but the URL model rates the "
                            f"URL as very risky (score {score:.3f}) — be careful.")
            return done(verdict, "D_llm", d["explanation"] or (
                f"The LLM classified the page as {d['verdict']} "
                "(ask for the detailed explanation if needed)."))

    # Fallback: no usable LLM verdict
    score = (c or {}).get("score")
    if b["flag"] is True:
        return done("phishing", "fallback_B", b["reason"])
    if score is None:  # classifier unavailable or host lookup failed: fail closed, never "clean"
        return done("suspicious", "fallback_lookup_failed",
                    "The domain could not be checked and the page content could not be analysed — be careful.")
    if score >= FALLBACK_PHISHING_SCORE:
        return done("phishing", "fallback_C",
                    f"URL model rates the risk as very high (score {score:.3f}).")
    return done("suspicious", "fallback_C",
                f"The URL looks suspicious (score {score:.3f}) but the page content could not be "
                "analysed — be careful.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--no-llm", action="store_true", help="skip Layer D (no API cost)")
    ap.add_argument("--no-host", action="store_true", help="URL-only classifier (no network lookups)")
    ap.add_argument("--explain", action="store_true", help="ask the LLM for an explanation")
    ap.add_argument("--log", action="store_true", help="log the verdict for false-alarm monitoring")
    args = ap.parse_args()
    for u in args.urls:
        res = analyze_url(u, use_llm=not args.no_llm, use_host_features=not args.no_host,
                          explain=args.explain)
        if args.log:
            from monitoring import record_verdict
            record_verdict(res)
        print(json.dumps(res, ensure_ascii=False, indent=2))
