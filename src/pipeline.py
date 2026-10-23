"""Full detection pipeline for one clicked URL: A -> B -> C -> D.

  A. blocklist      known phishing URL/host (also redirect targets)     -> phishing (stop)
                    (a listed plain page on the apex / www host of a verified Vietnamese brand
                    domain, likely a feed error: every layer, never "clean")
     verified official domain of a Vietnamese brand, or official domain of an international
     brand that hosts no third-party content (no user content, no hidden redirect)
                                                                        -> clean (stop)
  B. typosquatting  brand impersonation (also in redirect targets)      -> suspicious (needs D)
     Before D the page is downloaded once; when it redirects elsewhere, the final URL goes
     through A (listed -> phishing, stop) and B as well
  C. URL classifier DL score + ML classifier (+ host features), high-recall gate; when the
                    page can be downloaded, the DOM model's score joins the decision
                    (models/url_classifier/dom_config.json)        -> clean (stop) / suspicious
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
from functools import lru_cache  # noqa: E402

from typosquatting.brands import (  # noqa: E402
    all_official_domains, intl_source_categories, label_of_domain, load_vn_brands,
    verified_official_domains,
)
from typosquatting.paas_domains import is_untrusted_hosting, is_user_content_host  # noqa: E402
from typosquatting.typosquatting import check_typosquatting_deep  # noqa: E402
from url_classifier import dom_features  # noqa: E402
from url_classifier import predict as classifier  # noqa: E402
from url_utils import (  # noqa: E402
    MAX_URL_LEN, embedded_urls, ensure_scheme, host_of, is_ip_host, normalize_url,
    registrable_domain, split_host,
)

# Classifier score above which the fallback labels a URL "phishing" without the LLM
FALLBACK_PHISHING_SCORE = 0.9
# An LLM "safe" verdict is overruled to "suspicious" when the URL classifier is this sure: chosen
# on the Layer D dev set (catches 26/40 of D's phishing misses, touches 4/291 clean pages).
# Not applied on official brand domains: there a high score is a classifier error (google.com
# login pages), and no dev / test phishing page fell in that case
LLM_SAFE_OVERRIDE_SCORE = 0.99

# Messages when a URL may not be cleared even if the LLM finds the page safe
HOLD_EVASION = ("The page looks safe to the LLM, but the link hides many other destinations, "
                "a pattern used to evade checks — be careful.")
HOLD_LISTED_OFFICIAL = ("This page is on the brand's official domain but a public phishing list "
                        "reports it (possibly a list error, possibly a hacked page) — be careful.")

# Verified official domains that still go through B/C/D: marketplaces and classifieds host
# third-party content (fake shops, scam listings); zalo.me serves user profiles and group links
TRUST_EXCLUDED_CATEGORIES = frozenset({"ecommerce"})
TRUST_EXCLUDED_DOMAINS = frozenset({"zalo.me"})

# International brands: their official domains are trusted the same way, except where anyone can
# publish a page under the brand's domain. Excluded by catalogue category (marketplaces, retail,
# travel listings, social / video / music / gaming platforms, cloud storage, hosting) and by name
# where the category does not say it: user-content or file-sharing sites, hosting / domain
# providers, form / document services, and domains with phishing in the train / val URLs
# (orange.fr, venmo.com, walletconnect.com). pinterest.com, token.im: excluded on review.
# An excluded domain excludes every domain of its brand (google.com -> googlevideo.com, google.dev).
# Evidence: src/evaluate_trusted_shortcut.py -> reports/pipeline/trusted_shortcut.json
TRUST_INTL_EXCLUDED_CATEGORIES = ("commerce", "shopping", "retail", "travel", "social", "streaming",
                                  "music", "gaming", "cloud storage", "hosting", "classified")
TRUST_INTL_EXCLUDED_DOMAINS = frozenset({
    # user content, file sharing, forms, chat / developer platforms
    "google.com", "microsoft.com", "live.com", "onedrive.live.com", "office.com", "sharepoint.com",
    "apple.com", "icloud.com", "yahoo.com", "aol.com", "adobe.com", "intuit.com", "github.com",
    "googleapis.com", "gstatic.com", "zoom.us", "dropbox.com", "reddit.com", "youtube.com",
    "youtube.vn", "discord.com", "twitch.tv", "canva.com", "docusign.com", "naver.com",
    "slack.com", "okta.com", "salesforce.com", "shopify.com", "opensea.io", "leboncoin.fr",
    "biglobe.ne.jp",
    # payment links / profile pages anyone can create (buy.stripe.com/..., cash.app/$name)
    "stripe.com", "squareup.com", "cash.app",
    # hosting and domain providers
    "godaddy.com", "hostinger.com", "hostinger.vn", "ionos.com", "namecheap.com", "arubacloud.com",
    "cloudflare.com",
    # phishing found on the domain (train / val) or excluded on review
    "orange.fr", "venmo.com", "walletconnect.com", "pinterest.com", "token.im",
})

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


@lru_cache(maxsize=1)
def trusted_intl_domains() -> frozenset[str]:
    """Official domains of international brands that host no third-party pages (see above)."""
    vn_labels = {b["label"] for b in load_vn_brands()}
    cats = intl_source_categories()
    domains = all_official_domains() - verified_official_domains()
    excluded_labels = {label_of_domain(d) for d in domains
                       if d in TRUST_INTL_EXCLUDED_DOMAINS or is_untrusted_hosting(d)}
    out = set()
    for d in domains:
        label = label_of_domain(d)
        cat = cats.get(label, "").lower()
        if (label in vn_labels or label in excluded_labels or d in TRUST_INTL_EXCLUDED_DOMAINS
                or is_untrusted_hosting(d) or any(c in cat for c in TRUST_INTL_EXCLUDED_CATEGORIES)):
            continue
        out.add(d)
    return frozenset(out)


def trusted_official_domain(url: str, include_intl: bool = True) -> str | None:
    """Registrable domain when the URL is a page of a verified official Vietnamese brand domain or
    of a trusted international brand domain, that can be trusted without further checks, else
    None. include_intl=False: Vietnamese domains only (the shortcut before the extension)."""
    host = host_of(url)
    domain = registrable_domain(host)
    vn = domain in verified_official_domains() and domain not in TRUST_EXCLUDED_DOMAINS
    if not vn and not (include_intl and domain in trusted_intl_domains()):
        return None
    label = label_of_domain(domain)
    if vn and any(b["label"] == label and b["category"] in TRUST_EXCLUDED_CATEGORIES
                  for b in load_vn_brands()):
        return None
    # User-published content or a hidden destination (open redirect) is not the brand's own page
    if "\\" in url or is_user_content_host(host) or embedded_urls(url) or may_redirect(url):
        return None
    return domain


def _feed_error_candidate(url: str) -> bool:
    """Feed hit that is warned instead of blocked: a plain page (trusted-shortcut rules: no
    redirect form, no user content) on the apex or www host of a hand-verified Vietnamese brand
    domain. Not the derived international list, not other subdomains (microsites get hacked)."""
    domain = trusted_official_domain(url, include_intl=False)
    return domain is not None and host_of(url) in (domain, f"www.{domain}")


def analyze_url(url: str, use_llm: bool = True, llm_timeout_s: float | None = None,
                use_host_features: bool = True, explain: bool = False,
                lang: str = "vi", use_dom: bool = True, llm_model: str | None = None) -> dict:
    """Run the layers in order and return {verdict, decided_by, explanation, layers, latency_ms}.

    explain=True asks the LLM for an explanation in the user's language `lang` (user's choice,
    extra cost); by default Layer D only classifies. llm_model overrides the Layer D model.
    VirusTotal is not a layer: it only serves as a reference to check the pipeline (relabel_vt.py).
    """
    t0 = time.perf_counter()
    layers: dict = {}

    def done(verdict: str, decided_by: str, explanation: str) -> dict:
        return {"url": url, "verdict": verdict, "decided_by": decided_by,
                "explanation": explanation, "layers": layers,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # Reasons a URL may not be cleared, whatever later layers say: (kind, message for the user)
    holds: list[tuple[str, str]] = []

    a = check_blocklist(url)
    layers["A_blocklist"] = a
    official = trusted_official_domain(url)
    if a["flag"] is True:
        if not _feed_error_candidate(url):
            return done("phishing", "A_blocklist", a["reason"])
        # A plain page on the apex / www host of a hand-verified Vietnamese brand domain in a
        # public feed: usually a feed false positive, sometimes a hacked page. Not blocked
        # outright, never cleared: it goes through every layer and the user is at least warned
        holds.append(("listed_official", HOLD_LISTED_OFFICIAL))
    elif official:
        return done("clean", "verified_official_domain",
                    f"{official} is a verified official domain of '{label_of_domain(official)}'.")

    b = check_typosquatting_deep(url)
    layers["B_typosquatting"] = b
    # More hidden destinations than Layer B checks: an evasion pattern (decoys, long redirect
    # chains), never confirmed clean by Layer C; Layer D then checks the real final URL
    # (a URL longer than MAX_URL_LEN is only read in part: same treatment)
    if b.get("hidden_destinations") or len(url) > MAX_URL_LEN:
        holds.append(("evasion", HOLD_EVASION))
        layers["hidden_destinations"] = b.get("hidden_destinations", 0)

    c = (classifier.classify_url(url, use_host_features=use_host_features)
         if classifier.is_available() else None)
    layers["C_classifier"] = c or {"label": "unknown", "reason": "model not trained"}
    # DOM model: one download, reused by Layer D. Never applied when the host lookup failed
    # (score None): that URL stays fail-closed whatever its page looks like
    page = None
    if (use_dom and c is not None and c["score"] is not None and b["flag"] is not True
            and dom_features.is_active()):
        from llm.llm_client import fetch_html, page_from_html
        t_p = time.perf_counter()
        html, final_url = fetch_html(url)
        p_dom = dom_features.dom_score(html, final_url)
        c.update({"label_url_only": c["label"], "dom_score": p_dom,
                  "label": dom_features.combined_label(c["label"], p_dom),
                  "page_ms": round((time.perf_counter() - t_p) * 1000, 1)})
        page = page_from_html(html, final_url)
    if b["flag"] is not True and not holds and c is not None and c["label"] == "clean":
        return done("clean", "C_classifier",
                    f"URL model rates it safe (score {c['score']:.3f} < threshold {c['threshold']:.3f}).")

    if use_llm:
        from llm.llm_client import (
            DEFAULT_LLM_TIMEOUT_S, DEFAULT_MODEL, analyze_suspicious, cached_verdict, fetch,
        )
        model = llm_model or DEFAULT_MODEL
        t_d = time.perf_counter()
        # Layer D downloads the page anyway: done here so the redirect target (shortener, open
        # redirect) also goes through Layers A and B — bit.ly/x -> a listed phishing page is
        # blocked instead of being left to the LLM. On a cached verdict the stored target is
        # checked again (no download): the cache must not skip the blocklist / evasion rules
        hit = cached_verdict(url, model, explain, lang)
        # A cached 'safe' is only reused for unchanged content (page swapped after the check):
        # the page is downloaded again and its fingerprint compared in analyze_suspicious
        if page is None and (not hit or hit["verdict"] == "safe"):
            page = fetch(url)
        final_url = page[3] if page and len(page) > 3 else (hit or {}).get("final_url")
        # The target comes from the site's Location header: same length bound and same
        # "too long = evasion" rule as the clicked URL
        if final_url and len(final_url) > MAX_URL_LEN:
            holds.append(("evasion", HOLD_EVASION))
            layers["redirect_target_too_long"] = len(final_url)
            final_url = final_url[:MAX_URL_LEN]
        if final_url and normalize_url(final_url) != normalize_url(url):
            a2 = check_blocklist(final_url)
            layers["A_redirect_target"] = a2
            if a2["flag"] is True:
                return done("phishing", "A_blocklist",
                            f"URL redirects to {final_url[:80]} — {a2['reason']}")
            if b["flag"] is not True:
                b2 = check_typosquatting_deep(final_url)
                if b2["flag"] is True:  # used by the prompt and by the fallback below
                    b = {**b2, "reason": f"URL redirects to {final_url[:80]} — {b2['reason']}"}
                    layers["B_typosquatting"] = b
                if b2.get("hidden_destinations"):  # same evasion rule as for the clicked URL
                    holds.append(("evasion", HOLD_EVASION))
                    layers["hidden_destinations_redirect_target"] = b2["hidden_destinations"]
        d = analyze_suspicious(url, rule_results=[b], explain=explain, lang=lang, page=page,
                               cache=True, model=model,
                               timeout_s=llm_timeout_s or DEFAULT_LLM_TIMEOUT_S)
        d["latency_ms"] = round((time.perf_counter() - t_d) * 1000, 1)
        layers["D_llm"] = d
        if not d["timed_out"]:
            # LLM 'suspicious' = not enough evidence either way: warn the user, never 'clean'
            verdict = {"phishing": "phishing", "suspicious": "suspicious"}.get(d["verdict"], "clean")
            score = (c or {}).get("score")
            if verdict == "clean" and holds:  # see HOLD_*: the LLM alone cannot clear it
                kind, message = holds[0]
                return done("suspicious", f"D_llm+{kind}", message)
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
    ap.add_argument("--lang", default="vi", help="explanation language (ISO code), default vi")
    ap.add_argument("--log", action="store_true", help="log the verdict for false-alarm monitoring")
    args = ap.parse_args()
    for u in args.urls:
        res = analyze_url(u, use_llm=not args.no_llm, use_host_features=not args.no_host,
                          explain=args.explain, lang=args.lang)
        if args.log:
            from monitoring import record_verdict
            record_verdict(res)
        print(json.dumps(res, ensure_ascii=False, indent=2))
