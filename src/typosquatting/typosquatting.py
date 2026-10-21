"""Layer B — typosquatting / brand-impersonation detection.

Keywords are extracted from the WHOLE URL (domain label, its hyphen parts, subdomains, path and
query), not only from the registrable domain, then compared with the brand catalogue using:
  - Damerau-Levenshtein distance: number of edits, a transposition ("paypla") counts as 1 edit;
  - Jaro-Winkler similarity: proportional similarity that rewards a matching prefix, used as a
    second gate so that short random words within 1 edit of a brand are not flagged.
Unicode hosts (IDN / punycode) are reduced to an ASCII skeleton first to catch homoglyphs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import RuleResult  # noqa: E402
from typosquatting.brands import (  # noqa: E402
    INTL_CATEGORY, all_official_domains, brand_by_label, label_of_domain, load_all_brands,
)
from typosquatting.paas_domains import is_user_content_host  # noqa: E402
from url_utils import (  # noqa: E402
    all_embedded_urls, ensure_scheme, host_of, is_ip_host, split_host,
)

# Minimum brand-label length for each kind of match (shorter labels cause mass false positives)
MIN_LEN_EMBEDDED = 6      # brand as a substring of a longer label: "vietcombankvn"
MIN_LEN_SUBDOMAIN = 4     # brand as a whole subdomain label: "paypal.evil.com"
MIN_LEN_PATH = 4          # brand + fake TLD in the path: "/www-paypal-com/"
MIN_LEN_PATH_LURE = 5     # brand token in the path next to a credential keyword
MIN_TOKEN_LEN = 4
MIN_LEN_VN_PART = 3       # VN brand as an exact hyphen-separated part of the domain label

# Jaro-Winkler gate for fuzzy matches; stricter for short labels where 1 edit is a big change
JW_MIN_SHORT = 0.90       # brand length <= 6
JW_MIN_LONG = 0.85

# A token used as a domain-label part by >= this many Tranco top-1M domains is an ordinary word
# ("stream", "team", "telecom"), not a misspelt brand, so it is not fuzzy-matched
COMMON_TOKEN_MIN = 10
TRANCO_PATH = Path(__file__).resolve().parents[2] / "data" / "tranco_top1m.csv"
# VN brand labels that are also ordinary English words ("managed-vps.net", "grab-bat.net"):
# embedded in a domain they need a lure word, like international brands
VN_BRANDS_THAT_ARE_WORDS = frozenset({"vps", "grab", "onepay", "cake"})

PRIORITY = {
    "homoglyph": 0,
    "exact_label": 1,
    "subdomain": 2,
    "path_fake_domain": 3,
    "typosquat": 4,
    "embedded": 5,
    "path_brand_lure": 6,
}
DESCRIPTION = {
    "homoglyph": "uses Unicode characters that look like Latin letters (homoglyphs) to fake the brand name",
    "exact_label": "uses the exact brand name but is not an official domain",
    "subdomain": "puts the brand name (or a variant) in the subdomain",
    "path_fake_domain": "URL path disguised as the brand's domain (like www-brand-com)",
    "typosquat": "small misspelling of the brand name",
    "embedded": "embeds the whole brand name in the domain label",
    "path_brand_lure": "brand name in the path together with a login/verification keyword",
}

# Fake TLDs attackers append to a brand inside a path ("paypal-com", "www.vietcombank.com.vn")
_FAKE_TLD = "com|net|org|vn|info|io|xyz|top|site|online|shop|biz|edu|gov"
# Credential/urgency words that turn a brand mention in a path into a lure
_LURE_WORDS = frozenset({
    "login", "logon", "signin", "sign", "verify", "verification", "account", "secure", "security",
    "update", "auth", "otp", "password", "passwd", "banking", "wallet", "confirm", "unlock",
    "recover", "recovery", "validate", "billing", "payment", "suspend", "suspended", "webscr",
    "dangnhap", "xacminh", "xacthuc", "taikhoan", "matkhau", "capnhat", "khoa", "mokhoa",
})
# Words that make "brand + word" domains a lure (paypal-support, amazonrefund)
_DOMAIN_LURE_WORDS = (_LURE_WORDS - {"sign", "otp", "khoa", "auth"}) | {
    "support", "helpdesk", "service", "customer", "official", "refund", "reward", "gift",
    "promo", "bonus", "claim", "prize", "giveaway", "airdrop", "authentication",
}
# Common homoglyphs not removed by NFKD (Cyrillic/Greek look-alikes, digit swaps in IDNs)
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "ѕ": "s", "ԝ": "w", "ν": "v", "ο": "o",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "τ": "t", "ı": "i", "ł": "l", "ø": "o", "đ": "d",
})


def jw_min(brand: str) -> float:
    return JW_MIN_SHORT if len(brand) <= 6 else JW_MIN_LONG


def typosquat_threshold(brand: str) -> int:
    """Max Damerau-Levenshtein distance for a fuzzy match, by brand length.

    <=3 chars: 0 (one edit already changes 33-50% of the label: bit.ly~'bt', ft.com~'fpt');
    4-10 chars: 1; >10 chars: 2.
    """
    if len(brand) <= 3:
        return 0
    return 1 if len(brand) <= 10 else 2


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ascii_skeleton(label: str) -> str:
    """Decode punycode, map look-alike characters to ASCII and drop diacritics."""
    try:
        if "xn--" in label:
            label = label.encode("ascii").decode("idna")
    except UnicodeError:
        pass
    label = label.lower().translate(_CONFUSABLES)
    label = unicodedata.normalize("NFKD", label)
    return "".join(c for c in label if not unicodedata.combining(c))


def _path_and_query(url: str) -> str:
    raw = ensure_scheme(url).split("://", 1)[1]
    rest = raw.split("/", 1)[1] if "/" in raw else ""
    if "?" in raw.split("/", 1)[0]:  # "host?x=1" without a path
        rest = raw.split("?", 1)[1]
    return rest.split("#", 1)[0].lower()


@lru_cache(maxsize=2048)  # one per brand: must hold the whole catalogue (361 brands) or it never hits
def _fake_domain_regex(brand: str) -> re.Pattern:
    return re.compile(
        rf"(?:^|[/\-_.=])(?:www[\-_.])?{re.escape(brand)}[\-_.](?:{_FAKE_TLD})(?:[/\-_.?&]|$)")


@lru_cache(maxsize=1)
def _token_frequency() -> Counter:
    """How many Tranco top-1M domains use each hyphen-separated label part."""
    freq: Counter = Counter()
    if TRANCO_PATH.exists():
        with TRANCO_PATH.open(encoding="utf-8") as f:
            for line in f:
                label = line.rstrip().split(",", 1)[-1].split(".", 1)[0]
                freq.update(p for p in label.split("-") if p)
    return freq


def is_common_token(token: str) -> bool:
    return _token_frequency()[token] >= COMMON_TOKEN_MIN


def _is_foreign_cctld(suffix: str) -> bool:
    """'com.mx', 'ph', 'co.uk' -> True; 'vn', 'com.vn', 'com', 'online' -> False."""
    last = suffix.rsplit(".", 1)[-1]
    return len(last) == 2 and last != "vn"


@lru_cache(maxsize=1)
def _brands_by_len() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for b in load_all_brands():
        out.setdefault(len(b["label"]), []).append(b)
    return out


def _fuzzy_hits(token: str) -> list[tuple[str, int, float]]:
    """Brands within the Damerau-Levenshtein threshold of `token` that also pass the Jaro-Winkler gate."""
    if is_common_token(token):
        return []
    hits = []
    n = len(token)
    for length in range(max(1, n - 2), n + 3):
        for b in _brands_by_len().get(length, ()):
            label = b["label"]
            thr = typosquat_threshold(label)
            if thr == 0 or token == label:
                continue
            # Short brands: one inserted/deleted char turns a real word into the brand
            # (visma~visa, zoomg~zoom, mitel~itel) — only same-length edits count
            if len(label) <= 5 and len(token) != len(label):
                continue
            d = DamerauLevenshtein.distance(token, label, score_cutoff=thr)
            if d > thr:
                continue
            jw = JaroWinkler.similarity(token, label)
            if jw >= jw_min(label):
                hits.append((label, d, jw))
    return hits


def _embedding_is_lure(brand: str, label: str, path_tokens: set[str], suffix: str) -> bool:
    """A brand embedded in a longer domain label counts only if it looks like a lure.

    VN brands outside foreign ccTLDs: always (the system protects them first). Otherwise only
    with a lure word next to the brand ("paypal-support") or in the path, because brand + word
    compounds are common on legit sites (contentsquare, orangepage, otpbank.hu ~ tpbank).
    """
    if (brand_by_label()[brand]["category"] != INTL_CATEGORY and not _is_foreign_cctld(suffix)
            and brand not in VN_BRANDS_THAT_ARE_WORDS):
        return True
    rest = label.replace(brand, "")
    return any(w in rest for w in _DOMAIN_LURE_WORDS) or bool(path_tokens & _LURE_WORDS)


def extract_tokens(url: str) -> dict:
    """Split a URL into the keyword groups compared against brands."""
    host = host_of(url)
    sub, dom, _ = split_host(host)
    sub_labels = [s for s in sub.split(".") if s and s != "www"]
    path = _path_and_query(url)
    return {
        "host": host,
        "domain_label": dom,
        "main": _alnum(dom),
        "main_parts": [p for p in (_alnum(x) for x in re.split(r"[-_]", dom)) if p],
        "sub_labels": [_alnum(s) for s in sub_labels if _alnum(s)],
        "sub_parts": [p for s in sub_labels for p in (_alnum(x) for x in re.split(r"[-_]", s)) if p],
        "path": path,
        "path_tokens": [t for t in re.split(r"[^a-z0-9]+", path) if t],
    }


def check_typosquatting(url: str) -> RuleResult:
    """Layer B entry point: flag URLs that impersonate a known brand."""
    t0 = time.perf_counter()

    def result(flag, reason, brand=None, kind=None) -> RuleResult:
        r = RuleResult(name="typosquatting", flag=flag, reason=reason,
                       latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        if brand:
            r["brand"], r["kind"] = brand, kind
        return r

    tk = extract_tokens(url)
    host = tk["host"]
    if not host:
        return result("unknown", "Could not extract a domain from the URL.")
    if is_ip_host(host):
        return result(False, f"{host} is an IP address — typosquatting check does not apply.")
    _, dom, suf = split_host(host)
    domain = f"{dom}.{suf}" if dom and suf else host
    user_content = is_user_content_host(host)
    if domain in all_official_domains() and not user_content:
        return result(False, f"{domain} is an official domain of '{label_of_domain(domain)}'.")
    cand = tk["main"]
    if not cand:
        return result(False, "Could not extract the domain label — typosquatting check skipped.")

    labels = brand_by_label()
    cand_is_brand = cand in labels
    ptoks = set(tk["path_tokens"])
    candidates: list[tuple[str, int, str, str]] = []  # (kind, distance, brand, matched token)

    # Domain-label checks are skipped on user-content hosts of official domains
    # (sites.google.com, bucket.s3.amazonaws.com...): there only subdomain/path matter
    if not user_content:
        # Homoglyph / IDN: the ASCII skeleton of a Unicode label equals or nearly equals a brand
        skeleton = _alnum(_ascii_skeleton(tk["domain_label"]))
        if skeleton and skeleton != cand:
            if skeleton in labels:
                candidates.append(("homoglyph", 0, skeleton, skeleton))
            candidates += [("homoglyph", d, b, skeleton) for b, d, _ in _fuzzy_hits(skeleton)]

        # Exact brand label on a non-official domain (binance.info, vietcombank.xyz). Excluded:
        # foreign country TLDs (shopee.com.mx, google.de are the brands' own regional sites)
        if cand_is_brand and not _is_foreign_cctld(suf):
            candidates.append(("exact_label", 0, cand, cand))

        # Fuzzy match on the main label and on each hyphen part ("vietcombamk-online")
        if not cand_is_brand:
            for tok in {cand, *tk["main_parts"]}:
                if len(tok) >= MIN_TOKEN_LEN and tok not in labels:
                    # Near-misses of VN brands on foreign ccTLDs are other countries' own brands
                    # (tbank.ru ~ tpbank, mbank.pl ~ mbbank), not attacks on Vietnamese users
                    candidates += [("typosquat", d, b, tok) for b, d, _ in _fuzzy_hits(tok)
                                   if not (_is_foreign_cctld(suf)
                                           and labels[b]["category"] != INTL_CATEGORY)]
        for part in tk["main_parts"]:
            # VN brands as a whole hyphen part may be 3 chars ("acb-baomat", "fpt-cuocnet")
            min_len = (MIN_LEN_VN_PART if labels.get(part, {}).get("category") != INTL_CATEGORY
                       else MIN_LEN_SUBDOMAIN)
            if (part != cand and part in labels and len(part) >= min_len
                    and _embedding_is_lure(part, cand, ptoks, suf)):
                candidates.append(("embedded", 0, part, part))

    subs = set(tk["sub_labels"]) | set(tk["sub_parts"])
    for b in load_all_brands():
        label = b["label"]
        if (not user_content and len(label) >= MIN_LEN_EMBEDDED and label in cand
                and cand != label and _embedding_is_lure(label, cand, ptoks, suf)):
            candidates.append(("embedded", 0, label, cand))
        if len(label) >= MIN_LEN_SUBDOMAIN and label in subs:
            candidates.append(("subdomain", 0, label, label))
        elif len(label) >= MIN_LEN_EMBEDDED and any(label in s for s in subs):
            candidates.append(("subdomain", 0, label, label))
        if len(label) >= MIN_LEN_PATH and tk["path"] and _fake_domain_regex(label).search(tk["path"]):
            candidates.append(("path_fake_domain", 0, label, label))
    for s in subs:
        if len(s) >= MIN_LEN_EMBEDDED and s not in labels:
            candidates += [("subdomain", d, b, s) for b, d, _ in _fuzzy_hits(s)]

    # Brand named in the path together with a credential keyword ("/paypal/login/verify")
    if ptoks & _LURE_WORDS:
        for t in ptoks:
            if len(t) >= MIN_LEN_PATH_LURE and t in labels and t != cand:
                candidates.append(("path_brand_lure", 0, t, t))

    if not candidates:
        return result(False, f"Label '{cand}' matches no brand within the matching thresholds.")

    kind, d, brand, token = min(candidates, key=lambda x: (PRIORITY[x[0]], x[1]))
    where = {"subdomain": f"Subdomain of '{domain}'",
             "path_fake_domain": f"URL path of '{domain}'",
             "path_brand_lure": f"URL path of '{domain}'"}.get(kind, f"Label '{cand}'")
    detail = ""
    if d:
        detail = (f" (Damerau-Levenshtein = {d}, Jaro-Winkler = "
                  f"{JaroWinkler.similarity(token, brand):.2f}"
                  f" against '{brand}')")
    tail = (f"'{host}' is a platform where users publish their own content, not a site of "
            f"'{brand}'." if user_content
            else f"domain '{domain}' is not in the official list.")
    return result(True, f"{where} — {DESCRIPTION[kind]}: suspected impersonation of '{brand}'{detail}; {tail}",
                  brand, kind)


# Hidden destinations checked by Layer B (~1-20 ms each). Real redirect / tracking URLs carry 1-3;
# a URL with more is treated as evasion by the pipeline (never "clean" before Layer D), so this
# cap cannot hide a target. Without it, a 4 KB chain 'www.a/www.b/www.a/...' cost ~9 s of CPU
MAX_TARGETS_CHECKED = 20


def check_typosquatting_deep(url: str) -> RuleResult:
    """Check the URL and the destinations hidden in it (open redirects, click trackers):
    'tracker.com/r?u=https://paypa1-login.com' is judged on 'paypa1-login.com' as well.
    Shallow ones first (the redirect parameters themselves), at most MAX_TARGETS_CHECKED.
    More than that is an evasion pattern: every result then carries 'hidden_destinations'
    (count), which callers must treat as suspicious (pipeline: never clean before Layer D)."""
    targets = all_embedded_urls(url)
    evasive = {"hidden_destinations": len(targets)} if len(targets) > MAX_TARGETS_CHECKED else {}
    r = check_typosquatting(url)
    if r["flag"] is True:
        return {**r, **evasive}
    for target in targets[:MAX_TARGETS_CHECKED]:
        t = check_typosquatting(target)
        if t["flag"] is True:
            t["reason"] = f"URL redirects to {target[:80]} — {t['reason']}"
            t["latency_ms"] = round(t["latency_ms"] + r["latency_ms"], 2)
            return {**t, **evasive}
    return {**r, **evasive}




if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urls", nargs="+", help="URL(s) to check")
    for u in ap.parse_args().urls:
        print(json.dumps(check_typosquatting(u), ensure_ascii=False, indent=2))
