"""Hand-crafted lexical URL features for the XGBoost stage of Layer C.

Computed from the URL string only (no network), so classification stays fast and works offline.
The scheme (http/https) and a leading 'www.' are ignored on purpose: in the public datasets
legit URLs almost always start with 'https://www.', a shortcut that does not hold in real traffic.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typosquatting.paas_domains import (  # noqa: E402
    is_free_hosting, is_shortener, is_untrusted_hosting, is_user_content_host,
)
from url_utils import (  # noqa: E402
    REDIRECT_PARAMS, embedded_urls, ensure_scheme, host_of, is_ip_host, registrable_domain,
    split_host, strip_scheme_www,
)

SENSITIVE_WORDS = (
    "login", "logon", "signin", "verify", "account", "secure", "update", "banking", "confirm",
    "password", "wallet", "auth", "billing", "payment", "invoice", "suspend", "unlock",
    "recover", "support", "webscr", "otp", "dangnhap", "xacminh", "xacthuc", "taikhoan",
)
COMMON_TLDS = ("com", "net", "org", "vn", "com.vn", "edu.vn", "gov.vn", "io", "co", "info",
               "xyz", "top", "online", "site", "shop", "app", "dev", "ru", "cn", "de", "uk",
               "br", "in", "jp", "fr", "it", "nl", "pl", "au", "ca", "us", "me", "cc", "tk")
_RE_HEX = re.compile(r"[0-9a-f]{16,}", re.I)
_RE_FILE_EXT = re.compile(r"\.(php|html?|aspx?|jsp|cgi)$", re.I)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


FEATURE_NAMES: list[str] = [
    "url_len", "host_len", "path_len", "query_len", "n_host_dots", "n_host_hyphens",
    "n_host_digits", "n_subdomain_labels", "domain_label_len", "is_ip", "has_port",
    "has_at", "has_double_slash_path", "n_path_segments", "n_query_params", "digit_ratio",
    "upper_ratio", "n_hyphen", "n_underscore", "n_percent", "n_equals", "n_ampersand",
    "n_tilde", "host_entropy", "url_entropy", "is_free_hosting", "is_shortener",
    "has_punycode", "longest_token_len", "n_sensitive_words", "has_hex_string",
    "has_file_ext", "tld_len", "domain_has_digit",
    # Redirect / click-tracker URLs hiding the real destination
    "has_redirect_param", "n_embedded_urls", "embedded_on_untrusted_host",
] + [f"tld_{t}" for t in COMMON_TLDS] + ["tld_other"]


def url_features(url: str) -> list[float]:
    """Feature vector aligned with FEATURE_NAMES."""
    text = strip_scheme_www(url)
    try:
        parts = urlsplit(ensure_scheme(url))
        host = (parts.hostname or "").lower()
        path, query = parts.path, parts.query
        has_port = parts.port is not None
    except ValueError:
        host, path, query, has_port = "", "", "", False
    host = host.removeprefix("www.")
    sub, dom, suf = split_host(host)
    low = text.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    tld_vec = [float(suf == t) for t in COMMON_TLDS]
    feats = [
        len(text), len(host), len(path), len(query), host.count("."), host.count("-"),
        sum(c.isdigit() for c in host), len([s for s in sub.split(".") if s]), len(dom),
        float(is_ip_host(host)), float(has_port),
        float("@" in text), float("//" in path), len([p for p in path.split("/") if p]),
        len([q for q in query.split("&") if q]),
        _ratio(sum(c.isdigit() for c in text), len(text)),
        _ratio(sum(c.isupper() for c in text), len(text)),
        text.count("-"), text.count("_"), text.count("%"), text.count("="), text.count("&"),
        text.count("~"), _entropy(host), _entropy(text),
        float(is_free_hosting(f"{dom}.{suf}")), float(is_shortener(f"{dom}.{suf}")),
        float("xn--" in host), max((len(t) for t in tokens), default=0),
        sum(w in low for w in SENSITIVE_WORDS), float(bool(_RE_HEX.search(text))),
        float(bool(_RE_FILE_EXT.search(path))), len(suf), float(any(c.isdigit() for c in dom)),
    ]
    emb = embedded_urls(url)
    params = {p.split("=", 1)[0].lower() for p in query.split("&") if "=" in p}
    feats += [
        float(bool(params & REDIRECT_PARAMS)), len(emb),
        float(any(is_untrusted_hosting(registrable_domain(e)) or is_user_content_host(host_of(e))
                  for e in emb)),
    ]
    return feats + tld_vec + [float(not any(tld_vec))]


def feature_matrix(urls: list[str]) -> np.ndarray:
    return np.asarray([url_features(u) for u in urls], dtype=np.float32)
