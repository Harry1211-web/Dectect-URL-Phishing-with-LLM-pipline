"""Shared URL helpers used by every layer (blocklist, typosquatting, classifier, data collection)."""

from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

import tldextract

# Offline extractor: uses the public suffix list bundled with tldextract (no network call at runtime)
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

_RE_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def ensure_scheme(url: str) -> str:
    """Add 'http://' when the URL has no scheme so urlsplit parses the host correctly."""
    url = url.strip()
    return url if _RE_SCHEME.match(url) else "http://" + url


def host_of(url: str) -> str:
    """Lower-cased hostname without credentials, port or trailing dot ('' if unparsable)."""
    try:
        host = urlsplit(ensure_scheme(url)).hostname or ""
    except ValueError:
        return ""
    return host.strip(".").lower()


@lru_cache(maxsize=65536)
def _split_host(host: str) -> tuple[str, str, str]:
    ext = _EXTRACT(host)
    return ext.subdomain, ext.domain, ext.suffix


def registrable_domain(url_or_host: str) -> str:
    """Registrable domain (eTLD+1), e.g. 'a.b.example.co.uk' -> 'example.co.uk'; falls back to the host."""
    host = host_of(url_or_host)
    _, dom, suf = _split_host(host)
    if dom and suf:
        return f"{dom}.{suf}"
    return host


def split_host(url_or_host: str) -> tuple[str, str, str]:
    """(subdomain, domain label, public suffix) of the URL's host."""
    return _split_host(host_of(url_or_host))


def normalize_url(url: str) -> str:
    """Canonical form for exact matching: lower-case host, drop scheme, 'www.', default port,
    fragment and trailing slash. Path and query keep their case (they are case-sensitive on servers).
    """
    try:
        parts = urlsplit(ensure_scheme(url))
        host = (parts.hostname or "").strip(".").lower()
        port = parts.port
    except ValueError:
        return url.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit(("", netloc, path, parts.query, "")).lstrip("/")


def strip_scheme_www(url: str) -> str:
    """URL text without scheme and leading 'www.' — the form fed to the URL classifier, so the model
    cannot learn dataset artefacts such as 'legit URLs always start with https://www.'."""
    url = _RE_SCHEME.sub("", url.strip())
    return url[4:] if url.lower().startswith("www.") else url


REDIRECT_PARAMS = frozenset({
    "url", "u", "ru", "r", "redirect", "redirect_uri", "redirect_url", "redir", "target", "to",
    "continue", "next", "dest", "destination", "goto", "go", "return", "returnurl", "return_to",
    "link", "out", "q", "af_r", "uddg", "ref",
})
_RE_EMBEDDED = re.compile(r"(?:https?(?::|%3a)(?:/|%2f){2}|www\.)[^\s&\"'<>]+", re.I)


def embedded_urls(url: str) -> list[str]:
    """Destination URLs hidden in the query/path (open redirects, click trackers, deep links):
    'lijit.com/r?ru=https%3A%2F%2Fnew.express.adobe.com%2F...' -> ['https://new.express.adobe.com/...']."""
    from urllib.parse import parse_qsl, unquote
    try:
        parts = urlsplit(ensure_scheme(url))
    except ValueError:
        return []
    out: list[str] = []
    for key, val in parse_qsl(parts.query, keep_blank_values=False):
        val = unquote(val).strip()
        if _RE_EMBEDDED.match(val) or (key.lower() in REDIRECT_PARAMS and "." in val and " " not in val
                                       and "/" in val):
            out.append(val)
    out += [unquote(m) for m in _RE_EMBEDDED.findall(parts.path)]
    own = host_of(url)
    return [u for u in dict.fromkeys(out) if host_of(u) and host_of(u) != own]


def is_ip_host(host: str) -> bool:
    """True if the host is an IPv4/IPv6 literal."""
    import ipaddress
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False
