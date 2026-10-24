"""Host-based features for Layer C: DNS, TLS certificate, domain age (RDAP) and popularity (Tranco).

The URL string alone cannot tell a fresh phishing domain from a small legit site
('abc-xyz.com'). What the host looks like on the network can: phishing domains are days old,
use a certificate issued hours ago, have no MX record and are absent from popularity lists.

Lookups run concurrently with short timeouts; a failed lookup leaves the feature missing (NaN),
which XGBoost handles natively. Results are cached per host/domain in data/host_cache.sqlite.
A Layer C model trained with host_lookups = "none" (models/url_classifier/config.json) uses only
the Tranco features (local file, no network); the network lookups then only serve experiments.
"""

from __future__ import annotations

import ipaddress
import json
import math
import socket
import sqlite3
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import dns.exception
import dns.resolver
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typosquatting.paas_domains import (  # noqa: E402
    is_shortener, is_untrusted_hosting, is_user_content_host)
from url_utils import (  # noqa: E402
    ensure_scheme, file_version, host_of, is_ip_host, registrable_domain)

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "host_cache.sqlite"
TRANCO_PATH = ROOT / "data" / "tranco_top1m.csv"
CACHE_TTL_DAYS = 7
DNS_TIMEOUT_S = 2.0
TLS_TIMEOUT_S = 3.0
RDAP_TIMEOUT_S = 4.0
UA = "Mozilla/5.0 (compatible; dacntt-phishing-research/1.0)"

# Free/automated certificate authorities (used by most phishing kits, but also by many legit sites)
_FREE_CA = ("let's encrypt", "zerossl", "cpanel", "google trust services", "cloudflare",
            "buypass", "ssl.com free", "actalis")

HOST_FEATURE_NAMES = [
    "tranco_log_rank", "in_tranco",
    "dns_resolves", "dns_n_a", "dns_min_ttl", "dns_has_mx", "dns_n_ns",
    "tls_ok", "tls_cert_age_days", "tls_validity_days", "tls_free_ca", "tls_wildcard",
    "tls_n_san", "tls_cn_matches",
    "rdap_found", "domain_age_days", "days_to_expiry",
]
# Read from local files only: the features of a Layer C model without network lookups
OFFLINE_FEATURE_NAMES = ["tranco_log_rank", "in_tranco"]
NO_LOOKUP: dict = {"dns": {}, "tls": {}, "rdap": {}}  # raw lookup stand-in when nothing is looked up

_local = threading.local()
_db_lock = threading.Lock()


# ----------------------------------------------------------------------------- cache
def _db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(CACHE_PATH, timeout=60)
        # WAL lets many lookup threads read while one writes (default mode serialises them)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS cache (kind TEXT, key TEXT, ts REAL, value TEXT, "
                    "PRIMARY KEY (kind, key))")
        _local.con = con
    return con


def _cached(kind: str, key: str, fn, ttl_days: float = CACHE_TTL_DAYS,
            offline: bool = False) -> dict | None:
    """Cached lookup; with offline=True never touches the network (None when not cached)."""
    row = _db().execute("SELECT ts, value FROM cache WHERE kind=? AND key=?", (kind, key)).fetchone()
    if row and (offline or time.time() - row[0] < ttl_days * 86400):
        return json.loads(row[1])
    if offline:
        return None
    value = fn(key)
    with _db_lock:
        _db().execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                      (kind, key, time.time(), json.dumps(value)))
        _db().commit()
    return value


# ----------------------------------------------------------------------------- lookups
def _tranco_rank() -> dict[str, int]:
    return _tranco_rank_at(file_version(TRANCO_PATH))  # reloaded after the monthly refresh


@lru_cache(maxsize=1)
def _tranco_rank_at(_version: int) -> dict[str, int]:
    if not TRANCO_PATH.exists():
        return {}
    with TRANCO_PATH.open(encoding="utf-8") as f:
        return {ln.rstrip().split(",", 1)[1]: i + 1 for i, ln in enumerate(f)}


def _resolver() -> dns.resolver.Resolver:
    r = getattr(_local, "resolver", None)
    if r is None:
        r = dns.resolver.Resolver()
        r.nameservers = ["1.1.1.1", "8.8.8.8"]
        r.timeout = r.lifetime = DNS_TIMEOUT_S
        _local.resolver = r
    return r


class LookupUnavailable(Exception):
    """Transient failure (timeout, resolver overload): the result must NOT be cached."""


def _dns(host: str) -> dict:
    """A/MX/NS records. 'resolves': 0 only on a definitive answer (NXDOMAIN / no A record);
    timeouts are retried with the system resolver and, if still failing, raise
    LookupUnavailable so a dead-looking host is never cached because of a busy resolver."""
    out = {"resolves": 0}
    try:
        a = _resolver().resolve(host, "A")
        out.update(resolves=1, n_a=len(a), min_ttl=a.rrset.ttl)
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return out
    except (dns.exception.DNSException, OSError):
        try:
            addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None, socket.AF_INET)}
            out.update(resolves=1, n_a=len(addrs))
        except socket.gaierror as e:
            if e.errno in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", -5), 11001):
                return out
            raise LookupUnavailable(host) from e
        except OSError as e:
            raise LookupUnavailable(host) from e
    rd = registrable_domain(host)
    for rtype, key in (("MX", "has_mx"), ("NS", "n_ns")):
        try:
            ans = _resolver().resolve(rd, rtype)
            out[key] = 1 if rtype == "MX" else len(ans)
        except (dns.exception.DNSException, OSError):
            out[key] = 0
    return out


def _public_ip(host: str) -> str | None:
    """SSRF guard: an address of `host` when every address it resolves to is public, else None
    (a URL sent to the API must not make the server connect to its own internal network)."""
    try:
        ips = [i[4][0].split("%")[0] for i in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)]
    except (OSError, UnicodeError):
        return None
    return ips[0] if ips and all(ipaddress.ip_address(ip).is_global for ip in ips) else None


def _tls(host: str) -> dict:
    """Certificate of host:443. A cert that fails verification is still read (tls_ok=0).
    Hosts resolving to a non-public address are not contacted (empty result)."""
    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID, NameOID
    ip = _public_ip(host)
    if ip is None:
        return {}
    for verify in (True, False):
        ctx = ssl.create_default_context()
        if not verify:
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        try:
            # Connect to the checked IP (no second lookup: DNS rebinding); SNI / hostname check use host
            with socket.create_connection((ip, 443), timeout=TLS_TIMEOUT_S) as sock, \
                    ctx.wrap_socket(sock, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
        except ssl.SSLCertVerificationError:
            continue
        except (OSError, ssl.SSLError, ValueError):
            return {}
        if not der:
            return {}
        cert = x509.load_der_x509_certificate(der)
        issuer = " ".join(a.value for a in cert.issuer).lower()
        try:
            san = cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value.get_values_for_type(x509.DNSName)
        except x509.ExtensionNotFound:
            san = []
        cn = [a.value for a in cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)]
        names = [n.lower() for n in san + cn]
        now = time.time()
        return {
            "ok": int(verify),
            "age_days": (now - cert.not_valid_before_utc.timestamp()) / 86400,
            "validity_days": (cert.not_valid_after_utc - cert.not_valid_before_utc).days,
            "free_ca": int(any(c in issuer for c in _FREE_CA)),
            "wildcard": int(any(n.startswith("*.") for n in names)),
            "n_san": len(san),
            "cn_matches": int(any(n == host or (n.startswith("*.") and host.endswith(n[1:]))
                                  for n in names)),
        }
    return {}


RDAP_BOOTSTRAP_RETRY_S = 600
_bootstrap_failed_at = -float("inf")


def _rdap_bootstrap() -> dict[str, str]:
    """TLD -> RDAP base URL from the IANA bootstrap file. A failed download raises (nothing cached)
    instead of returning {}: an empty table would store 'RDAP unsupported' for every domain for
    30 days. After a failure, lookups fail fast for RDAP_BOOTSTRAP_RETRY_S instead of each one
    waiting 20 s and filling the shared live-lookup pool."""
    global _bootstrap_failed_at
    if time.monotonic() - _bootstrap_failed_at < RDAP_BOOTSTRAP_RETRY_S:
        raise LookupUnavailable("IANA RDAP bootstrap (recent failure)")
    try:
        return _rdap_bootstrap_table()
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        _bootstrap_failed_at = time.monotonic()
        raise LookupUnavailable("IANA RDAP bootstrap") from e


@lru_cache(maxsize=1)
def _rdap_bootstrap_table() -> dict[str, str]:
    data = requests.get("https://data.iana.org/rdap/dns.json", timeout=20).json()
    return {tld: urls[0] for tlds, urls in data["services"] for tld in tlds}


def _rdap(domain: str) -> dict:
    tld = domain.rsplit(".", 1)[-1]
    base = _rdap_bootstrap().get(tld)
    if not base:
        return {"supported": 0}
    try:
        r = requests.get(base.rstrip("/") + "/domain/" + domain, timeout=RDAP_TIMEOUT_S,
                         headers={"User-Agent": UA, "Accept": "application/rdap+json"})
    except requests.RequestException:
        return {"supported": 1, "error": 1}
    if r.status_code == 404:
        return {"supported": 1, "found": 0}
    if r.status_code != 200:
        return {"supported": 1, "error": r.status_code}
    try:
        events = {e.get("eventAction"): e.get("eventDate") for e in r.json().get("events", [])}
    except (ValueError, AttributeError):  # not RDAP JSON
        return {"supported": 1, "error": 1}
    out = {"supported": 1, "found": 1}
    for action, key in (("registration", "registered"), ("expiration", "expires")):
        if events.get(action):
            try:
                out[key] = datetime.fromisoformat(events[action].replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return out


# ----------------------------------------------------------------------------- features
def lookup_host(host: str, offline: bool = False) -> dict | None:
    """Raw (cached) DNS/TLS/RDAP results for one host; offline=True reads the cache only and
    returns None when the host was never looked up."""
    if not host or is_ip_host(host):
        return {"dns": {}, "tls": {}, "rdap": {}}
    d = _cached("dns", host, _dns, offline=offline)
    if d is None:
        return None
    if not d.get("resolves"):
        return {"dns": d, "tls": {}, "rdap": {}, "complete": True}
    t = _cached("tls", host, _tls, offline=offline)
    rdap = _cached("rdap", registrable_domain(host), _rdap, ttl_days=30, offline=offline)
    # complete = every lookup was attempted (an empty result is a real "no certificate" answer;
    # a missing cache entry only means the collector has not reached this host yet)
    return {"dns": d, "tls": t or {}, "rdap": rdap or {},
            "complete": t is not None and rdap is not None}


def is_live(raw: dict | None) -> bool:
    return bool(raw and raw["dns"].get("resolves"))


def is_complete(raw: dict | None) -> bool:
    return bool(raw and raw.get("complete", True))


def host_feature_vector(url: str, raw: dict) -> list[float]:
    """Vector aligned with HOST_FEATURE_NAMES (NaN = unknown)."""
    nan = float("nan")
    host = host_of(url)
    rd = registrable_domain(host)
    # A tenant of a shared platform does not inherit the platform's popularity or age; the
    # platform's own homepage (bare apex / www) does, except a shortener's (its links redirect)
    own_home = host in (rd, "www." + rd) and is_bare(url) and not is_shortener(rd)
    shared = not own_home and (is_untrusted_hosting(rd) or is_user_content_host(host))
    rank = None if shared else _tranco_rank().get(rd)
    d, t, r = raw.get("dns", {}), raw.get("tls", {}), raw.get("rdap", {})
    now = time.time()

    def g(src: dict, k: str) -> float:
        v = src.get(k)
        return nan if v is None else float(v)

    found = r.get("found")
    return [
        math.log10(rank) if rank else nan, float(rank is not None),
        g(d, "resolves"), g(d, "n_a"), g(d, "min_ttl"), g(d, "has_mx"), g(d, "n_ns"),
        g(t, "ok") if t else (0.0 if d.get("resolves") else nan),  # 0 = resolves, no HTTPS
        g(t, "age_days"), g(t, "validity_days"), g(t, "free_ca"), g(t, "wildcard"),
        g(t, "n_san"), g(t, "cn_matches"),
        nan if shared or found is None else float(found),
        nan if shared or "registered" not in r else (now - r["registered"]) / 86400,
        nan if shared or "expires" not in r else (r["expires"] - now) / 86400,
    ]


def is_bare(url: str) -> bool:
    """Nothing after the host: no path, query, fragment or credentials. Parsed like host_of: a
    browser reads a backslash as '/' in http(s) URLs, so host + backslash + path is not bare."""
    u = ensure_scheme(url)
    if u[:5].lower() in ("http:", "https"):
        u = u.replace("\\", "/")
    try:
        p = urlsplit(u)
    except ValueError:
        return False
    return p.path in ("", "/") and not p.query and not p.fragment and "@" not in p.netloc


def segment_of(url: str, raw: dict | None, host_branch: bool) -> str:
    """Threshold segment: branch x popularity/age bucket x (bare domain | URL with path).
    Phishing is far rarer on Tranco-ranked domains than on young unranked ones, so each segment
    gets its own threshold (chosen on validation data by train.py). A bare subdomain of a ranked
    domain (x.tilda.ws) may be a tenant of a platform not yet listed: it gets its own
    'ranked_sub' bucket instead of the rank bucket, whose bare threshold can be "never flag"
    because the domain's own homepage is never phishing."""
    shape = "bare" if is_bare(url) else "path"
    if not host_branch:
        return f"lexical_{shape}"
    v = host_feature_vector(url, raw or {})
    rank, age = v[0], v[15]
    host = host_of(url)
    rd = registrable_domain(host)
    if rank == rank and shape == "bare" and host not in (rd, "www." + rd):
        bucket = "ranked_sub"
    elif rank == rank:  # not NaN
        bucket = "top10k" if rank <= 4 else "top100k" if rank <= 5 else "ranked"
    else:
        bucket = "old_unranked" if age == age and age > 3 * 365 else "other"
    return f"host_{bucket}_{shape}"


def raw_lookups(urls: list[str], offline: bool = False, workers: int = 16) -> dict:
    """{host: raw lookup or None} for the hosts of `urls`, looked up concurrently."""
    hosts = list(dict.fromkeys(host_of(u) for u in urls))
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(hosts, pool.map(lambda h: lookup_host(h, offline), hosts)))


# Live (single-URL) lookups: DNS on its own pool, TLS and RDAP on another. TLS / RDAP may keep
# running after the budget (up to ~6 s against a slow host); sharing one pool would let a stream
# of such URLs queue every other user's DNS lookup past its budget (-> fail closed -> LLM)
_DNS_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dns-lookup")
_LIVE_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="host-lookup")


def lookup_host_live(host: str, budget_s: float = 2.5) -> dict | None:
    """Single-host lookup for the pipeline: DNS first, then TLS and RDAP concurrently, all within
    `budget_s`. Parts that miss the budget are left empty (features become NaN)."""
    if not host or is_ip_host(host):
        return {"dns": {}, "tls": {}, "rdap": {}}
    t0 = time.perf_counter()
    try:
        d = _DNS_POOL.submit(_cached, "dns", host, _dns).result(timeout=budget_s)
    except Exception:
        return None
    if not d.get("resolves"):
        return {"dns": d, "tls": {}, "rdap": {}}
    ft = _LIVE_POOL.submit(_cached, "tls", host, _tls)
    fr = _LIVE_POOL.submit(_cached, "rdap", registrable_domain(host), _rdap, 30)
    out = {"dns": d, "tls": {}, "rdap": {}}
    for key, fut in (("tls", ft), ("rdap", fr)):
        try:
            out[key] = fut.result(timeout=max(0.05, budget_s - (time.perf_counter() - t0))) or {}
        except Exception:
            pass  # still running in the background; it will be cached for the next click
    return out
