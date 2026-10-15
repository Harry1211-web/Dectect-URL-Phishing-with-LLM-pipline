"""Load the brand catalogue from ``dataset/brands/``.

- Used by the typosquatting layer (B), the LLM prompt notes (D) and the trusted-domain shortcut.
- Vietnamese brands: `vn_brand_domains.csv` (cross-checked against official sources).
- International brands: `brands.csv` + `brands_extra.csv`.
- Secondary legitimate domains (CDN, infrastructure, regional sites): `secondary_official_domains.csv`.
"""

from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_utils import registrable_domain  # noqa: E402

BRANDS_DIR = Path(__file__).resolve().parents[2] / "dataset" / "brands"
VN_BRANDS_PATH = BRANDS_DIR / "vn_brand_domains.csv"
INTL_BRANDS_PATH = BRANDS_DIR / "brands.csv"
INTL_BRANDS_EXTRA_PATH = BRANDS_DIR / "brands_extra.csv"
SECONDARY_DOMAINS_PATH = BRANDS_DIR / "secondary_official_domains.csv"

VALID_CATEGORIES = {"bank", "e_wallet", "ecommerce", "telecom", "airline", "public_service",
                    "logistics", "insurance", "securities", "consumer_finance", "transport",
                    "conglomerate"}
VALID_VERIFIED = {"verified", "unverified"}
INTL_CATEGORY = "international"


class Brand(TypedDict):
    label: str           # Unique brand label used for string matching (alphanumeric)
    category: str        # Industry group; 'international' for brands.csv entries
    domains: list[str]   # Official registrable domains
    verified: str        # 'verified' only if every domain was checked against an official source
    note: str            # Free-text note (renames, scope), empty if none


@lru_cache(maxsize=1)
def load_vn_brands() -> tuple[Brand, ...]:
    """Read the Vietnamese brand CSV, group rows by brand and validate integrity."""
    with VN_BRANDS_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["brand", "domain", "category", "verified", "note"], \
            f"Unexpected CSV header: {reader.fieldnames}"
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]

    grouped: dict[str, Brand] = {}
    domain_owner: dict[str, str] = {}
    for r in rows:
        label, domain = r["brand"].lower(), r["domain"].lower()
        assert label.isalnum(), f"Brand label must be alphanumeric: {label!r}"
        assert r["category"] in VALID_CATEGORIES, f"Invalid category {r['category']!r} ({label})"
        assert r["verified"] in VALID_VERIFIED, f"Invalid verified value {r['verified']!r} ({label})"
        assert "." in domain and "/" not in domain and " " not in domain, f"Bad domain: {domain!r}"
        assert domain_owner.get(domain, label) == label, \
            f"Domain '{domain}' assigned to two brands: {domain_owner.get(domain)} and {label}"
        domain_owner[domain] = label

        b = grouped.get(label)
        if b is None:
            grouped[label] = Brand(label=label, category=r["category"], domains=[domain],
                                   verified=r["verified"], note=r["note"])
            continue
        assert b["category"] == r["category"], f"{label}: inconsistent category across rows"
        b["domains"].append(domain)
        # A brand is 'verified' only when all of its domains are verified
        if r["verified"] != "verified":
            b["verified"] = "unverified"
        if r["note"] and r["note"] not in b["note"]:
            b["note"] = f"{b['note']}; {r['note']}".strip("; ")
    return tuple(grouped.values())


def _read_intl_file(path: Path, note: str, out: dict[str, Brand],
                    domain_owner: dict[str, str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            label = (r.get("identifier") or "").strip().lower()
            website = (r.get("website") or "").strip()
            if not label or not label.isalnum() or not website:
                continue
            host = urlparse(website).hostname
            if not host:
                continue
            domain = host.lower().removeprefix("www.")
            if domain in domain_owner or label in out:
                continue  # skip duplicate domain or label
            domain_owner[domain] = label
            out[label] = Brand(label=label, category=INTL_CATEGORY, domains=[domain],
                               verified="not_applicable", note=note)


@lru_cache(maxsize=1)
def load_intl_brands() -> tuple[Brand, ...]:
    """Read international brands from `brands.csv` and `brands_extra.csv`."""
    out: dict[str, Brand] = {}
    owner: dict[str, str] = {}
    _read_intl_file(INTL_BRANDS_PATH, "Source: standard international brand catalogue", out, owner)
    if INTL_BRANDS_EXTRA_PATH.exists():
        _read_intl_file(INTL_BRANDS_EXTRA_PATH, "Source: brands_extra.csv", out, owner)
    return tuple(out.values())


@lru_cache(maxsize=1)
def intl_source_categories() -> dict[str, str]:
    """International brand label -> category of its source CSV ("Finance", "Retail", ...)."""
    out: dict[str, str] = {}
    for path in (INTL_BRANDS_PATH, INTL_BRANDS_EXTRA_PATH):
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    out.setdefault((r.get("identifier") or "").strip().lower(),
                                   (r.get("category") or "").strip())
    return out


@lru_cache(maxsize=1)
def load_all_brands() -> tuple[Brand, ...]:
    """Vietnamese + international brands (input of the typosquatting check)."""
    return load_vn_brands() + load_intl_brands()


@lru_cache(maxsize=1)
def brand_by_label() -> dict[str, Brand]:
    return {b["label"]: b for b in load_all_brands()}


@lru_cache(maxsize=1)
def load_secondary_domains() -> dict[str, str]:
    """Legitimate secondary domains of known brands -> {domain: brand label}."""
    if not SECONDARY_DOMAINS_PATH.exists():
        return {}
    labels = set(brand_by_label())
    out: dict[str, str] = {}
    with SECONDARY_DOMAINS_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["brand", "domain", "verified", "note"], \
            f"Unexpected CSV header: {reader.fieldnames}"
        for r in reader:
            label = (r["brand"] or "").strip().lower()
            domain = (r["domain"] or "").strip().lower()
            assert label in labels, f"Brand '{label}' is not in the brand catalogue"
            assert (r["verified"] or "").strip() in VALID_VERIFIED, \
                f"Invalid verified value {r['verified']!r} ({domain})"
            assert "." in domain and "/" not in domain and " " not in domain, f"Bad domain: {domain!r}"
            out[domain] = label
    return out


@lru_cache(maxsize=1)
def _domain_to_label() -> dict[str, str]:
    m = {d.lower(): b["label"] for b in load_all_brands() for d in b["domains"]}
    for d, label in load_secondary_domains().items():
        m.setdefault(d, label)
    return m


def label_of_domain(domain: str) -> str | None:
    """Brand label owning an official (primary or secondary) domain."""
    return _domain_to_label().get(domain.lower())


def all_official_domains() -> set[str]:
    """Every official domain of every brand, secondary domains included."""
    return set(_domain_to_label())


@lru_cache(maxsize=1)
def verified_official_domains() -> frozenset[str]:
    """Official domains of Vietnamese brands that were verified against official sources."""
    return frozenset(d.lower() for b in load_vn_brands() if b["verified"] == "verified"
                     for d in b["domains"])


def normalize_domain(url_or_domain: str) -> str:
    """URL or hostname -> lower-case registrable domain."""
    return registrable_domain(url_or_domain)


if __name__ == "__main__":
    from collections import Counter
    vn = load_vn_brands()
    n_verified = sum(1 for b in vn if b["verified"] == "verified")
    print(f"{len(vn)} VN brands ({n_verified} verified), {len(load_intl_brands())} international, "
          f"{len(all_official_domains())} official domains")
    for cat, n in sorted(Counter(b["category"] for b in vn).items()):
        print(f"  {cat:16s} {n}")
