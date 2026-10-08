"""RAG tier 1 — structured lookup in the official brand-domain table.

Only Vietnamese brands verified against official sources (`verified_official_domains()`) are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typosquatting.brands import (  # noqa: E402
    brand_by_label, label_of_domain, normalize_domain, verified_official_domains,
)
from typosquatting.typosquatting import suspected_brand  # noqa: E402


def lookup_tier1(url: str) -> dict | None:
    """Brand facts the LLM can compare against.

    1. The domain is a verified official domain -> confirm it.
    2. The URL impersonates a brand -> return that brand's real official domains.
    """
    domain = normalize_domain(url)
    verified = verified_official_domains()
    brands = brand_by_label()

    if domain in verified:
        label = label_of_domain(domain)
        return {
            "match": "official_domain",
            "brand": label,
            "official_domains": sorted(brands[label]["domains"]) if label else [domain],
            "note": f"'{domain}' là một domain chính thức đã xác minh của thương hiệu '{label}'.",
        }

    label = suspected_brand(url)
    if label and label in brands:
        real = sorted(d for d in brands[label]["domains"] if d in verified)
        if not real:
            return None
        return {
            "match": "suspected_impersonation",
            "brand": label,
            "official_domains": real,
            "note": (f"'{domain}' KHÔNG nằm trong danh sách domain chính thức đã xác minh "
                     f"của '{label}' — domain chính thức: {', '.join(real)}."),
        }
    return None
