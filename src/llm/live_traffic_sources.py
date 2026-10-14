"""Live traffic sources for LLM benchmarks: Tranco (legit domains) and OpenPhish (phishing URLs)."""

from __future__ import annotations

import io
import random
import re
import zipfile

import pandas as pd
import requests

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
OPENPHISH_URL = "https://openphish.com/feed.txt"

# Infrastructure domains in Tranco (DNS, CDN, API back-ends...) that serve no web page
_RE_INFRA = re.compile(
    r"(^awsdns-|-?cdn-?|(^|[.-])api([.-]|$)|dns\d*\.|traffic-?manager|"
    r"webvisor|park-my-domain|^yourdomain\.|apm\.|-apm\.)",
    re.IGNORECASE,
)


def drop_infrastructure(domains: list[str]) -> list[str]:
    return [d for d in domains if not _RE_INFRA.search(d)]


def fetch_tranco_domains(n: int, pool: int = 50_000, seed: int = 42,
                         drop_infra: bool = True) -> list[str]:
    """`n` random legit domains from the Tranco top `pool`."""
    r = requests.get(TRANCO_URL, timeout=30)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=["rank", "domain"], nrows=pool)
    domains = df["domain"].tolist()
    if drop_infra:
        domains = drop_infrastructure(domains)
    random.Random(seed).shuffle(domains)
    return domains[:n]


def fetch_openphish_urls(n: int, seed: int = 42) -> list[str]:
    """`n` random live phishing URLs from the OpenPhish public feed."""
    r = requests.get(OPENPHISH_URL, timeout=15)
    r.raise_for_status()
    urls = [ln.strip() for ln in r.text.splitlines() if ln.strip().startswith("http")]
    random.Random(seed).shuffle(urls)
    return urls[:n]
