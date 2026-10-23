"""Build the Vietnamese evaluation set of Layer C (data/test_sets/vi/). Evaluation only: never read
by build_dataset.py or train.py.

Phishing: domains / URLs blacklisted by the Chong Lua Dao project (chongluadao.vn), taken from a
public machine-readable mirror pinned to one commit (the project's own API is offline). Only
hosts that are still alive are kept: DNS resolves AND the domain was registered before the
snapshot date (a domain registered later has a new owner, so the blacklist no longer proves it).
Clean: highest-ranked .vn domains of the Tranco list + verified official domains of Vietnamese
brands (dataset/brands/vn_brand_domains.csv), DNS must resolve; reviewed look-alikes and
reseller sites (EXCLUDED_CLEAN) are left out.

Zero overlap: a URL is dropped when its registrable domain appears anywhere in the training
data (urls_labeled.csv, noisy_labels.csv) or in the external test set.

Run:    python src/data_collection/vi_testset.py   (network: GitHub, Tranco, DNS / TLS / RDAP)
Output: data/test_sets/vi/phishing.csv, data/test_sets/vi/clean.csv (audited labels applied)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection.sources import RAW_DIR, ROOT, _download, _lines, tranco_domains  # noqa: E402
from typosquatting.brands import verified_official_domains  # noqa: E402
from url_classifier import host_features as hf  # noqa: E402
from url_utils import ensure_scheme, registrable_domain  # noqa: E402

OUT_DIR = ROOT / "data" / "test_sets" / "vi"
# Mirror of the Chong Lua Dao blacklist, pinned to the last commit that changed the data
CLD_REPO = "elliotwutingfeng/ChongLuaDao-Phishing-Blocklist"
CLD_SNAPSHOT = dt.date(2024, 5, 16)
# Every file whose domains must never appear in this set
TRAINING_FILES = [ROOT / "dataset" / "urls" / "urls_labeled.csv",
                  ROOT / "dataset" / "urls" / "noisy_labels.csv",
                  ROOT / "dataset" / "external" / "phreshphish_urls.csv"]
MAX_CLEAN_TRANCO = 2000
# Tranco-ranked .vn domains that are not reliably clean (manual review 2026-10-12)
EXCLUDED_CLEAN = {
    **{d: "reseller / info site named after a brand, not the brand's own site"
       for d in ("dangky3gmobifone.vn", "dangkymobifone.vn", "dangky5gviettel.com.vn",
                 "vietteldata.vn", "vietteltop.vn", "viettelup.vn", "vinaphonedata.vn",
                 "vinfastgiare.vn", "sieuthidienmayxanh.vn")},
    **{d: "brand look-alike whose owner could not be verified (TLS error / connection reset)"
       for d in ("vietel.com.vn", "jntexpress.vn", "revolt.vn")},
}


def training_domains() -> set[str]:
    """Registrable domains of every URL used for training or other evaluations."""
    out: set[str] = set()
    for f in TRAINING_FILES:
        if f.exists():
            out |= set(pd.read_csv(f, usecols=["url"]).url.map(registrable_domain))
    return out


def as_url(entry: str) -> str:
    """Same shape as the training URLs: scheme added, '/' for a bare host."""
    url = ensure_scheme(entry)
    return url if "/" in url.split("://", 1)[1] else url + "/"


def lookup(hosts: list[str], workers: int = 16) -> dict[str, dict | None]:
    """Live DNS / TLS / RDAP lookups (stored in the host cache); None when the lookup errors."""
    def one(h: str) -> dict | None:
        try:
            return hf.lookup_host(h)
        except Exception:  # resolver / network error: treated as not alive
            return None
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(hosts, pool.map(one, hosts)))


def latest_data_commit() -> tuple[str, str]:
    """(sha, date) of the last mirror commit that changed urls.txt."""
    import requests
    r = requests.get(f"https://api.github.com/repos/{CLD_REPO}/commits",
                     params={"path": "urls.txt", "per_page": 1}, timeout=30)
    r.raise_for_status()
    c = r.json()[0]
    return c["sha"], c["commit"]["committer"]["date"][:10]


def build_phishing(exclude: set[str]) -> pd.DataFrame:
    sha, date = latest_data_commit()
    if date != CLD_SNAPSHOT.isoformat():
        raise SystemExit(f"Mirror data changed ({date}); review it and update CLD_SNAPSHOT.")
    raw_url = f"https://raw.githubusercontent.com/{CLD_REPO}/{sha}/urls.txt"
    p = _download(raw_url, RAW_DIR / f"chongluadao_urls_{sha[:7]}.txt", max_age_days=10_000)
    if p is None:
        raise SystemExit("Chong Lua Dao mirror could not be downloaded.")
    d = pd.DataFrame({"entry": _lines(p)})
    d["url"] = d.entry.map(as_url)
    d["domain"] = d.url.map(registrable_domain)
    d["host"] = d.url.map(lambda u: u.split("://", 1)[1].split("/", 1)[0].lower())
    n0 = len(d)
    d = d[~d.domain.isin(exclude)].drop_duplicates("url")
    print(f"phishing: {n0} listed, {len(d)} after removing training domains", flush=True)

    raw = lookup(d.host.unique().tolist())
    d["resolves"] = d.host.map(lambda h: hf.is_live(raw.get(h)))
    reg = {h: (r or {}).get("rdap", {}).get("registered") for h, r in raw.items()}
    d["registered"] = d.host.map(lambda h: dt.date.fromtimestamp(reg[h]).isoformat()
                                 if reg.get(h) else "")
    # Registered after the snapshot = new owner; unknown date is kept and flagged
    new_owner = d.registered.ne("") & (d.registered > CLD_SNAPSHOT.isoformat())
    print(f"phishing: {d.resolves.sum()} resolve, {(d.resolves & new_owner).sum()} of them "
          f"re-registered after {CLD_SNAPSHOT}", flush=True)
    d = d[d.resolves & ~new_owner].copy()
    d["source"] = "chongluadao"
    d["evidence"] = raw_url
    d["listed_by"] = CLD_SNAPSHOT.isoformat()
    return d[["url", "source", "evidence", "listed_by", "registered"]]


def build_clean(exclude: set[str]) -> pd.DataFrame:
    rank = {dom: i + 1 for i, dom in enumerate(tranco_domains())}
    vn = [(dom, "tranco_vn", f"tranco rank {r}") for dom, r in rank.items()
          if dom.endswith(".vn") and registrable_domain(dom) == dom]
    vn = [x for x in vn if x[0] not in exclude and x[0] not in EXCLUDED_CLEAN][:MAX_CLEAN_TRANCO]
    brands = [(dom, "vn_brand_verified", "dataset/brands/vn_brand_domains.csv")
              for dom in sorted(verified_official_domains()) if dom not in exclude]
    d = pd.DataFrame(brands + vn, columns=["domain", "source", "evidence"]).drop_duplicates("domain")
    print(f"clean: {len(brands)} brand + {len(vn)} Tranco .vn domains outside training", flush=True)
    raw = lookup(d.domain.tolist())
    d = d[d.domain.map(lambda h: hf.is_live(raw.get(h)))].copy()
    d["url"] = d.domain.map(as_url)
    return d[["url", "source", "evidence"]]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    exclude = training_domains()
    print(f"{len(exclude)} registrable domains excluded (training + external)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("clean", build_clean(exclude)), ("phishing", build_phishing(exclude))):
        df.sort_values("url").to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows -> {OUT_DIR / (name + '.csv')}", flush=True)
    from data_collection.audit_labels import apply_eval_set_fixes
    apply_eval_set_fixes(("vi",))  # audited labels (label_fixes.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
