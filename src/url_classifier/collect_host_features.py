"""Collect host features (DNS -> TLS -> RDAP) for the dataset rows used by the XGBoost stage.

Staged so slow lookups only run where they matter: TLS only for hosts that resolve, RDAP once per
registrable domain that resolves. Everything goes to the cache (data/host_cache.sqlite), so the
script can be stopped and resumed, and train.py reads features from the cache afterwards.

Run: python src/url_classifier/collect_host_features.py [--max-xgb-rows 80000]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from url_classifier import host_features as hf  # noqa: E402
from url_classifier.train import DATA_PATH, _cnn_part  # noqa: E402
from url_utils import host_of, is_ip_host, registrable_domain  # noqa: E402


def _run(stage: str, fn, keys: list[str], workers: int) -> dict:
    t0, out = time.time(), {}
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fn, k): k for k in keys}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = {}
            if i % 5000 == 0 or i == len(keys):
                print(f"  {stage}: {i}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def rows_needing_features(max_xgb_rows: int, seed: int = 42, all_train: bool = False,
                          cnn_part_sample: int = 0) -> pd.DataFrame:
    """Rows of the XGBoost train part (sampled) + all val/test rows. all_train=True takes every
    train row; cnn_part_sample adds a sample of the CNN train part (for stacking experiments)."""
    df = pd.read_csv(DATA_PATH)
    if all_train:
        return df
    xgb_part = df[(df.split == "train") & ~df.group.map(_cnn_part)]
    if len(xgb_part) > max_xgb_rows:
        xgb_part = xgb_part.sample(max_xgb_rows, random_state=seed)
    extra = df.iloc[0:0]
    if cnn_part_sample:
        cnn_part = df[(df.split == "train") & df.group.map(_cnn_part)]
        extra = cnn_part.sample(min(cnn_part_sample, len(cnn_part)), random_state=seed)
    return pd.concat([xgb_part, extra, df[df.split != "train"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-xgb-rows", type=int, default=80_000)
    ap.add_argument("--dns-workers", type=int, default=48)
    ap.add_argument("--tls-workers", type=int, default=96)
    ap.add_argument("--rdap-workers", type=int, default=24)
    ap.add_argument("--all-train", action="store_true", help="look up every train row too")
    ap.add_argument("--cnn-part-sample", type=int, default=0,
                    help="also look up this many rows of the CNN train part")
    args = ap.parse_args()

    rows = rows_needing_features(args.max_xgb_rows, all_train=args.all_train,
                                 cnn_part_sample=args.cnn_part_sample)
    hosts = [h for h in dict.fromkeys(rows.url.map(host_of)) if h and not is_ip_host(h)]
    print(f"{len(rows)} rows, {len(hosts)} hosts", flush=True)

    dns = _run("dns", lambda h: hf._cached("dns", h, hf._dns), hosts, args.dns_workers)
    live = [h for h in hosts if dns.get(h, {}).get("resolves")]
    print(f"live hosts: {len(live)}/{len(hosts)}", flush=True)
    _run("tls", lambda h: hf._cached("tls", h, hf._tls), live, args.tls_workers)
    domains = list(dict.fromkeys(registrable_domain(h) for h in live))
    hf._rdap_bootstrap()
    _run("rdap", lambda d: hf._cached("rdap", d, hf._rdap, ttl_days=30), domains, args.rdap_workers)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
