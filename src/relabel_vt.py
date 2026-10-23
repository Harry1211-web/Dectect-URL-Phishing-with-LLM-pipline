"""Fill the relabelling subset with VirusTotal labels (the reference labels; no hand labels).

Rule (virustotal.verdict_of): >= 2 malicious engines -> phishing, 0 malicious and
0 suspicious -> clean, otherwise (1 engine, suspicious only, or URL unknown to VirusTotal) ->
unsure. URLs are never submitted. Adds the vt_* columns; vt_label is what score compares with.
These are VirusTotal labels, not hand labels; they must not be used to evaluate Layer A, which
itself queries VirusTotal.
Run:    python src/relabel_vt.py          (~15 s per uncached URL, free quota)
Output: reports/relabel/relabel_subset.csv (updated in place)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from virustotal import lookup_batch, verdict_of  # noqa: E402
from relabel_subset import CSV_PATH  # noqa: E402

STAT_KEYS = ("malicious", "suspicious", "harmless", "undetected")


def main() -> int:
    d = pd.read_csv(CSV_PATH)
    res = lookup_batch(d.url.tolist(), progress=lambda m: print(m, flush=True))
    d["vt_status"] = [res[u]["status"] for u in d.url]
    for k in STAT_KEYS:
        d[f"vt_{k}"] = [(res[u].get("stats") or {}).get(k) for u in d.url]
    d["vt_label"] = [verdict_of(res[u]) for u in d.url]
    done = d.vt_status != "error"
    d.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(d.vt_status.value_counts().to_dict(), d.vt_label.value_counts().to_dict())
    return 0 if done.all() else 1


if __name__ == "__main__":
    sys.exit(main())
