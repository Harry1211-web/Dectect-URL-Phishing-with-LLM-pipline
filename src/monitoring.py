"""False-alarm monitoring on real traffic: log every pipeline verdict, collect user feedback
("this page is not phishing" / "this page is phishing") and report the reported error rates.

Privacy: no URL is stored. Each entry keeps the registrable domain and a keyed hash of the URL
(HMAC-SHA256 with a local secret in data/monitoring/secret.key), so feedback can be matched to its
verdict but paths and query strings (tokens, e-mails) never reach the log.

Files: data/monitoring/verdicts.jsonl, data/monitoring/feedback.jsonl (one JSON object per line).

Run:
  python src/pipeline.py <url> --log                         (logs the verdict)
  python src/monitoring.py feedback <url> clean|phishing     (user report on a page)
  python src/monitoring.py report [--days 7]
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from url_utils import normalize_url, registrable_domain  # noqa: E402

MONITORING_DIR = Path(__file__).resolve().parents[1] / "data" / "monitoring"
VERDICTS_PATH = MONITORING_DIR / "verdicts.jsonl"
FEEDBACK_PATH = MONITORING_DIR / "feedback.jsonl"
SECRET_PATH = MONITORING_DIR / "secret.key"
WARNED = ("phishing", "suspicious")  # verdicts that show the user a warning


def _secret() -> bytes:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_PATH.exists():
        SECRET_PATH.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(SECRET_PATH.read_text(encoding="utf-8").strip())


def url_id(url: str) -> str:
    """Keyed hash of the normalized URL: same page -> same id, URL not recoverable."""
    return hmac.new(_secret(), normalize_url(url).encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def _append(path: Path, entry: dict) -> None:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_verdict(result: dict) -> None:
    """Log one analyze_url() result (no URL, no page text)."""
    layers = result.get("layers", {})
    c = layers.get("C_classifier") or {}
    d = layers.get("D_llm") or {}
    _append(VERDICTS_PATH, {
        "time": _now(), "url_id": url_id(result["url"]), "domain": registrable_domain(result["url"]),
        "verdict": result["verdict"], "decided_by": result["decided_by"],
        "B_flag": (layers.get("B_typosquatting") or {}).get("flag"),
        "C_score": c.get("score"), "D_verdict": d.get("verdict"),
        "latency_ms": result.get("latency_ms"),
    })


def record_feedback(url: str, label: str) -> None:
    """User report: label 'clean' (false alarm) or 'phishing' (missed / confirmed)."""
    if label not in ("clean", "phishing"):
        raise ValueError("label must be 'clean' or 'phishing'")
    _append(FEEDBACK_PATH, {"time": _now(), "url_id": url_id(url),
                            "domain": registrable_domain(url), "label": label})


def _read(path: Path, since: datetime) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r for r in rows if datetime.fromisoformat(r["time"]) >= since]


def report(days: int = 7) -> dict:
    """Reported error rates over the last `days`. Feedback is matched to the latest verdict of the
    same URL; a rate is a lower bound (most users never report)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    verdicts, feedback = _read(VERDICTS_PATH, since), _read(FEEDBACK_PATH, since)
    latest = {v["url_id"]: v for v in verdicts}  # file order = time order
    false_alarms, misses, confirmed = [], [], 0
    for fb in {f["url_id"]: f for f in feedback}.values():  # last report per URL
        v = latest.get(fb["url_id"])
        if v is None:
            continue
        if fb["label"] == "clean" and v["verdict"] in WARNED:
            false_alarms.append(v)
        elif fb["label"] == "phishing" and v["verdict"] == "clean":
            misses.append(v)
        elif fb["label"] == "phishing":
            confirmed += 1
    n = len(verdicts)
    n_warned = sum(v["verdict"] in WARNED for v in verdicts)
    return {
        "days": days, "n_verdicts": n,
        "verdicts": dict(Counter(v["verdict"] for v in verdicts)),
        "decided_by": dict(Counter(v["decided_by"] for v in verdicts)),
        "warning_rate": round(n_warned / n, 4) if n else None,
        "n_feedback": len(feedback),
        "reported_false_alarms": len(false_alarms),
        "reported_false_alarm_rate_of_warnings": round(len(false_alarms) / n_warned, 4) if n_warned else None,
        "false_alarms_by_layer": dict(Counter(v["decided_by"] for v in false_alarms)),
        "false_alarm_domains": Counter(v["domain"] for v in false_alarms).most_common(10),
        "reported_misses": len(misses),
        "misses_by_layer": dict(Counter(v["decided_by"] for v in misses)),
        "confirmed_phishing": confirmed,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fb = sub.add_parser("feedback", help="record a user report on a page")
    fb.add_argument("url")
    fb.add_argument("label", choices=("clean", "phishing"))
    rp = sub.add_parser("report", help="reported error rates")
    rp.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    if args.cmd == "feedback":
        record_feedback(args.url, args.label)
        print("recorded")
    else:
        print(json.dumps(report(args.days), ensure_ascii=False, indent=2))
