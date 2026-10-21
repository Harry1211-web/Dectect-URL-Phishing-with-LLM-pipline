"""False-alarm monitoring: no URL in the log, feedback matched to its verdict."""

import monitoring


def test_feedback_is_matched_without_storing_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "MONITORING_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "SECRET_PATH", tmp_path / "k")
    secret_url = "https://shop.example.com/account?token=abc123&email=a@b.c"
    for url, verdict, by in ((secret_url, "suspicious", "D_llm"),
                             ("https://news.example.org/a", "clean", "C_classifier"),
                             ("https://evil.top/login", "clean", "C_classifier")):
        monitoring.record_verdict({"url": url, "verdict": verdict, "decided_by": by, "layers": {}})
    monitoring.record_feedback(secret_url, "clean")
    monitoring.record_feedback("https://evil.top/login", "phishing")
    log = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.jsonl"))
    assert "token" not in log and "abc123" not in log and "/account" not in log
    r = monitoring.report(days=1)
    assert r["n_verdicts"] == 3 and r["reported_false_alarms"] == 1
    assert r["false_alarms_by_layer"] == {"D_llm": 1} and r["false_alarm_domains"] == [("example.com", 1)]
    assert r["reported_misses"] == 1 and r["misses_by_layer"] == {"C_classifier": 1}


def test_logs_rotate_per_month_and_old_months_are_deleted(tmp_path, monkeypatch):
    """One file per month; months past KEEP_MONTHS are deleted; the older single-file format
    is still read; a report only opens the months it covers."""
    import json
    from datetime import datetime, timezone
    monkeypatch.setattr(monitoring, "MONITORING_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "SECRET_PATH", tmp_path / "k")
    monkeypatch.setattr(monitoring, "KEEP_MONTHS", 12)
    now = datetime.now(timezone.utc)
    monitoring.record_verdict({"url": "https://a.example/", "verdict": "clean",
                               "decided_by": "C_classifier", "layers": {}})
    assert (tmp_path / f"verdicts-{now:%Y-%m}.jsonl").exists()
    legacy = {"time": now.isoformat(), "url_id": "x", "domain": "b.example", "verdict": "phishing",
              "decided_by": "A_blocklist"}
    (tmp_path / "verdicts.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    old = tmp_path / f"verdicts-{now.year - 2}-01.jsonl"
    old.write_text(json.dumps({**legacy, "time": "2000-01-01T00:00:00+00:00"}) + "\n", encoding="utf-8")
    assert monitoring.report(days=1)["n_verdicts"] == 2
    assert monitoring.prune(now) == [old.name] and not old.exists()
    assert (tmp_path / f"verdicts-{now:%Y-%m}.jsonl").exists()
