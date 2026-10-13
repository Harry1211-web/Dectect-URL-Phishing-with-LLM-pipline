"""False-alarm monitoring: no URL in the log, feedback matched to its verdict."""

import monitoring


def test_feedback_is_matched_without_storing_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "MONITORING_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "VERDICTS_PATH", tmp_path / "v.jsonl")
    monkeypatch.setattr(monitoring, "FEEDBACK_PATH", tmp_path / "f.jsonl")
    monkeypatch.setattr(monitoring, "SECRET_PATH", tmp_path / "k")
    secret_url = "https://shop.example.com/account?token=abc123&email=a@b.c"
    for url, verdict, by in ((secret_url, "suspicious", "D_llm"),
                             ("https://news.example.org/a", "clean", "C_classifier"),
                             ("https://evil.top/login", "clean", "C_classifier")):
        monitoring.record_verdict({"url": url, "verdict": verdict, "decided_by": by, "layers": {}})
    monitoring.record_feedback(secret_url, "clean")
    monitoring.record_feedback("https://evil.top/login", "phishing")
    log = (tmp_path / "v.jsonl").read_text(encoding="utf-8") + (tmp_path / "f.jsonl").read_text(encoding="utf-8")
    assert "token" not in log and "abc123" not in log and "/account" not in log
    r = monitoring.report(days=1)
    assert r["n_verdicts"] == 3 and r["reported_false_alarms"] == 1
    assert r["false_alarms_by_layer"] == {"D_llm": 1} and r["false_alarm_domains"] == [("example.com", 1)]
    assert r["reported_misses"] == 1 and r["misses_by_layer"] == {"C_classifier": 1}
