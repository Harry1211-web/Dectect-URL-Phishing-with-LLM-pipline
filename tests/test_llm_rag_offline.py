"""Layer D logic that runs without calling the OpenAI API."""

from html_sanitize import sanitize_html
from llm_client import (
    PROMPT_TEMPLATE_RAG, WARNING_INJECTION_DETECTED, analyze_suspicious, build_prompt,
    check_typosquatting_with_redirect, parse_verdict,
)
from rag.pipeline import BLOCK_END, BLOCK_START
from rag.tier1 import lookup_tier1
from redteam_prompt_injection import BASE_SCENARIOS, INJECTIONS


def test_sanitizer_removes_every_injection_technique():
    sc = BASE_SCENARIOS[0]
    for name, inject in INJECTIONS.items():
        out = sanitize_html(inject(sc["content"]), url=sc["url"])
        assert out["prompt_injection_suspected"], name
        assert "OVERRIDE" not in out["text"] and "VERDICT" not in out["text"], name


def test_sanitizer_keeps_legit_warning_article():
    html = ("<p>Cảnh giác với trang tự nhận đã được xác minh an toàn bởi ngân hàng.</p>"
            "<p style='opacity:0.5'>Nội dung mờ nhưng vẫn hiển thị.</p>")
    out = sanitize_html(html, url="https://chongluadao.vn/a")
    assert not out["prompt_injection_suspected"]
    assert "hiển thị" in out["text"]


def test_parse_verdict():
    assert parse_verdict("VERDICT: phishing\nGIAI_THICH: giả mạo") == "phishing"
    assert parse_verdict("VERDICT: safe") == "safe"
    assert parse_verdict("Trang này không có dấu hiệu lừa đảo, an toàn.") == "safe"
    assert parse_verdict("???") is None


def test_prompt_frames_page_and_rag_as_data():
    p = build_prompt("https://x.com", "Tiêu đề", "Nội dung", True, True, PROMPT_TEMPLATE_RAG,
                     f"{BLOCK_START}\nRAG\n{BLOCK_END}")
    assert "<<<NỘI DUNG TRANG BẮT ĐẦU" in p and BLOCK_START in p
    assert WARNING_INJECTION_DETECTED in p and "{" not in p


def test_tier1_lookup():
    assert lookup_tier1("https://vietcombank.com.vn")["match"] == "official_domain"
    r = lookup_tier1("https://vietcombamk.com/login")
    assert r["match"] == "suspected_impersonation" and "vietcombank.com.vn" in r["official_domains"]


def test_redirect_target_is_checked():
    r = check_typosquatting_with_redirect("https://bit.ly/abc", "https://vietcombamk.com/login")
    assert r["flag"] is True


def test_unfetchable_page_is_reported_as_unusable():
    r = analyze_suspicious("https://x.invalid", page=("", "", False, "https://x.invalid"))
    assert r["timed_out"] is True and r["verdict"] is None
