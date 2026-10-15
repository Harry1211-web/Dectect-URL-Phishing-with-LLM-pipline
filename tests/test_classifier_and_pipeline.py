import pytest
import torch

from url_classifier import predict
from url_classifier.features import FEATURE_NAMES, url_features
from url_classifier.model import EMBED_DIM, MAX_LEN, CharCnnLstm, encode_urls


def test_feature_vector_shape_and_scheme_invariance():
    a = url_features("https://www.example.com/login.php?x=1")
    b = url_features("http://example.com/login.php?x=1")
    assert len(a) == len(FEATURE_NAMES) and a == b


def test_encoder_keeps_head_and_tail_of_long_urls():
    x = encode_urls(["a.com/" + "x" * 500 + "TAIL"])
    assert x.shape == (1, MAX_LEN) and x[0, -1] != 0


def test_model_shapes():
    m = CharCnnLstm()
    x = torch.from_numpy(encode_urls(["https://a.com", "b.org/x?y=1"]))
    assert m(x).shape == (2,) and m.embed(x).shape == (2, EMBED_DIM)


needs_model = pytest.mark.skipif(not predict.is_available(), reason="Layer C model not trained")


@needs_model
def test_classifier_output():
    r = predict.classify_url("http://paypal-verify-account.xyz/login.php", use_host_features=False)
    assert r["label"] in ("clean", "suspicious") and 0 <= r["score"] <= 1
    assert r["branch"] == "lexical"


@needs_model
def test_pipeline_without_llm():
    from pipeline import analyze_url
    r = analyze_url("https://vietcombamk.com/login", use_llm=False, use_host_features=False)
    assert r["verdict"] == "phishing" and r["decided_by"] in ("A_blocklist", "fallback_B")
    r = analyze_url("https://vnexpress.net/", use_llm=False, use_host_features=False)
    assert r["verdict"] in ("clean", "suspicious")


def test_llm_safe_is_overruled_when_classifier_is_very_sure(monkeypatch):
    import pipeline
    from llm import llm_client

    def fake_classifier(score):
        return lambda url, use_host_features=True: {"label": "suspicious", "score": score,
                                                    "threshold": 0.5}
    monkeypatch.setattr(pipeline.classifier, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "analyze_suspicious", lambda *a, **k: {
        "verdict": "safe", "explanation": "", "timed_out": False})
    url = "https://some-unknown-site.example/page"
    monkeypatch.setattr(pipeline.classifier, "classify_url", fake_classifier(0.995))
    r = pipeline.analyze_url(url)
    assert r["verdict"] == "suspicious" and r["decided_by"] == "D_llm+C_override"
    monkeypatch.setattr(pipeline.classifier, "classify_url", fake_classifier(0.7))
    assert pipeline.analyze_url(url)["verdict"] == "clean"
    # Official brand domain: a very high score is a classifier error, the LLM verdict stands
    monkeypatch.setattr(pipeline.classifier, "classify_url", fake_classifier(0.999))
    assert pipeline.analyze_url("https://accounts.google.com/ServiceLogin")["verdict"] == "clean"


def test_verified_official_domain_is_trusted_without_loopholes():
    from pipeline import analyze_url, trusted_official_domain
    r = analyze_url("https://www.vietcombank.com.vn/vi-VN/KHCN", use_llm=False)
    assert r["verdict"] == "clean" and r["decided_by"] == "verified_official_domain"
    assert trusted_official_domain("https://ebank.bidv.vn/login") == "bidv.vn"
    # Hidden destination (open redirect), marketplace content, user profiles, look-alikes
    assert trusted_official_domain(
        "https://vietcombank.com.vn/redirect?url=https://vietcombamk.com/login") is None
    assert trusted_official_domain("https://shopee.vn/some-shop") is None
    assert trusted_official_domain("https://zalo.me/g/abcdef") is None
    assert trusted_official_domain("https://vietcombank.com.vn.evil.top/") is None
    assert trusted_official_domain("https://vietcombamk.com/") is None
    # Host read like a browser ('\' ends the host) and redirect values that are not full URLs
    for u in ("https://evil.com\\@vietcombank.com.vn/", "https://evil.com\\.vietcombank.com.vn/",
              "https://vietcombank.com.vn/r?next=//evil.com", "https://vietcombank.com.vn/r?url=evil.com",
              "https://vietcombank.com.vn/r?to=%2F%2Fevil.com", "https://vietcombank.com.vn/go/https:/evil.com",
              "https://vietcombank.com.vn/r?x=1;next=//evil.com", "https://vietcombank.com.vn/r?//evil.com",
              "https://vietcombank.com.vn/go/evil.com", "https://vietcombank.com.vn/r#//evil.com",
              "https://vietcombank.com.vn/r?u=%252F%252Fevil.com", "https://vietcombank.com.vn/r?u=evil。com",
              "https://vietcombank.com.vn/r?u=evil％2Ecom", "https://vietcombank.com.vn//evil.com",
              "https://vietcombank.com.vn/r?next=/%09/3232235777", "https://vietcombank.com.vn/r?u=1.2.3.4",
              "https://vietcombank.com.vn/r?u=evil.com+", "https://vietcombank.com.vn/r?u=(evil.com)",
              "https://vietcombank.com.vn/r?u=evil.com%00", "https://vietcombank.com.vn/vi?utm_source=x",
              "https://vietcombank.com.vn/go/evil.com~", "https://vietcombank.com.vn/go/%E2%80%A4evil.com"):
        assert trusted_official_domain(u) is None, u
    for u in ("https://vietcombank.com.vn/vi/the-tin-dung", "https://vietcombank.com.vn/a/report.pdf"):
        assert trusted_official_domain(u) == "vietcombank.com.vn", u


def test_trusted_shortcut_international_brands():
    from pipeline import trusted_intl_domains, trusted_official_domain
    assert trusted_official_domain("https://www.paypal.com/signin") == "paypal.com"
    assert trusted_official_domain("https://www.paypal.com/signin", include_intl=False) is None
    # Same redirect rules as the Vietnamese domains
    assert trusted_official_domain("https://paypal.com/r?u=https://evil.com") is None
    # Platforms where anyone can publish under the brand's domain stay out
    trusted = trusted_intl_domains()
    for d in ("canva.com", "github.com", "pinterest.com", "token.im", "amazon.com", "airbnb.com",
              "facebook.com", "godaddy.com", "googleapis.com", "shopee.sg", "google.com",
              # every domain of an excluded brand, payment links
              "googlevideo.com", "google.dev", "microsoft365.com", "stripe.com", "cash.app"):
        assert d not in trusted, d
    assert trusted_official_domain("https://shopee.vn/some-shop") is None


def test_host_feature_vector_handles_missing_data():
    import math
    from url_classifier.host_features import HOST_FEATURE_NAMES, host_feature_vector
    v = host_feature_vector("https://unknown-host.example/", {})
    assert len(v) == len(HOST_FEATURE_NAMES) and math.isnan(v[2])
    v = host_feature_vector("https://abc.vercel.app/", {"dns": {"resolves": 1}, "tls": {}, "rdap": {}})
    assert math.isnan(v[0])  # a tenant of a shared platform does not inherit its Tranco rank


def test_bare_subdomain_of_ranked_domain_has_own_segment():
    from url_classifier.host_features import host_feature_vector, segment_of
    raw = {"dns": {"resolves": 1}, "tls": {}, "rdap": {}}
    if host_feature_vector("https://wikipedia.org/", raw)[0] != host_feature_vector("https://wikipedia.org/", raw)[0]:
        pytest.skip("Tranco list not available")
    assert segment_of("https://www.wikipedia.org/", raw, True) == "host_top10k_bare"
    # x.tilda.ws-like tenants must not get the "never flag" threshold of the brand's homepage
    assert segment_of("https://wallet-x.wikipedia.org/", raw, True) == "host_ranked_sub_bare"
    assert segment_of("https://en.wikipedia.org/wiki/Phishing", raw, True) == "host_top10k_path"


@needs_model
def test_failed_host_lookup_fails_closed(monkeypatch):
    """A resolver timeout (attacker-provokable) must never turn into a 'clean' verdict."""
    from url_classifier import host_features
    monkeypatch.setattr(host_features, "lookup_host_live", lambda host, budget_s=2.5: None)
    r = predict.classify_url("https://example-unknown-site.com/login")
    assert r["label"] == "suspicious" and r["branch"] == "lookup_failed"
    from pipeline import analyze_url
    out = analyze_url("https://example-unknown-site.com/login", use_llm=False)
    assert out["verdict"] in ("suspicious", "phishing")


def test_segment_thresholds_are_never_fail_open():
    import numpy as np
    from url_classifier.train import MAX_SEGMENT_THRESHOLD, segment_thresholds
    seg = np.array(["a"] * 4 + ["b"] * 4)
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.9, 0.2, 0.95])
    y = np.array([0, 0, 0, 0, 0, 1, 0, 1])  # segment "a" has no phishing at all
    thr, _ = segment_thresholds(seg, score, y, np.ones(8, bool), n_clean=6, min_clean=0.5)
    assert all(t <= MAX_SEGMENT_THRESHOLD for t in thr.values())
