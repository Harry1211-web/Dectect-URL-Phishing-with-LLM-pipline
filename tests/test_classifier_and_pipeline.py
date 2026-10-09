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
