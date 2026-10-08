from url_utils import host_of, is_ip_host, normalize_url, registrable_domain, strip_scheme_www


def test_registrable_domain_handles_multi_part_suffix():
    assert registrable_domain("https://a.b.vietcombank.com.vn/x") == "vietcombank.com.vn"
    assert registrable_domain("evil.co.uk") == "evil.co.uk"


def test_host_strips_credentials_and_port():
    assert host_of("http://user:pw@Example.COM:8080/a") == "example.com"


def test_normalize_url_is_scheme_www_and_slash_insensitive():
    assert normalize_url("https://www.Example.com/a/") == normalize_url("http://example.com/a")
    assert normalize_url("example.com:443/a#frag") == "example.com/a"
    assert normalize_url("http://x.com/a?b=1") == "x.com/a?b=1"


def test_strip_scheme_www():
    assert strip_scheme_www("https://www.abc.com/x") == "abc.com/x"


def test_embedded_urls():
    from url_utils import embedded_urls
    u = "https://t.lijit.com/r?ru=https%3A%2F%2Fnew.express.adobe.com%2Fwebpage%2Fx"
    assert embedded_urls(u) == ["https://new.express.adobe.com/webpage/x"]
    assert embedded_urls("https://site.com/login?next=/account") == []


def test_ip_host():
    assert is_ip_host("192.168.1.1") and is_ip_host("[::1]") and not is_ip_host("a.com")
