import pytest

from typosquatting.brands import all_official_domains, load_vn_brands
from typosquatting.typosquatting import check_typosquatting, suspected_brand


@pytest.mark.parametrize("url,brand,kind", [
    ("https://vietcombamk.com/login", "vietcombank", "typosquat"),        # substitution
    ("https://agirbank.vn", "agribank", "typosquat"),                     # transposition (DL = 1)
    ("https://paypa1.com", "paypal", "typosquat"),
    ("https://xn--pypal-4ve.com/", "paypal", "homoglyph"),                # Cyrillic 'а'
    ("https://vietcombank-online.top", "vietcombank", "embedded"),
    ("https://paypal-support-center.com", "paypal", "embedded"),
    ("http://paypal.secure-login.xyz/", "paypal", "subdomain"),
    ("https://evil.com/www-paypal-com/signin", "paypal", "path_fake_domain"),
    ("http://abc.vercel.app/paypal/login", "paypal", "path_brand_lure"),
    ("https://paypal-login.s3.amazonaws.com/index.html", "paypal", "subdomain"),
    ("https://momo.online/claim", "momo", "exact_label"),
])
def test_detects_impersonation(url, brand, kind):
    r = check_typosquatting(url)
    assert r["flag"] is True, r["reason"]
    assert (r["brand"], r["kind"]) == (brand, kind)


@pytest.mark.parametrize("url", [
    "https://vietcombank.com.vn/personal",       # official
    "https://www.google.com/search?q=paypal",    # official, no lure
    "https://stream.cz/",                         # common word close to 'steam'
    "https://orangepage.net/",                    # intl brand compound without lure
    "https://shopee.com.mx/",                     # brand's own regional site
    "https://otpbank.hu/",                        # foreign bank close to 'tpbank'
    "https://dantri.com.vn/",
    "http://192.168.1.10/login",
])
def test_no_false_positive(url):
    assert check_typosquatting(url)["flag"] is False


def test_user_content_on_official_domain_is_not_auto_safe():
    r = check_typosquatting("https://docs.google.com/forms/d/x/viewform?paypal=login")
    assert r["flag"] is True and r["brand"] == "paypal"


def test_redirect_target_is_checked():
    from typosquatting.typosquatting import check_typosquatting_deep
    r = check_typosquatting_deep("https://www.google.com/url?q=https://vietcombamk.com/login")
    assert r["flag"] is True and r["brand"] == "vietcombank"


def test_short_vn_brand_as_hyphen_part():
    r = check_typosquatting("https://acb-baomat.online/secure")
    assert r["flag"] is True and r["brand"] == "acb"


def test_suspected_brand_and_catalogue():
    assert suspected_brand("https://vietcombamk.com") == "vietcombank"
    assert suspected_brand("https://dantri.com.vn") is None
    assert len(load_vn_brands()) > 50 and "vietcombank.com.vn" in all_official_domains()
