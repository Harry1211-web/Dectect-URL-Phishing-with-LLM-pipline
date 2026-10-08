"""Free hosting / PaaS platforms and public URL shorteners.

- A subdomain of a PaaS (e.g. `phish-abc.vercel.app`) can be created in minutes for phishing.
- Shorteners hide the real destination.
- Used to add context warnings to the LLM prompt and as a feature of the URL classifier.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# PaaS, site builders, free hosting and dynamic DNS providers
FREE_HOSTING: frozenset[str] = frozenset({
    # Site builders / free static hosting
    "vercel.app", "pages.dev", "netlify.app", "github.io", "gitlab.io",
    "herokuapp.com", "onrender.com", "surge.sh", "glitch.me",
    "repl.co", "replit.dev", "replit.app", "firebaseapp.com", "web.app",
    "azurewebsites.net", "000webhostapp.com", "typedream.app",
    "streamlitapp.com", "streamlit.app", "vusercontent.net",
    "gitbook.io", "railway.app", "up.railway.app",
    # Free blog / CMS
    "blogspot.com", "wordpress.com", "weebly.com", "wixsite.com",
    "tumblr.com", "square.site", "myshopify.com", "webador.com",
    "jimdofree.com", "yolasite.com", "strikingly.com", "carrd.co",
    "godaddysites.com",
    "craftum.io",
    # Platforms seen with hundreds of tenants in phishing feeds
    "ewp.live", "codeanyapp.com", "webwave.dev", "easywp.com", "fleek.co", "brizy.site",
    "start.page", "ukit.me", "w3spaces.com", "ubpages.com", "umbler.net", "bolt.host",
    "preview-domain.com", "page.gd", "builderallwppro.com", "freewebhostmost.com",
    "wcomhost.com", "io.vn", "co.vu", "pages.net.br", "line.pm", "bero-webspace.de",
    # Dynamic DNS
    "duckdns.org", "ddns.net", "no-ip.org", "no-ip.info",
})

# Public URL shorteners
SHORTENERS: frozenset[str] = frozenset({
    "bit.ly", "bit.do", "tinyurl.com", "is.gd", "v.gd", "ow.ly", "t.co",
    "goo.gl", "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy",
    "tiny.cc", "s.id", "adf.ly", "lnkd.in", "qr.ae", "tr.im", "u.to",
    "j.mp", "po.st", "bc.vc", "x.co", "shorten.tv", "lnk.ink", "ig.do",
    "1url.at",
})


# Hosts on OFFICIAL brand domains where anyone can publish content (Google Sites/Docs/Forms,
# cloud storage buckets, Office Forms, Adobe Express...). Being an official domain does NOT make
# a page there safe — phishing kits are routinely hosted on them.
USER_CONTENT_HOSTS: frozenset[str] = frozenset({
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "storage.cloud.google.com", "firebasestorage.googleapis.com",
    "script.google.com", "googleusercontent.com", "forms.office.com", "onedrive.live.com",
    "1drv.ms", "sharepoint.com", "express.adobe.com", "new.express.adobe.com",
    "spark.adobe.com", "indd.adobe.com", "acrobat.adobe.com", "postoffice.adobe.com",
    "dropbox.com", "dl.dropboxusercontent.com", "notion.site", "webflow.io", "linktr.ee",
})
# Registrable domains whose every subdomain is a tenant bucket (e.g. bucket.s3.amazonaws.com)
USER_CONTENT_SUFFIXES: tuple[str, ...] = (
    ".amazonaws.com", ".googleusercontent.com", ".sharepoint.com", ".blob.core.windows.net",
    ".r2.dev", ".workers.dev",
)


def is_user_content_host(host: str) -> bool:
    """True when `host` serves user-published content even though its domain is official."""
    host = host.lower().strip(".").removeprefix("www.")
    return host in USER_CONTENT_HOSTS or host.endswith(USER_CONTENT_SUFFIXES)


def is_free_hosting(registrable_domain: str) -> bool:
    d = registrable_domain.lower().strip(".")
    return d in FREE_HOSTING or d.startswith("blogspot.")  # blogspot.com, blogspot.sn, ...


def is_shortener(registrable_domain: str) -> bool:
    return registrable_domain.lower().strip(".") in SHORTENERS


ABUSED_PLATFORMS_PATH = (Path(__file__).resolve().parents[2] / "data" / "blocklist"
                         / "abused_platforms.txt")


@lru_cache(maxsize=1)
def _abused_platforms() -> frozenset[str]:
    """Platforms derived from the phishing feeds by blocklist/platforms.py (rebuilt on update)."""
    if not ABUSED_PLATFORMS_PATH.exists():
        return frozenset()
    return frozenset(ln.strip() for ln in ABUSED_PLATFORMS_PATH.read_text(encoding="utf-8").split()
                     if ln.strip())


def is_abused_platform(registrable_domain: str) -> bool:
    return registrable_domain.lower().strip(".") in _abused_platforms()


def is_untrusted_hosting(registrable_domain: str) -> bool:
    """True for free hosting platforms, URL shorteners and feed-derived abused platforms."""
    d = registrable_domain.lower().strip(".")
    return is_free_hosting(d) or d in SHORTENERS or is_abused_platform(d)
