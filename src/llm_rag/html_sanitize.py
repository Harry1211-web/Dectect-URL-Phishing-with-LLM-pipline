"""Clean page HTML before it enters the LLM prompt, removing prompt-injection carriers.

Four defence layers:
  1. Drop `<script>`, `<style>` and HTML comments.
  2. Drop elements hidden by attribute or CSS (`display:none`, `visibility:hidden`, `opacity:0`,
     `font-size:0`, `hidden`, `aria-hidden="true"`).
  3. Drop text whose colour equals its background colour.
  4. Strip zero-width / invisible characters.

All text (visible and removed) is also scanned for fake instructions; when found, the sentences
are removed and a flag is raised so the prompt can warn the model.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

# Zero-width / invisible characters commonly used to split or hide text
_INVISIBLE_CHARS = "​‌‍⁠﻿"
_RE_INVISIBLE = re.compile(f"[{_INVISIBLE_CHARS}]")

# opacity/font-size count as hidden only when exactly 0 (0, 0.0, 0px...) — keeps visible
# elements such as opacity:0.5 or font-size:0.8em
_RE_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|"
    r"opacity\s*:\s*0*(?:\.0+)?\s*(?:!important)?\s*(?:;|$)|"
    r"font-size\s*:\s*0+(?:\.0+)?(?:px|em|rem|pt|%)?\s*(?:!important)?\s*(?:;|$)",
    re.I,
)
_RE_COLOR = re.compile(r"(?<![-\w])color\s*:\s*([^;]+)", re.I)
_RE_BG = re.compile(r"background(?:-color)?\s*:\s*([^;]+)", re.I)

# Sentences that try to instruct the analysing model (Vietnamese + English)
_RE_FAKE_INSTRUCTION = re.compile(
    r"bỏ qua (?:mọi|các|hết)?\s*(?:hướng dẫn|chỉ thị)|"
    r"system\s*override|ignore\s*(?:all|previous|the\s*above)\s*instructions?|"
    r"đã được xác minh an toàn|verified.{0,20}safe|"
    r"(?:bạn phải|hãy)\s*(?:trả lời|xác nhận|kết luận).{0,30}(?:an toàn|safe)|"
    r"you must (?:respond|answer|reply|confirm|conclude).{0,40}"
    r"(?:safe|verified|legitimate|no (?:danger|risk|threat)|not (?:a )?(?:threat|risk))",
    re.I,
)

# Narrative/quoting phrases of legitimate warning articles (exempt unless a direct command)
_RE_NARRATIVE = re.compile(
    r"kẻ (?:gian|lừa đảo|xấu)|đối tượng lừa đảo|hacker|tin tặc|"
    r"chiêu trò|thủ đoạn|dấu hiệu (?:lừa đảo|nhận biết|đáng ngờ)|"
    r"cách nhận (?:biết|diện)|nếu (?:thấy|nhận được|gặp|bạn nhận được)|"
    r"thường (?:yêu cầu|giả mạo|dùng|dụ dỗ)|cảnh giác|lưu ý\s*:?|"
    r"giải thích cách|hướng dẫn nhận diện",
    re.I,
)
# Direct commands aimed at the model — never exempt, even inside narrative text
# (blocks bypasses such as "Lưu ý: ... bỏ qua mọi hướng dẫn")
_RE_DIRECT_COMMAND = re.compile(
    r"verdict\s*[:=]|system\s*override|"
    r"bỏ qua (?:mọi|các|hết)?\s*(?:hướng dẫn|chỉ thị)|"
    r"ignore\s*(?:all|previous|the\s*above)\s*instructions?|"
    r"(?:bạn phải|hãy)\s*(?:trả lời|xác nhận|kết luận)|"
    r"you must (?:respond|answer|reply|confirm|conclude)",
    re.I,
)
_RE_TRUSTED_SOURCE = re.compile(
    r"bởi\s+(?:chống lừa đảo|ncsc|cục an toàn thông tin|bộ công an|công an|"
    r"ngân hàng nhà nước|sbv|bộ thông tin)",
    re.I,
)

# A quoted authority is trusted only when the page really is on that authority's domain
_GOV_VN_SUFFIX = ".gov.vn"
_CHONGLUADAO_DOMAIN = "chongluadao.vn"
_GOV_SOURCE_KEYWORDS = (
    "ncsc", "cục an toàn thông tin", "bộ công an", "công an",
    "ngân hàng nhà nước", "sbv", "bộ thông tin",
)


def _page_host(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "//" + url
    netloc = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return netloc.removeprefix("www.")


def _quoted_source_matches_domain(sentence: str, host: str) -> bool:
    """True when the authority quoted in the sentence owns the page's domain."""
    if not host:
        return False
    s = sentence.lower()
    if "chống lừa đảo" in s and (host == _CHONGLUADAO_DOMAIN
                                 or host.endswith("." + _CHONGLUADAO_DOMAIN)):
        return True
    return any(k in s for k in _GOV_SOURCE_KEYWORDS) and host.endswith(_GOV_VN_SUFFIX)


def is_fake_instruction(sentence: str, host: str = "") -> bool:
    """Decide whether a sentence is an injected instruction that must be removed."""
    if not _RE_FAKE_INSTRUCTION.search(sentence):
        return False
    if _RE_DIRECT_COMMAND.search(sentence):
        return True
    if _RE_NARRATIVE.search(sentence):
        return False
    if _RE_TRUSTED_SOURCE.search(sentence) and _quoted_source_matches_domain(sentence, host):
        return False
    return True


def _is_hidden(tag) -> bool:
    if tag.get("hidden") is not None:
        return True
    if (tag.get("aria-hidden") or "").strip().lower() == "true":
        return True
    style = tag.get("style") or ""
    if _RE_HIDDEN_STYLE.search(style):
        return True
    m_color, m_bg = _RE_COLOR.search(style), _RE_BG.search(style)
    return bool(m_color and m_bg
                and m_color.group(1).strip().lower() == m_bg.group(1).strip().lower())


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?\n])\s+", text)


def sanitize_html(html: str, url: str = "") -> dict:
    """Clean HTML and detect prompt injection.

    Returns {"title", "text", "hidden_elements_removed", "prompt_injection_suspected"}.
    """
    host = _page_host(url)
    soup = BeautifulSoup(html or "", "html.parser")
    title_tag = soup.find("title")
    title = _RE_INVISIBLE.sub("", title_tag.get_text(" ")) if title_tag else ""
    title = re.sub(r"\s+", " ", title).strip()

    removed_text: list[str] = []
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        removed_text.append(str(c))
        c.extract()
    for tag in soup(["script", "style"]):
        tag.decompose()

    n_hidden = 0
    for tag in soup.find_all(True):
        if tag.parent is None:  # already removed together with a hidden ancestor
            continue
        if _is_hidden(tag):
            removed_text.append(tag.get_text(" "))
            tag.decompose()
            n_hidden += 1

    visible = _RE_INVISIBLE.sub("", soup.get_text(" "))
    visible = re.sub(r"\s+", " ", visible).strip()

    everything = _RE_INVISIBLE.sub("", title + "\n" + visible + " " + " ".join(removed_text))
    injection = any(is_fake_instruction(s, host) for s in _split_sentences(everything))

    # Remove the injected sentences from the visible text and the title as well
    if injection:
        visible = " ".join(s for s in _split_sentences(visible)
                           if not is_fake_instruction(s, host)).strip()
        title = " ".join(s for s in _split_sentences(title)
                         if not is_fake_instruction(s, host)).strip()

    return {
        "title": title[:200],
        "text": visible,
        "hidden_elements_removed": n_hidden,
        "prompt_injection_suspected": injection,
    }
