"""Clean page HTML before it enters the LLM prompt, removing prompt-injection carriers.

Four defence layers:
  1. Drop `<script>`, `<style>` and HTML comments.
  2. Drop elements hidden by attribute or CSS (`display:none`, `visibility:hidden`, `opacity:0`,
     `font-size:0`, `hidden`, `aria-hidden="true"`).
  3. Drop text whose colour equals its background colour.
  4. Strip zero-width / invisible characters.

All text (visible and removed) is also scanned for fake instructions (multilingual patterns loaded
from data/injection_patterns/); when found, the sentences are removed and a flag is raised so the
prompt can warn the model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, CData, Comment, NavigableString, Tag

_VISIBLE, _HIDDEN, _DROPPED = 0, 1, 2

# Zero-width / invisible characters used to split or hide text: zero-width and joiner characters,
# soft hyphen, bidi controls, invisible operators, variation selectors
_INVISIBLE_CHARS = ("­͏؜ᅟᅠ឴឵᠎​-‏‪-‮"
                    "⁠-⁤⁦-⁯ㅤ︀-️﻿ﾠ"
                    "\U000e0100-\U000e01ef")
_RE_INVISIBLE = re.compile(f"[{_INVISIBLE_CHARS}\U000e0000-\U000e007f]")
# Unicode tag characters (U+E0000-E007F) mirror ASCII and are invisible on screen, but LLMs read
# them ("ASCII smuggling"): they are decoded so the injection scan sees the hidden sentence
_RE_TAG_RUN = re.compile("[\U000e0020-\U000e007e]+")

# opacity/font-size count as hidden only when exactly 0 (0, 0.0, 0px...) — keeps visible
# elements such as opacity:0.5 or font-size:0.8em. Also: transparent text, element moved far
# off screen (left/top/text-indent of -999px or more), clipped to nothing, scaled to 0
_RE_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|"
    r"opacity\s*:\s*0*(?:\.0+)?\s*(?:!important)?\s*(?:;|$)|"
    r"font-size\s*:\s*0+(?:\.0+)?(?:px|em|rem|pt|%)?\s*(?:!important)?\s*(?:;|$)|"
    r"(?<![-\w])color\s*:\s*transparent|"
    r"(?<![-\w])(?:left|top|right|text-indent|margin-left|margin-top)\s*:\s*-\d{3,}|"
    r"clip\s*:\s*rect\(\s*0[^)]*\)|clip-path\s*:\s*inset\(\s*(?:50|100)%|"
    r"transform\s*:\s*scale\(\s*0(?:\.0+)?\s*[,)]",
    re.I,
)


def _smuggled_text(text: str) -> list[str]:
    """ASCII hidden as Unicode tag characters, decoded (flag emoji use only a few, harmlessly)."""
    return ["".join(chr(ord(c) - 0xE0000) for c in m) for m in _RE_TAG_RUN.findall(text)]
_RE_COLOR = re.compile(r"(?<![-\w])color\s*:\s*([^;]+)", re.I)
_RE_BG = re.compile(r"background(?:-color)?\s*:\s*([^;]+)", re.I)

# Language data, one file per language (data/injection_patterns/<lang>.json). No language-specific
# pattern lives in the code: a new language is a new file. Keys:
#   fake_instruction / direct_command / narrative / trusted_source : regexes (sanitizer)
#   gov_source_keywords : names of anti-scam authorities quoted on pages
#   authority_domains   : domain suffixes reserved for those authorities (explicit list: a generic
#                         "gov.<cc>" rule would trust zones anyone can register under some ccTLDs)
#   verdict_phishing_words / verdict_safe_words / negations : fallback verdict parsing (llm_client)
#   bait_words          : urgency / credential lures in page text (DOM features of Layer C)
#   language_name       : English name of the language (LLM explanation language)
PATTERNS_DIR = Path(__file__).resolve().parents[2] / "data" / "injection_patterns"
_PATTERN_KEYS = ("fake_instruction", "direct_command", "narrative", "trusted_source",
                 "gov_source_keywords", "authority_domains", "verdict_phishing_words",
                 "verdict_safe_words", "negations", "bait_words")
_META_KEYS = ("language_name",)


class PatternLoader:
    """Loads and merges every language file; every page is checked against every language."""

    def __init__(self, directory: Path = PATTERNS_DIR):
        self.directory = directory
        self.languages: list[str] = []
        self.patterns: dict[str, list[str]] = {k: [] for k in _PATTERN_KEYS}
        self.language_names: dict[str, str] = {}
        for f in sorted(directory.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            unknown = set(data) - set(_PATTERN_KEYS) - set(_META_KEYS)
            if unknown:
                raise ValueError(f"{f.name}: unknown keys {sorted(unknown)}")
            self.languages.append(f.stem)
            self.language_names[f.stem] = data.get("language_name", f.stem)
            for k in _PATTERN_KEYS:
                self.patterns[k].extend(x for x in data.get(k, []) if x not in self.patterns[k])
        if not self.patterns["fake_instruction"]:
            raise RuntimeError(f"no injection patterns found in {directory}")

    def __getitem__(self, key: str) -> list[str]:
        return self.patterns[key]

    def regex(self, key: str) -> re.Pattern:
        """One case-insensitive alternation of all languages' patterns for `key`."""
        return _compile(self.patterns[key])


def _load_patterns(directory: Path = PATTERNS_DIR) -> dict[str, list[str]]:
    return PatternLoader(directory).patterns


def _compile(patterns: list[str]) -> re.Pattern:
    # An empty group must match nothing (an empty alternation would match everything)
    return re.compile("|".join(f"(?:{p})" for p in patterns) or r"(?!x)x", re.I)


PATTERNS = PatternLoader()
_PATTERNS = PATTERNS.patterns
# Sentences that try to instruct the analysing model
_RE_FAKE_INSTRUCTION = _compile(_PATTERNS["fake_instruction"])
# Narrative/quoting phrases of legitimate warning articles (exempt unless a direct command)
_RE_NARRATIVE = _compile(_PATTERNS["narrative"])
# Direct commands aimed at the model — never exempt, even inside narrative text
# (blocks bypasses such as "Note: ... ignore all instructions")
_RE_DIRECT_COMMAND = _compile(_PATTERNS["direct_command"])
# "... by <authority>" phrases that quote an anti-scam authority
_RE_TRUSTED_SOURCE = _compile(_PATTERNS["trusted_source"])
_GOV_SOURCE_KEYWORDS = tuple(k.lower() for k in _PATTERNS["gov_source_keywords"])

# A quoted authority is trusted only when the page really is on an authority's domain
_AUTHORITY_SUFFIXES = tuple(_PATTERNS["authority_domains"])


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
    host = host.rstrip(".").lower()
    authority_host = any(host == d or host.endswith("." + d) for d in _AUTHORITY_SUFFIXES)
    s = sentence.lower()
    return authority_host and any(k in s for k in _GOV_SOURCE_KEYWORDS)


# Cyrillic / Greek letters that look Latin ("ignоre" with a Cyrillic о). Only these: folding
# "đ" or Cyrillic words would break the Vietnamese and Russian patterns, so the original sentence
# is checked as well
_LOOKALIKES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԝ": "w", "ν": "v", "ο": "o", "α": "a", "ε": "e",
    "ι": "i", "κ": "k", "τ": "t",
})


def is_fake_instruction(sentence: str, host: str = "") -> bool:
    """Decide whether a sentence is an injected instruction that must be removed."""
    folded = sentence.translate(_LOOKALIKES)
    return _is_fake(sentence, host) or (folded != sentence and _is_fake(folded, host))


def _is_fake(sentence: str, host: str) -> bool:
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

    removed_text: list[str] = _smuggled_text(html or "")
    # One pass over the tree, nothing is removed from it: removing nodes one by one costs a
    # search in the parent each time, so 100k hidden elements or comments took ~20 s (DoS).
    # State of a tag: visible, hidden (text kept for the injection scan) or dropped
    # (script / style: not page text). Descendants come after their parent.
    state: dict[int, int] = {id(soup): _VISIBLE}
    visible_parts: list[str] = []
    n_hidden = 0
    for el in soup.descendants:
        parent = state[id(el.parent)]
        if isinstance(el, Tag):
            if el.name in ("script", "style"):
                s = _DROPPED
            elif parent != _VISIBLE:
                s = parent
            elif _is_hidden(el):
                s, n_hidden = _HIDDEN, n_hidden + 1
            else:
                s = _VISIBLE
            state[id(el)] = s
        elif isinstance(el, Comment):
            if parent != _DROPPED:
                removed_text.append(str(el))
        elif type(el) in (NavigableString, CData) and parent != _DROPPED:
            (visible_parts if parent == _VISIBLE else removed_text).append(str(el))

    visible = _RE_INVISIBLE.sub("", " ".join(visible_parts))
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
