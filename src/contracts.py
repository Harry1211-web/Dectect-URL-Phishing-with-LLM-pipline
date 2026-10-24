"""Shared result types passed between the pipeline layers."""

from typing import Literal, NotRequired, TypedDict

# Outcome of a technical rule: True = violation, False = looks fine, "unknown" = could not check
CheckFlag = Literal[True, False, "unknown"]


class RuleResult(TypedDict):
    """Result of one technical rule (blocklist, typosquatting, ...)."""
    name: str                  # rule id, e.g. "typosquatting"
    flag: CheckFlag
    reason: str                # short human-readable explanation (shown to users)
    latency_ms: float
    brand: NotRequired[str]    # impersonated brand label, when relevant
    kind: NotRequired[str]     # sub-type of the violation, e.g. "typosquat", "path_brand_lure"
