"""Calibrate the RAG tier-2 relevance threshold per language (Vietnamese vs English).

Checks whether one global cosine-distance threshold works for both languages or whether a
language-conditional threshold is needed (tier2.py uses the latter).
Run:    python src/llm_rag/rag/threshold_relevance_en.py
Output: reports/llm_rag/rag_pipeline/threshold_relevance_en.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import LONG_QUERIES, QUERIES  # noqa: E402
from threshold_relevance import UNRELATED_QUERIES, top1_distances, youden_j  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "reports" / "llm_rag" / "rag_pipeline" / "threshold_relevance_en.json"

# English paraphrases of the same corpus topics, written like real international phishing pages
EN_RELATED_QUERIES: list[dict] = [
    {"file": "001_25_fraud_scenarios_2026.md",
     "query": "Police cyber security warning about the most common online scam tricks this year"},
    {"file": "002_fake_momo_zalopay_vnpay.md",
     "query": "Your e-wallet account has a security alert, verify your OTP code now to avoid suspension"},
    {"file": "003_fake_hotels_travel.md",
     "query": "Book your hotel room now, pay a deposit to secure your reservation for this trip"},
    {"file": "004_deepfake_haotian_ai.md",
     "query": "Video call from a familiar face asking to transfer money urgently, real-time deepfake scam"},
    {"file": "005_pirated_sites_malware.md",
     "query": "Watch the latest movies online for free, disable your ad blocker to continue streaming"},
    {"file": "006_signs_of_scam_websites.md",
     "query": "How to tell if a website is fake or fraudulent before entering personal information"},
    {"file": "007_fake_ncsc_policeonline.md",
     "query": "Official recovery service claims it can help get your scammed money back, is this trustworthy"},
    {"file": "008_fake_bank_brand_agribank.md",
     "query": "Bank login page domain looks almost identical to the real one but with extra letters added"},
    {"file": "009_ai_fraud_2025_2026.md",
     "query": "How AI powered scams and chatbot fraud are expected to evolve in the near future"},
    {"file": "010_back_to_school_fraud_2026.md",
     "query": "New student loses deposit money for a rental room found through a social media listing"},
    {"file": "011_fake_vneid.md",
     "query": "Someone claiming to be a government official asks you to install an unknown identity app"},
    {"file": "012_fake_vietcombank_crypto_investment.md",
     "query": "High return crypto investment opportunity endorsed by a major bank logo to build trust"},
    {"file": "013_fake_shopee_ecommerce.md",
     "query": "Your order is pending payment confirmation, enter your card details to receive a refund"},
    {"file": "014_fake_tax_social_insurance.md",
     "query": "Caller claims to be a tax or social insurance officer asking for your ID card photo"},
    {"file": "015_fake_telecom_carrier.md",
     "query": "Telecom customer service call asks you to read out the verification code to extend your plan"},
]

# The same everyday unrelated topics as threshold_relevance.py, in English
EN_UNRELATED_QUERIES: list[str] = [
    "traditional northern-style beef pho recipe",
    "last night's world cup qualifier football results",
    "how to install python on windows 11",
    "will it rain in Hanoi next week, weather forecast",
    "how to take care of indoor plants in winter",
    "review of the newest smartphone released this month",
    "CGV cinema showtimes this weekend",
    "yoga exercises for lower back pain at home",
    "top 10 most beautiful travel spots in Da Lat",
    "how to properly boil chicken for Lunar New Year offerings",
    "how to grow vegetables on your rooftop",
    "this week's professional basketball league schedule",
    "how to choose a laptop for new university students",
    "coffee shops with a nice view in district 1",
    "gym workout guide to build muscle for beginners",
]


def _group(pos: list[float], neg: list[float]) -> dict:
    return {"n_related": len(pos), "n_unrelated": len(neg),
            "related_distance": {"min": round(min(pos), 4), "max": round(max(pos), 4)},
            "unrelated_distance": {"min": round(min(neg), 4), "max": round(max(neg), 4)},
            "youden_threshold": youden_j(pos, neg)}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    vn_rel_q = [m["query"] for m in QUERIES] + [m["query"] for m in LONG_QUERIES]
    en_rel_q = [m["query"] for m in EN_RELATED_QUERIES]
    vn_pos, vn_neg = top1_distances(vn_rel_q), top1_distances(UNRELATED_QUERIES)
    en_pos, en_neg = top1_distances(en_rel_q), top1_distances(EN_UNRELATED_QUERIES)
    summary = {
        "vietnamese": _group(vn_pos, vn_neg),
        "english": _group(en_pos, en_neg),
        "combined": {"youden_threshold": youden_j(vn_pos + en_pos, vn_neg + en_neg),
                     "note": "One global threshold for both languages."},
        "note": ("Small N, starting thresholds only. If each language separates well but the "
                 "combined threshold is clearly worse, a language-conditional threshold is needed."),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({**summary, "details": {
        "vn_related": [{"query": q, "distance": round(d, 4)} for q, d in zip(vn_rel_q, vn_pos)],
        "vn_unrelated": [{"query": q, "distance": round(d, 4)}
                         for q, d in zip(UNRELATED_QUERIES, vn_neg)],
        "en_related": [{"query": q, "distance": round(d, 4)} for q, d in zip(en_rel_q, en_pos)],
        "en_unrelated": [{"query": q, "distance": round(d, 4)}
                         for q, d in zip(EN_UNRELATED_QUERIES, en_neg)],
    }}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
