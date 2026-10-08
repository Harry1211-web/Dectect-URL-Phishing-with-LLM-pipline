"""Calibrate the RAG tier-2 relevance threshold (cosine distance) with Youden's J.

- Related group: paraphrased corpus queries from eval_retrieval.py.
- Unrelated group: everyday topics unrelated to scams/cyber security (normal legit pages).
Run:    python src/llm_rag/rag/threshold_relevance.py
Output: reports/llm_rag/rag_pipeline/threshold_relevance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import LONG_QUERIES, QUERIES  # noqa: E402
from tier2 import query_tier2  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "reports" / "llm_rag" / "rag_pipeline" / "threshold_relevance.json"

# Everyday queries unrelated to the corpus — tier 2 should NOT attach documents to these
UNRELATED_QUERIES = [
    "công thức nấu phở bò truyền thống miền Bắc",
    "kết quả bóng đá vòng loại world cup tối qua",
    "hướng dẫn cài đặt python trên windows 11",
    "dự báo thời tiết Hà Nội tuần tới có mưa không",
    "cách chăm sóc cây cảnh trong nhà mùa đông",
    "review điện thoại mới ra mắt tháng này",
    "lịch chiếu phim rạp CGV cuối tuần",
    "bài tập yoga giảm đau lưng tại nhà",
    "top 10 địa điểm du lịch Đà Lạt đẹp nhất",
    "cách luộc gà cúng đúng chuẩn ngày tết",
    "hướng dẫn trồng rau sạch trên sân thượng",
    "lịch thi đấu giải bóng rổ nhà nghề tuần này",
    "cách chọn mua laptop cho sinh viên mới nhập học",
    "địa chỉ quán cà phê view đẹp ở quận 1",
    "hướng dẫn tập gym tăng cơ cho người mới bắt đầu",
]


def top1_distances(queries: list[str]) -> list[float]:
    """Raw top-1 distance per query (no threshold: this script is where it is derived)."""
    out = []
    for q in queries:
        hits = query_tier2(q, k=1, max_distance=None)
        out.append(hits[0]["distance"] if hits else float("inf"))
    return out


def youden_j(pos: list[float], neg: list[float]) -> dict:
    """Try every midpoint between sorted distances; maximise J = sensitivity + specificity - 1.
    Sensitivity = related kept (distance <= T); specificity = unrelated filtered (distance > T)."""
    vals = sorted(set(pos + neg))
    if not vals:
        return {"error": "no data"}
    cands = [vals[0] - 0.01] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 0.01]
    best = None
    for t in cands:
        sens = sum(d <= t for d in pos) / len(pos)
        spec = sum(d > t for d in neg) / len(neg)
        j = sens + spec - 1
        if best is None or j > best["youden_j"]:
            best = {"threshold": round(t, 4), "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4), "youden_j": round(j, 4)}
    return best


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    related = [m["query"] for m in QUERIES] + [m["query"] for m in LONG_QUERIES]
    pos = top1_distances(related)
    neg = top1_distances(UNRELATED_QUERIES)
    summary = {
        "n_related": len(pos), "n_unrelated": len(neg),
        "related_distance": {"min": round(min(pos), 4), "max": round(max(pos), 4)},
        "unrelated_distance": {"min": round(min(neg), 4), "max": round(max(neg), 4)},
        "chosen_threshold": youden_j(pos, neg),
        "note": ("Small N: a data-backed starting threshold, not a final one. The two groups "
                 "overlap, so the threshold filters weakly (prefers keeping related content)."),
        "details": {
            "related": [{"query": q, "distance": round(d, 4)} for q, d in zip(related, pos)],
            "unrelated": [{"query": q, "distance": round(d, 4)}
                          for q, d in zip(UNRELATED_QUERIES, neg)],
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
