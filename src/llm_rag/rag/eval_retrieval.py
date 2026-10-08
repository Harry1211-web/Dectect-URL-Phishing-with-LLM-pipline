"""Measure RAG tier-2 semantic retrieval quality (top-1 / top-3 accuracy).

Each query paraphrases the topic of one corpus file WITHOUT reusing its wording, so the score
reflects semantic retrieval rather than keyword overlap.
Run:    python src/llm_rag/rag/eval_retrieval.py
Output: reports/llm_rag/rag_pipeline/eval_retrieval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier2 import query_tier2  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "reports" / "llm_rag" / "rag_pipeline" / "eval_retrieval.json"

# One short paraphrased query per corpus file
QUERIES: list[dict] = [
    {"file": "001_25_fraud_scenarios_2026.md",
     "query": "công an cảnh báo những chiêu lừa đảo phổ biến nhất trong năm nay"},
    {"file": "002_fake_momo_zalopay_vnpay.md",
     "query": "tin nhắn giả mạo ví điện tử yêu cầu nhập mã xác thực"},
    {"file": "003_fake_hotels_travel.md",
     "query": "đặt phòng khách sạn online bị mất tiền cọc vì trang giả"},
    {"file": "004_deepfake_haotian_ai.md",
     "query": "phần mềm giả giọng nói khuôn mặt trong cuộc gọi video để lừa đảo"},
    {"file": "005_pirated_sites_malware.md",
     "query": "xem phim lậu miễn phí trên mạng có nguy cơ dính virus không"},
    {"file": "006_signs_of_scam_websites.md",
     "query": "làm sao để phân biệt một trang web có phải lừa đảo hay không"},
    {"file": "007_fake_ncsc_policeonline.md",
     "query": "trang web tự nhận hỗ trợ lấy lại tiền bị lừa có đáng tin không"},
    {"file": "008_fake_bank_brand_agribank.md",
     "query": "tên miền gần giống ngân hàng nông nghiệp nhưng thêm chữ vào sau"},
    {"file": "009_ai_fraud_2025_2026.md",
     "query": "xu hướng lừa đảo dùng trí tuệ nhân tạo sẽ diễn biến ra sao thời gian tới"},
    {"file": "010_back_to_school_fraud_2026.md",
     "query": "sinh viên năm nhất bị lừa tiền đặt cọc phòng trọ đầu năm học"},
    {"file": "011_fake_vneid.md",
     "query": "người tự xưng công an gọi điện yêu cầu cài ứng dụng định danh lạ"},
    {"file": "012_fake_vietcombank_crypto_investment.md",
     "query": "quảng cáo đầu tư tiền điện tử dùng hình ảnh ngân hàng thật để tạo lòng tin"},
    {"file": "013_fake_shopee_ecommerce.md",
     "query": "nhắn tin giả sàn mua sắm online yêu cầu nhập thông tin thẻ để nhận hoàn tiền"},
    {"file": "014_fake_tax_social_insurance.md",
     "query": "người gọi điện tự xưng cán bộ thuế yêu cầu gửi ảnh căn cước công dân"},
    {"file": "015_fake_telecom_carrier.md",
     "query": "nhân viên tổng đài điện thoại đòi mã xác thực để gia hạn gói cước"},
    {"file": "016_fake_vingroup.md",
     "query": "trang web mạo danh tập đoàn bất động sản ô tô lớn nhất Việt Nam mời đầu tư"},
    {"file": "017_fake_fpt.md",
     "query": "tuyển dụng giả danh công ty viễn thông công nghệ yêu cầu đóng tiền trước"},
    {"file": "018_fake_vng.md",
     "query": "website nạp thẻ game online giá rẻ bị mất tiền mà không nhận được xu"},
    {"file": "019_fake_agency_business_sites.md",
     "query": "trang web sao chép giao diện cổng thông tin của cơ quan nhà nước"},
    {"file": "020_counterfeit_goods_ecommerce.md",
     "query": "mua đồ hiệu online giá rẻ bất thường nhận về sản phẩm kém chất lượng"},
    {"file": "021_fake_finance_company.md",
     "query": "ứng dụng cho vay tiền nhanh đòi phí giải ngân trước khi nhận khoản vay"},
]

# Long queries simulating the real retrieval input (URL + title + page snippet)
LONG_QUERIES: list[dict] = [
    {"file": "001_25_fraud_scenarios_2026.md",
     "query": "Cổng thông tin an ninh mạng - Tổng hợp cảnh báo lừa đảo. Trang tổng hợp danh sách "
                "các chiêu trò lừa đảo trực tuyến đang phổ biến hiện nay, từ giả mạo cơ quan nhà "
                "nước, deepfake, dụ dỗ đầu tư đến mua bán hàng online, kèm khuyến nghị phòng tránh "
                "cho người dân."},
    {"file": "002_fake_momo_zalopay_vnpay.md",
     "query": "Ví điện tử của bạn có tin nhắn lạ - Kiểm tra ngay. Gần đây nhiều người dùng ví điện "
                "tử phản ánh nhận được tin nhắn/cuộc gọi tự xưng nhân viên yêu cầu cung cấp mã xác "
                "thực để xử lý sự cố tài khoản hoặc nhận thưởng."},
    {"file": "003_fake_hotels_travel.md",
     "query": "Đặt phòng nghỉ dưỡng giá rẻ mùa cao điểm. Trang quảng cáo phòng khách sạn giá ưu "
                "đãi, yêu cầu khách chuyển khoản đặt cọc trước để giữ chỗ, không qua nền tảng đặt "
                "phòng trung gian nào."},
    {"file": "004_deepfake_haotian_ai.md",
     "query": "Cuộc gọi video từ người quen nhờ chuyển tiền gấp. Nạn nhân nhận cuộc gọi video call "
                "thấy đúng khuôn mặt và giọng nói người quen nhưng thực chất là công nghệ giả mạo "
                "hình ảnh/âm thanh thời gian thực."},
    {"file": "005_pirated_sites_malware.md",
     "query": "Xem phim bộ mới nhất, chất lượng cao, hoàn toàn miễn phí. Trang chia sẻ phim lậu "
                "không bản quyền, yêu cầu tắt trình chặn quảng cáo và cài thêm tiện ích mở rộng để "
                "xem được video."},
    {"file": "006_signs_of_scam_websites.md",
     "query": "Hướng dẫn an toàn khi mua sắm trực tuyến. Bài viết tổng hợp các dấu hiệu chung để "
                "nhận biết một trang web có đáng ngờ hay không trước khi nhập thông tin cá nhân/"
                "thanh toán."},
    {"file": "007_fake_ncsc_policeonline.md",
     "query": "Dịch vụ hỗ trợ lấy lại tiền đã chuyển nhầm cho lừa đảo. Trang tự giới thiệu là kênh "
                "hỗ trợ chính thức của cơ quan an ninh mạng, nhận xử lý hồ sơ lấy lại tiền bị lừa "
                "đảo qua mạng cho nạn nhân."},
    {"file": "008_fake_bank_brand_agribank.md",
     "query": "Ngân hàng Nông nghiệp - Cổng đăng nhập Internet Banking. Địa chỉ trang gần giống "
                "tên miền chính thức của ngân hàng nhưng có thêm vài ký tự lạ phía sau, yêu cầu "
                "đăng nhập lại tài khoản."},
    {"file": "009_ai_fraud_2025_2026.md",
     "query": "Báo cáo xu hướng tội phạm mạng năm tới. Phân tích cách tội phạm mạng ứng dụng công "
                "cụ trí tuệ nhân tạo để tự động hoá và tăng quy mô các chiến dịch lừa đảo trong thời "
                "gian tới."},
    {"file": "010_back_to_school_fraud_2026.md",
     "query": "Tân sinh viên tìm phòng trọ gần trường. Nhiều tân sinh viên chuyển khoản đặt cọc "
                "phòng trọ qua tin đăng trên mạng xã hội đầu năm học rồi mất liên lạc với người cho "
                "thuê."},
    {"file": "011_fake_vneid.md",
     "query": "Cán bộ công an yêu cầu cập nhật ứng dụng định danh điện tử. Người gọi điện tự xưng "
                "công an, yêu cầu người dân cài đặt một ứng dụng lạ để \"đồng bộ\" dữ liệu định danh "
                "cá nhân."},
    {"file": "012_fake_vietcombank_crypto_investment.md",
     "query": "Cơ hội đầu tư sinh lời cao được ngân hàng bảo trợ. Quảng cáo mời gọi đầu tư tài "
                "chính/tiền điện tử, sử dụng logo và hình ảnh của ngân hàng lớn để tạo lòng tin cho "
                "nhà đầu tư."},
    {"file": "013_fake_shopee_ecommerce.md",
     "query": "Đơn hàng của bạn đang chờ xác nhận thanh toán. Tin nhắn tự xưng từ sàn thương mại "
                "điện tử, thông báo đơn hàng gặp sự cố hoặc được hoàn tiền, kèm liên kết yêu cầu "
                "nhập thông tin thẻ ngân hàng."},
    {"file": "014_fake_tax_social_insurance.md",
     "query": "Thông báo từ cơ quan quản lý thuế/bảo hiểm về hồ sơ của bạn. Người gọi điện tự "
                "xưng cán bộ nhà nước, yêu cầu cung cấp thông tin cá nhân, hình ảnh giấy tờ tuỳ thân "
                "hoặc cài ứng dụng kê khai lạ."},
    {"file": "015_fake_telecom_carrier.md",
     "query": "Thuê bao của bạn đủ điều kiện nâng cấp gói cước miễn phí. Cuộc gọi tự xưng tổng đài "
                "chăm sóc khách hàng nhà mạng, đề nghị hỗ trợ gia hạn/nâng cấp dịch vụ và yêu cầu "
                "đọc mã xác thực gửi qua tin nhắn."},
    {"file": "016_fake_vingroup.md",
     "query": "Tập đoàn Vingroup International - Cơ hội đầu tư cổ phần ưu đãi. Trang tự nhận là "
                "chi nhánh quốc tế của tập đoàn, mời góp vốn vào dự án bất động sản và xe điện với "
                "lợi nhuận cam kết hàng tháng."},
    {"file": "017_fake_fpt.md",
     "query": "FPT Telecom tuyển cộng tác viên làm việc tại nhà. Email mời phỏng vấn trực tuyến, "
                "yêu cầu ứng viên cài ứng dụng riêng và nộp phí đào tạo, đồng phục trước khi nhận "
                "việc."},
    {"file": "018_fake_vng.md",
     "query": "Nạp thẻ game chiết khấu cao - Zing Card. Trang nạp tiền vào game với giá rẻ hơn "
                "trang chính thức, yêu cầu đăng nhập tài khoản game và chuyển khoản trước."},
    {"file": "019_fake_agency_business_sites.md",
     "query": "Cổng thông tin điện tử - Tra cứu hồ sơ trực tuyến. Trang có giao diện giống hệt "
                "website chính thức của cơ quan, được gửi qua tin nhắn kèm yêu cầu khai báo thông "
                "tin cá nhân."},
    {"file": "020_counterfeit_goods_ecommerce.md",
     "query": "Thanh lý hàng hiệu chính hãng giảm 80%. Cửa hàng online đăng ảnh sản phẩm đẹp, "
                "nhiều đánh giá năm sao, chỉ nhận chuyển khoản trước và giục khách đặt nhanh vì sắp "
                "hết hàng."},
    {"file": "021_fake_finance_company.md",
     "query": "Vay tiêu dùng không cần thế chấp, giải ngân trong 30 phút. Ứng dụng tự nhận là "
                "công ty tài chính uy tín, yêu cầu nộp phí bảo hiểm khoản vay trước khi giải ngân."},
]


def score(items: list[dict]) -> dict:
    details = []
    top1 = top3 = 0
    for it in items:
        # max_distance=None: raw ranking quality, independent of the relevance filter
        # (the filter itself is measured by threshold_relevance.py)
        hits = query_tier2(it["query"], k=3, max_distance=None)
        files = [h["file"] for h in hits]
        ok1 = bool(files) and files[0] == it["file"]
        ok3 = it["file"] in files
        top1 += ok1
        top3 += ok3
        details.append({"query": it["query"], "expected_file": it["file"], "top1_ok": ok1,
                        "top3_ok": ok3, "returned_files": files,
                        "distances": [h["distance"] for h in hits]})
        print(f"  top1={'OK' if ok1 else 'MISS':4s} top3={'OK' if ok3 else 'MISS':4s}  "
              f"{it['query'][:55]}", file=sys.stderr)
    n = len(items)
    return {"n_queries": n, "top1_accuracy": round(top1 / n, 4),
            "top3_accuracy": round(top3 / n, 4), "details": details}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    print("=== Short queries ===", file=sys.stderr)
    short = score(QUERIES)
    print("=== Long queries (url + title + snippet) ===", file=sys.stderr)
    long_ = score(LONG_QUERIES)
    summary = {"short_queries": short, "long_queries": long_,
               "note": f"One paraphrased query per corpus file, N={len(QUERIES)}."}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "details"}
                      for k, v in summary.items() if k != "note"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
