"""Layer D (LLM) accuracy on Vietnamese brands.

- Legit set: live `.vn` domains from Tranco.
- Phishing set: simulated attacks on popular Vietnamese brands/services (not fetchable, so the
  prompt is built from the scenario text through the same path as production).
Run: python src/llm/vn_benchmark.py   (calls the OpenAI API)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_traffic_sources import fetch_tranco_domains  # noqa: E402
from llm_benchmark import compute_metrics, prepare_sample  # noqa: E402
from llm_client import (  # noqa: E402
    DEFAULT_MODEL, build_prompt, call_openai, check_typosquatting_with_redirect,
)
from redteam_prompt_injection import BASE_SCENARIOS  # noqa: E402

ROOT = _HERE.parents[1]
OUT_DIR = ROOT / "reports" / "llm" / "llm_benchmark"

# Extra scenarios widening brand coverage (airlines, ride-hailing, insurance, more banks...)
EXTRA_SCENARIOS: list[dict] = [
    {"id": "vietnamairlines_hoanve", "url": "https://vietnamairlines-hoanve.info/claim",
     "title": "Vietnam Airlines - Hoàn vé do hủy chuyến",
     "content": ("Chuyến bay của quý khách đã bị hủy. Vui lòng nhập số CCCD, số thẻ "
                 "ngân hàng và mã OTP để nhận hoàn tiền vé trong vòng 24 giờ, nếu "
                 "không yêu cầu sẽ bị huỷ.")},
    {"id": "vietjet_khuyenmai", "url": "https://vietjet-flash-sale.online/deal",
     "title": "Vietjet Air - Săn vé 0 đồng nhân dịp lễ",
     "content": ("Chương trình vé 0 đồng giới hạn số lượng. Đăng nhập tài khoản "
                 "ngân hàng liên kết và xác nhận mã OTP để giữ chỗ ngay, ưu đãi kết "
                 "thúc sau 15 phút.")},
    {"id": "grab_trian", "url": "https://grab-rewards-vn.site/claim",
     "title": "Grab - Tri ân tài xế và khách hàng thân thiết",
     "content": ("Nhận phần quà tri ân trị giá 5.000.000đ từ Grab. Nhập số điện thoại "
                 "và mật khẩu tài khoản Grab kèm mã OTP để nhận thưởng qua ví điện tử.")},
    {"id": "baoviet_boi_thuong", "url": "https://baoviet-claim-online.net/login",
     "title": "Bảo Việt - Chi trả quyền lợi bảo hiểm trực tuyến",
     "content": ("Hồ sơ bồi thường bảo hiểm đã được duyệt chi 12.500.000đ. Nhập tên đăng "
                 "nhập ngân hàng và mã OTP Smart để tiền về tài khoản ngay trong ngày.")},
    {"id": "agribank_nongnghiep", "url": "https://agribank-ebanking-portal.cc/ibk",
     "title": "Agribank E-Mobile Banking - Nâng cấp dịch vụ",
     "content": ("Dịch vụ Agribank E-Mobile Banking của quý khách cần nâng cấp bảo mật sinh "
                 "trắc học. Đăng nhập tài khoản và mã OTP để hoàn tất, tránh gián đoạn dịch vụ.")},
    {"id": "hdbank_vaynhanh", "url": "https://hdbank-vaytien-online.top/apply",
     "title": "HDBank - Gói vay ưu đãi lãi suất 0%",
     "content": ("Gói vay tiêu dùng giải ngân trong 30 phút. Quý khách vui lòng cung cấp "
                 "thông tin tài khoản ngân hàng và mã OTP xác thực để hoàn tất thủ tục giải ngân.")},
    {"id": "shb_tindung", "url": "https://shb-mo-the-online.xyz/register",
     "title": "SHB - Phát hành thẻ tín dụng hạn mức 50 triệu",
     "content": ("Thẻ tín dụng SHB đã được phê duyệt sơ bộ. Đăng nhập ngân hàng điện tử và "
                 "cung cấp mã OTP để xác nhận địa chỉ nhận thẻ.")},
    {"id": "vinaphone_simchinhchu", "url": "https://vinaphone-chuanhoathongtin.biz/verify",
     "title": "VinaPhone - Chuẩn hóa thông tin thuê bao",
     "content": ("Thuê bao của quý khách chưa chuẩn hóa thông tin với CSDLQG về dân cư. Quý "
                 "khách cần cung cấp ảnh CCCD 2 mặt và mã OTP ngân hàng để xác thực danh tính.")},
    {"id": "vnpay_trungthuong", "url": "https://vnpay-qr-rewards.club/spin",
     "title": "VNPAY-QR - Vòng quay may mắn trúng xe SH",
     "content": ("Chúc mừng bạn đã trúng xe máy Honda SH 125i từ chương trình VNPAY-QR. Đăng "
                 "nhập ví điện tử hoặc tài khoản ngân hàng và nhập OTP để nhận mã nhận thưởng.")},
    {"id": "seabank_canhbaotaikhoan", "url": "https://seabank-security-check.site/auth",
     "title": "SeABank - Cảnh báo giao dịch bất thường",
     "content": ("Tài khoản SeABank của quý khách vừa phát sinh giao dịch 15.000.000đ tại "
                 "Hà Nội. Nếu không phải bạn, hãy đăng nhập ngay và nhập mã OTP để hủy giao dịch.")},
    {"id": "nhnn_dautu_traiphiu", "url": "https://nganhangnhanuoc-dautu.com/bond",
     "title": "Ngân hàng Nhà nước Việt Nam - Cổng đầu tư trái phiếu chính phủ",
     "content": ("Cơ hội đầu tư trái phiếu chính phủ lãi suất 15%/năm được NHNN bảo lãnh. "
                 "Chuyển tiền vào tài khoản chỉ định để mở sổ đầu tư trực tuyến.")},
    {"id": "fpt_tuyendung_online", "url": "https://fpt-software-recruitment.online/apply",
     "title": "FPT Software - Tuyển dụng cộng tác viên nhập liệu tại nhà",
     "content": ("Việc làm nhập liệu tại nhà lương 500k-1 triệu/ngày cho FPT Software. Đặt cọc "
                 "chuyển khoản để được nhận việc ngay, không cần phỏng vấn.")},
]


def phishing_sample(sc: dict) -> dict:
    """Build the prompt for a simulated (unfetchable) phishing page like `prepare_sample` does."""
    url, title, content = sc["url"], sc["title"], sc["content"]
    typo_flag = check_typosquatting_with_redirect(url)["flag"]
    prompt = build_prompt(url, title, content, injection_suspected=False, typosquatting=typo_flag)
    return {"url": url, "label": "phishing", "prompt": prompt, "typosquatting": typo_flag,
            "origin": "vn_scenario"}


def legit_sample(domain: str) -> dict | None:
    s = prepare_sample(f"https://{domain}", "legit")
    if s is not None:
        s["origin"] = "tranco_vn"
    return s


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    scenarios = list(BASE_SCENARIOS) + EXTRA_SCENARIOS
    phishing = [phishing_sample(sc) for sc in scenarios]
    vn_domains = [d for d in fetch_tranco_domains(1_000_000, pool=1_000_000, seed=args.seed)
                  if d.endswith(".vn")]
    legit = []
    for d in vn_domains:
        if len(legit) >= len(scenarios):
            break
        s = legit_sample(d)
        if s is not None:
            legit.append(s)
            print(f"  legit: {d}", flush=True)
    print(f"{len(phishing)} phishing scenarios, {len(legit)} reachable .vn domains.", file=sys.stderr)

    rows = []
    for i, s in enumerate(phishing + legit, 1):
        res = call_openai(s["prompt"], args.model, temperature=args.temperature)
        rows.append({"url": s["url"], "label": s["label"], "origin": s["origin"],
                     "typosquatting": s.get("typosquatting"), **res})
        print(f"  [{i}/{len(phishing) + len(legit)}] {s['url'][:60]} -> {res.get('verdict')}",
              flush=True)

    summary = {
        "time": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "temperature": args.temperature,
        "n_phishing_vn": len(phishing), "n_legit_vn": len(legit),
        "metrics": compute_metrics(rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "vn_benchmark_summary.json"
    out.write_text(json.dumps({**summary, "details": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
