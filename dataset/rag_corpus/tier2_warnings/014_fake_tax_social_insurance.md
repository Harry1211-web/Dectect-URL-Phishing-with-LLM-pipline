---
id: 014_fake_tax_social_insurance
title: Xuất hiện tình trạng mạo danh ngành thuế, bảo hiểm xã hội để lừa đảo
publisher: Khanh Hoa Provincial Social Insurance, Khanh Hoa Provincial Tax Department (via Vietnam+/VNA)
source_url: https://www.vietnamplus.vn/khanh-hoa-xuat-hien-tinh-trang-mao-danh-nganh-thue-bao-hiem-xa-hoi-de-lua-dao-post956514.vnp
published_date: 2024-05-31
verified_date: 2026-09-21
verified_by: Teammate B (read directly via WebFetch)
topics: [impersonating_authority, tax, social_insurance, fake_app, citizen_id]
related_vn_brands: []
---

# Cảnh báo mạo danh cơ quan Thuế và Bảo hiểm Xã hội

Bảo hiểm Xã hội tỉnh Khánh Hòa và Cục Thuế tỉnh Khánh Hòa (đăng qua Vietnam+/TTXVN) cảnh báo thủ
đoạn mạo danh cán bộ thuế và bảo hiểm xã hội để lừa đảo chiếm đoạt tài sản.

## Thủ đoạn mạo danh Bảo hiểm xã hội

Kẻ lừa đảo gọi điện yêu cầu cung cấp "thông tin cá nhân để đồng bộ dữ liệu, cài đặt VssID-BHXH" và
hiệu chỉnh sổ bảo hiểm. Một số trường hợp đe doạ nếu không cung cấp thông tin thì "sổ bảo hiểm xã
hội, thẻ bảo hiểm y tế không còn giá trị sử dụng".

## Thủ đoạn mạo danh ngành Thuế

- Giả mạo văn bản mời tham dự hội nghị, tập huấn.
- Gửi đường link kèm hướng dẫn cài đặt ứng dụng giả mạo (đóng vai ứng dụng thuế điện tử/hoá đơn
  điện tử).
- Tự xưng cán bộ thuế, yêu cầu cung cấp thông tin cá nhân, gửi hình ảnh căn cước công dân.
- Gọi điện hăm doạ (truy thu thuế, phạt) để gây áp lực chiếm đoạt tài sản.

## Khuyến nghị phòng tránh

- Không kết bạn Zalo hoặc bấm vào đường link lạ tự xưng cơ quan thuế/bảo hiểm.
- Không cung cấp mã OTP, mật khẩu, hoặc dữ liệu cá nhân (căn cước công dân...) cho người lạ qua
  điện thoại/tin nhắn.
- Để điều chỉnh thông tin thật, đến trực tiếp cơ quan bảo hiểm/thuế hoặc dùng cổng dịch vụ công
  chính thức, không qua ứng dụng/link do người gọi điện gửi.
- Lưu giữ bằng chứng (tin nhắn, ghi âm) và báo cáo công an, cơ quan thuế/bảo hiểm khi phát hiện.

## Ghi chú liên quan hệ thống

Nhóm chủ đề "giả mạo cơ quan thuế/bảo hiểm xã hội" trước đây chưa có tài liệu Tầng 2 tương ứng dù
đã xuất hiện trong kịch bản red-team (`src/LLM + RAG/redteam_prompt_injection.py::KICH_BAN_GOC`, id
`thue_hoanthue`). Không có brand VN cụ thể trong Typosquatting Levenshtein (cơ quan nhà nước, không phải doanh
nghiệp cung cấp dịch vụ tài chính/viễn thông) — tài liệu này bổ sung ngữ cảnh cho mô hình LLM khi phân
tích domain giả mạo kiểu "hoàn thuế"/"cập nhật sổ BHXH".
