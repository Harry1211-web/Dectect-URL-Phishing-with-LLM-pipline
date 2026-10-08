---
id: 015_fake_telecom_carrier
title: Cảnh báo thủ đoạn giả danh nhân viên Viettel lừa chiếm đoạt mã OTP
publisher: Công an tỉnh Tây Ninh - Phòng An ninh mạng và phòng chống tội phạm sử dụng công nghệ cao
source_url: https://congan.tayninh.gov.vn/vi/chi-tiet-tin-tuc?Id=255836
published_date: 2025-07-30
verified_date: 2026-09-21
verified_by: Bạn B (đọc trực tiếp qua WebFetch)
topics: [brand_impersonation, telecom, otp, fake_app]
related_vn_brands: [viettel]
---

# Cảnh báo giả danh nhân viên Viettel chiếm đoạt mã OTP

Công an tỉnh Tây Ninh (Phòng An ninh mạng và phòng chống tội phạm sử dụng công nghệ cao) cảnh báo
thủ đoạn giả danh nhân viên chăm sóc khách hàng Viettel để chiếm đoạt tài sản.

## Thủ đoạn lừa đảo

Kẻ lừa đảo giả danh nhân viên chăm sóc khách hàng Viettel, dùng số điện thoại giả để liên hệ nạn
nhân trên toàn quốc, với các chiêu trò:

- Đề nghị gia hạn gói cước hoặc tặng dung lượng data miễn phí.
- Cảnh báo giả về cước phí còn nợ, nguy cơ khoá SIM.
- Yêu cầu cung cấp mã OTP để "hoàn tất nâng cấp".
- Hướng dẫn nạn nhân thao tác trên ứng dụng My Viettel (hoặc ứng dụng giả mạo tương tự).

Mục tiêu cuối cùng: chiếm quyền truy cập tài khoản ngân hàng/ví điện tử sau khi lấy được mã OTP.

## Khuyến nghị phòng tránh

- Không cung cấp mã OTP qua điện thoại/tin nhắn cho bất kỳ ai, kể cả người tự xưng nhân viên nhà
  mạng, khi chưa xác minh được danh tính người gọi.
- Chỉ thực hiện thay đổi gói cước/dịch vụ qua kênh chính thức: cửa hàng Viettel, ứng dụng/website
  chính thức, hoặc tổng đài đã lưu sẵn.
- Báo ngay cho công an địa phương khi gặp trường hợp nghi ngờ.
- Thay đổi dịch vụ hợp pháp luôn qua kênh đã xác minh, không qua cuộc gọi/tin nhắn không mời mà đến
  yêu cầu thông tin nhạy cảm.

## Ghi chú liên quan hệ thống

`viettel.vn`/`vietteltelecom.vn` đã có trong danh sách brand VN của Typosquatting Levenshtein
(`dataset/brands/vn_brand_domains.csv`, nhóm `vien_thong`). Nhóm chủ đề "giả mạo nhà mạng viễn
thông" trước đây chưa có tài liệu Tầng 2 tương ứng dù đã xuất hiện trong kịch bản red-team
(`src/LLM + RAG/redteam_prompt_injection.py::KICH_BAN_GOC`, id `vnpt_khuyenmai`) — tài liệu này bổ sung ngữ
cảnh cụ thể (giả danh nhân viên qua điện thoại, không chỉ URL giả) cho mô hình LLM khi giải thích verdict.
