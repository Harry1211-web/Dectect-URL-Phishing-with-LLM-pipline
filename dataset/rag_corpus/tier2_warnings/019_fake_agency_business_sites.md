---
id: 019_fake_agency_business_sites
title: Nhận biết chiêu trò giả mạo trang thông tin điện tử của cơ quan, doanh nghiệp để lừa đảo
publisher: Chongluadao.vn
source_url: https://chongluadao.vn/blog/en/2023/07/nhan-biet-chieu-tro-gia-mao-trang-thong-tin-dien-tu-cua-co-quan-doanh-nghiep-de-lua-dao/
published_date: 2023-07-11
verified_date: 2026-09-23
verified_by: Bạn B (đọc trực tiếp qua WebFetch)
topics: [overview, impersonating_authority, fake_website, url_signs, technical_signs]
related_vn_brands: []
---

# Nhận biết chiêu trò giả mạo trang thông tin điện tử cơ quan/doanh nghiệp

Chongluadao.vn ghi nhận 531 vụ lừa đảo dùng thủ đoạn này tính đến tháng 6/2023.

## Thủ đoạn lừa đảo

- Tạo website có giao diện giống hệt trang chính thức của cơ quan, doanh nghiệp.
- Gửi tin nhắn giả mạo yêu cầu người dùng truy cập liên kết giả.
- Yêu cầu khai báo thông tin cá nhân, tài khoản ngân hàng để đánh cắp dữ liệu.
- Mạo danh cán bộ nhà nước, gửi liên kết cài đặt ứng dụng giả để kiểm soát thiết bị.

## Dấu hiệu cảnh báo kỹ thuật (KHÔNG gắn brand cụ thể — áp dụng chung)

- **Tên miền**: đuôi miền bất thường (ví dụ `vn-cbs.xyz`, `vn-ms.top`), chứa ký tự lạ hoặc lỗi
  chính tả so với tên cơ quan/doanh nghiệp thật.
- **Chứng chỉ SSL**: thiếu biểu tượng khóa hoặc không có `https://` ở đầu URL.
- **Thiết kế**: giao diện không chuyên nghiệp, hoặc là bản sao chép lỗi từ trang chính thức.
- **Nội dung**: lỗi chính tả, ngôn ngữ không chuẩn mực.
- **Yêu cầu**: đòi cung cấp tài khoản ngân hàng, số thẻ tín dụng, mật khẩu.

## Khuyến nghị phòng tránh

- Kiểm tra kỹ URL trước khi cung cấp bất kỳ thông tin nào.
- Dùng trình duyệt an toàn (Chrome, Firefox, Safari).
- Tránh nhấp liên kết trong email/tin nhắn lạ.
- Chỉ cung cấp thông tin nhạy cảm trên trang đã xác minh đáng tin cậy.
- Dùng xác thực 2 yếu tố.
- Gọi điện xác minh trực tiếp với công ty/ngân hàng qua số trong danh bạ chính thức (không dùng số
  ghi trên trang nghi ngờ).

## Ghi chú liên quan hệ thống

Tài liệu TỔNG QUÁT, không gắn 1 brand cụ thể — bổ sung cho nhóm case giả mạo brand CHƯA có trong
`vn_brand_domains.csv`/brand quốc tế, hoặc case tên miền lạ mạo danh cơ quan nhà nước nói chung
(không phải NCSC/policeonline đã có ở tài liệu 007). Giúp mô hình LLM có căn cứ giải thích dấu hiệu kỹ
thuật (URL/SSL/thiết kế/nội dung) ngay cả khi không tra được brand cụ thể trong Tầng 1.
