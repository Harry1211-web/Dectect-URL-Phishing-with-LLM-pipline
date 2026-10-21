# Đề tài

## Tên đề tài

**Tiếng Việt:**
> Hệ thống phát hiện và cảnh báo website lừa đảo đa tầng ưu tiên không bỏ sót, kết hợp học sâu trên URL, đặc trưng hạ tầng tên miền và mô hình ngôn ngữ lớn chống prompt injection, đánh giá trên dữ liệu thực tế Việt Nam

**Tiếng Anh (cho bài báo):**
> A Recall-First Multi-Layer Phishing Detection System Combining Deep URL Models, Domain Infrastructure Evidence and an Injection-Resistant LLM, Evaluated on Real-World Vietnamese Data

**Tên ngắn:** Phát hiện website lừa đảo đa tầng: từ blocklist đến LLM

---

## Mô tả đề tài

Website lừa đảo chỉ tồn tại vài giờ đến vài ngày, nên các danh sách đen công khai luôn đi sau kẻ tấn
công. Ngược lại, dùng mô hình ngôn ngữ lớn (LLM) đọc mọi trang thì chậm, tốn kém và dễ bị chính trang
độc thao túng qua prompt injection. Đề tài xây dựng một hệ thống phân tầng, trong đó mỗi URL người dùng
bấm vào chỉ đi qua đúng mức kiểm tra cần thiết:

| Lớp | Vai trò | Thời gian |
|---|---|---|
| **A** – Blocklist + VirusTotal | Chặn tức thì URL đã biết là lừa đảo (OpenPhish, PhishTank, Phishing.Database, cập nhật mỗi giờ) | < 1 ms |
| **B** – Typosquatting | Phát hiện mạo danh thương hiệu (Damerau-Levenshtein + Jaro-Winkler trên toàn bộ URL, kể cả subdomain, path, URL chuyển hướng ẩn) | ~0,5 ms |
| **C** – BiLSTM + XGBoost | Cổng ưu tiên recall: điểm học sâu của URL kết hợp đặc trưng hạ tầng (DNS, chứng chỉ TLS, tuổi tên miền RDAP, thứ hạng Tranco). Xác nhận ngay khoảng một nửa số URL sạch, phần còn lại chuyển lên D | ~32 ms |
| **D** – LLM | Đọc nội dung trang đã lọc chỉ thị ẩn, kết luận phishing / suspicious / safe | ~2 s |

Nguyên tắc thiết kế: **không để URL lừa đảo nào bị gắn nhãn sạch**. Khi một lớp gặp lỗi, hết thời gian
hoặc thiếu bằng chứng, hệ thống nghiêng về cảnh báo chứ không bỏ qua. Hệ thống được đóng gói thành HTTP API
(Docker, xác thực bằng khoá, giới hạn tần suất) để giao diện web hoặc tiện ích trình duyệt gọi trực tiếp, và
được gia cố trước các kỹ thuật né tránh thực tế: URL rút gọn, chuyển hướng lồng nhau, host mã hoá hoặc dùng
ký tự full-width, cloaking, đầu độc cache.

---

## Ba đóng góp chính

### 1. Kiến trúc phân tầng ưu tiên recall: gần như không bỏ sót, nhưng chỉ cần LLM cho một nửa lưu lượng

- Trên tập test 81.633 URL, **Lớp A + B + C chỉ bỏ sót 5 trên 39.266 URL phishing (0,013 %)**.
- Đồng thời hệ thống **xác nhận sạch 50,9 % URL sạch trong khoảng 33 ms, không gọi LLM lần nào**. LLM chỉ
  xử lý nửa "khó" của lưu lượng, cắt khoảng một nửa chi phí và độ trễ.
- Cặp mô hình được **chọn bằng thực nghiệm** theo cùng một quy tắc trên tập val (ít bỏ sót nhất với điều
  kiện nhận ra ≥ 50 % URL sạch), từ 5 kiến trúc DL × 3 bộ phân loại, và so với 8 baseline. Mô hình chỉ
  dùng URL bỏ sót nhiều hơn **30–160 lần** so với khi có đặc trưng hạ tầng.
- Ablation 3 seed: **khối đặc trưng hạ tầng là yếu tố quyết định**. Bỏ nó thì số bỏ sót trên val tăng từ 18
  lên 329; riêng bỏ Tranco đã tăng lên 72.

### 2. Bộ benchmark lừa đảo tiếng Việt thực tế, không rò rỉ dữ liệu, cho thấy giới hạn của các giải pháp toàn cầu

- **4.290 URL**: phishing lấy từ Chống Lừa Đảo; URL sạch lấy từ tên miền `.vn` trong Tranco và các tên miền
  thương hiệu đã xác minh. Đã kiểm tra **không trùng tên miền** với dữ liệu huấn luyện.
- **Các feed blocklist toàn cầu chỉ bắt được 0,4 % số phishing tiếng Việt**; VirusTotal bắt được 70,5 %
  (trên mẫu 190 URL).
- Lớp C **khái quát được sang dữ liệu chưa từng thấy**: bỏ sót **1,07 %**, nhận ra 52,7 % URL sạch. Điểm
  BiLSTM là yếu tố giúp ngưỡng chọn trên val vẫn đúng với URL tiếng Việt: cùng đặc trưng nhưng LightGBM chỉ
  nhận ra 16–31 % URL sạch.
- Toàn pipeline chạy trên trang thật, đang hoạt động, đạt **warn-F1 0,90–0,91**.

### 3. Lớp LLM chống prompt injection, không phụ thuộc mô hình, có kiểm chứng đối kháng

- **Bộ lọc dựa trên dữ liệu (16 ngôn ngữ)** gỡ văn bản ẩn, trùng màu nền, đẩy ra ngoài màn hình, ký tự vô
  hình và "ASCII smuggling". Nội dung trang được đóng khung là DATA, với marker có nonce ngẫu nhiên mà trang
  không thể giả mạo.
- **Red team** với các tấn công ẩn, đa bước và đa ngôn ngữ: lật tối đa **1 trên 324 trang thật** thành *safe*.
- **Kết hợp chéo giữa các lớp:** LLM nói *safe* nhưng Lớp C chắc chắn ≥ 0,99 thì hạ xuống *suspicious*.
  Nhờ đó tỉ lệ phishing bị kết luận *safe* trên 2.004 trang giữ riêng giảm từ **7,6 % xuống 5,1 %**.
- **Cùng một prompt dùng được cho hai họ LLM.** Kiểm định McNemar cho thấy đổi mô hình là đánh đổi giữa bỏ
  sót và báo nhầm: Haiku bỏ sót ít hơn (p = 0,004) nhưng chặn nhầm nhiều hơn (p = 0,005). Hai lần chạy lặp
  lại khớp 95–97 % kết luận.

---

## Hạn chế về dữ liệu (giữ nguyên, không sửa nhãn)

Nhãn của các nguồn công khai có nhiễu. Đề tài không sửa nhãn thủ công mà báo cáo nhiễu như một hạn chế
đã đo được. Nhãn tham chiếu duy nhất là của VirusTotal (≥ 2 engine báo độc → phishing, 0 → sạch, còn
lại → không chắc chắn); không có nhãn gán tay.

| File | Nhiễu đo được | Nguyên nhân |
|---|---|---|
| `dataset/urls/urls_labeled.csv` (537.613 URL) | 92 URL nhãn *clean* đang nằm trong feed phishing (train 70, val 16, test 6); 1.442 dòng train nghi sai nhãn (đã loại khỏi train, val/test vẫn còn nhiễu cùng loại); một ít URL tài nguyên (vd `googletagmanager.com/gtm.js`) bị gắn *phishing* | Nhãn sạch lấy tại một thời điểm, sau đó site bị hack hoặc tên miền hết hạn bị mua lại; crawler Tranco gom cả trang cờ bạc / IP lạ; Phishing.Database không xác minh từng URL; feed ghi cả script mà trang lừa đảo nhúng vào |
| `data/test_sets/vi/phishing.csv` (2.710 URL) | Khoảng 32 % host mang dáng cờ bạc, khoảng 3 % forex / crypto (ước lượng theo từ khoá, có thể đếm dư); toàn bộ được liệt kê từ 5/2024 | Chống Lừa Đảo định nghĩa "lừa đảo" rộng hơn phishing (cờ bạc, đầu tư bất hợp pháp tại Việt Nam); sau hơn một năm nhiều tên miền đã chết hoặc đổi chủ |
| `data/test_sets/vi/clean.csv` (1.580 URL) | 1 URL đang nằm trong feed phishing | Nhãn sạch theo thứ hạng Tranco `.vn`; phổ biến không đồng nghĩa an toàn |
| `dataset/external/phreshphish_urls.csv` và các tập con (`layer_d_dev.csv`, `layer_d_test.csv`, `dom_pages.csv`) | Khoảng 6,7 % lệch so với VirusTotal | Nhãn gán tự động khi thu thập: trang chủ nền tảng bị gắn theo trang lừa đảo của một tenant; adware, cửa hàng lừa đảo lọt vào nhóm *clean* |
| `reports/relabel/relabel_subset.csv` (300 trang) | Nhãn dataset lệch 7 trên 149 trang VirusTotal kết luận được: **4,7 %** (khoảng tin cậy 95 %: 2,3–9,4 %); 151 trang VirusTotal không chắc chắn | Bản thân VirusTotal cũng nhiễu: "0 engine báo độc" không có nghĩa là sạch, phishing mới thường chưa bị phát hiện |

**Ảnh hưởng:** nhiễu làm lệch số liệu theo cả hai chiều. URL "phishing" thực ra là cờ bạc hợp pháp ở nước
ngoài làm số bỏ sót trông cao hơn thực tế; URL "clean" thực ra đã bị hack làm số báo nhầm trông cao hơn
thực tế.

---

## Lưu ý khi viết báo cáo

- Kết quả Lớp A trên tập test của dataset bị **lạc quan**, vì feed vừa là nguồn nhãn vừa là bộ chặn. Nên
  nhấn mạnh con số "B + C, không có A" (bỏ sót 10) và kết quả trên tập tiếng Việt.
- Phép đo VirusTotal chỉ trên mẫu 190 URL; Lớp D vẫn để lọt 4–5 % phishing thành *safe*. Hai điểm này nên
  ghi ở mục **hạn chế**, cùng với nhiễu nhãn ở mục trên.
- Phần gia cố bảo mật (né tránh bằng chuyển hướng, host mã hoá, đầu độc cache, cloaking) **chưa có số đo
  định lượng**. Có thể nêu như đóng góp về kỹ thuật hệ thống, không nên xếp vào nhóm đóng góp khoa học có
  số liệu.

Nguồn số liệu: `IMPROVEMENTS_NOTE.md`, `reports/`.
