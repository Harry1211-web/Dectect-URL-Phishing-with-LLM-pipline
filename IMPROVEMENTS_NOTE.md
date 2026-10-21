# Ghi chú cải tiến

Ghi chú này gồm các thay đổi đã làm để tối ưu hệ thống, các phương pháp đã đo rồi bỏ, số liệu
benchmark và các đóng góp mà những số liệu đó chứng minh. Mọi con số đều lấy từ
`notebooks/pipeline_experiment.ipynb` hoặc từ các file báo cáo mà notebook dẫn tới. Mọi lựa chọn
đều được đưa ra trên dữ liệu validation / dev, và mỗi tập test chỉ được chấm một lần cho mỗi cấu
hình đã chọn.

## 1. Các phương pháp đã áp dụng

| Phương pháp | Nội dung | Bằng chứng |
|---|---|---|
| **Bộ lọc dùng mẫu động** | `html_sanitize.PatternLoader` gộp mỗi ngôn ngữ một file JSON (`data/injection_patterns/<lang>.json`, 16 ngôn ngữ). Mỗi file chứa: regex prompt injection, các cơ quan được trích dẫn và tên miền riêng của họ, từ kết luận, từ phủ định, từ mồi nhử. Bộ lọc và bộ đọc kết luận không còn mẫu riêng của ngôn ngữ nào. | Phát hiện giống hệt trước (bộ test không đổi). Thêm ngôn ngữ chỉ cần thêm file. Red team khó: tối đa 1 trang bị lật thành *safe* trên 324 trang thật bị tấn công (`reports/llm/redteam_injection/`). |
| **VirusTotal trong Lớp A** | Chạy sau blocklist và lối tắt tên miền tin cậy. Tra báo cáo URL trên VirusTotal: ≥ 2 engine báo độc hại → phishing. Có cache, không bao giờ gửi URL lên. Lỗi quota / mạng thì bỏ qua. | Trên một mẫu cố định của tập tiếng Việt (95 phishing từ Chống Lừa Đảo + 95 sạch): các feed công khai bắt được 1,1 % phishing, VirusTotal bắt 70,5 % (precision 0,971, gắn cờ nhầm 2 trang sạch). |
| **Chống Lừa Đảo làm nhãn chuẩn tiếng Việt** | Nhãn phishing tiếng Việt lấy từ blacklist Chống Lừa Đảo (một bản mirror công khai cố định, chỉ giữ host còn sống và đăng ký trước thời điểm chụp). Chống Lừa Đảo *không* phải feed của Lớp A, nên đánh giá Lớp A trên tập này không bị vòng lặp. | 2.710 URL phishing (1.604 tên miền). |
| **Tập test tiếng Việt không rò rỉ** | Chuyển sang `data/test_sets/vi/`, tách khỏi dữ liệu huấn luyện. Đã kiểm tra trùng tên miền đăng ký với mọi file huấn luyện, mọi file đánh giá và các trang DOM. | 0 tên miền trùng với file huấn luyện và trang DOM. Có 2 tên miền phishing trùng với các tập trang của Lớp D, nhưng đó là tập đánh giá, chưa từng dùng để huấn luyện. |
| **Một ngôn ngữ prompt, ngôn ngữ đầu ra động** | Prompt viết bằng tiếng Anh. Tham số `lang` (mặc định `vi`) chọn ngôn ngữ giải thích. Từ khóa VERDICT vẫn giữ tiếng Anh. | Dev, 651 trang: giải thích bằng tiếng Việt 100 % (gpt) / 99,7 % (Haiku). Kết luận khớp với lần chạy chỉ ra kết luận: 97,2 % / 95,9 %. Phishing → safe: 15,7 → 15,4 % (gpt, McNemar p = 1,0) và 16,5 → 14,8 % (Haiku, p = 0,07). Giữ lại vì không có thay đổi đáng kể. |
| **Chọn lại Lớp C** | 5 mô hình DL cho URL, cross-fit như bản production. Sau đó RF / XGBoost / LightGBM dùng điểm của mô hình DL thắng + đặc trưng URL + đặc trưng host. Cả hai bước đều chọn theo quy tắc "ít bỏ sót nhất trên val khi nhận ra ≥ 50 % sạch trên val". | BiLSTM thắng với 424 lần bỏ sót trên val (CNN/BiLSTM 428, char-CNN 478, URLNet 667, Transformer 1.128). XGBoost thắng với 18 (LightGBM 24, RF 32). Bỏ sót trên test của bản production giảm 12 → 10 ở mức 51,1 % sạch. |
| **Nhãn VirusTotal cho tập gán nhãn lại** | 300 trang gán nhãn bằng VirusTotal thay cho nhãn tay: ≥ 2 engine → phishing, 0 → sạch, còn lại → không chắc. | 149 trang có nhãn: 4,7 % nhãn của dataset không khớp [KTC 95 %: 2,3–9,4 %]. 124 URL VirusTotal không biết. |

## 2. Các phương pháp đã bỏ và lý do

| Phương pháp | Trạng thái | Lý do |
|---|---|---|
| Regex viết cứng trong Python | Dùng đến commit f82f78b (regex tiếng Việt + tiếng Anh trong `html_sanitize.py`). Chuyển sang dữ liệu ở d733ab8. Các danh sách cuối cùng (tên miền cơ quan, từ kết luận, từ phủ định) chuyển trong đợt này. | Thêm ngôn ngữ phải sửa code. Một lần "dịch toàn bộ" trước đây đã dịch luôn mẫu phát hiện tiếng Việt, và lẽ ra sẽ làm hỏng khả năng phát hiện trên trang tiếng Việt. |
| System prompt hai ngôn ngữ | **Chưa bao giờ triển khai.** Dự án từng dùng prompt chỉ tiếng Việt (f82f78b), sau đó là một prompt tiếng Anh. | Hai prompt sẽ nhân đôi công tinh chỉnh và đánh giá. Ngôn ngữ đầu ra động đã cho giải thích tiếng Việt từ một prompt mà kết luận không thay đổi đáng kể (bảng trên). |
| URL phishing tổng hợp / tự tạo làm nhãn | **Không bao giờ dùng để huấn luyện hay kiểm thử.** Trang tổng hợp chỉ có trong red team và các kịch bản thương hiệu Việt, và được báo cáo rõ là tổng hợp. | URL tự tạo đo bộ sinh chứ không đo kẻ tấn công thật. Dự án dùng feed thật thay thế (PhishTank, OpenPhish, Chống Lừa Đảo). |
| Dữ liệu gộp chung, không tách biệt | Tập tiếng Việt từng nằm ở `dataset/urls/vi/` cạnh dữ liệu huấn luyện (dù đã tách theo tên miền). Nay nằm ở `data/test_sets/vi/`, có kiểm tra trùng lặp trong notebook và trong test. | Tách vật lý và có kiểm tra thì rò rỉ sẽ lộ ra. |
| Cho mô hình DOM xóa cờ của URL (luật "and") | Đo trên val DOM: 1 bỏ sót, 52,6 % sạch (chỉ URL: 1 bỏ sót, 36,3 %). **Bị loại.** | Trang do kẻ tấn công kiểm soát: một trang cloaking trông sạch sẽ biến URL đã bị gắn cờ thành "sạch". |
| Mô hình DOM chỉ thêm nghi ngờ (luật "or") | Được phép, nhưng không đạt ≥ 50 % sạch trên val (riêng URL đã dưới mức này trên các trang đó). **Không bật.** | Lớp C production vẫn chỉ dùng URL. Mô hình DOM (PR-AUC val 0,974, ROC-AUC trên tập giữ riêng 0,935) được giữ lại và đo đạc. |
| Số engine VirusTotal làm đặc trưng ML | Không xây dựng | Số engine gắn cờ một URL gần như chính là nhãn của nó (rò rỉ). Quota miễn phí (500 / ngày) không đủ cho tập huấn luyện. |
| Các ứng viên Lớp C khác | Đã đo, thua trên val | CNN/BiLSTM, char-CNN, URLNet, Transformer; LightGBM, random forest (mục 3). |
| Các vòng trước | Đã đo, bỏ | RAG cho Lớp D (kết luận không tốt hơn, `reports/llm/rag_pipeline/`). Các biến thể tên miền ghép của Lớp B (báo động nhầm nhiều hơn mà bỏ sót của pipeline không giảm). |

## 3. Tóm tắt benchmark

**Cách đọc.** Phishing là lớp dương.

- **Lớp A–C:** một URL bị *gắn cờ* khi bị chặn hoặc bị chuyển tiếp sang Lớp D.
- **Lớp C** là cổng ưu tiên recall. Theo thiết kế, nó chuyển khoảng một nửa số URL sạch sang LLM,
  nên precision và F1 thấp là có chủ đích. Con số quan trọng là tỉ lệ bỏ sót.
- **Lớp D và toàn pipeline:** *strict* = kết luận *phishing*. *warn* = *phishing* hoặc
  *suspicious*, tức người dùng được cảnh báo.

### 3.1 Lớp C và các baseline — tập test của dataset (81.633 URL), cùng một quy tắc cho mọi mô hình

| Mô hình | % bỏ sót test | Precision | Recall | F1 | ROC-AUC | VN: % bỏ sót / % sạch |
|---|---|---|---|---|---|---|
| Logistic regression (URL) | 5,37 | — | 0,946 | 0,762 | 0,918 | 30,6 / 97,5 |
| Random forest (URL) | 3,57 | — | 0,964 | 0,772 | 0,940 | 50,1 / 92,7 |
| LightGBM (URL) | 2,85 | — | 0,972 | 0,776 | 0,946 | 44,5 / 98,0 |
| Transformer (URL) | 2,80 | — | 0,972 | 0,777 | 0,946 | 27,3 / 78,5 |
| URLNet (URL) | 1,63 | — | 0,984 | 0,785 | 0,963 | 13,5 / 98,4 |
| char-CNN (URL) | 1,10 | — | 0,989 | 0,787 | 0,971 | 18,2 / 99,0 |
| BiLSTM (URL) | 1,02 | — | 0,990 | 0,787 | 0,972 | 17,4 / 98,0 |
| LightGBM (URL + host) | 0,033 | — | 0,9997 | 0,791 | 0,995 | 0,78 / 16,3 |
| Lớp C cũ: CNN/BiLSTM + XGBoost | 0,033 | — | 0,9997 | 0,791 | 0,994 | 0,66 / 51,1 |
| **Lớp C cuối: BiLSTM + XGBoost** | **0,033** | — | **0,9997** | **0,790** | **0,994** | **0,59 / 50,1** |

- Precision ≈ 0,65 ở mọi dòng, vì quy tắc chuyển tiếp ≈ 50 % URL sạch. Recall = 1 − tỉ lệ bỏ sót.
- Các ứng viên được chọn trên val (`reports/layer_c_selection/`):
  - Mô hình DL: 424–1.128 bỏ sót.
  - Mô hình ML trên điểm BiLSTM: XGBoost 18, LightGBM 24, random forest 32.
- Với cùng đặc trưng và ngưỡng val, LightGBM chỉ nhận ra 31 % URL sạch tiếng Việt, random forest
  21 %, còn XGBoost 50 %.

### 3.2 Các lớp của pipeline — ngưỡng production, dữ liệu host lấy từ cache, không gọi LLM

| Tập | Lớp | Bỏ sót | Nhận ra sạch | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Test của dataset (39.266 phishing / 42.367 sạch) | A | 10.065 | 100 % | 1,000 | 0,744 | 0,853 |
| | B | 36.206 | 99,7 % | 0,964 | 0,078 | 0,144 |
| | C (cũ) | 12 | 51,1 % | — | 0,9997 | — |
| | **C (cuối)** | **10** | **51,1 %** | 0,654 | 0,9997 | 0,791 |
| | **A + B + C** | **5** | **50,9 %** | 0,654 | 0,9999 | 0,791 |
| Tập tiếng Việt (2.710 / 1.580) | A (feed) | 2.700 | 99,9 % | 0,909 | 0,004 | 0,007 |
| | B | 2.653 | 99,8 % | 0,950 | 0,021 | 0,041 |
| | C (cũ) | 28 (1,03 %) | 48,4 % | — | 0,990 | — |
| | **C (cuối)** | **29 (1,07 %)** | **52,7 %** | 0,782 | 0,989 | 0,874 |
| | **A + B + C** | **29** | **54,4 %** | 0,788 | 0,989 | 0,877 |
| Mẫu VN, 95 + 95 | Chỉ blocklist | 94 | 100 % | 1,000 | 0,011 | 0,021 |
| | **Blocklist + VirusTotal** | **28** | **97,9 %** | **0,971** | **0,705** | **0,817** |

Các feed blocklist cũng là nguồn nhãn phishing của dataset, nên kết quả của Lớp A trên tập test
của dataset bị lạc quan. Riêng "B + C (không có A)" bỏ sót 10.

### 3.3 Lớp D và toàn pipeline

| Thiết lập | Mô hình | Phishing → safe | Sạch → phishing | Strict P / R / F1 | Warn P / R / F1 | p50 / p95 |
|---|---|---|---|---|---|---|
| Lớp D, test giữ riêng (2.004 trang), có override của C | gpt-6-luna | 5,08 % | 2,90 % | 0,959 / 0,678 / 0,795 | 0,929 / 0,949 / 0,939 | 2,0 / 4,8 s |
| | claude-haiku-5-5 | 3,98 % | 4,80 % | 0,940 / 0,745 / 0,831 | 0,888 / 0,960 / 0,923 | 1,5 / 3,2 s |
| Toàn pipeline, VN live (300 + 300), Lớp C cũ | gpt-6-luna | — | — | 0,954 / 0,417 / 0,580 | 0,899 / 0,890 / 0,895 | 2,9 / 8,7 s |
| Toàn pipeline, VN live, **bản cuối** | gpt-6-luna | — | — | 0,957 / 0,517 / 0,671 | 0,900 / 0,900 / **0,900** | 2,7 / 8,0 s |
| | claude-haiku-5-5 | — | — | 0,969 / 0,523 / 0,680 | 0,910 / 0,907 / **0,908** | 2,3 / 7,4 s |

**Chạy lại lần hai trên cùng các trang:**
- Kết luận khớp 95,4–97,4 %.
- Phishing → safe: gpt (test) 7,57 / 7,67 %; Haiku (test) 5,98 / 5,58 %.
- Không có McNemar p nào < 0,12.

**Chi phí mỗi trang:** gpt $0,000129, Haiku $0,000224.

**Các lần chạy VN live** bỏ qua VirusTotal vì quota. Nhiều trang tiếng Việt trong blacklist đã
offline, nên kết luận của chúng lấy từ luật dự phòng. Đó là lý do recall strict thấp.

### 3.4 Độ trễ (một URL, không LLM; `reports/pipeline/metrics.json`)

| Giai đoạn | p50 | p95 |
|---|---|---|
| A blocklist | 0,11 ms | 0,15 ms |
| B typosquatting | 0,50 ms | 2,4 ms |
| C (dữ liệu host đã cache) | 31,7 ms | 34,6 ms |
| A + B + C | 32,5 ms | 35,2 ms |
| VirusTotal (cache / live) | ~1 ms / timeout tối đa 2 s | — |
| D (LLM) | 1,5–2,7 s | 3,2–8,0 s |

## 4. Đóng góp cho bài báo nghiên cứu

Mỗi luận điểm đều kèm bằng chứng. Luận điểm nào chưa đủ bằng chứng được ghi ở phần giới hạn.

1. **Chuỗi lớp ưu tiên recall, chỉ dùng LLM cho nửa khó.**
   - Lớp A + B + C bỏ sót 5 trên 39.266 URL phishing test (0,013 %).
   - Đồng thời xác nhận sạch 50,9 % URL sạch trong khoảng 35 ms, không gọi LLM lần nào.
   - Trên các trang giữ riêng có đi tới LLM, override của C giảm phishing → safe còn 5,1 % (gpt)
     / 4,0 % (Haiku).
2. **Bằng chứng host cộng điểm URL từ DL, chọn theo một quy tắc ưu tiên bỏ sót.**
   - So trên 5 mô hình DL × 3 bộ phân loại và 8 baseline, mô hình chỉ dùng URL bỏ sót phishing
     test nhiều hơn 30–160 lần so với mô hình dùng host.
   - Thêm điểm BiLSTM vào XGBoost là yếu tố giúp ngưỡng val áp dụng được cho URL tiếng Việt chưa
     từng thấy: nhận ra 50 % sạch ở mức bỏ sót 0,59 %.
   - Với cùng đặc trưng host, LightGBM / random forest chỉ nhận ra 16–31 % sạch.
3. **Benchmark tiếng Việt thực tế, không rò rỉ, cho thấy các feed bỏ sót rất nhiều.**
   - 4.290 URL: phishing từ Chống Lừa Đảo; sạch từ Tranco `.vn` và các tên miền thương hiệu đã
     xác minh. Đã kiểm tra tách biệt tên miền.
   - Feed toàn cầu bắt được 0,4 % số phishing này; VirusTotal bắt 70,5 % (trên mẫu).
   - Cổng ML bỏ sót 1,07 %; toàn pipeline live đạt warn-F1 0,90–0,91.
4. **Lớp LLM không phụ thuộc mô hình, chống prompt injection, có đo độ ổn định.**
   - Một bộ lọc (dựa trên dữ liệu, 16 ngôn ngữ) và một prompt tiếng Anh dùng được cho hai họ LLM.
   - Tấn công prompt injection khó lật tối đa 1 trên 324 trang thật thành *safe*.
   - Hai lần chạy giống hệt nhau khớp 95–97 % kết luận, không có khác biệt đáng kể.
   - Đổi mô hình là đánh đổi giữa bỏ sót và báo động nhầm: Haiku bỏ sót ít hơn (p = 0,004)
     nhưng chặn nhầm trang sạch nhiều hơn (p = 0,005).
5. **Dùng cấu trúc trang có ràng buộc bảo mật.**
   - Mô hình DOM huấn luyện trên trang PhreshPhish tách rời mọi tập đánh giá, xếp hạng tốt các
     trang giữ riêng (ROC-AUC 0,935).
   - Cho nó xóa cờ URL bị loại, vì trang do kẻ tấn công kiểm soát (cloaking).
   - Theo luật "chỉ thêm nghi ngờ", nó không đạt mục tiêu về lưu lượng sạch.
   - Đây là kết quả âm, ủng hộ việc dùng bằng chứng URL / host làm cổng.

**Giới hạn.**
- Nhãn của các feed công khai bị nhiễu: lệch 4,7 % so với VirusTotal trên tập gán nhãn lại, và một
  nửa số URL đó VirusTotal không biết.
- Phép đo VirusTotal chỉ trên mẫu 190 URL.
- Lớp C cần tra host trực tiếp khi chạy.
- Lớp D vẫn để 4–5 % trang phishing nó xem lọt thành *safe*.
- Mỗi mô hình DL chỉ huấn luyện một lần (một seed), nên chênh lệch BiLSTM / CNN-BiLSTM trên val
  (424 so với 428 bỏ sót) nằm trong mức nhiễu.
