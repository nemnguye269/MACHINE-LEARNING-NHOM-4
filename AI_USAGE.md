# Khai báo sử dụng công cụ AI

Học phần: Nhập môn Học máy (FIT4013) — Đề tài 2: Phát hiện tin tức giả mạo
Nhóm 4 — Lớp 18-11, Trường Đại học Đại Nam.

Theo yêu cầu tại mục 2.3 của *Hướng dẫn thực hiện bài tập lớn học phần FIT4013*, nhóm khai báo
ngắn gọn phạm vi sử dụng công cụ AI trong đề tài như sau.

## 1. Công cụ đã dùng

Trợ lý AI dạng mô hình ngôn ngữ lớn (giao diện hội thoại), dùng trong quá trình soạn báo cáo và
viết giao diện demo.

## 2. Đã dùng AI cho việc gì

| Hạng mục | Cách sử dụng | Mức độ |
|---|---|---|
| Tìm hiểu lý thuyết | Giải thích lại KNN, cosine similarity, TF-IDF và cách MultinomialNB tính xác suất để nhóm đối chiếu với giáo trình | Tham khảo |
| Gợi ý mã nguồn | Gợi ý cấu trúc hàm tiền xử lý, cách đặt tham số `TfidfVectorizer`, cách dùng `kneighbors()` để lấy chỉ số 5 bài gần nhất | Gợi ý, nhóm sửa lại |
| Sửa lỗi | Đọc thông báo lỗi khi chạy pipeline, gợi ý nguyên nhân (ví dụ lệch thứ tự hàng giữa ma trận huấn luyện và DataFrame) | Gợi ý |
| Trình bày | Gợi ý bố cục bảng, biểu đồ, cách viết chú thích hình và cách diễn đạt một số đoạn văn | Gợi ý |
| Giao diện demo | Gợi ý bảng màu, bố cục trang Streamlit và cách viết CSS | Gợi ý, nhóm chỉnh lại |

## 3. Những gì nhóm tự làm

- Toàn bộ số liệu trong báo cáo do nhóm tự chạy trên máy của mình bằng `train_and_evaluate.py`,
  `experiments.py`, `experiments_representation.py` và `build_reference.py`. Không có số liệu nào
  do AI tạo ra.
- Nhóm tự chọn dữ liệu, tự chia tập huấn luyện/kiểm tra, tự quyết định cấu hình mô hình và tự
  kiểm tra chéo kết quả với các lần chạy lại.
- Nhóm đọc lại và chịu trách nhiệm toàn bộ nội dung báo cáo, mã nguồn và kết quả đã nộp; có thể
  giải thích từng bước khi bảo vệ.

## 4. Kiểm chứng

Mọi con số trong báo cáo đều tái tạo được bằng ba lệnh trong thư mục `code/`:

```bash
python train_and_evaluate.py            # bốn mô hình, ma trận nhầm lẫn, kịch bản E3-E5
python experiments_representation.py    # so sánh Bag-of-Words và TF-IDF
python build_reference.py               # danh sách bài huấn luyện cho panel 5 bài gần nhất
```

Kết quả được ghi vào `code/results/` dưới dạng JSON và CSV; báo cáo chỉ trích dẫn từ các tệp này.
