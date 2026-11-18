# MACHINE-LEARNING-NHOM-4 — Phát hiện tin tức giả mạo sử dụng thuật toán KNN

Báo cáo học phần **Machine Learning** — Trường Đại học Đại Nam, Khoa Công nghệ Thông tin.

- **Học phần:** Machine Learning (Học máy)
- **Đề tài:** Phát hiện tin tức giả mạo sử dụng thuật toán KNN (K-Nearest Neighbors)
- **Giảng viên hướng dẫn:** Th.S. Lê Thị Thùy Trang

## Thành viên nhóm (lớp 18-11)

| Mã sinh viên | Họ và tên |
|---|---|
| 1871020409 | Nguyễn Bắc Nam |
| 1871020544 | Đỗ Minh Thành |
| 1871020282 | Nguyễn Phi Hùng |
| 1871020528 | Nguyễn Đức Thái |

## Kết quả chính (bộ dữ liệu ISOT Fake News — 39.105 bài sau lọc trùng, test 9.777 bài)

| Mô hình | Accuracy | Precision | Recall | F1 | Huấn luyện | Dự đoán |
|---|---|---|---|---|---|---|
| Naive Bayes + TF-IDF | 95,81% | 95,31% | 95,39% | 95,35% | 0,11s | 0,02s |
| Logistic Regression + TF-IDF | 98,84% | 99,29% | 98,14% | 98,71% | 0,42s | 0,02s |
| Passive Aggressive + TF-IDF | **99,37%** | **99,48%** | **99,11%** | **99,30%** | 0,20s | 0,02s |
| **KNN (mô hình chính)** | 89,35% | 93,24% | 82,34% | 87,45% | 0,03s | 183,29s |

- **KNN cấu hình:** k = 5, khoảng cách cosine, trọng số theo khoảng cách (distance), TF-IDF n-gram 1–2 (50.000 đặc trưng).
- **Thí nghiệm E3–E5:**
  - E3 — số láng giềng k: 87,48% (k=1) → **89,76% (k=9)** → 89,63% (k=11); vùng k = 5–9 ổn định.
  - E4 — phép đo khoảng cách: cosine 89,35% vs euclidean 55,61% (nguyên nhân: các bài rỗng sau tiền xử lý tạo vector 0 "lọt" vào top-k euclidean; khi loại 3 bài rỗng khỏi tập train, euclidean trở về ≈ cosine).
  - E5 — cơ chế bỏ phiếu: distance 89,35% vs uniform 89,25%.
- Điểm thảo luận đặc trưng của KNN: **lazy learning** — "huấn luyện" 0,03s (chỉ ghi nhớ dữ liệu) nhưng dự đoán 183,29s (~9.000 lần chậm mô hình tuyến tính, ≈ 19 ms/bài).

## Cấu trúc repo

```
├── main.tex              # Tệp chính của báo cáo LaTeX
├── settings.tex          # Cấu hình + thông tin nhóm (sửa tên/MSV/GVHD tại đây)
├── cover.tex             # Trang bìa (mẫu BÁO CÁO HỌC PHẦN)
├── frontmatter/          # Lời nói đầu, kí hiệu viết tắt, danh mục
├── chapters/             # Chương 1–3
├── backmatter/           # Kết luận, phụ lục, tài liệu tham khảo
├── images/               # Logo, biểu đồ kết quả, ảnh demo
├── code/                 # Mã nguồn thí nghiệm
│   ├── train_and_evaluate.py   # Tiền xử lý + TF-IDF + KNN + 3 mô hình tham chiếu
│   ├── experiments.py          # Kịch bản thí nghiệm E3–E5
│   ├── app.py                  # Demo Streamlit (load knn_model.pkl)
│   ├── run_train.bat / run_app.bat
│   ├── data/                   # Fake.csv + True.csv (ISOT, tải từ Kaggle)
│   ├── knn_model.pkl           # Mô hình KNN đã xây dựng (cho demo)
│   ├── tfidf_vectorizer.pkl    # Vectorizer TF-IDF (cho demo)
│   └── results/                # Bảng số liệu + biểu đồ kết quả
└── main.pdf              # Báo cáo biên dịch xong
```

## Cách chạy thí nghiệm

1. Cài Python 3.12, tạo môi trường:
   ```powershell
   cd code
   py -3.12 -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```
2. Tải bộ dữ liệu [ISOT Fake News (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), đặt `Fake.csv` + `True.csv` vào `code\data\`.
3. Huấn luyện + đánh giá: double-click `run_train.bat` (hoặc `python train_and_evaluate.py`) — vài phút trên CPU.
4. Thí nghiệm E3–E5: `python experiments.py` (khoảng 25 phút do chi phí dự đoán KNN).
5. Demo Streamlit: double-click `run_app.bat` (hoặc `streamlit run app.py`).

## Biên dịch báo cáo

Biên dịch `main.tex` bằng **XeLaTeX 2–3 lần** (MiKTeX/TeX Live) để cập nhật mục lục và tham chiếu.
