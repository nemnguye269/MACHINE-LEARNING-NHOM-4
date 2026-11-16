# MACHINE-LEARNING-NHOM-4 — Phát hiện tin tức giả mạo sử dụng mạng nơ-ron học sâu (LSTM)

Báo cáo học phần **Machine Learning** — Trường Đại học Đại Nam, Khoa Công nghệ Thông tin.

- **Học phần:** Machine Learning (Học máy)
- **Đề tài:** Phát hiện tin tức giả mạo sử dụng mạng nơ-ron học sâu (LSTM)
- **Giảng viên hướng dẫn:** Th.S. Lê Thị Thùy Trang

## Thành viên nhóm (lớp 18-11)

| Mã sinh viên | Họ và tên |
|---|---|
| 1871020409 | Nguyễn Bắc Nam |
| 1871020544 | Đỗ Minh Thành |
| 1871020282 | Nguyễn Phi Hùng |
| 1871020528 | Nguyễn Đức Thái |

## Kết quả chính (bộ dữ liệu ISOT Fake News — 39.105 bài sau lọc trùng)

| Mô hình | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes + TF-IDF | 95,81% | 95,31% | 95,39% | 95,35% |
| Logistic Regression + TF-IDF | 98,84% | 99,29% | 98,14% | 98,71% |
| Passive Aggressive + TF-IDF | **99,42%** | **99,54%** | **99,16%** | **99,35%** |
| LSTM (mô hình chính) | 99,20% | 99,47% | 98,75% | 99,11% |

- LSTM hội tụ nhanh, dừng sớm ở epoch 5 (best epoch 3: 99,20%).
- Các kịch bản E3–E5 xác nhận: từ điển 10.000 từ + chuỗi 256 + Dropout/Early Stopping là cấu hình tốt nhất.

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
│   ├── train_and_evaluate.py   # Huấn luyện 4 mô hình + xuất biểu đồ
│   ├── experiments.py          # Kịch bản thí nghiệm E3–E5
│   ├── app.py                  # Demo Streamlit
│   ├── run_train.bat / run_app.bat
│   ├── data/                   # Fake.csv + True.csv (ISOT, tải từ Kaggle)
│   └── results/                # Bảng số liệu + biểu đồ kết quả
└── main.pdf              # Báo cáo biên dịch xong (51 trang)
```

## Cách chạy thí nghiệm

1. Cài Python 3.12, tạo môi trường:
   ```powershell
   cd code
   py -3.12 -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt tensorflow
   ```
2. Tải bộ dữ liệu [ISOT Fake News (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), đặt `Fake.csv` + `True.csv` vào `code\data\`.
3. Huấn luyện + đánh giá: double-click `run_train.bat` (hoặc `python train_and_evaluate.py`) — khoảng 15–30 phút trên CPU.
4. Demo Streamlit: double-click `run_app.bat` (hoặc `streamlit run app.py`).

## Biên dịch báo cáo

Biên dịch `main.tex` bằng **XeLaTeX 2 lần** (MiKTeX/TeX Live) để cập nhật mục lục và tham chiếu.
