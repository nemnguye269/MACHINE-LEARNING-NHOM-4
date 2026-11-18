# Code — Phát hiện tin tức giả mạo bằng thuật toán KNN

## Chuẩn bị (chỉ cần làm 1 lần)

```powershell
# 1. Tạo môi trường Python 3.12
py -3.12 -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# 2. Tải dữ liệu Kaggle ISOT (cần tài khoản Kaggle miễn phí)
#    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
#    Đặt Fake.csv và True.csv vào thư mục data\ ở đây.
```

## Chạy

| Việc cần làm | Cách chạy |
|---|---|
| Huấn luyện + đánh giá + xuất biểu đồ | Double-click `run_train.bat` (hoặc `python train_and_evaluate.py` sau khi activate venv) |
| Thí nghiệm E3–E5 (k, khoảng cách, trọng số) | `python experiments.py` |
| Demo app kiểm tra tin thật/giả | Double-click `run_app.bat` (hoặc `streamlit run app.py`) |

## Kịch bản thí nghiệm

| Kịch bản | Nội dung |
|---|---|
| E1 | 3 mô hình tham chiếu TF-IDF: Naive Bayes, Logistic Regression, Passive Aggressive |
| E2 | KNN cấu hình chuẩn: k = 5, khoảng cách cosine, trọng số distance |
| E3 | Đổi số lang gieng k = 1 / 3 / 5 / 7 / 9 / 11 |
| E4 | Đổi đo khoảng cách: cosine vs euclidean |
| E5 | Đổi trọng số: distance vs uniform |

## Kết quả đầu ra (thư mục `results/`)

| File | Dùng làm gì trong báo cáo |
|---|---|
| `confusion-matrix.png` | Hình "Ma trận nhầm lẫn" của KNN (chương 3) |
| `roc-curve.png` | Hình "Đường cong ROC" của KNN (chương 3) |
| `do-chinh-xac-theo-k.png` | Hình "Độ chính xác theo k" (chương 3) |
| `ket-qua.csv`, `ket-qua.json` | Số liệu bảng kết quả 5 mô hình (chương 3) |
| `kich-ban.json` | Số liệu bảng kịch bản E3–E5 (chương 3) |

Model + vectorizer cho demo (`knn_model.pkl`, `tfidf_vectorizer.pkl`) được lưu ngay thư mục `code\` sau khi chạy `train_and_evaluate.py`.
