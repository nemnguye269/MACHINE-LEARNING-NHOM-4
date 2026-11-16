# Code — Phát hiện tin tức giả mạo (LSTM)

## Chuẩn bị (chỉ cần làm 1 lần)

```powershell
# 1. Tạo môi trường Python 3.12 (TensorFlow chưa hỗ trợ Python 3.14)
py -3.12 -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt tensorflow

# 2. Tải dữ liệu Kaggle ISOT (cần tài khoản Kaggle miễn phí)
#    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
#    Đặt Fake.csv và True.csv vào thư mục data\ ở đây.
```

## Chạy

| Việc cần làm | Cách chạy |
|---|---|
| Huấn luyện + đánh giá + xuất biểu đồ | Double-click `run_train.bat` (hoặc `python train_and_evaluate.py` sau khi activate venv) |
| Demo app kiểm tra tin thật/giả | Double-click `run_app.bat` (hoặc `streamlit run app.py`) |

## Kết quả đầu ra (thư mục `results/`)

| File | Dùng làm gì trong báo cáo |
|---|---|
| `loss-curve.png` | Hình "Biểu đồ lỗi và độ chính xác theo vòng huấn luyện" (chương 3) |
| `confusion-matrix.png` | Hình "Ma trận nhầm lẫn" (chương 3) |
| `roc-curve.png` | Hình "Đường cong ROC" (chương 3) |
| `ket-qua.csv`, `ket-qua.json` | Số liệu bảng kết quả 4 mô hình (chương 3) |

Sau khi chạy xong: chép 3 file PNG vào thư mục `..\images\` của báo cáo, gửi `ket-qua.csv` để cập nhật bảng số liệu.
