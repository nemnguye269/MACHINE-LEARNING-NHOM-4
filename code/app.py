# -*- coding: utf-8 -*-
"""
Giao dien demo kiem tra tin that/gia bang Streamlit (mo hinh KNN).
Cach chay:
    streamlit run app.py
(Yeu cau da chay train_and_evaluate.py de co knn_model.pkl va tfidf_vectorizer.pkl)
"""

import pickle
import re
import string
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Kiem tra tin that/gia", page_icon="📰")
st.title("📰 Kiểm tra tin thật hay tin giả")
st.write("Dán nội dung (tiêu đề + nội dung bài viết) vào khung dưới rồi bấm **Kiểm tra**.")

# =========================
# Tai mo hinh KNN + vectorizer TF-IDF
# =========================
@st.cache_resource
def load_assets():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

if not (Path("knn_model.pkl").exists() and Path("tfidf_vectorizer.pkl").exists()):
    st.error("Chưa tìm thấy knn_model.pkl / tfidf_vectorizer.pkl. Hãy chạy `python train_and_evaluate.py` trước.")
    st.stop()

model, vectorizer = load_assets()

# =========================
# Ham tien xu ly (giong luc huan luyen)
# =========================
NLTK_DIR = Path(__file__).resolve().parent / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)
nltk.data.path[:] = [str(NLTK_DIR)] + [
    p for p in nltk.data.path if not str(p).upper().startswith("D:\\")
]
nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
DIGIT_RE = re.compile(r"\d+")


def clean_text(s: str) -> str:
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = s.lower().translate(PUNCT_TABLE)
    s = DIGIT_RE.sub(" ", s)
    words = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(STEMMER.stem(w) for w in words)


# =========================
# Giao dien
# =========================
text = st.text_area("Nội dung bài viết:", height=250,
                    placeholder="Dán tiêu đề và nội dung bài viết vào đây...")

if st.button("🔍 Kiểm tra"):
    if not text.strip():
        st.warning("Bạn chưa nhập nội dung nào.")
    else:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        p_fake = float(model.predict_proba(X)[0][1])
        p_real = 1 - p_fake

        st.subheader("Kết quả")
        col1, col2 = st.columns(2)
        col1.metric("Xác suất TIN THẬT", f"{p_real:.1%}")
        col2.metric("Xác suất TIN GIẢ", f"{p_fake:.1%}")

        if p_fake >= 0.5:
            st.error(f"⚠️ Mô hình dự đoán đây là **TIN GIẢ** (độ tin cậy {p_fake:.1%})")
        else:
            st.success(f"✅ Mô hình dự đoán đây là **TIN THẬT** (độ tin cậy {p_real:.1%})")
        st.caption("Kết quả chỉ mang tính tham khảo, không thay thế kiểm chứng sự thật.")
