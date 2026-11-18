# -*- coding: utf-8 -*-
"""
Phat hien tin tuc gia mao su dung thuat toan KNN — Bo du lieu ISOT (Kaggle)
Quy trinh: tai du lieu -> tien xu ly -> TF-IDF ->
  (1) 3 mo hinh tham chieu: Naive Bayes, Logistic Regression, Passive Aggressive
  (2) mo hinh chinh: KNN (k = 5, khoang cach cosine, trong so distance)
-> danh gia -> xuat bieu do + luu model cho demo Streamlit.

Cach chay:
    python train_and_evaluate.py

Yeu cau:
    - Dat 2 tep Fake.csv va True.csv trong thu muc data/ (cung cap voi tep nay)
    - pip install -r requirements.txt
"""

import json
import pickle
import re
import string
import time
from pathlib import Path

import pandas as pd
import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

# =========================
# 0. Cau hinh
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.25            # 75% train / 25% test
K_NEIGHBORS = 5             # so lang gieng k (mo hinh chinh KNN)
K_METRIC = "cosine"         # khoang cach cosine — phu hop voi vector TF-IDF
K_WEIGHTS = "distance"      # lang gieng gan nang trong so hon

DATA_DIR = Path("data")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

results = {}

# =========================
# 1. Tai du lieu
# =========================
def load_data():
    """Doc Fake.csv (nhan 1) va True.csv (nhan 0), gop va xao tron."""
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = 1
    true["label"] = 0
    data = pd.concat([fake, true]).sample(frac=1, random_state=RANDOM_STATE)
    data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data = data[["text", "label"]]
    # Loai bai trong / trung lap
    data = data.dropna()
    data = data.drop_duplicates(subset="text")
    print(f"[1] So bai viet: {len(data)}  (Fake={sum(data.label)}, True={len(data)-sum(data.label)})")
    return data


# =========================
# 2. Tien xu ly
# =========================
# NLTK tren Windows co dinh duong dan mac dinh C:\nltk_data, D:\nltk_data.
# Neu o D:\ khong doc duoc se gay loi, nen chi giu thu muc nltk_data cua du an.
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
# 3. Vector hoa TF-IDF (dung chung cho moi mo hinh)
# =========================
def build_tfidf(X_train, X_test):
    """TF-IDF voi n-gram 1..2, gioi han 50.000 dac trung pho bien nhat."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)
    print(f"[3] TF-IDF: {Xtr.shape[0]} x {Xtr.shape[1]} dac trung (thua)")
    return tfidf, Xtr, Xte


# =========================
# 4. Danh gia mot mo hinh
# =========================
def evaluate_model(name, model, Xtr, y_train, Xte, y_test, use_proba=False):
    """Fit + du doan + tinh cac do do; tra ve (y_pred, y_prob)."""
    t0 = time.time()
    model.fit(Xtr, y_train)          # voi KNN: chi la "hoc ghi nho" du lieu
    fit_s = time.time() - t0

    t0 = time.time()
    if use_proba:
        # Xac suat lop 1 (Fake) — dung cho ROC; nguong 0.5 ra y_pred
        y_prob = model.predict_proba(Xte)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = None
        y_pred = model.predict(Xte)
    pred_s = time.time() - t0

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fit_time_s": round(fit_s, 2),
        "predict_time_s": round(pred_s, 2),
    }
    print(f"    Thoi gian: fit {fit_s:.2f}s | du doan {pred_s:.2f}s")
    print(classification_report(y_test, y_pred, target_names=["True", "Fake"]))
    return y_pred, y_prob


# =========================
# 5. Cac do do bieu do
# =========================
def plot_confusion(cm, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["True", "Fake"],
                yticklabels=["True", "Fake"])
    plt.xlabel("Du doan"); plt.ylabel("Thuc te")
    plt.title("Confusion Matrix - KNN")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Da luu {path}")


def plot_roc(y_test, y_prob, path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ngau nhien")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - KNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Da luu {path}")


# =========================
# 6. Tong hop ket qua
# =========================
def summarize():
    print("\n========= TONG HOP KET QUA (tap kiem tra 25%) =========")
    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    df.columns = ["Accuracy", "Precision", "Recall", "F1"]
    print((df * 100).round(2).astype(str) + " %")
    df.to_csv(OUT_DIR / "ket-qua.csv")
    with open(OUT_DIR / "ket-qua.json", "w", encoding="utf-8") as f:
        json.dump({k: {kk: round(vv, 4) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"    Da luu ket qua vao {OUT_DIR}/ket-qua.csv va ket-qua.json")


if __name__ == "__main__":
    data = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"[2] Train={len(X_train)}, Test={len(X_test)}")

    X_train = X_train.apply(clean_text)
    X_test = X_test.apply(clean_text)

    tfidf, Xtr, Xte = build_tfidf(X_train, X_test)

    # --- (1) Ba mo hinh tham chieu ---
    reference_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
    }
    for name, model in reference_models.items():
        print(f"[4] Mo hinh tham chieu: {name} ...")
        evaluate_model(name, model, Xtr, y_train, Xte, y_test)

    # --- (2) Mo hinh chinh: KNN ---
    print(f"[4] Mo hinh chinh: KNN (k={K_NEIGHBORS}, metric={K_METRIC}, "
          f"weights={K_WEIGHTS}) ...")
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric=K_METRIC,
                               weights=K_WEIGHTS, algorithm="brute")
    y_pred_knn, y_prob_knn = evaluate_model(
        "KNN", knn, Xtr, y_train, Xte, y_test, use_proba=True)

    # --- Luu model + vectorizer cho demo Streamlit (app.py) ---
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("    Da luu knn_model.pkl va tfidf_vectorizer.pkl (cho app.py)")

    # --- Bieu do danh gia cua KNN ---
    plot_confusion(confusion_matrix(y_test, y_pred_knn),
                   OUT_DIR / "confusion-matrix.png")
    plot_roc(y_test, y_prob_knn, OUT_DIR / "roc-curve.png")

    summarize()
