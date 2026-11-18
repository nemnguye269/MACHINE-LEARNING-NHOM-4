# -*- coding: utf-8 -*-
"""Thi nghiem so sanh cach BIEU DIEN van ban (yeu cau 3.1.3 cua de cuong):
  - Bag-of-Words (CountVectorizer) vs TF-IDF
  - unigram vs unigram+bigram
  - min_df, so luong dac trung
voi 2 mo hinh: Multinomial Naive Bayes (mo hinh baseline cua de cuong)
va KNN (mo hinh chinh cua nhom).

Ket qua: results/bieu-dien.json, results/bieu-dien.csv,
         results/so-sanh-bieu-dien.png

Chay: python experiments_representation.py   (~10 phut, phan lon la KNN predict)
"""
import json
import re
import string
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

RANDOM_STATE = 42
TEST_SIZE = 0.25
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

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


print("Dang tai va tien xu ly du lieu...")
fake = pd.read_csv("data/Fake.csv"); fake["label"] = 1
true = pd.read_csv("data/True.csv"); true["label"] = 0
data = pd.concat([fake, true]).sample(frac=1, random_state=RANDOM_STATE)
data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
data = data[["text", "label"]].dropna().drop_duplicates(subset="text")

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)
print(f"Train={len(X_train)}  Test={len(X_test)}")

CONFIGS = [
    # (ten, vectorizer, model)
    ("BoW · unigram", CountVectorizer(ngram_range=(1, 1)), "nb"),
    ("BoW · 1-2gram", CountVectorizer(ngram_range=(1, 2)), "nb"),
    ("TF-IDF · unigram", TfidfVectorizer(ngram_range=(1, 1)), "nb"),
    ("TF-IDF · 1-2gram", TfidfVectorizer(ngram_range=(1, 2), max_features=50000), "nb"),
    ("TF-IDF · min_df=3", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=3), "nb"),
    ("TF-IDF · 20.000 đặc trưng", TfidfVectorizer(ngram_range=(1, 2), max_features=20000), "nb"),
    ("BoW · 1-2gram + KNN", CountVectorizer(ngram_range=(1, 2)), "knn"),
    ("TF-IDF · 1-2gram + KNN", TfidfVectorizer(ngram_range=(1, 2), max_features=50000), "knn"),
]

results = []
for name, vec, kind in CONFIGS:
    print(f"\n[{name}] vector hoa...")
    t0 = time.time()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    vec_s = time.time() - t0

    if kind == "nb":
        model = MultinomialNB()
    else:
        model = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                     weights="distance", algorithm="brute")

    t0 = time.time()
    model.fit(Xtr, y_train)
    fit_s = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(Xte)
    pred_s = time.time() - t0

    row = {
        "cau_hinh": name,
        "bieu_dien": "BoW" if name.startswith("BoW") else "TF-IDF",
        "mo_hinh": "Naive Bayes" if kind == "nb" else "KNN",
        "so_dac_trung": int(Xtr.shape[1]),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "vector_time_s": round(vec_s, 1),
        "fit_time_s": round(fit_s, 2),
        "predict_time_s": round(pred_s, 2),
    }
    results.append(row)
    with open(OUT_DIR / "bieu-dien.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(OUT_DIR / "bieu-dien.csv", index=False,
                                 encoding="utf-8")
    print(f"    acc={row['accuracy']:.4f}  F1={row['f1']:.4f}  "
          f"({row['so_dac_trung']} dac trung, vector {vec_s:.0f}s, "
          f"predict {pred_s:.0f}s)", flush=True)


with open(OUT_DIR / "bieu-dien.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
pd.DataFrame(results).to_csv(OUT_DIR / "bieu-dien.csv", index=False,
                             encoding="utf-8")

# --- bieu do so sanh ---
nb = [r for r in results if r["mo_hinh"] == "Naive Bayes"]
knn = [r for r in results if r["mo_hinh"] == "KNN"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, rows, title in ((axes[0], nb, "MultinomialNB (baseline)"),
                        (axes[1], knn, "KNN (k=5, cosine)")):
    labels = [r["cau_hinh"].replace(" + KNN", "") for r in rows]
    vals = [r["accuracy"] * 100 for r in rows]
    colors = ["#8A6E3B" if r["bieu_dien"] == "BoW" else "#1D6B4A" for r in rows]
    bars = ax.barh(range(len(rows)), vals, color=colors, height=0.55)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(50, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=8)

fig.suptitle("So sánh cách biểu diễn văn bản (BoW và TF-IDF)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "so-sanh-bieu-dien.png", dpi=160)
plt.close(fig)

print("\nDa luu results/bieu-dien.json, .csv va so-sanh-bieu-dien.png")
