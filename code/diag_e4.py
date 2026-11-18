# -*- coding: utf-8 -*-
"""Chan doan E4: vi sao KNN euclidean (55,61%) khac KNN cosine (89,35%)
tren cung ma tran TF-IDF?

Kiem tra:
  1. Vector TF-IDF co thuc su chuan hoa L2 (norm = 1) hay khong?
  2. KNN euclidean cua sklearn tra ve lang gieng co dung toan hoc khong?
     (tren vector chuan hoa, top-k euclidean PHAI giong top-k cosine)
"""
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

NLTK_DIR = Path(__file__).resolve().parent / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)
nltk.data.path[:] = [str(NLTK_DIR)] + [
    p for p in nltk.data.path if not str(p).upper().startswith("D:\\")
]
nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)
STOP = set(stopwords.words("english"))
STEMMER = PorterStemmer()
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
PUNCT = str.maketrans("", "", string.punctuation)
DIGIT = re.compile(r"\d+")


def clean(s):
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = s.lower().translate(PUNCT)
    s = DIGIT.sub(" ", s)
    return " ".join(STEMMER.stem(w) for w in s.split() if w not in STOP)


print("Dang tai du lieu...")
fake = pd.read_csv("data/Fake.csv"); fake["label"] = 1
true = pd.read_csv("data/True.csv"); true["label"] = 0
data = pd.concat([fake, true]).sample(frac=1, random_state=42)
data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
data = data.dropna().drop_duplicates(subset="text")
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.25, random_state=42)
print("Dang tien xu ly...")
X_train = X_train.apply(clean)
X_test = X_test.apply(clean)
print("Dang build TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)
ytr = np.array(y_train)
yte = np.array(y_test)

# --- 1) Kiem tra norm L2 ---
norms = np.sqrt(Xtr.multiply(Xtr).sum(axis=1)).A1
print(f"[1] Train norms: min={norms.min():.6f} max={norms.max():.6f} "
      f"max|norm-1|={np.abs(norms - 1).max():.2e}")

# --- 2) Chay 2 metric tren 300 mau test dau ---
m = 300
Xq, yq = Xte[:m], yte[:m]
knn_e = KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                             weights="distance", algorithm="brute")
knn_e.fit(Xtr, ytr)
pred_e = knn_e.predict(Xq)
print(f"[2] euclidean acc (300 mau) = {accuracy_score(yq, pred_e):.4f}")

knn_c = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                             weights="distance", algorithm="brute")
knn_c.fit(Xtr, ytr)
pred_c = knn_c.predict(Xq)
print(f"[2] cosine    acc (300 mau) = {accuracy_score(yq, pred_c):.4f}")

# --- 3) So sanh lang gieng cua mau dau tien ---
d_e, i_e = knn_e.kneighbors(Xq[:1])
d_c, i_c = knn_c.kneighbors(Xq[:1])
print(f"[3] knn_e idx = {i_e[0]}")
print(f"[3] knn_e dist = {np.round(d_e[0], 6)}")
print(f"[3] knn_c idx = {i_c[0]}")
print(f"[3] knn_c dist = {np.round(d_c[0], 6)}")

# --- 4) Tinh tay cho mau dau: euclidean & cosine dung toan hoc ---
D_manual = pairwise_distances(Xq[:1], Xtr, metric="euclidean").ravel()
D_cos = cosine_distances(Xq[:1], Xtr).ravel()
top_manual = np.argsort(D_manual)[:5]
top_cos = np.argsort(D_cos)[:5]
print(f"[4] manual euclid top5 idx = {top_manual}, dist = {np.round(D_manual[top_manual], 6)}")
print(f"[4] manual cosine top5 idx = {top_cos}, dist = {np.round(D_cos[top_cos], 6)}")

# Kiem tra dang toan hoc: tren vector chuan hoa, euclid = sqrt(2 - 2*cos)
check = np.sqrt(np.maximum(0, 2 - 2 * D_cos[:5]))
print(f"[4] sqrt(2-2cos) 5 phan tu dau = {np.round(check, 6)}")
print(f"[4] manual euclid 5 phan tu dau = {np.round(D_manual[:5], 6)}")

# Dist ma knn_e tra ve co bang manual euclid tai cung chi so khong?
print(f"[4] knn_e dist tai idx cua no vs manual: "
      f"{np.round(d_e[0], 6)} vs {np.round(D_manual[i_e[0]], 6)}")
