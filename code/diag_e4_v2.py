# -*- coding: utf-8 -*-
"""Chan doan E4 buoc 2: khi LOAI BO cac bai rong (vector 0) khoi tap huan
luyen, KNN euclidean co ve voi muc cua cosine khong? Dem so bai rong."""
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

# --- Dem so vector 0 ---
n_tr = np.sqrt(Xtr.multiply(Xtr).sum(axis=1)).A1
n_te = np.sqrt(Xte.multiply(Xte).sum(axis=1)).A1
z_tr = int((n_tr == 0).sum())
z_te = int((n_te == 0).sum())
print(f"[dem] Bai rong (vector 0): train = {z_tr}/{Xtr.shape[0]}, test = {z_te}/{Xte.shape[0]}")

# --- Loc bo cac bai rong trong tap train ---
keep = n_tr > 0
Xtr2 = Xtr[keep]
ytr2 = ytr[keep]
print(f"[loc] Tap train sau loc: {Xtr2.shape[0]} bai")

m = 300
Xq, yq = Xte[:m], yte[:m]

knn_e = KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                             weights="distance", algorithm="brute")
knn_e.fit(Xtr2, ytr2)
pred_e = knn_e.predict(Xq)
print(f"[2] euclidean KHONG bai rong acc (300 mau) = {accuracy_score(yq, pred_e):.4f}")

knn_c = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                             weights="distance", algorithm="brute")
knn_c.fit(Xtr2, ytr2)
pred_c = knn_c.predict(Xq)
print(f"[2] cosine    KHONG bai rong acc (300 mau) = {accuracy_score(yq, pred_c):.4f}")
