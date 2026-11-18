# -*- coding: utf-8 -*-
"""
Chay cac kich ban thi nghiem voi thuat toan KNN (dong bo voi bao cao):
  E3: so lang gieng k = 1 / 3 / 5 / 7 / 9 / 11
  E4: do khoang cach: cosine / euclidean (k = 5)
  E5: trong so lang gieng: distance / uniform (k = 5, cosine)
Ket qua: results/kich-ban.json + results/do-chinh-xac-theo-k.png
"""

import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# --- NLTK: tranh o D:\ hu hong ---
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


def clean_text(s):
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = s.lower().translate(PUNCT_TABLE)
    s = DIGIT_RE.sub(" ", s)
    words = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(STEMMER.stem(w) for w in words)


def load_split():
    fake = pd.read_csv("data/Fake.csv"); fake["label"] = 1
    true = pd.read_csv("data/True.csv"); true["label"] = 0
    data = pd.concat([fake, true]).sample(frac=1, random_state=RANDOM_STATE)
    data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data = data.dropna().drop_duplicates(subset="text")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=RANDOM_STATE)
    X_train = X_train.apply(clean_text)
    X_test = X_test.apply(clean_text)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)
    return Xtr, Xte, np.array(y_train), np.array(y_test)


def run_knn(Xtr, Xte, y_train, y_test, k=5, metric="cosine", weights="distance"):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                               weights=weights, algorithm="brute")
    knn.fit(Xtr, y_train)
    prob = knn.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    print(f"[KNN] k={k} metric={metric} weights={weights} -> acc={acc:.4f}")
    return round(acc, 4)


if __name__ == "__main__":
    Xtr, Xte, y_train, y_test = load_split()
    print(f"Train={Xtr.shape[0]}, Test={Xte.shape[0]}")

    results = {}

    # --- E3: do chinh xac theo k (k le tu 1 den 11) ---
    k_list = [1, 3, 5, 7, 9, 11]
    k_acc = {}
    for k in k_list:
        k_acc[k] = run_knn(Xtr, Xte, y_train, y_test, k=k)
    results["E3_k3"] = k_acc[3]
    results["E3_k5_chuan"] = k_acc[5]
    results["E3_k7"] = k_acc[7]
    results["E3_k_curve"] = {str(k): v for k, v in k_acc.items()}

    # --- E4: do khoang cach ---
    results["E4_euclidean"] = run_knn(Xtr, Xte, y_train, y_test,
                                      k=5, metric="euclidean")

    # --- E5: trong so lang gieng ---
    results["E5_uniform"] = run_knn(Xtr, Xte, y_train, y_test,
                                    k=5, weights="uniform")

    with open(OUT / "kich-ban.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Da luu results/kich-ban.json")

    # --- Bieu do do chinh xac theo k ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(int(k) for k in results["E3_k_curve"])
    accs = [results["E3_k_curve"][str(k)] * 100 for k in ks]
    ax.plot(ks, accs, "o-", color="tab:blue")
    best_k = max(k_acc, key=k_acc.get)
    ax.axvline(best_k, color="tab:red", linestyle="--",
               label=f"k tot nhat = {best_k}")
    ax.set_xlabel("So lang gieng k"); ax.set_ylabel("Do chinh xac (%)")
    ax.set_title("Do chinh xac cua KNN theo so lang gieng k")
    ax.set_xticks(ks); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "do-chinh-xac-theo-k.png", dpi=150)
    print("Da luu results/do-chinh-xac-theo-k.png")
