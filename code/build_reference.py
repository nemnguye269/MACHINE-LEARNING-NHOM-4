# -*- coding: utf-8 -*-
"""Tao knn_ref.pkl: tieu de + nhan cua tung bai trong TAP HUAN LUYEN,
theo dung thu tu hang cua ma tran Xtr da fit trong train_and_evaluate.py.

Muc dich: app.py dung model.kneighbors() de chi ra 5 bai gan nhat
(chi so hang trong Xtr), tra cuu sang file nay de hien thi tieu de.

Chay: python build_reference.py
"""
import pickle
import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.25
DATA_DIR = Path("data")

NLTK_DIR = Path(__file__).resolve().parent / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)
nltk.data.path[:] = [str(NLTK_DIR)] + [
    p for p in nltk.data.path if not str(p).upper().startswith("D:\\")
]
nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)

# ---- giong het train_and_evaluate.py -------------------------------------
fake = pd.read_csv(DATA_DIR / "Fake.csv")
true = pd.read_csv(DATA_DIR / "True.csv")
fake["label"] = 1
true["label"] = 0
data = pd.concat([fake, true]).sample(frac=1, random_state=RANDOM_STATE)
data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")
data = data[["title", "text", "label"]]
data = data.dropna()
data = data.drop_duplicates(subset="text")

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


# Chia DataFrame truc tiep: cung so mau + cung seed => dung thu tu hang nhu
# khi chia Series data["text"] o train_and_evaluate.py
train_df, test_df = train_test_split(
    data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"Train={len(train_df)}  Test={len(test_df)}")

with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

y_model = list(getattr(model, "_y", []))
y_file = list(train_df["label"].values)
same = len(y_model) == len(y_file) and all(
    int(a) == int(b) for a, b in zip(y_model, y_file))
print(f"Do dai khop: {len(y_model)} vs {len(y_file)} | nhan khop: {same}")

if not same:
    mism = sum(1 for a, b in zip(y_model, y_file) if int(a) != int(b))
    raise SystemExit(f"LOI: nhan lech ({mism} dong). Thu tu hang khong khop, dung tao file.")

rows = []
for title, text, label in zip(train_df["title"], train_df["text"], train_df["label"]):
    t = " ".join(str(title).split())
    if not t:
        t = " ".join(str(text).split()[:14]) + "..."
    snippet = " ".join(str(text).split())
    rows.append({
        "title": t[:120],
        "snippet": snippet[:160],
        "label": int(label),
    })

with open("knn_ref.pkl", "wb") as f:
    pickle.dump(rows, f, protocol=4)

mb = Path("knn_ref.pkl").stat().st_size / 1e6
print(f"Da luu knn_ref.pkl: {len(rows)} bai, {mb:.1f} MB")

# Kiem tra nhanh: 5 lang gieng cua 1 bai test phai tra ve dung chi so
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

with open("tfidf_vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)
sample = clean_text(test_df.iloc[0]["text"])
dist, idx = model.kneighbors(vec.transform([sample]), n_neighbors=5)
print("Vi du 5 lang gieng:")
for d, i in zip(dist[0], idx[0]):
    print(f"  d={d:.4f}  nhan={rows[i]['label']}  {rows[i]['title'][:70]}")
