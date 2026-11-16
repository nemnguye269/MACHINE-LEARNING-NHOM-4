# -*- coding: utf-8 -*-
"""
Phat hien tin tuc gia mao — Bo du lieu ISOT (Kaggle)
Quy trinh: tai du lieu -> tien xu ly -> vector hoa ->
huang luyen 3 mo hinh truyen thong + LSTM -> danh gia -> xuat bieu do.

Cach chay:
    python train_and_evaluate.py

Yeu cau:
    - Dat 2 tep Fake.csv va True.csv trong thu muc data/ (cung cap voi tep nay)
    - pip install -r requirements.txt
"""

import json
import re
import string
from pathlib import Path

import numpy as np
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 0. Cau hinh
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.25          # 75% train / 25% test
VOCAB_SIZE = 10000        # tu dien toi da
MAX_LEN = 256             # do dai chuoi co dinh
EMBED_DIM = 64            # kich thuoc vector embedding
LSTM_UNITS = 64           # so neron LSTM
DROPOUT = 0.5
EPOCHS = 10
BATCH_SIZE = 64

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
import nltk
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
# 3. Cac mo hinh truyen thong (TF-IDF)
# =========================
def run_traditional_models(X_train, X_test, y_train, y_test):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50),
    }

    for name, model in models.items():
        print(f"[3] Dang huan luyen: {name} ...")
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        print(classification_report(y_test, y_pred, target_names=["True", "Fake"]))


# =========================
# 4. Mo hinh LSTM
# =========================
def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def run_lstm(X_train, X_test, y_train, y_test):
    print("[4] Dang vector hoa cho LSTM (Tokenizer + Padding) ...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    Xtr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    Xte = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    print("[4] Dang huan luyen mo hinh LSTM ...")
    model = build_lstm_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=2,
                               restore_best_weights=True)
    history = model.fit(
        Xtr, np.array(y_train),
        validation_data=(Xte, np.array(y_test)),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2,
    )

    # Xac suat du doan (dung cho ROC + ma tran nham lan)
    y_prob = model.predict(Xte).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    results["LSTM"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(classification_report(y_test, y_pred, target_names=["True", "Fake"]))

    # --- Luu model + tokenizer cho app.py ---
    model.save("lstm_fake_news.keras")
    import pickle
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # --- Bieu do loss / accuracy ---
    plot_history(history, OUT_DIR / "loss-curve.png")

    # --- Ma tran nham lan ---
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, OUT_DIR / "confusion-matrix.png")

    # --- Duong cong ROC ---
    plot_roc(y_test, y_prob, OUT_DIR / "roc-curve.png")
    return history


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="test loss")
    axes[0].set_title("Loss theo epochs"); axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="test acc")
    axes[1].set_title("Accuracy theo epochs"); axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Da luu {path}")


def plot_confusion(cm, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["True", "Fake"],
                yticklabels=["True", "Fake"])
    plt.xlabel("Du doan"); plt.ylabel("Thuc te")
    plt.title("Confusion Matrix - LSTM")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Da luu {path}")


def plot_roc(y_test, y_prob, path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"LSTM (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ngau nhien")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Da luu {path}")


# =========================
# 5. Tong hop ket qua
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

    run_traditional_models(X_train, X_test, y_train, y_test)
    run_lstm(X_train, X_test, y_train, y_test)
    summarize()
