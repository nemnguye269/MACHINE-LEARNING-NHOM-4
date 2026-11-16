# -*- coding: utf-8 -*-
"""
Chay cac kich ban thi nghiem E3-E5 voi mo hinh LSTM (dong bo voi bao cao):
  E3: tu dien 20.000 tu
  E4: do dai chuoi 128 / 400
  E5: bo Dropout + Early Stopping (chay du 10 epochs)
Ket qua: results/kich-ban.json
"""

import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

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
    return (X_train.apply(clean_text), X_test.apply(clean_text),
            np.array(y_train), np.array(y_test))


def run_lstm(X_train, X_test, y_train, y_test, vocab=10000, maxlen=256,
             use_dropout=True, use_earlystop=True, epochs=10, name="E"):
    tok = Tokenizer(num_words=vocab, oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    Xtr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=maxlen)
    Xte = pad_sequences(tok.texts_to_sequences(X_test), maxlen=maxlen)

    layers = [Embedding(vocab, 64), LSTM(64)]
    if use_dropout:
        layers.append(Dropout(0.5))
    layers += [Dense(32, activation="relu"), Dense(1, activation="sigmoid")]
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = ([EarlyStopping(monitor="val_loss", patience=2,
                                restore_best_weights=True)]
                 if use_earlystop else [])
    hist = model.fit(Xtr, y_train, validation_data=(Xte, y_test),
                     epochs=epochs, batch_size=64, callbacks=callbacks, verbose=2)
    prob = model.predict(Xte, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    n_epochs = len(hist.history["loss"])
    print(f"[{name}] vocab={vocab} maxlen={maxlen} dropout={use_dropout} "
          f"earlystop={use_earlystop} -> acc={acc:.4f} ({n_epochs} epochs)")
    return {"accuracy": round(acc, 4), "epochs": n_epochs}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train={len(X_train)}, Test={len(X_test)}")
    results = {}
    results["E3_vocab20000"] = run_lstm(X_train, X_test, y_train, y_test,
                                        vocab=20000, name="E3")
    results["E4_maxlen128"] = run_lstm(X_train, X_test, y_train, y_test,
                                       maxlen=128, name="E4a")
    results["E4_maxlen400"] = run_lstm(X_train, X_test, y_train, y_test,
                                       maxlen=400, name="E4b")
    results["E5_noDropout_noEarlyStop"] = run_lstm(
        X_train, X_test, y_train, y_test,
        use_dropout=False, use_earlystop=False, name="E5")
    with open(OUT / "kich-ban.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Da luu results/kich-ban.json")
