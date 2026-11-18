# -*- coding: utf-8 -*-
"""Lay 2 bai mau (tin that + tin gia) tu bo du lieu de test app demo."""
import pandas as pd

true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

def make_sample(df, path):
    row = df.iloc[0]
    title = str(row["title"])
    text = str(row["text"])
    s = title + "\n\n" + text[:1400]
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("=====", path, "=====")
    print(s)
    print()

make_sample(true_df, "sample_true.txt")
make_sample(fake_df, "sample_fake.txt")
