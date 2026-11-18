# -*- coding: utf-8 -*-
"""
Giao dien demo kiem tra tin that/gia bang Streamlit (mo hinh KNN).

Cach chay:
    streamlit run app.py

Yeu cau: da chay train_and_evaluate.py (co knn_model.pkl + tfidf_vectorizer.pkl)
va build_reference.py (co knn_ref.pkl — danh sach tieu de bai trong tap huan luyen,
dung de hien thi 5 bai gan nhat).
"""

import html
import json
import pickle
import re
import string
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent

# =========================
# Cau hinh trang
# =========================
FAVICON = BASE_DIR / "assets" / "favicon.png"
st.set_page_config(
    page_title="Kiem chung tin tuc · KNN",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

K_DEFAULT = 50          # nguong canh bao (%) — trung vi, nguoi dung chinh duoc
N_NEIGHBORS = 5

N_TRAIN = 29_328
N_TEST = 9_777

# =========================
# CSS
# =========================
CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

:root{
  --paper:#FBFAF7;
  --surface:#FFFFFF;
  --line:#E5E1D8;
  --line-2:#D6D0C4;
  --ink:#16181C;
  --ink-2:#3A3E45;
  --muted:#6E727A;
  --fake:#B3261E;
  --fake-bg:#FCEFEE;
  --fake-line:#EFD2CF;
  --real:#1D6B4A;
  --real-bg:#EBF3EF;
  --real-line:#CFE3D8;
  --font:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
}

html, body, .stApp{ background:var(--paper) !important; }
.stApp, .stApp *{ font-family:var(--font) !important; }
.stApp{ color:var(--ink); }

/* khung chrome cua Streamlit: an de trang sach */
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
#MainMenu, footer, [data-testid="stHeaderActionElements"]{ display:none !important; }

[data-testid="stMainBlockContainer"]{
  max-width:1140px;
  padding:34px 28px 56px 28px;
}

/* ---------------- masthead ---------------- */
.masthead{ padding-bottom:16px; border-bottom:1px solid var(--line); }
.eyebrow{
  display:flex; align-items:center; gap:9px;
  font-size:11px; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); margin:0 0 10px;
}
.eyebrow i{ width:5px; height:5px; border-radius:50%; background:var(--ink); display:block; }
.masthead h1{
  font-size:clamp(26px,3.2vw,35px) !important; line-height:1.12 !important;
  letter-spacing:-.034em; font-weight:650 !important; margin:0 0 10px !important;
  padding:0 !important; color:var(--ink);
}
.lede{ font-size:14.5px; line-height:1.55; color:var(--ink-2); max-width:640px; margin:0; }

.facts{ display:flex; flex-wrap:wrap; gap:0; margin:16px 0 0; }
.fact{ padding:0 22px; border-left:1px solid var(--line); }
.fact:first-child{ padding-left:0; border-left:0; }
.fact b{
  display:block; font-size:19px; font-weight:640; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; color:var(--ink);
}
.fact span{ font-size:11.5px; color:var(--muted); }

/* ---------------- tabs ---------------- */
[data-testid="stTabs"] [role="tablist"]{
  gap:26px; border-bottom:1px solid var(--line); background:transparent;
}
[data-testid="stTabs"] [data-testid="stTab"]{
  background:transparent !important;
  padding:0 0 10px !important;
  font-size:13px !important; font-weight:560; letter-spacing:-.005em;
  color:var(--muted) !important;
  border-bottom:2px solid transparent;
  transition:color .16s ease, border-color .16s ease;
}
[data-testid="stTabs"] [data-testid="stTab"] p{
  font-size:13px !important; font-weight:inherit; margin:0;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover{ color:var(--ink) !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]{
  color:var(--ink) !important; border-bottom-color:var(--ink);
}
[data-testid="stTabs"] [data-testid="stTabPanel"]{ padding-top:10px; }

/* ---------------- the nhap lieu ---------------- */
.st-key-inputcard{
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:14px;
  padding:16px 18px 18px;
  box-shadow:0 1px 2px rgba(22,24,28,.04), 0 22px 44px -38px rgba(22,24,28,.45);
}
.st-key-inputcard [data-testid="stVerticalBlock"]{ gap:6px; }
.form-label{
  font-size:12px; font-weight:600; letter-spacing:.02em;
  color:var(--ink); margin:2px 0 2px;
}
[data-testid="stWidgetLabel"] p{ font-size:12.5px; font-weight:600; color:var(--ink); }

[data-testid="stTextAreaRootElement"]{
  background:var(--paper) !important;
  border:1px solid var(--line) !important;
  border-radius:10px !important;
  transition:border-color .16s ease, box-shadow .16s ease;
}
[data-testid="stTextAreaRootElement"]:focus-within{
  border-color:#A8A193 !important;
  box-shadow:0 0 0 3px rgba(22,24,28,.06);
}
[data-testid="stTextArea"] textarea{
  background:transparent !important;
  font-size:14.5px !important;
  line-height:1.62 !important;
  color:var(--ink) !important;
  padding:11px 15px !important;
}
[data-testid="stTextArea"] textarea::placeholder{ color:#9B9D9F !important; }
.counter{
  font-size:11.5px; color:var(--muted); margin:1px 0 2px;
  font-variant-numeric:tabular-nums;
}

/* ---------------- nut ---------------- */
.stButton button{
  border-radius:9px;
  border:1px solid var(--line-2);
  background:var(--surface);
  color:var(--ink);
  font-size:13.5px; font-weight:560; letter-spacing:-.01em;
  min-height:40px;
  transition:transform .16s cubic-bezier(.2,.7,.3,1), background .16s ease,
             border-color .16s ease, box-shadow .16s ease;
}
.stButton button:hover{
  transform:translateY(-1px);
  border-color:#9E9789;
  background:var(--surface);
  color:var(--ink);
  box-shadow:0 7px 16px -12px rgba(22,24,28,.5);
}
.stButton button:active{ transform:translateY(0) scale(.993); }
.stButton button:focus-visible{
  outline:2px solid var(--ink); outline-offset:2px;
}
.stButton button[kind="primary"],
[data-testid="stBaseButton-primary"]{
  background:var(--ink) !important;
  border-color:var(--ink) !important;
  color:#FBFAF7 !important;
}
.stButton button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover{
  background:#2B2E35 !important;
  border-color:#2B2E35 !important;
}

.st-key-presets button{ min-height:34px; font-size:12.5px; padding:0 10px; }

/* ---------------- slider ---------------- */
[data-testid="stSlider"]{ padding-bottom:0; margin-top:-4px; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  box-shadow:0 0 0 2px var(--paper);
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"]{
  font-size:11px; font-weight:600; font-variant-numeric:tabular-nums;
}
[data-testid="stSliderTickBar"]{
  opacity:1 !important; color:var(--muted) !important;
  font-size:10.5px !important; font-variant-numeric:tabular-nums;
}

/* ---------------- the ket qua ---------------- */
.rp{
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:14px;
  padding:20px;
  min-height:414px;
  box-shadow:0 1px 2px rgba(22,24,28,.04), 0 22px 44px -38px rgba(22,24,28,.45);
}
.rc-head{ display:flex; align-items:center; justify-content:space-between; gap:12px; }
.chip{
  display:inline-flex; align-items:center; gap:7px;
  padding:5px 11px; border-radius:99px;
  font-size:11.5px; font-weight:650; letter-spacing:.03em;
  border:1px solid transparent;
}
.chip::before{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }
.chip-fake{ background:var(--fake-bg); color:var(--fake); border-color:var(--fake-line); }
.chip-real{ background:var(--real-bg); color:var(--real); border-color:var(--real-line); }
.chip-neutral{ background:#F3F1EC; color:var(--ink-2); border-color:var(--line); }
.rc-meta{ font-size:11.5px; color:var(--muted); text-align:right; line-height:1.4; }

.rc-score{ margin-top:20px; display:flex; align-items:baseline; gap:2px; }
.rc-score .num{
  font-size:60px; line-height:.95; font-weight:640; letter-spacing:-.045em;
  font-variant-numeric:tabular-nums;
}
.rc-score .pct{ font-size:23px; font-weight:600; color:var(--muted); letter-spacing:-.02em; }
.rc-cap{ font-size:13px; color:var(--muted); margin-top:8px; }
.rc-cap b{ color:var(--ink-2); font-weight:600; }

.splitwrap{ position:relative; margin:20px 0 9px; }
.split{ display:flex; height:8px; border-radius:99px; overflow:hidden; background:var(--fake-bg); }
.split i{ display:block; height:100%; transition:width .35s ease; }
.split .s-real{ background:var(--real); }
.split .s-fake{ background:var(--fake); }
.split-mark{
  position:absolute; top:-4px; height:16px; width:2px; border-radius:2px;
  background:var(--ink); opacity:.45;
}
.split-legend{
  display:flex; justify-content:space-between;
  font-size:11px; color:var(--muted); letter-spacing:.02em;
  font-variant-numeric:tabular-nums;
}
.rc-note{
  font-size:12.5px; line-height:1.6; color:var(--muted);
  margin:16px 0 0; padding-top:14px; border-top:1px solid var(--line);
}
.rc-note b{ color:var(--ink-2); font-weight:600; }

/* ---------------- 5 bai gan nhat ---------------- */
.ev{ margin-top:16px; }
.ev-head{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; gap:10px; }
.ev-head h3{ font-size:12.5px; font-weight:600; margin:0; letter-spacing:-.005em; }
.ev-head span{ font-size:10.5px; color:var(--muted); text-align:right; }
.ev ol{ list-style:none; margin:0; padding:0; }
.ev li{
  display:grid; grid-template-columns:18px 1fr auto auto;
  gap:10px; align-items:center;
  padding:8px 0; border-bottom:1px solid #F1EEE8;
}
.ev li:last-child{ border-bottom:0; padding-bottom:0; }
.ev-rank{ font-size:11px; color:#A9A294; font-variant-numeric:tabular-nums; }
.ev-text{
  font-size:12.5px; line-height:1.45; color:var(--ink-2);
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden;
}
.ev-tag{ font-size:10px; font-weight:650; padding:2px 7px; border-radius:5px; letter-spacing:.02em; }
.tag-fake{ background:var(--fake-bg); color:var(--fake); }
.tag-real{ background:var(--real-bg); color:var(--real); }
.ev-d{ font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.ev-foot{ font-size:11px; line-height:1.55; color:var(--muted); margin:10px 0 0; }
.ev-foot b{ color:var(--ink-2); font-weight:600; }

/* ---------------- trang thai rong ---------------- */
.es-title{ font-size:15px; font-weight:620; letter-spacing:-.015em; margin:0 0 6px; }
.es-sub{ font-size:13px; line-height:1.65; color:var(--muted); margin:0; max-width:420px; }
.steps{ list-style:none; padding:0; margin:20px 0 0; display:grid; gap:14px; }
.steps li{ display:grid; grid-template-columns:24px 1fr; gap:12px; align-items:start; }
.steps .n{
  width:24px; height:24px; border-radius:50%;
  border:1px solid var(--line-2); background:var(--paper);
  font-size:11px; font-weight:600; color:var(--ink-2);
  display:flex; align-items:center; justify-content:center;
}
.steps .t{ font-size:13px; font-weight:600; letter-spacing:-.01em; margin:2px 0 3px; }
.steps .d{ font-size:12.5px; line-height:1.55; color:var(--muted); margin:0; }

/* ---------------- tab ket qua thuc nghiem ---------------- */
.exp{ padding-bottom:8px; }
.exp-h2{
  font-size:19px !important; line-height:1.3 !important; font-weight:640 !important;
  letter-spacing:-.02em; margin:0 0 6px !important; padding:0 !important;
  color:var(--ink) !important;
}
.exp .sub{ font-size:13px; line-height:1.6; color:var(--muted); margin:0 0 14px; max-width:720px; }
.exp .block{ margin-bottom:34px; }
.tbl{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
.tbl th{
  font-size:10.5px; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
  color:var(--muted); text-align:right; padding:0 0 8px;
  border-bottom:1px solid var(--line-2); white-space:nowrap;
}
.tbl th:first-child{ text-align:left; }
.tbl td{
  font-size:12.5px; color:var(--ink-2); text-align:right;
  padding:10px 0; border-bottom:1px solid #F1EEE8;
}
.tbl td:first-child{ text-align:left; color:var(--ink); font-weight:520; }
.tbl tr.hl td{ background:#FAF8F2; }
.tbl tr.hl td:first-child{ font-weight:620; }
.tbl .tbl-tag{
  font-size:9.5px; font-weight:650; letter-spacing:.03em;
  padding:1px 6px; border-radius:4px; margin-left:7px;
  background:#F1EEE6; color:var(--muted); vertical-align:1px;
}
.tbl .tbl-tag-ink{ background:var(--ink); color:#FBFAF7; }
.tbl-wrap{ overflow-x:auto; }
.exp .note{
  font-size:12px; line-height:1.65; color:var(--muted);
  margin:12px 0 0; padding-left:12px; border-left:2px solid var(--line-2);
}
.exp .note b{ color:var(--ink-2); font-weight:600; }
.exp ul.findings{
  margin:14px 0 0; padding:0; list-style:none;
  display:grid; gap:10px; max-width:720px;
}
.exp ul.findings li{
  font-size:12.5px; line-height:1.62; color:var(--ink-2);
  padding-left:16px; position:relative;
}
.exp ul.findings li::before{
  content:""; position:absolute; left:0; top:8px;
  width:5px; height:5px; border-radius:50%; background:var(--line-2);
}
.exp ul.findings b{ font-weight:620; }

/* ---------------- footer ---------------- */
.foot{
  margin-top:36px; padding-top:16px; border-top:1px solid var(--line);
  display:flex; justify-content:space-between; gap:18px; flex-wrap:wrap;
  font-size:11.5px; color:var(--muted);
}
.foot a{ color:var(--ink-2); text-decoration:none; border-bottom:1px solid var(--line-2); }
.foot a:hover{ border-bottom-color:var(--ink); }

@media (max-width:640px){
  [data-testid="stMainBlockContainer"]{ padding:26px 18px 48px; }
  .facts{ display:grid; grid-template-columns:1fr 1fr; gap:12px 14px; }
  .fact{ padding:0; border-left:0; }
  .rc-score .num{ font-size:52px; }
  .rp{ min-height:0; }
}
</style>
"""

EMPTY = """
<div class="rp">
  <div class="rc-head">
    <span class="chip chip-neutral">Chưa có kết quả</span>
    <span class="rc-meta">Sẵn sàng phân tích</span>
  </div>
  <p class="es-title" style="margin-top:20px">Kết quả sẽ hiện ở đây</p>
  <p class="es-sub">Mỗi lần phân tích trả về nhãn dự đoán, xác suất tin giả
  và 5 bài viết gần nhất để bạn tự đối chiếu.</p>
  <ol class="steps">
    <li><span class="n">1</span><div><p class="t">Dán bài viết</p><p class="d">Tiêu đề và nội dung, tối thiểu vài câu.</p></div></li>
    <li><span class="n">2</span><div><p class="t">Đo độ giống bằng cosine similarity</p>
      <p class="d">Mỗi bài thành một vector từ. Cosine similarity đo góc giữa hai vector:
      1,00 là dùng cùng bộ từ, 0 là không có từ nào chung. Bài của bạn được so với
      cả 29.328 bài đã biết.</p></div></li>
    <li><span class="n">3</span><div><p class="t">5 bài gần nhất bỏ phiếu</p>
      <p class="d">KNN lấy 5 bài có similarity cao nhất, tức distance = 1 − similarity
      nhỏ nhất; bài càng giống thì phiếu càng nặng.</p></div></li>
  </ol>
</div>
"""

FOOTER = """
<div class="foot">
  <span>Demo học phần Machine Learning — Đề tài 2, KNN trên TF-IDF, dữ liệu ISOT Fake News.</span>
  <span>Kết quả chỉ mang tính tham khảo, không thay thế kiểm chứng sự thật.</span>
</div>
"""


# =========================
# Tai mo hinh
# =========================
@st.cache_resource(show_spinner=False)
def load_assets():
    with open(BASE_DIR / "knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_resource(show_spinner=False)
def load_reference():
    path = BASE_DIR / "knn_ref.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_result(name: str):
    path = BASE_DIR / "results" / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# =========================
# Tien xu ly (giong luc huan luyen)
# =========================
NLTK_DIR = BASE_DIR / "nltk_data"
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


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}"


def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_dist(d: float) -> str:
    return f"{d:.2f}".replace(".", ",")


def analyze(text: str) -> dict:
    model, vectorizer = load_assets()
    ref = load_reference()

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    p_fake = float(model.predict_proba(X)[0][1])

    neighbors = []
    if ref is not None and X.nnz > 0:
        dist, idx = model.kneighbors(X, n_neighbors=N_NEIGHBORS)
        for d, i in zip(dist[0], idx[0]):
            row = ref[int(i)]
            neighbors.append({"title": row["title"], "label": row["label"],
                              "d": float(d)})

    return {"p_fake": p_fake, "tokens": int(X.nnz), "neighbors": neighbors,
            "has_ref": ref is not None}


# =========================
# Ve ket qua
# =========================
def render_result(res: dict, threshold_pct: int) -> str:
    p_fake = res["p_fake"]
    p_real = 1.0 - p_fake
    thr = threshold_pct / 100.0
    flagged = p_fake >= thr
    no_signal = res["tokens"] == 0

    if no_signal:
        chip = '<span class="chip chip-neutral">Không đủ tín hiệu</span>'
        num, pct = "—", ""
        cap = ("Sau tiền xử lý, bài viết không còn từ nào khớp từ điển TF-IDF "
               "(quá ngắn, hoặc chỉ gồm từ dừng và chữ số).")
        bar_real, bar_fake = 50, 50
        note = ("Không nên đọc con số xác suất ở đây: vector đặc trưng bằng 0 nên "
                "mọi khoảng cách đều như nhau. Hãy dán thêm nội dung.")
    else:
        if flagged:
            chip = '<span class="chip chip-fake">Tin giả</span>'
        else:
            chip = '<span class="chip chip-real">Tin thật</span>'
        num = fmt_pct(p_fake)
        pct = "%"
        cap = 'xác suất mô hình gán cho lớp <b>tin giả</b>'
        if not flagged:
            num = fmt_pct(p_real)
            cap = 'xác suất mô hình gán cho lớp <b>tin thật</b>'
        bar_real = round(p_real * 100)
        bar_fake = 100 - bar_real
        if flagged:
            note = (f"Bị gắn cờ vì xác suất tin giả {fmt_pct(p_fake)}% vượt ngưỡng "
                    f"{threshold_pct}%. Con số là tỷ lệ trọng số của 5 bài gần nhất "
                    f"mang nhãn giả.")
        else:
            note = (f"Không bị gắn cờ vì xác suất tin giả {fmt_pct(p_fake)}% thấp hơn "
                    f"ngưỡng {threshold_pct}%. Con số là tỷ lệ trọng số của 5 bài "
                    f"gần nhất mang nhãn thật.")

    rows = []
    for i, nb in enumerate(res["neighbors"], start=1):
        tag_cls = "tag-fake" if nb["label"] == 1 else "tag-real"
        tag_txt = "giả" if nb["label"] == 1 else "thật"
        rows.append(
            f'<li><span class="ev-rank">{i}</span>'
            f'<span class="ev-text">{html.escape(nb["title"])}</span>'
            f'<span class="ev-tag {tag_cls}">{tag_txt}</span>'
            f'<span class="ev-d">d = {fmt_dist(nb["d"])}</span></li>'
        )

    if res["neighbors"]:
        evidence = (
            '<div class="ev"><div class="ev-head"><h3>5 bài gần nhất trong tập huấn luyện</h3>'
            '<span>distance = 1 − cosine similarity</span></div><ol>' + "".join(rows) + "</ol>"
            '<p class="ev-foot"><b>d</b> càng nhỏ càng giống: 0 là trùng khít, '
            '1 là không có từ nào chung. Đây cũng chính là 5 phiếu tạo nên con số '
            'xác suất ở trên.</p></div>'
        )
    elif res["has_ref"]:
        evidence = ""
    else:
        evidence = ('<div class="ev"><div class="ev-head"><h3>5 bài gần nhất</h3></div>'
                    '<p class="rc-note" style="border-top:0;padding-top:0">Chưa có '
                    '<code>knn_ref.pkl</code>. Chạy <code>python build_reference.py</code> '
                    'để xem 5 bài gần nhất.</p></div>')

    return f"""
<div class="rp">
  <div class="rc-head">
    {chip}
    <span class="rc-meta">ngưỡng cảnh báo {threshold_pct}%<br>{res["tokens"]} đặc trưng khác 0</span>
  </div>
  <div class="rc-score"><span class="num">{num}</span><span class="pct">{pct}</span></div>
  <div class="rc-cap">{cap}</div>
  <div class="splitwrap">
    <div class="split"><i class="s-real" style="width:{bar_real}%"></i><i class="s-fake" style="width:{bar_fake}%"></i></div>
    <div class="split-mark" style="left:{threshold_pct}%"></div>
  </div>
  <div class="split-legend"><span>tin thật {bar_real}%</span><span>tin giả {bar_fake}%</span></div>
  <p class="rc-note">{note}</p>
  {evidence}
</div>
"""


# =========================
# Tab ket qua thuc nghiem
# =========================
MODEL_ORDER = ["Naive Bayes", "KNN", "Logistic Regression", "Passive Aggressive"]


def model_table(kq: dict) -> str:
    rows = []
    for name in MODEL_ORDER:
        if name not in kq:
            continue
        r = kq[name]
        hl = " class=\"hl\"" if name == "KNN" else ""
        if name == "Naive Bayes":
            label = 'Naive Bayes<span class="tbl-tag">mốc</span>'
        elif name == "KNN":
            label = 'KNN<span class="tbl-tag tbl-tag-ink">chính</span>'
        else:
            label = name
        rows.append(
            f"<tr{hl}><td>{label}</td>"
            f"<td>{fmt_num(r['accuracy'] * 100)}</td>"
            f"<td>{fmt_num(r['precision'] * 100)}</td>"
            f"<td>{fmt_num(r['recall'] * 100)}</td>"
            f"<td>{fmt_num(r['f1'] * 100)}</td>"
            f"<td>{fmt_num(r['fit_time_s'])}</td>"
            f"<td>{fmt_num(r['predict_time_s'])}</td></tr>"
        )
    return (
        '<div class="tbl-wrap"><table class="tbl">'
        "<thead><tr><th>Mô hình</th><th>Accuracy</th><th>Precision</th>"
        "<th>Recall</th><th>F1</th><th>Fit (s)</th><th>Predict (s)</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def representation_table(rows: list) -> str:
    out = []
    for r in rows:
        hl = " class=\"hl\"" if r["mo_hinh"] == "KNN" else ""
        out.append(
            f"<tr{hl}><td>{html.escape(r['cau_hinh'])}</td>"
            f"<td>{fmt_int(int(r['so_dac_trung']))}</td>"
            f"<td>{fmt_num(r['accuracy'] * 100)}</td>"
            f"<td>{fmt_num(r['f1'] * 100)}</td>"
            f"<td>{fmt_num(r['vector_time_s'], 1)}</td>"
            f"<td>{fmt_num(r['predict_time_s'], 1)}</td></tr>"
        )
    return (
        '<div class="tbl-wrap"><table class="tbl">'
        "<thead><tr><th>Cấu hình</th><th>Đặc trưng</th><th>Accuracy</th>"
        "<th>F1</th><th>Vector (s)</th><th>Predict (s)</th></tr></thead>"
        "<tbody>" + "".join(out) + "</tbody></table></div>"
    )


def render_experiments(kq: dict, bd: list | None) -> None:
    st.markdown('<div class="exp">', unsafe_allow_html=True)

    st.markdown(
        '<div class="block"><h2 class="exp-h2">Bốn mô hình trên cùng một cách chia dữ liệu</h2>'
        '<p class="sub">39.105 bài sau khi bỏ trùng (17.908 tin giả, 21.197 tin thật), '
        'chia 75/25 với random_state = 42: 29.328 bài huấn luyện, 9.777 bài kiểm tra. '
        'Cả bốn mô hình dùng chung vector TF-IDF 1–2gram, 50.000 đặc trưng.</p>',
        unsafe_allow_html=True,
    )
    if kq:
        st.markdown(model_table(kq), unsafe_allow_html=True)
        st.markdown(
            '<p class="note">Naive Bayes là mốc so sánh theo đề cương. KNN có precision '
            'cao hơn nhưng recall thấp: mô hình bỏ sót 778 tin giả, trong khi chỉ gắn nhãn '
            'giả nhầm cho 263 tin thật. Thời gian dự đoán của KNN là 183,29 s cho 9.777 bài, '
            'gấp khoảng 9.000 lần Naive Bayes, vì mỗi bài phải so với toàn bộ 29.328 bài '
            'huấn luyện.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<p class="sub">Chưa có <code>results/ket-qua.json</code>. '
                    'Chạy <code>python train_and_evaluate.py</code> để sinh số liệu.</p>',
                    unsafe_allow_html=True)

    if kq:
        st.markdown(
            '<div class="block" style="margin-top:30px"><h2 class="exp-h2">Sai ở đâu</h2>'
            '<p class="sub">Ma trận nhầm lẫn của KNN (k = 5) và đường cong độ chính xác '
            'khi tăng k. Trục hoành của ma trận: dự đoán; trục tung: nhãn thật.</p></div>',
            unsafe_allow_html=True,
        )
        img_a, img_b = st.columns(2, gap="medium")
        cm = BASE_DIR / "results" / "confusion-matrix.png"
        kc = BASE_DIR / "results" / "do-chinh-xac-theo-k.png"
        with img_a:
            if cm.exists():
                st.image(str(cm), width="stretch")
        with img_b:
            if kc.exists():
                st.image(str(kc), width="stretch")

        st.markdown(
            '<ul class="findings">'
            '<li><b>Bỏ sót nhiều hơn báo nhầm.</b> 778 tin giả được đoán là thật, gấp ba lần '
            '263 tin thật bị gắn nhãn giả. Với bài toán kiểm chứng tin, đây là kiểu sai '
            'nguy hiểm hơn: người đọc vẫn tin một bài đã bị mô hình cho qua.</li>'
            '<li><b>Vì sao KNN bỏ sót.</b> Tin giả trong ISOT được viết theo lối tin thật '
            '(trích nguồn, nêu số liệu, dùng từ thời sự), nên vector TF-IDF của chúng nằm gần '
            'cụm tin thật. KNN lấy 5 bài gần nhất nên chỉ cần 3 trong 5 bài là tin thật thì '
            'bài đang xét bị đoán là thật.</li>'
            '<li><b>Tăng k không chữa được lỗi này.</b> k = 9 cho accuracy cao nhất (89,76%) '
            'nhưng recall vẫn là điểm yếu, vì lỗi đến từ cách biểu diễn văn bản chứ không từ '
            'giá trị k. Muốn giảm bỏ sót, hạ ngưỡng cảnh báo ở tab Kiểm chứng — đánh đổi '
            'bằng nhiều báo nhầm hơn.</li>'
            '<li><b>Khoảng cách Euclid không phù hợp.</b> Khi thử metric euclidean, ba bài '
            'rỗng sau tiền xử lý (vector 0) lọt vào top-5 của mọi bài khác và kéo accuracy '
            'xuống 55,61%. Lọc ba bài đó ra thì euclidean đạt 89,00%, gần bằng cosine '
            '(89,33%), nhưng vẫn không ổn định bằng.</li>'
            '</ul>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="block" style="margin-top:34px"><h2 class="exp-h2">Bag-of-Words hay TF-IDF?</h2>'
        '<p class="sub">Cùng dữ liệu, cùng phép tiền xử lý; chỉ đổi cách biểu diễn văn bản '
        'và số đặc trưng. Bag-of-Words đếm số lần xuất hiện, TF-IDF nhân thêm trọng số nghịch '
        'với tần suất trong toàn bộ tập, nên từ hiếm và có tính phân biệt được nâng giá trị.</p>',
        unsafe_allow_html=True,
    )
    if bd:
        st.markdown(representation_table(bd), unsafe_allow_html=True)
        st.markdown(
            '<p class="note">Đọc theo cặp: 1–2gram luôn tốt hơn unigram vì bắt được cụm từ '
            'như “hillary clinton” hay “fake news”. Điểm đáng chú ý: với MultinomialNB, '
            '<b>BoW 1–2gram cho accuracy cao hơn TF-IDF</b> (96,68% so với 95,81%), dù phải '
            'dùng tới 2,8 triệu chiều thay vì 50.000 — trong bộ ISOT, sự lặp lại của tên hãng '
            'tin và cú pháp giật gân chính là tín hiệu phân loại, trong khi TF-IDF lại hạ '
            'trọng số từ hay gặp. min_df = 3 gần như không đổi accuracy vì max_features '
            'đã cắt bớt từ hiếm. Với KNN thì ngược lại: TF-IDF đạt 89,35%, cao hơn BoW '
            '(87,94%) mà dự đoán còn nhanh hơn (44 s so với 78 s) vì chỉ dùng 50.000 chiều. '
            'Đó là lý do mô hình chính và giao diện demo dùng TF-IDF.</p>',
            unsafe_allow_html=True,
        )
        biu = BASE_DIR / "results" / "so-sanh-bieu-dien.png"
        if biu.exists():
            st.image(str(biu), width="stretch")
    else:
        st.markdown('<p class="sub">Chưa có <code>results/bieu-dien.json</code>. '
                    'Chạy <code>python experiments_representation.py</code> '
                    '(khoảng 10 phút) để sinh bảng so sánh.</p>',
                    unsafe_allow_html=True)

    st.markdown(
        '<p class="note" style="margin-top:6px">Sinh lại toàn bộ số liệu: '
        '<code>python train_and_evaluate.py</code> rồi '
        '<code>python experiments_representation.py</code>. '
        'Chi tiết từng kịch bản E1–E5 nằm trong báo cáo, chương 3.</p></div>',
        unsafe_allow_html=True,
    )


# =========================
# Giao dien
# =========================
if not ((BASE_DIR / "knn_model.pkl").exists()
        and (BASE_DIR / "tfidf_vectorizer.pkl").exists()):
    st.markdown(
        '<div class="masthead"><h1>Chưa có mô hình</h1>'
        '<p class="lede">Thiếu <code>knn_model.pkl</code> hoặc '
        '<code>tfidf_vectorizer.pkl</code>. Chạy <code>python train_and_evaluate.py</code> '
        'rồi mở lại trang.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(CSS, unsafe_allow_html=True)

KQ = load_result("ket-qua.json") or {}
BD = load_result("bieu-dien.json")

knn_acc = f"{KQ['KNN']['accuracy'] * 100:.2f}".replace(".", ",") if "KNN" in KQ else "89,35"

st.markdown(
    '<div class="masthead">'
    '<p class="eyebrow"><i></i>Đề tài 2 · Phát hiện tin tức giả mạo</p>'
    '<h1>Kiểm chứng một bài tin trước khi chia sẻ</h1>'
    '<p class="lede">Dán tiêu đề và nội dung bài viết tiếng Anh. Mô hình KNN so bài viết '
    'với 29.328 bài đã gán nhãn, rồi trả về nhãn dự đoán, xác suất và 5 bài giống nhất '
    'làm căn cứ.</p>'
    '<div class="facts">'
    f'<div class="fact"><b>{knn_acc}%</b><span>độ chính xác trên 9.777 bài kiểm tra</span></div>'
    '<div class="fact"><b>k = 5</b><span>KNN, cosine similarity</span></div>'
    '<div class="fact"><b>50.000</b><span>đặc trưng TF-IDF (unigram + bigram)</span></div>'
    '<div class="fact"><b>39.105</b><span>bài huấn luyện từ bộ ISOT</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)


def load_preset(filename: str):
    def _cb():
        path = BASE_DIR / filename
        if path.exists():
            st.session_state["article"] = path.read_text(encoding="utf-8").strip()
            st.session_state["autorun"] = True
    return _cb


if "res" not in st.session_state:
    st.session_state["res"] = None

tab_check, tab_exp = st.tabs(["Kiểm chứng", "Kết quả thực nghiệm"])

with tab_check:
    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        with st.container(key="inputcard"):
            st.markdown('<p class="form-label">Bài viết cần kiểm tra</p>',
                        unsafe_allow_html=True)
            text = st.text_area(
                "Nội dung bài viết",
                key="article",
                height=118,
                label_visibility="collapsed",
                placeholder="Dán tiêu đề và nội dung bài viết (tiếng Anh) vào đây…",
            )

            words = len(text.split())
            chars = len(text)
            st.markdown(
                f'<p class="counter">{fmt_int(words)} từ · {fmt_int(chars)} ký tự</p>',
                unsafe_allow_html=True,
            )

            with st.container(key="presets"):
                p1, p2, p3 = st.columns([1.15, 1.15, 0.7], gap="small")
                p1.button("Tin thật mẫu", on_click=load_preset("sample_true.txt"),
                          width="stretch")
                p2.button("Tin giả mẫu", on_click=load_preset("sample_fake.txt"),
                          width="stretch")
                p3.button("Xoá", on_click=lambda: st.session_state.update(
                    article="", res=None), width="stretch")

            threshold = st.slider("Ngưỡng cảnh báo tin giả", 30, 70, K_DEFAULT, 1,
                                  format="%d%%")

            run = st.button("Phân tích bài viết", type="primary",
                            width="stretch")

    autorun = st.session_state.pop("autorun", False)
    if (run or autorun):
        if not text.strip():
            with right:
                st.markdown(
                    '<div class="rp"><div class="rc-head"><span class="chip chip-neutral">'
                    'Chưa có nội dung</span></div><p class="es-title" style="margin-top:22px">'
                    'Bạn chưa dán bài viết nào</p><p class="es-sub">Dán nội dung ở cột bên trái, '
                    'hoặc bấm “Tin thật mẫu” / “Tin giả mẫu” để thử ngay.</p></div>',
                    unsafe_allow_html=True,
                )
            st.session_state["res"] = None
        else:
            with st.spinner("Đang so khớp 29.328 bài huấn luyện…"):
                st.session_state["res"] = analyze(text)

    with right:
        res = st.session_state.get("res")
        st.markdown(render_result(res, threshold) if res else EMPTY,
                    unsafe_allow_html=True)

with tab_exp:
    render_experiments(KQ, BD)

st.markdown(FOOTER, unsafe_allow_html=True)

