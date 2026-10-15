"""포스터(poster.html)에 넣을 그래프 3개와 QR 코드 2개를 SVG로 만들어 끼워 넣는 스크립트.

저장소 맨 위 폴더에서 실행한다: python3 code/poster_figs.py
check.py와 같은 전처리·분할·모델을 사용하므로 그래프의 값은 check.py 수치와 같다.
포스터를 만들 때만 사용하는 도구라 matplotlib·segno·xgboost가 필요하다(앱 실행에는 필요 없음).
  pip install matplotlib segno xgboost

poster.html의 <!-- FIG:이름 --> 과 <!-- /FIG:이름 --> 사이를 새 SVG로 바꾼다.
  FIG:hist     2. 데이터  분포 그래프
  FIG:bars     4. 결과    평가지표 비교 막대 그래프(기준선 + 후보 모델 전부, 선택 모델만 강조 색)
  FIG:result   4. 결과    결과 그래프(분류: 혼동행렬 히트맵)
  FIG:qr_repo  제목띠     저장소 QR
  FIG:qr_app   발치       앱 QR

[다른 샘플에서 이 파일을 쓸 때]
  · 공통 부분(설정·svg_of·qr_svg·inject)은 그대로 둔다.
  · "이 샘플의 데이터와 그래프" 부분만 바꾼다. 그래프 이름(FIG:이름) 다섯 개는 그대로 사용한다.
  · 그래프 폭은 COL_W(단 폭)로 고정한다. 폭을 맞춰야 그래프 글자 20pt가 포스터에서도 20pt로 보인다.
"""
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 끄기
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import segno

# ───────────── 공통 설정(다섯 샘플이 같다) ─────────────
ROOT = Path(__file__).resolve().parent.parent
POSTER = ROOT / "poster.html"
ACCENT = "#1f3d7a"      # poster.html의 --accent와 같은 남색
GRAY = "#9aa0aa"        # 선택하지 않은 모델·나머지 점
DARK = "#4a4f57"        # 기준선
# 단 폭(인치) = (594mm - 좌우 여백 2×26mm - 단 사이 22mm) / 2 = 260mm. poster.html의 --pad-x·--gap을 바꾸면 다시 계산한다
COL_W = 260 / 25.4

plt.rcParams.update({
    "svg.fonttype": "none",          # 글자를 글자로 남겨 브라우저 글꼴로 그린다
    "font.family": ["Apple SD Gothic Neo", "Noto Sans KR", "sans-serif"],
    "font.size": 20,                 # 포스터의 표·그림 설명 20pt와 맞춘다
    "axes.edgecolor": "#000", "axes.labelcolor": "#000",
    "xtick.color": "#000", "ytick.color": "#000",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "svg.hashsalt": "poster",        # 다시 만들어도 같은 SVG가 나오게 한다
})


def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]                       # XML 머리말·DOCTYPE 제거
    s = re.sub(r'\swidth="[^"]+"\sheight="[^"]+"', "", s, count=1)  # 크기는 CSS가 정한다
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    return s


def qr_svg(url):
    qr = segno.make(url, error="m")
    n = qr.symbol_size(scale=1, border=2)[0]
    buf = io.BytesIO()
    qr.save(buf, kind="svg", xmldecl=False, svgns=True, scale=1, border=2, dark="#000")
    s = buf.getvalue().decode()
    s = re.sub(r'\swidth="\d+"\sheight="\d+"', f' viewBox="0 0 {n} {n}" shape-rendering="crispEdges"', s, count=1)
    return s.replace("<svg ", f'<svg role="img" aria-label="{url}" ', 1)


def inject(figs):
    html = POSTER.read_text(encoding="utf-8")
    for name, svg in figs.items():
        pat = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"poster.html에 FIG:{name} 표시가 없습니다")
        html = pat.sub(lambda m: m.group(1) + svg + m.group(2), html)
    POSTER.write_text(html, encoding="utf-8")


def bar_chart(names, values, pick, base, fmt, xlabel, height=4.6):
    """평가지표 비교 가로 막대. pick=선택 모델 이름, base=기준선 이름."""
    fig, ax = plt.subplots(figsize=(COL_W, height))
    y = np.arange(len(names))[::-1]
    colors = [ACCENT if n == pick else DARK if n == base else GRAY for n in names]
    ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        if n == pick:
            t.set_color(ACCENT)
            t.set_fontweight("bold")
    vmax = max(values)
    for yi, v, n in zip(y, values, names):
        ax.text(v + vmax * 0.012, yi, fmt(v), va="center", fontsize=20,
                color=ACCENT if n == pick else "#000", fontweight="bold" if n == pick else "normal")
    ax.set_xlim(0, vmax * 1.16)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 or t == 0 else f"{t:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return svg_of(fig)


# ───────────── 이 샘플의 데이터와 그래프(샘플마다 바꾼다) ─────────────
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA = ROOT / "data" / "secom.csv"
REPO_URL = "https://github.com/greatsong/ds-sample-4-semiconductor"
APP_URL = "https://greatsong.github.io/ds-sample-4-semiconductor/"

# check.py와 같은 데이터 준비: 상수 열과 빈칸이 절반을 넘는 열을 빼고 446개를 남긴다
df = pd.read_csv(DATA)
sensors = [c for c in df.columns if c.startswith("sensor_")]
nunique = df[sensors].nunique(dropna=True)
rest = [c for c in sensors if nunique[c] > 1]
miss_rest = df[rest].isna().mean()
keep = [c for c in rest if miss_rest[c] <= 0.5]
y = df["fail"]
X_tr, X_te, y_tr, y_te = train_test_split(df[keep], y, test_size=0.2, random_state=42, stratify=y)

# 그래프 1(FIG:hist): 왼쪽 판정별 웨이퍼 수, 오른쪽 센서 590개의 열별 빈칸 비율
counts = y.value_counts()
ratio = df[sensors].isna().mean()
cats = ["0%", "0~5%", "5~50%", "50% 초과"]
ncol = pd.cut(ratio, [-0.001, 0, 0.05, 0.5, 1.0], labels=cats).value_counts().reindex(cats)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.1), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.35})
a1.bar(["양품", "불량"], [counts[0], counts[1]], color=[GRAY, ACCENT], width=0.6)
for i, v in enumerate([counts[0], counts[1]]):
    a1.text(i, v + 25, f"{v:,}", ha="center", va="bottom", fontsize=20,
            color=ACCENT if i == 1 else "#000", fontweight="bold" if i == 1 else "normal")
a1.set_ylim(0, counts[0] * 1.2)
a1.set_ylabel("웨이퍼 수(장)")
a1.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}"))
a1.tick_params(axis="x", length=0)
colors = [ACCENT if c == "50% 초과" else GRAY for c in cats]
a2.bar(cats, ncol.values, color=colors, width=0.6)
for i, v in enumerate(ncol.values):
    a2.text(i, v + 8, f"{v}", ha="center", va="bottom", fontsize=20,
            color=ACCENT if cats[i] == "50% 초과" else "#000", fontweight="bold" if cats[i] == "50% 초과" else "normal")
a2.set_ylim(0, ncol.max() * 1.2)
a2.set_ylabel("센서 열 수(개)")
a2.set_xlabel("열별 빈칸 비율")
a2.tick_params(axis="x", length=0)
svg_hist = svg_of(fig)

# 그래프 2(FIG:bars): 기준선과 후보 모델 다섯 개의 정확도·재현율·F1(check.py 후보 모델 비교와 같은 모델)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
PICK, BASE = "로지스틱(선택)", "기준선(전부 양품)"
candidates = [
    (PICK, LogisticRegression(max_iter=3000, class_weight="balanced")),
    ("k-NN(k=5)", KNeighborsClassifier(n_neighbors=5)),
    ("결정 트리(깊이 4)", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)),
    ("랜덤 포레스트", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)),
    ("XGBoost", XGBClassifier(scale_pos_weight=spw, random_state=0)),
]
base_pred = np.zeros(len(y_te), dtype=int)
rows = [(BASE, base_pred)]
for name, clf in candidates:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf).fit(X_tr, y_tr)
    rows.append((name, model.predict(X_te)))
    if name == PICK:
        pick_pred = rows[-1][1]
names = [n for n, _ in rows]
metrics = [("정확도", accuracy_score), ("재현율", lambda a, b: recall_score(a, b, zero_division=0)),
           ("F1", lambda a, b: f1_score(a, b, zero_division=0))]
table = {m: [f(y_te, p) for _, p in rows] for m, f in metrics}

# 지표별로 묶은 가로 막대 세 묶음(모델 이름은 왼쪽 묶음에만)
fig, axes = plt.subplots(1, 3, figsize=(COL_W, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
yy = np.arange(len(names))[::-1]
colors = [ACCENT if n == PICK else DARK if n == BASE else GRAY for n in names]
for ax, (m, _) in zip(axes, metrics):
    vals = table[m]
    ax.barh(yy, vals, color=colors, height=0.66)
    for yi, v, n in zip(yy, vals, names):
        ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=20,
                color=ACCENT if n == PICK else "#000", fontweight="bold" if n == PICK else "normal")
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_title(m, fontsize=22, fontweight="bold", loc="left")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(yy)
axes[0].set_yticklabels(names)
for t, n in zip(axes[0].get_yticklabels(), names):
    if n == PICK:
        t.set_color(ACCENT)
        t.set_fontweight("bold")
svg_bars = svg_of(fig)

# 그래프 3(FIG:result): 로지스틱 회귀(가중치 켬)의 혼동행렬
cm = confusion_matrix(y_te, pick_pred, labels=[0, 1])
cmap = LinearSegmentedColormap.from_list("accent", ["#ffffff", ACCENT])
fig, ax = plt.subplots(figsize=(COL_W, 3.5))
ax.imshow(np.log1p(cm), cmap=cmap, vmin=0, vmax=np.log1p(cm.max()), aspect="auto")
labels = [["양품을 양품으로", "양품을 불량으로"], ["불량을 양품으로(놓침)", "불량을 불량으로(잡음)"]]
for i in range(2):
    for j in range(2):
        dark = np.log1p(cm[i, j]) > np.log1p(cm.max()) * 0.6
        c = "#fff" if dark else "#000"
        ax.text(j, i - 0.1, f"{cm[i, j]}장", ha="center", va="center", fontsize=40, fontweight="bold", color=c)
        ax.text(j, i + 0.25, labels[i][j], ha="center", va="center", fontsize=20, color=c)
ax.set_xticks([0, 1])
ax.set_xticklabels(["예측 양품", "예측 불량"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["실제 양품", "실제 불량"])
ax.xaxis.tick_top()
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([0.5], minor=True)
ax.set_yticks([0.5], minor=True)
ax.grid(which="minor", color="#fff", linewidth=4)
ax.tick_params(which="minor", length=0)
svg_result = svg_of(fig)

inject({"hist": svg_hist, "bars": svg_bars, "result": svg_result,
        "qr_repo": qr_svg(REPO_URL), "qr_app": qr_svg(APP_URL)})
for m, _ in metrics:
    print(m + ":", ", ".join(f"{n} {v:.3f}" for n, v in zip(names, table[m])))
print("혼동행렬(로지스틱 · 가중치 켬):", cm.tolist())
print(f"poster.html 갱신: 그래프 3개, QR 2개 (시험 {len(y_te)}장, 불량 {int(y_te.sum())}장, 남긴 센서 {len(keep)}개)")
