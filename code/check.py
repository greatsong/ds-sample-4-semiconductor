# 검증 스크립트: main.py와 같은 계산을 터미널에서 돌려 README·log·slides에 적은 모든 수치를 출력한다.
# 실행: 저장소 맨 위 폴더에서  python3 code/check.py
# 후보 모델 비교에 XGBoost가 들어가므로 이 파일은 xgboost가 설치된 컴퓨터에서 돌린다(앱에는 넣을 수 없음).
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = Path(__file__).resolve().parent.parent / "data" / "secom.csv"


def pick_columns(df):
    """정보가 없는 열(상수 열)과 빈칸이 절반을 넘는 열을 빼고 남길 센서 열을 고른다."""
    sensors = [c for c in df.columns if c.startswith("sensor_")]
    nunique = df[sensors].nunique(dropna=True)
    constant = [c for c in sensors if nunique[c] <= 1]
    rest = [c for c in sensors if c not in set(constant)]
    miss = df[rest].isna().mean()
    half = [c for c in rest if miss[c] > 0.5]
    keep = [c for c in rest if miss[c] <= 0.5]
    return sensors, constant, half, keep


def logit(weight):
    """빈칸을 학습 데이터의 중앙값으로 채우고, 표준화한 뒤 로지스틱 회귀를 학습한다."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced" if weight else None),
    )


def scores(y_true, pred):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "acc": accuracy_score(y_true, pred),
        "rec": recall_score(y_true, pred, zero_division=0),
        "prec": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "cm": cm,
    }


def show(name, s):
    (tn, fp), (fn, tp) = s["cm"]
    print(f"{name}: 정확도 {s['acc']:.3f} ({tn + tp}/{tn + fp + fn + tp}장 맞힘) · "
          f"재현율 {s['rec']:.3f} (불량 {fn + tp}장 중 {tp}장 잡음) · "
          f"정밀도 {s['prec']:.3f} (불량이라고 답한 {fp + tp}장 중 {tp}장이 실제 불량) · F1 {s['f1']:.3f}")
    print(f"    혼동행렬 [[양품→양품 {tn}, 양품→불량 {fp}], [불량→양품 {fn}, 불량→불량 {tp}]]")


def experiment(title, X_tr, X_te, y_tr, y_te):
    print(f"\n### {title}")
    print(f"학습 {len(y_tr)}장(불량 {int(y_tr.sum())}, {y_tr.mean():.1%}) · 시험 {len(y_te)}장(불량 {int(y_te.sum())}, {y_te.mean():.1%})")
    base = pd.Series(0, index=y_te.index)
    base_s = scores(y_te, base)
    show("기준선(전부 양품이라고 찍기)", base_s)
    results = {}
    for weight in [False, True]:
        model = logit(weight).fit(X_tr, y_tr)
        results[weight] = scores(y_te, model.predict(X_te))
        show(f"로지스틱 회귀 · 가중치 {'켬' if weight else '끔'}", results[weight])
    print(f"불량 한 장 차이로 바뀌는 재현율: {1 / y_te.sum():.3f}")
    print(f"기준선과 가중치 켬 모델의 정확도 차이: {results[True]['acc'] - base_s['acc']:.3f}")


df = pd.read_csv(DATA)
sensors, constant, half, keep = pick_columns(df)
print("## 데이터")
print(f"행 {len(df)} · 열 {df.shape[1]} (timestamp 1 · fail 1 · 센서 {len(sensors)})")
ts = pd.to_datetime(df["timestamp"], dayfirst=True)
print(f"측정 기간: {ts.min()} ~ {ts.max()} · 시각 순서대로 정렬되어 있음: {ts.is_monotonic_increasing}")
counts = df["fail"].value_counts()
print(f"양품 {counts[0]}장 · 불량 {counts[1]}장 (불량 비율 {df['fail'].mean():.1%})")
print(f"센서 칸 {df[sensors].size:,}개 중 빈칸 {int(df[sensors].isna().sum().sum()):,}개")

ratio = df[sensors].isna().mean()
bins = pd.cut(ratio, [-0.001, 0, 0.05, 0.5, 1.0], labels=["0%", "0~5%", "5~50%", "50% 초과"])
print("열별 빈칸 비율 분포:", bins.value_counts().reindex(["0%", "0~5%", "5~50%", "50% 초과"]).to_dict())

print("\n## 전처리 판단")
all_na = [c for c in sensors if df[c].isna().all()]
print(f"모든 칸이 빈칸인 열: {len(all_na)}개")
print(f"모든 행이 같은 값인 열(빈칸 제외, 상수 열): {len(constant)}개 → 제거")
print(f"상수 열을 뺀 뒤 빈칸이 절반을 넘는 열: {len(half)}개 → 제거 "
      f"(빈칸 비율 {df[half].isna().mean().min():.1%} ~ {df[half].isna().mean().max():.1%})")
rest_miss = df[[c for c in sensors if c not in set(constant)]].isna().mean()
print(f"(비교) 기준을 30%로 하면 빠지는 열: {int((rest_miss > 0.3).sum())}개")
print(f"남긴 센서 열 가운데 값 0이 들어 있는 열: {int((df[keep] == 0).any().sum())}개 · "
      f"최댓값이 1,000 이상인 열: {int((df[keep].max() >= 1000).sum())}개")
print(f"남긴 센서 열: {len(keep)}개 · 남은 빈칸 {int(df[keep].isna().sum().sum()):,}개 → 학습 데이터의 열별 중앙값으로 채움")

X = df[keep]
y = df["fail"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
experiment("본 분석: 무작위 분할(시험 20%, 불량 비율 유지, 분할 난수 42)", X_tr, X_te, y_tr, y_te)

print("\n## 후보 모델 비교(같은 전처리: 중앙값 채우기 → 표준화 · 같은 무작위 분할 · 시험 314장)")
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"XGBoost 가중치(학습 데이터의 양품 수 / 불량 수): {(y_tr == 0).sum()} / {(y_tr == 1).sum()} = {spw:.2f}")
candidates = [
    ("로지스틱 회귀(가중치 켬)", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ("k-NN(k=5, 가중치 없음)", KNeighborsClassifier(n_neighbors=5)),
    ("결정 트리(깊이 4, 가중치 켬)", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0)),
    ("랜덤 포레스트(300그루, 가중치 켬)", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)),
    ("XGBoost(기본값, 가중치 켬)", XGBClassifier(scale_pos_weight=spw, random_state=0)),
]
for name, clf in candidates:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf).fit(X_tr, y_tr)
    s = scores(y_te, model.predict(X_te))
    show(name, s)
    print(f"    기준선 정확도와의 차이: {s['acc'] - (y_te == 0).mean():.3f}")

cut = int(len(df) * 0.8)
print(f"\n시간 순 분할 경계: 학습 마지막 {df['timestamp'].iloc[cut - 1]} · 시험 처음 {df['timestamp'].iloc[cut]}")
experiment("반론 확인: 시간 순 분할(앞 80% 학습 · 뒤 20% 시험)",
           X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:])

print("\n## 참고: 열을 빼지 않고 센서 590개를 모두 넣은 경우(무작위 분할)")
X_tr, X_te, y_tr, y_te = train_test_split(df[sensors], y, test_size=0.2, random_state=42, stratify=y)
for weight in [False, True]:
    model = logit(weight).fit(X_tr, y_tr)
    show(f"로지스틱 회귀 · 가중치 {'켬' if weight else '끔'}", scores(y_te, model.predict(X_te)))
