# 센서 590개의 측정값으로 불량 웨이퍼를 미리 가려낼 수 있을까 (SECOM 반도체 공정 1,567장)
# 실행: 저장소 맨 위 폴더에서  streamlit run code/main.py
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/secom.csv"
LABELS = ["양품", "불량"]


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    sensors = [c for c in df.columns if c.startswith("sensor_")]
    nunique = df[sensors].nunique(dropna=True)
    constant = [c for c in sensors if nunique[c] <= 1]
    rest = [c for c in sensors if c not in set(constant)]
    miss = df[rest].isna().mean()
    half = [c for c in rest if miss[c] > 0.5]
    keep = [c for c in rest if miss[c] <= 0.5]
    return df, sensors, constant, half, keep


def logit(weight):
    """빈칸을 학습 데이터의 중앙값으로 채우고, 표준화한 뒤 로지스틱 회귀를 학습한다."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced" if weight else None),
    )


def scores(y_true, pred):
    return {
        "정확도": accuracy_score(y_true, pred),
        "재현율": recall_score(y_true, pred, zero_division=0),
        "정밀도": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "cm": confusion_matrix(y_true, pred, labels=[0, 1]),
    }


@st.cache_resource
def train(_df, keep, time_order=False):
    X = _df[keep]
    y = _df["fail"]
    if time_order:
        cut = int(len(_df) * 0.8)
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    res = {
        "n_test": len(y_te),
        "n_fail": int(y_te.sum()),
        "기준선": scores(y_te, pd.Series(0, index=y_te.index)),
    }
    for weight in [False, True]:
        model = logit(weight).fit(X_tr, y_tr)
        res["켬" if weight else "끔"] = scores(y_te, model.predict(X_te))
    res["data"] = (X_tr, X_te, y_tr, y_te)
    return res


def heatmap(cm, title):
    table = pd.DataFrame(cm, index=["실제 " + c for c in LABELS], columns=["예측 " + c for c in LABELS])
    fig = px.imshow(table, text_auto=True, color_continuous_scale="Blues", aspect="auto", title=title)
    fig.update_layout(coloraxis_showscale=False, height=330)
    return fig


def count_text(s):
    (tn, fp), (fn, tp) = s["cm"]
    return tn, fp, fn, tp


st.set_page_config(page_title="센서로 불량 웨이퍼 가려내기", page_icon="🔬", layout="wide")
df, sensors, constant, half, keep = load()
res = train(df, keep)

st.title("🔬 센서 590개로 불량 웨이퍼를 미리 가려낼 수 있을까")
st.caption("SECOM 반도체 공정 데이터 1,567장 · 분류(로지스틱 회귀) · 당곡고 데이터 과학 팀 프로젝트 샘플")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 질문", "② 데이터 보기", "③ 기준선", "④ 모델 결과", "⑤ 해석·한계"])

with tab1:
    st.subheader("우리 팀의 질문")
    st.markdown("> **센서 590개의 측정값으로 불량 웨이퍼를 미리 가려낼 수 있는가**")
    st.write("반도체 공장에는 센서가 수백 개 달려 있습니다. 그 숫자만으로 불량을 미리 알 수 있는지 알아보았습니다.")
    st.write

with tab2:
    counts = df["fail"].value_counts().reindex([0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("웨이퍼", f"{len(df):,}장", "2008년 7월 19일 ~ 10월 17일", delta_color="off")
    c2.metric("불량", f"{counts[1]}장", f"전체의 {df['fail'].mean():.1%}", delta_color="off")
    c3.metric("센서 빈칸", f"{int(df[sensors].isna().sum().sum()):,}칸", f"전체 {df[sensors].size:,}칸 중", delta_color="off")

    left, right = st.columns(2)
    with left:
        fig = px.bar(x=LABELS, y=counts.values, text=counts.values, color=LABELS,
                     color_discrete_sequence=["#9aa3b5", "#e0663a"],
                     labels={"x": "판정", "y": "웨이퍼 수"}, title="양품과 불량의 개수")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        ratio = (df[sensors].isna().mean() * 100).rename("빈칸 비율(%)").reset_index()
        fig = px.histogram(ratio, x="빈칸 비율(%)", nbins=50, title="센서 590개의 빈칸 비율 분포",
                           labels={"count": "센서 수"})
        fig.add_vline(x=50, line_dash="dash", line_color="#e0663a", annotation_text="50%")
        fig.update_yaxes(title="센서 수")
        st.plotly_chart(fig, width="stretch")

    st.subheader("전처리 판단")
    st.table(pd.DataFrame({
        "판단": ["모든 행이 같은 값인 열(상수 열)", "빈칸이 절반을 넘는 열", "남긴 센서 열"],
        "열 수": [len(constant), len(half), len(keep)],
        "처리": ["제거: 값이 바뀌지 않아 정보가 없음",
                 "제거: 절반 이상을 채워 넣으면 만든 값이 됨",
                 f"남은 빈칸 {int(df[keep].isna().sum().sum()):,}칸은 학습 데이터의 중앙값으로 채움"],
    }).set_index("판단"))
    with st.expander("원본 데이터 일부 보기"):
        st.dataframe(df.iloc[:30, :12], width="stretch")

with tab3:
    base = res["기준선"]
    st.subheader("기준선: 전부 양품이라고 찍기")
    st.write(f"시험 데이터 {res['n_test']}장 가운데 불량은 {res['n_fail']}장뿐입니다. "
             "센서를 보지 않고 모두 양품이라고 답해도 맞히는 비율이 기준선입니다.")
    c1, c2 = st.columns(2)
    c1.metric("기준선 정확도", f"{base['정확도']:.3f}")
    c2.metric("기준선 재현율", f"{base['재현율']:.3f}", f"불량 {res['n_fail']}장 중 0장", delta_color="off")
    st.write("정확도는 93%가 넘지만 불량은 하나도 잡지 못합니다. 그래서 정확도와 함께 재현율을 봐야 합니다.")

with tab4:
    weight = st.toggle("가중치 켬(불량에 더 큰 비중을 두고 학습)", value=True)
    key = "켬" if weight else "끔"
    s = res[key]
    tn, fp, fn, tp = count_text(s)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("정확도", f"{s['정확도']:.3f}", f"기준선 {res['기준선']['정확도']:.3f}", delta_color="off")
    c2.metric("재현율", f"{s['재현율']:.3f}", f"불량 {fn + tp}장 중 {tp}장", delta_color="off")
    c3.metric("정밀도", f"{s['정밀도']:.3f}", f"불량 판정 {fp + tp}장 중 {tp}장", delta_color="off")
    c4.metric("F1", f"{s['F1']:.3f}")
    st.info(f"가중치 {key}: 실제 불량 {fn + tp}장 가운데 {tp}장을 잡고 {fn}장을 놓쳤습니다. "
            f"양품 {tn + fp}장 가운데 {fp}장은 불량으로 잘못 판정했습니다.")

    left, right = st.columns(2)
    left.plotly_chart(heatmap(res["끔"]["cm"], "가중치 끔"), width="stretch")
    right.plotly_chart(heatmap(res["켬"]["cm"], "가중치 켬"), width="stretch")

    table = pd.DataFrame({
        name: [res[k]["정확도"], res[k]["재현율"], res[k]["정밀도"], res[k]["F1"]]
        for name, k in [("기준선(전부 양품)", "기준선"), ("가중치 끔", "끔"), ("가중치 켬", "켬")]
    }, index=["정확도", "재현율", "정밀도", "F1"]).T.rename_axis("방법")
    st.table(table.map(lambda v: f"{v:.3f}"))
    st.write("정확도: 전체 웨이퍼 가운데 판정이 맞은 비율")
    st.write("재현율: 실제 불량 가운데 모델이 불량이라고 잡아낸 비율")
    st.write("정밀도: 모델이 불량이라고 판정한 것 가운데 실제 불량인 비율")
    st.write("F1: 정밀도와 재현율을 함께 반영한 값(조화평균). 둘 중 하나라도 낮으면 낮아집니다")

    st.subheader("후보 모델 비교")
    st.caption("같은 전처리(중앙값으로 채우기 → 표준화) · 같은 무작위 분할(시험 314장, 불량 21장). "
               "로컬에서 계산한 값: XGBoost는 브라우저 앱에서 실행할 수 없어서 code/check.py로 계산한 값을 그대로 적었습니다.")
    st.table(pd.DataFrame(
        [
            ["기준선(전부 양품)", "0.933", "0.000", "0.000", "0.000", "21장 중 0장", "0장"],
            ["로지스틱 회귀(가중치 켬, 최종)", "0.838", "0.190", "0.105", "0.136", "21장 중 4장", "34장"],
            ["k-NN(k=5, 가중치 없음)", "0.933", "0.048", "0.500", "0.087", "21장 중 1장", "1장"],
            ["결정 트리(깊이 4, 가중치 켬)", "0.656", "0.381", "0.078", "0.129", "21장 중 8장", "95장"],
            ["랜덤 포레스트(300그루, 가중치 켬)", "0.933", "0.000", "0.000", "0.000", "21장 중 0장", "0장"],
            ["XGBoost(기본값, 가중치 켬)", "0.930", "0.000", "0.000", "0.000", "21장 중 0장", "1장"],
        ],
        columns=["후보 모델", "정확도", "재현율", "정밀도", "F1", "잡은 불량", "불량으로 판정한 양품"],
    ).set_index("후보 모델"))
    st.write("어느 모델도 정확도로 기준선을 넘지 못했습니다. 결정 트리는 불량을 가장 많이 잡았지만(8장) 양품 95장을 불량으로 판정했습니다. "
             "그래서 수업에서 배웠고 계수로 결과를 설명할 수 있는 로지스틱 회귀(가중치 켬)를 최종 모델로 골랐습니다.")

    with st.expander("랜덤 포레스트를 앱에서 직접 돌려 보기(1분쯤 걸릴 수 있음)"):
        if st.button("랜덤 포레스트 실행(가중치 켬, 300그루)"):
            X_tr, X_te, y_tr, y_te = res["data"]
            forest = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
            ).fit(X_tr, y_tr)
            f = scores(y_te, forest.predict(X_te))
            ftn, ffp, ffn, ftp = count_text(f)
            st.write(f"정확도 {f['정확도']:.3f} · 재현율 {f['재현율']:.3f} (불량 {ffn + ftp}장 중 {ftp}장)")
            st.write("정확도는 기준선과 같은 값이 나오지만, 불량을 잡은 수를 함께 확인해야 합니다.")

with tab5:
    on, off, base = res["켬"], res["끔"], res["기준선"]
    st.subheader("해석")
    st.write(f"두 모델 모두 정확도가 기준선 {base['정확도']:.3f}보다 낮습니다(가중치 끔 {off['정확도']:.3f}, 켬 {on['정확도']:.3f}). "
             f"재현율도 가중치를 켰을 때 {on['재현율']:.3f}에 그칩니다.")
    st.write("이 데이터로는 로지스틱 회귀로 불량을 미리 가려내기 어렵다는 것을 확인했습니다.")
    st.write("가중치를 켜면 불량을 한 장 더 잡지만, 양품을 불량으로 잘못 판정한 수도 함께 늘어납니다.")

    st.subheader("반론 확인: 나중 웨이퍼로 시험하면")
    t = train(df, keep, time_order=True)
    st.write(f"앞 80%로 학습하고 뒤 20%({t['n_test']}장, 불량 {t['n_fail']}장)로 시험했습니다.")
    st.table(pd.DataFrame({
        name: [t[k]["정확도"], t[k]["재현율"], t[k]["정밀도"], t[k]["F1"]]
        for name, k in [("기준선(전부 양품)", "기준선"), ("가중치 끔", "끔"), ("가중치 켬", "켬")]
    }, index=["정확도", "재현율", "정밀도", "F1"]).T.rename_axis("방법").map(lambda v: f"{v:.3f}"))
    st.write("시간 순으로 나눠도 두 모델 모두 기준선을 넘지 못했습니다.")

    st.subheader("한계")
    tn, fp, fn, tp = count_text(on)
    st.markdown(
        "- 센서 이름이 공개되지 않아 어느 공정 때문에 불량이 생기는지 말할 수 없습니다.\n"
        f"- 불량은 {int(df['fail'].sum())}장뿐이고, 시험 데이터의 불량은 {res['n_fail']}장입니다. 한 장 차이로 재현율이 크게 바뀝니다.\n"
        "- 2008년 한 공장의 약 석 달치 데이터입니다.\n"
        f"- **이 모델로 실제 라인의 불량을 걸러내면 양품 {fp}장을 불량으로 버리고 불량 {fn}장은 그대로 내보냅니다.**"
    )
