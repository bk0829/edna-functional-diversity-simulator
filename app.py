import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="eDNA Functional Diversity Simulator",
    page_icon="🌊",
    layout="wide",
)


# 종별 기능적 특성
# 순서: 식성, 내성도, 서식층, 유수성, 소형종 여부
TRAITS = {
    "피라미": ("잡식성", "중간종", "부유성", "유수성", False),
    "갈겨니": ("잡식성", "민감종", "부유성", "유수성", True),
    "버들치": ("잡식성", "민감종", "부유성", "유수성", True),
    "참붕어": ("잡식성", "내성종", "부유성", "정수성", True),
    "붕어": ("잡식성", "내성종", "부유성", "정수성", False),
    "잉어": ("잡식성", "내성종", "부유성", "정수성", False),
    "누치": ("육식성", "중간종", "부유성", "유수성", False),
    "참마자": ("육식성", "중간종", "저서성", "유수성", True),
    "모래무지": ("잡식성", "중간종", "저서성", "유수성", True),
    "돌고기": ("육식성", "중간종", "저서성", "유수성", True),
    "줄몰개": ("잡식성", "내성종", "부유성", "정수성", True),
    "납지리": ("잡식성", "중간종", "부유성", "정수성", True),
    "줄납자루": ("잡식성", "중간종", "부유성", "정수성", True),
    "은어": ("초식성", "민감종", "부유성", "유수성", False),
    "메기": ("육식성", "내성종", "저서성", "정수성", False),
    "가물치": ("육식성", "내성종", "부유성", "정수성", False),
    "동자개속": ("육식성", "중간종", "저서성", "정수성", True),
    "얼룩동사리": ("육식성", "중간종", "저서성", "유수성", True),
    "검정망둑": ("잡식성", "중간종", "저서성", "정수성", True),
    "밀망둑속": ("육식성", "중간종", "저서성", "정수성", True),
    "미꾸리": ("잡식성", "내성종", "저서성", "정수성", True),
    "미꾸라지": ("잡식성", "내성종", "저서성", "정수성", True),
    "배스": ("육식성", "내성종", "부유성", "정수성", False),
    "블루길": ("잡식성", "내성종", "부유성", "정수성", True),
    "백조어": ("육식성", "중간종", "부유성", "유수성", True),
}


# 보고서에서 설정한 핵심 가중치
BASE_WEIGHTS = {
    "수온 상승": {
        "민감종": 0.3,
        "중간종": 0.9,
        "내성종": 1.2,
    },
    "용존산소 감소": {
        "민감종": 0.2,
        "중간종": 0.8,
        "내성종": 1.1,
    },
    "유량 감소": {
        "민감종": 0.8,
        "중간종": 1.0,
        "내성종": 1.0,
    },
    "집중호우": {
        "민감종": 0.9,
        "중간종": 1.0,
        "내성종": 1.0,
    },
    "부영양화": {
        "민감종": 0.2,
        "중간종": 0.8,
        "내성종": 1.3,
    },
}


def load_data(uploaded_file):
    """eDNA 엑셀을 읽고 종별 read를 합산한다."""
    df = pd.read_excel(uploaded_file)
    df.columns = [str(column).strip() for column in df.columns]

    read_column = next(
        (
            column
            for column in df.columns
            if column.lower().replace(" ", "")
            in {"totalread", "read", "reads"}
        ),
        None,
    )

    korean_name_column = next(
        (
            column
            for column in df.columns
            if column in {"한국명", "Korean name", "Unnamed: 5"}
        ),
        None,
    )

    # 원본 파일처럼 여섯 번째 열에 한국명이 있는 경우
    if korean_name_column is None and len(df.columns) >= 6:
        korean_name_column = df.columns[5]

    if read_column is None or korean_name_column is None:
        raise ValueError(
            "엑셀에서 Total read 열과 한국명 열을 찾지 못했습니다."
        )

    result = df[[korean_name_column, read_column]].copy()
    result.columns = ["한국명", "Total read"]

    result["한국명"] = result["한국명"].astype(str).str.strip()
    result["Total read"] = pd.to_numeric(
        result["Total read"], errors="coerce"
    ).fillna(0)

    result = result[result["한국명"].isin(TRAITS)]

    result = (
        result.groupby("한국명", as_index=False)["Total read"]
        .sum()
    )

    if result.empty or result["Total read"].sum() == 0:
        raise ValueError(
            "기능적 특성 표와 일치하는 어종 또는 read 값이 없습니다."
        )

    result["현재 비율"] = (
        result["Total read"] / result["Total read"].sum()
    )

    trait_df = pd.DataFrame.from_dict(
        TRAITS,
        orient="index",
        columns=[
            "식성",
            "내성도",
            "서식층",
            "유수성",
            "소형종",
        ],
    ).reset_index(names="한국명")

    return result.merge(trait_df, on="한국명", how="left")


def calculate_weight(row, scenario):
    """시나리오와 종의 특성에 따라 복합 가중치를 계산한다."""
    weight = BASE_WEIGHTS[scenario][row["내성도"]]

    if scenario == "용존산소 감소":
        if row["서식층"] == "저서성":
            weight *= 0.75

    elif scenario == "유량 감소":
        if row["서식층"] == "저서성":
            weight *= 0.80
        if row["유수성"] == "유수성":
            weight *= 0.70

    elif scenario == "집중호우":
        if row["서식층"] == "저서성":
            weight *= 0.70
        if bool(row["소형종"]):
            weight *= 0.80

    elif scenario == "부영양화":
        if (
            row["식성"] == "초식성"
            and row["내성도"] == "민감종"
        ):
            weight *= 0.70

        if (
            row["식성"] == "잡식성"
            and row["내성도"] == "내성종"
        ):
            weight *= 1.10

    return weight


def run_simulation(data, scenario):
    """가중치를 적용한 뒤 결과 비율을 100%로 정규화한다."""
    result = data.copy()

    result["가중치"] = result.apply(
        lambda row: calculate_weight(row, scenario),
        axis=1,
    )

    result["가중 비율"] = (
        result["현재 비율"] * result["가중치"]
    )

    result["예측 비율"] = (
        result["가중 비율"] / result["가중 비율"].sum()
    )

    result["변화(%p)"] = (
        result["예측 비율"] - result["현재 비율"]
    ) * 100

    result["기능군"] = (
        result["식성"] + "-" + result["내성도"]
    )

    return result


def calculate_composition(data, category, value_column):
    """기능적 특성별 구성비를 계산한다."""
    return (
        data.groupby(category)[value_column]
        .sum()
        .mul(100)
        .sort_values(ascending=False)
    )


def calculate_shannon(series):
    """기능군 분포의 균형을 보기 위한 Shannon 지수."""
    proportions = series[series > 0]

    return float(
        -(proportions * np.log(proportions)).sum()
    )


def draw_comparison(before, after, title):
    """현재와 예측 구성을 비교하는 막대그래프."""
    labels = sorted(set(before.index) | set(after.index))

    before_values = before.reindex(labels, fill_value=0)
    after_values = after.reindex(labels, fill_value=0)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        x - width / 2,
        before_values.values,
        width,
        label="현재",
    )

    ax.bar(
        x + width / 2,
        after_values.values,
        width,
        label="예측",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("eDNA read 비율(%)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    return fig


st.title("🌊 eDNA Functional Diversity Simulator")

st.caption(
    "eDNA read 기반 규칙 모델을 이용하여 "
    "기후변화 시나리오에 따른 하천 어류 군집 변화를 비교합니다."
)

uploaded_file = st.file_uploader(
    "eDNA 엑셀 파일을 업로드하세요.",
    type=["xlsx"],
)

selected_scenario = st.selectbox(
    "기후변화 시나리오를 선택하세요.",
    list(BASE_WEIGHTS.keys()),
)

with st.expander("모델 해석 시 주의사항"):
    st.write(
        "eDNA read 비율은 실제 개체수나 생체량과 동일하지 않습니다."
    )
    st.write(
        "본 결과는 문헌 기반 가중치를 적용한 상대적 변화 비교이며, "
        "확정적인 미래 예측이 아닙니다."
    )


if uploaded_file is not None:
    try:
        current_data = load_data(uploaded_file)

        simulation_result = run_simulation(
            current_data,
            selected_scenario,
        )

        st.success(
            f"{len(current_data)}개 분류군을 분석했습니다."
        )

        current_shannon = calculate_shannon(
            simulation_result.groupby("기능군")[
                "현재 비율"
            ].sum()
        )

        predicted_shannon = calculate_shannon(
            simulation_result.groupby("기능군")[
                "예측 비율"
            ].sum()
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "현재 기능군 Shannon 지수",
            f"{current_shannon:.3f}",
        )

        metric2.metric(
            "예측 기능군 Shannon 지수",
            f"{predicted_shannon:.3f}",
            f"{predicted_shannon - current_shannon:+.3f}",
        )

        metric3.metric(
            "분석 시나리오",
            selected_scenario,
        )

        tabs = st.tabs(
            [
                "구성 비교",
                "기능군",
                "어종별 변화",
                "기능적 중복성",
            ]
        )

        with tabs[0]:
            for category in [
                "식성",
                "내성도",
                "서식층",
                "유수성",
            ]:
                before = calculate_composition(
                    simulation_result,
                    category,
                    "현재 비율",
                )

                after = calculate_composition(
                    simulation_result,
                    category,
                    "예측 비율",
                )

                st.pyplot(
                    draw_comparison(
                        before,
                        after,
                        f"{category} 구성 변화",
                    )
                )

        with tabs[1]:
            before = calculate_composition(
                simulation_result,
                "기능군",
                "현재 비율",
            )

            after = calculate_composition(
                simulation_result,
                "기능군",
                "예측 비율",
            )

            st.pyplot(
                draw_comparison(
                    before,
                    after,
                    "식성 × 내성도 기능군 변화",
                )
            )

            guild_table = pd.concat(
                [
                    before.rename("현재(%)"),
                    after.rename("예측(%)"),
                ],
                axis=1,
            )

            guild_table["변화(%p)"] = (
                guild_table["예측(%)"]
                - guild_table["현재(%)"]
            )

            st.dataframe(
                guild_table.round(3),
                use_container_width=True,
            )

        with tabs[2]:
            species_table = simulation_result[
                [
                    "한국명",
                    "식성",
                    "내성도",
                    "서식층",
                    "유수성",
                    "현재 비율",
                    "예측 비율",
                    "변화(%p)",
                    "가중치",
                ]
            ].copy()

            species_table["현재 비율"] *= 100
            species_table["예측 비율"] *= 100

            species_table = species_table.sort_values(
                "변화(%p)"
            )

            st.dataframe(
                species_table.round(4),
                use_container_width=True,
            )

            csv_data = species_table.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "결과 CSV 다운로드",
                csv_data,
                "simulation_result.csv",
                "text/csv",
            )

        with tabs[3]:
            redundancy = (
                simulation_result.groupby("기능군")[
                    "한국명"
                ]
                .nunique()
                .sort_values(ascending=False)
            )

            st.dataframe(
                redundancy.rename("분류군 수"),
                use_container_width=True,
            )

            vulnerable_guilds = redundancy[
                redundancy == 1
            ].index.tolist()

            if vulnerable_guilds:
                st.warning(
                    "기능적 중복성이 1인 취약 기능군: "
                    + ", ".join(vulnerable_guilds)
                )

        most_decreased = simulation_result.loc[
            simulation_result["변화(%p)"].idxmin()
        ]

        most_increased = simulation_result.loc[
            simulation_result["변화(%p)"].idxmax()
        ]

        st.subheader("📝 자동 해석")

        st.write(
            f"**{selected_scenario}** 시나리오에서 "
            f"가장 크게 감소한 어종은 "
            f"**{most_decreased['한국명']} "
            f"({most_decreased['변화(%p)']:.2f}%p)**이며, "
            f"가장 크게 증가한 어종은 "
            f"**{most_increased['한국명']} "
            f"({most_increased['변화(%p)']:+.2f}%p)**입니다."
        )

        if predicted_shannon < current_shannon:
            st.write(
                "기능군 Shannon 지수가 감소하여 "
                "기능군 분포가 현재보다 더 편중되는 방향으로 "
                "예측되었습니다."
            )

        elif predicted_shannon > current_shannon:
            st.write(
                "기능군 Shannon 지수가 증가하여 "
                "기능군 분포가 현재보다 더 균등해지는 방향으로 "
                "예측되었습니다."
            )

        else:
            st.write(
                "기능군 Shannon 지수에는 "
                "뚜렷한 변화가 나타나지 않았습니다."
            )

    except Exception as error:
        st.error(
            f"파일을 처리하지 못했습니다: {error}"
        )

else:
    st.info(
        "먼저 eDNA 엑셀 파일을 업로드하세요."
    )
