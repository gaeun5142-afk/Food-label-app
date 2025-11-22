import streamlit as st
import json
from server import MODEL_NAME

from server import *
from server import load_law_texts

# ------------------------------
# 🔵 Streamlit 페이지 설정
# ------------------------------

st.set_page_config(page_title="바른식품표시", layout="wide")


# ------------------------------
# 🔵 메뉴 구성
# ------------------------------

menu = st.sidebar.radio(
    "메뉴 선택",
    ["홈", "기준 데이터 생성", "오류 자동체크", "법령 보기"],
)


# ==========================================================
# 홈
# ==========================================================
if menu == "홈":
    st.title("🍱 바른식품표시 플랫폼")
    st.write("식품 표시사항 자동 생성 · 자동 검증 플랫폼입니다.")


# ==========================================================
# 기준 데이터 생성
# ==========================================================
elif menu == "기준 데이터 생성":
    st.title("📘 기준 데이터 생성")

    excel_file = st.file_uploader("배합비 엑셀 업로드", type=["xlsx", "xls"])
    raw_images = st.file_uploader(
        "원재료 사진 업로드 (여러 개 가능)", type=["png", "jpg"], accept_multiple_files=True
    )

    if st.button("기준 데이터 생성"):
        if excel_file:
            result = create_standard(
                excel_file,
                raw_images,
                prompt=PROMPT_CREATE_STANDARD,
                law_text=ALL_LAW_TEXT,
            )
            st.success("기준 데이터 생성 완료!")
            st.json(result)
        else:
            st.error("엑셀 파일이 필요합니다.")


# ==========================================================
# 오류 자동체크
# ==========================================================
elif menu == "오류 자동체크":
    st.title("🟥 오류 자동 검증")

    design_file = st.file_uploader("디자인 파일 업로드", type=["png", "jpg", "pdf"])
    standard_json = st.text_area("기준 데이터(JSON)", "")

    if st.button("오류 자동검증 실행"):
        if not design_file:
            st.error("디자인 파일을 업로드하세요.")
        else:
            try:
                standard_data = json.loads(standard_json)
                result = verify_design(
                    design_file,
                    standard_data,
                    prompt=PROMPT_VERIFY_DESIGN,
                    law_text=ALL_LAW_TEXT,
                )
                st.success("검증 완료!")
                st.json(result)
            except Exception as e:
                st.error(f"기준 데이터(JSON) 파싱 실패: {e}")


# ==========================================================
# 법령 보기
# ==========================================================
elif menu == "법령 보기":
    st.title("📚 식품 관련 법령")
    st.text(ALL_LAW_TEXT[:15000])




