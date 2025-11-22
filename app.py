import streamlit as st
import requests
from PIL import Image
import io

# -----------------------------
# Flask 서버 주소 적기 (중요)
# -----------------------------
FLASK_API_URL = "https://food-label-backend.onrender.com"
# 예: "https://foodchecker-backend.onrender.com"

st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 네비게이션 메뉴
# -----------------------------
menu = st.sidebar.radio("메뉴 선택", ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"])

# -----------------------------
# 1. 홈 화면
# -----------------------------
if menu == "홈":
    st.title("🏠 바른식품표시 플랫폼")
    st.write("식품 표시사항을 자동으로 변환하고, 오류를 검사하고, 식품 관련 사이트를 모아놓은 서비스입니다.")

# -----------------------------
# 2. 자동 변환 화면 (QA 자료 → 자동 라벨 생성)
# -----------------------------
elif menu == "자동 변환":
    st.title("📄 자동 변환 (QA 기반 표시사항 생성)")

    uploaded_files = st.file_uploader(
        "QA 자료 업로드 (여러 파일 가능)", 
        type=["pdf","jpg","png","jpeg","xlsx","xls"],
        accept_multiple_files=True
    )

    if st.button("결과 확인하기"):
        if not uploaded_files:
            st.error("파일을 업로드하세요.")
        else:
            files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]

            with st.spinner("AI가 분석 중입니다..."):
                response = requests.post(f"{FLASK_API_URL}/api/upload-qa", files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("분석 완료!")
                st.json(result)
            else:
                st.error("서버 오류 발생")
                st.write(response.text)

# -----------------------------
# 3. 오류 자동체크 화면
# -----------------------------
elif menu == "오류 자동체크":
    st.title("🔍 오류 자동체크 (기준 데이터 vs 디자인 검증)")

    standard_excel = st.file_uploader("📘 기준데이터(excel)", type=["xlsx"])
    design_file = st.file_uploader("🖼️ 디자인파일(PDF/이미지)", type=["pdf","jpg","png","jpeg"])

    if st.button("결과 확인하기"):
        if not design_file:
            st.error("디자인 파일을 업로드하세요.")
        else:
            files = {"design_file": (design_file.name, design_file.read(), design_file.type)}

            if standard_excel:
                files["standard_excel"] = (standard_excel.name, standard_excel.read(), standard_excel.type)

            with st.spinner("오류 검사 중..."):
                response = requests.post(f"{FLASK_API_URL}/api/verify-design", files=files)

            if response.status_code == 200:
                st.success("검사 완료!")
                st.json(response.json())
            else:
                st.error("서버 오류 발생")
                st.write(response.text)

# -----------------------------
# 4. 식품 관련 사이트
# -----------------------------
elif menu == "식품 관련 사이트":
    st.title("🔗 식품 관련 사이트 모음")

    st.markdown("""
    ### 📌 유용한 링크
    - **식약처 식품안전나라**  
      https://www.foodsafetykorea.go.kr  
    - **식품 표시 기준 고시**  
      https://www.foodsafetykorea.go.kr/foodcode/04_03.jsp  
    - **식품 영양성분 DB**  
      https://koreanfood.rda.go.kr/kfi/fct/fctList  
    - **부정불량식품 신고센터 (1399)**  
      https://www.mfds.go.kr
    """)

