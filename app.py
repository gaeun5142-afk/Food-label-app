import streamlit as st
import requests
from PIL import Image
import io
import json

# -----------------------------
# Flask 서버 주소 (Render 주소로 설정)
# -----------------------------
FLASK_API_URL = "https://food-label-app-4.onrender.com"  # 너가 만든 Render Flask URL

st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 사이드바 네비게이션 메뉴
# -----------------------------
menu = st.sidebar.radio(
    "메뉴 선택", 
    ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"]
)

# -----------------------------
# 1. 홈 화면
# -----------------------------
if menu == "홈":
    st.title("🏠 바른식품표시 플랫폼")
    st.write(
        """
        이 웹앱은 식품 표시사항을 **자동으로 생성**하고,  
        **디자인과 기준데이터를 비교해 오류를 자동으로 검출**하는 플랫폼입니다.
        """
    )

# -----------------------------
# 2. 자동 변환 화면 (QA 자료 → 자동 라벨 생성)
# -----------------------------
elif menu == "자동 변환":
    st.title("📄 자동 변환 (QA 기반 표시사항 생성)")

    uploaded_files = st.file_uploader(
        "QA 자료 업로드 (여러 파일 가능)", 
        type=["pdf", "jpg", "jpeg", "png", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if st.button("결과 확인하기"):
        if not uploaded_files:
            st.error("파일을 업로드하세요.")
        else:
            # Flask 서버로 보낼 파일 포맷 맞추기
            files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]

            with st.spinner("AI가 QA 자료를 분석 중입니다..."):
                try:
                    response = requests.post(
                        f"{FLASK_API_URL}/api/upload-qa",
                        files=files,
                        timeout=600
                    )
                except Exception as e:
                    st.error(f"서버 연결 오류: {e}")
                else:
                    if response.status_code == 200:
                        result = response.json()
                        st.success("분석 완료!")
                        st.subheader("📌 생성된 식품표시 기준 데이터 (JSON)")
                        st.json(result)
                    else:
                        st.error("서버에서 오류가 발생했습니다.")
                        st.write("상태 코드:", response.status_code)
                        st.write(response.text)

# -----------------------------
# 3. 오류 자동체크 화면
# -----------------------------
elif menu == "오류 자동체크":
    st.title("🔍 오류 자동체크 (기준 데이터 vs 디자인 검증)")

    standard_excel = st.file_uploader(
        "📘 기준데이터 (Excel / PDF)", 
        type=["xlsx", "xls", "pdf"]
    )
    design_file = st.file_uploader(
        "🖼️ 디자인 파일 (PDF / 이미지)", 
        type=["pdf", "jpg", "jpeg", "png"]
    )

    if st.button("결과 확인하기"):
        if not design_file:
            st.error("디자인 파일을 업로드하세요.")
        else:
            files = {
                "design_file": (design_file.name, design_file.read(), design_file.type)
            }

            if standard_excel:
                files["standard_excel"] = (
                    standard_excel.name,
                    standard_excel.read(),
                    standard_excel.type
                )

            with st.spinner("디자인과 기준 데이터를 비교 중입니다..."):
                try:
                    response = requests.post(
                        f"{FLASK_API_URL}/api/verify-design",
                        files=files,
                        timeout=600
                    )
                except Exception as e:
                    st.error(f"서버 연결 오류: {e}")
                else:
                    if response.status_code == 200:
                        st.success("검사 완료!")
                        result = response.json()

                        st.subheader("📌 총점 및 법규 준수 여부")
                        score = result.get("score", "N/A")
                        law = result.get("law_compliance", {})
                        st.write(f"**점수:** {score}")
                        st.write("**법규 상태:**", law.get("status", "N/A"))
                        if law.get("violations"):
                            st.write("**위반 사항:**")
                            for v in law["violations"]:
                                st.write("-", v)

                        st.subheader("📌 상세 이슈 목록")
                        issues = result.get("issues", [])
                        if not issues:
                            st.write("발견된 이슈가 없습니다. 👍")
                        else:
                            for i, issue in enumerate(issues, start=1):
                                st.markdown(f"#### 이슈 {i}")
                                st.write("유형:", issue.get("type"))
                                st.write("위치:", issue.get("location"))
                                st.write("설명:", issue.get("issue"))
                                st.write("기준값:", issue.get("expected"))
                                st.write("디자인 실제값:", issue.get("actual"))
                                st.write("수정 제안:", issue.get("suggestion"))
                                st.markdown("---")
                    else:
                        st.error("서버에서 오류가 발생했습니다.")
                        st.write("상태 코드:", response.status_code)
                        st.write(response.text)

# -----------------------------
# 4. 식품 관련 사이트
# -----------------------------
elif menu == "식품 관련 사이트":
    st.title("🔗 식품 관련 사이트 모음")

    st.markdown(
        """
        ### 📌 유용한 링크

        - **식약처 식품안전나라**  
          https://www.foodsafetykorea.go.kr  

        - **식품 표시 기준 고시**  
          https://www.foodsafetykorea.go.kr/foodcode/04_03.jsp  

        - **식품 영양성분 DB**  
          https://koreanfood.rda.go.kr/kfi/fct/fctList  

        - **부정불량식품 신고센터 (1399)**  
          https://www.mfds.go.kr
        """
    )
