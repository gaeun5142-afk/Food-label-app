import streamlit as st
import requests
from PIL import Image
import io
import json
from supabase import create_client, Client

# -----------------------------
# Supabase 설정 (Streamlit Secrets 사용)
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Flask 서버 주소 (Render)
# -----------------------------
FLASK_API_URL = "https://food-label-app-4.onrender.com"  # Render에 만든 Flask 서버 URL
 
# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 로그인 관련 유틸 함수
# -----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None  # 로그인된 유저 정보 저장용

if "login_error" not in st.session_state:
    st.session_state["login_error"] = None


def show_login_page():
    # 🔒 원래 쓰던 자물쇠 이모지로 변경
    st.title("🔒 바른식품표시 로그인")

    # 이전 에러 메시지 표시 (한 번만)
    if st.session_state["login_error"]:
        st.error(st.session_state["login_error"])
        st.session_state["login_error"] = None  # 표시 후 초기화

    email = st.text_input("이메일", key="login_email")
    password = st.text_input("비밀번호", type="password", key="login_password")

    # 버튼 눌렀을 때만 처리
    if st.button("로그인"):
        if not email or not password:
            st.session_state["login_error"] = "이메일과 비밀번호를 모두 입력해 주세요."
            st.rerun()
            return

        try:
            # Supabase 로그인
            res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            user = getattr(res, "user", None)

            # 로그인 실패 처리
            if user is None:
                st.session_state["login_error"] = "로그인 실패: 이메일/비밀번호를 확인해 주세요."
                st.rerun()
                return

            # 로그인 성공 처리 (메시지 없이 바로 리다이렉트)
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email
            }
            st.session_state["login_error"] = None  # 에러 초기화
            st.rerun()

        except Exception as e:
            # Supabase 내부 오류 또는 비번 불일치
            st.session_state["login_error"] = "로그인 실패: 이메일/비밀번호를 확인해 주세요."
            print("로그인 오류:", e)
            st.rerun()


def show_top_bar():
    """상단에 사용자 정보 + 로그아웃 버튼"""
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("### 바른식품표시 플랫폼")
        if st.session_state["user"]:
            st.markdown(f"**로그인된 사용자:** {st.session_state['user']['email']}")
    with cols[1]:
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.session_state["login_error"] = None
            st.rerun()


# -----------------------------
# 메인 콘텐츠 (로그인 후)
# -----------------------------
def show_main_app():
    show_top_bar()

    # 사이드바 메뉴
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"],
    )

    # 1. 홈
    if menu == "홈":
        st.title("🏠 바른식품표시 플랫폼")
        st.write(
            """
            이 웹앱은 식품 표시사항을 **자동으로 생성**하고,  
            **디자인과 기준데이터를 비교해 오류를 자동으로 검출**하는 플랫폼입니다.
            """
        )

    # 2. 자동 변환 (QA → 자동 라벨)
    elif menu == "자동 변환":
        st.title("📄 자동 변환 (QA 기반 표시사항 생성)")
        uploaded_files = st.file_uploader(
            "QA 자료 업로드 (여러 파일 가능)",
            type=["pdf", "jpg", "jpeg", "png", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        if st.button("결과 확인하기"):
            if not uploaded_files:
                st.error("파일을 업로드하세요.")
            else:
                files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]
                with st.spinner("AI가 QA 자료를 분석 중입니다..."):
                    try:
                        response = requests.post(
                            f"{FLASK_API_URL}/api/upload-qa",
                            files=files,
                            timeout=600,
                        )
                    except Exception as e:
                        st.error(f"서버 연결 오류: {e}")
                    else:
                        if response.status_code == 200:
                            result = response.json()

                            # ✅✅✅ 이 줄이 지금까지 없어서 전부 깨졌던 거다
                            st.session_state["standard_result"] = result

                            st.success("분석 완료!")
                            st.subheader("📌 생성된 식품표시 기준 데이터 (JSON)")
                            st.json(result)

                        else:
                            st.error("서버에서 오류가 발생했습니다.")
                            st.write("상태 코드:", response.status_code)
                            st.write(response.text)

    # 3. 오류 자동체크
    elif menu == "오류 자동체크":
        st.title("🔍 오류 자동체크 ")
        standard_excel = st.file_uploader(
            "📘 기준데이터 (Excel / PDF)", type=["xlsx", "xls", "pdf"]
        )
        design_file = st.file_uploader(
            "🖼️ 디자인 파일 (PDF / 이미지)",
            type=["pdf", "jpg", "jpeg", "png"],
        )

        if st.button("결과 확인하기"):

            # ✅✅✅ 기준 데이터 없으면 서버 요청 자체 차단
            if "standard_result" not in st.session_state:
                st.error("⚠️ 먼저 [자동 변환]에서 기준 데이터를 생성해야 합니다.")
                return

            if not design_file:
                st.error("디자인 파일을 업로드하세요.")
                return

            files = {
                 "design_file": (
                    design_file.name,
                    design_file.read(),
                    design_file.type,
              )
         }

        if standard_excel:
            files["standard_excel"] = (
                standard_excel.name,
                standard_excel.read(),
                standard_excel.type,
           )

    with st.spinner("디자인과 기준 데이터를 비교 중입니다..."):
        try:
            response = requests.post(
                f"{FLASK_API_URL}/api/verify-design-strict",
                files=files,
                data={
                    "standard_data": json.dumps(
                        st.session_state["standard_result"],
                        ensure_ascii=False
                    )
                },
                timeout=600,
            )



                    except Exception as e:
                        st.error(f"서버 연결 오류: {e}")
                    else:
                        if response.status_code == 200:
                            st.success("검사 완료!")
                            result = response.json()

                            # -----------------------
                            # 1) 총점 및 법규 준수 여부
                            # -----------------------
                            st.subheader("📌 총점 및 법규 준수 여부")
                            score = result.get("score", "N/A")
                            law = result.get("law_compliance", {})
                            st.write(f"**점수:** {score}")
                            st.write("**법규 상태:**", law.get("status", "N/A"))
                            if law.get("violations"):
                                st.write("**위반 사항:**")
                                for v in law["violations"]:
                                    st.write("-", v)

                            # -----------------------
                            # 2) 상세 이슈 목록
                            # -----------------------
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

                            # -----------------------
                            # 3) AI 정밀 분석 결과 (하이라이트)
                            #    server.py에서
                            #    result["design_ocr_highlighted_html"]
                            #    를 추가해 줬다는 가정
                            # -----------------------
                            highlight_html = result.get("design_ocr_highlighted_html")
                            if highlight_html:
                                st.subheader("🔎 AI 정밀 분석 결과 (하이라이트)")
                                st.markdown(
                                    """
                                    <div style="font-size:13px; color:#555; margin-bottom:8px;">
                                      * 붉은색으로 표시된 부분은 기준 정보와 다르거나 오타가 의심되는 곳입니다.
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(highlight_html, unsafe_allow_html=True)

                        else:
                            st.error("서버에서 오류가 발생했습니다.")
                            st.write("상태 코드:", response.status_code)
                            st.write(response.text)

    # 4. 식품 관련 사이트
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


# -----------------------------
# 앱 진입점
# -----------------------------
def main():
    # 아직 로그인 안 했으면 로그인 화면만 보여주기
    if st.session_state["user"] is None:
        show_login_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()
