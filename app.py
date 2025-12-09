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
FLASK_API_URL = "https://food-label-app-4.onrender.com"

# -----------------------------
# Streamlit 기본 설정
# -----------------------------
st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 로그인 상태
# -----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

if "login_error" not in st.session_state:
    st.session_state["login_error"] = None

# -----------------------------
# 로그인 페이지
# -----------------------------
def show_login_page():
    st.title("🔒 바른식품표시 로그인")

    if st.session_state["login_error"]:
        st.error(st.session_state["login_error"])
        st.session_state["login_error"] = None

    email = st.text_input("이메일")
    password = st.text_input("비밀번호", type="password")

    if st.button("로그인"):
        try:
            res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user = getattr(res, "user", None)

            if not user:
                st.session_state["login_error"] = "로그인 실패"
                st.rerun()

            st.session_state["user"] = {
                "id": user.id,
                "email": user.email
            }
            st.rerun()

        except Exception as e:
            st.session_state["login_error"] = "로그인 실패"
            st.rerun()

# -----------------------------
# 상단 바
# -----------------------------
def show_top_bar():
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("### 바른식품표시 플랫폼")
        if st.session_state["user"]:
            st.markdown(f"**로그인 사용자:** {st.session_state['user']['email']}")
    with cols[1]:
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.session_state["login_error"] = None
            st.rerun()

# -----------------------------
# 메인 앱
# -----------------------------
def show_main_app():
    show_top_bar()

    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"]
    )

    # -----------------------------
    # 홈
    # -----------------------------
    if menu == "홈":
        st.title("🏠 바른식품표시 플랫폼")

    # -----------------------------
    # 자동 변환 (QA → 기준 데이터 생성)
    # -----------------------------
    elif menu == "자동 변환":
        st.title("📄 자동 변환")

        uploaded_files = st.file_uploader(
            "QA 파일 업로드",
            type=["pdf", "jpg", "jpeg", "png", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if st.button("결과 확인하기"):
            if not uploaded_files:
                st.error("파일을 업로드하세요.")
                return

            files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]

            with st.spinner("AI가 QA 자료를 분석 중입니다..."):
                response = requests.post(
                    f"{FLASK_API_URL}/api/upload-qa",
                    files=files,
                    timeout=600
                )

            if response.status_code == 200:
                result = response.json()
                st.session_state["standard_result"] = result
                st.success("✅ 기준 데이터 생성 완료")
                st.json(result)
            else:
                st.error("서버 오류")
                st.write(response.text)

    # -----------------------------
    # ✅ 오류 자동체크 (최종 정상)
    # -----------------------------
    elif menu == "오류 자동체크":
        st.title("🔍 오류 자동체크")

        standard_excel = st.file_uploader(
            "📘 기준데이터 (선택)", type=["xlsx", "xls", "pdf"]
        )

        design_file = st.file_uploader(
            "🖼️ 디자인 파일", type=["pdf", "jpg", "jpeg", "png"]
        )

        if st.button("결과 확인하기"):

            # ✅ 기준 데이터 없으면 차단
            if "standard_result" not in st.session_state:
                st.error("⚠️ 먼저 [자동 변환]에서 기준 데이터를 생성하세요.")
                return

            # ✅ 디자인 파일 없으면 차단
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

            if response.status_code != 200:
                st.error("서버 오류 발생")
                st.write(response.text)
                return

            result = response.json()
            st.success("✅ 검사 완료")

            # ✅ 총점
            st.subheader("📌 총점 및 법규 준수 여부")
            st.write("점수:", result.get("score"))
            law = result.get("law_compliance", {})
            st.write("법규 상태:", law.get("status"))

            if law.get("violations"):
                for v in law["violations"]:
                    st.write("-", v)

            # ✅ 이슈 목록
            st.subheader("📌 상세 이슈 목록")
            issues = result.get("issues", [])

            if not issues:
                st.write("✅ 발견된 이슈 없음")
            else:
                for i, issue in enumerate(issues, 1):
                    st.markdown(f"### 이슈 {i}")
                    st.write("유형:", issue.get("type"))
                    st.write("설명:", issue.get("issue"))
                    st.write("기준값:", issue.get("expected"))
                    st.write("디자인값:", issue.get("actual"))
                    st.write("수정 제안:", issue.get("suggestion"))
                    st.markdown("---")

    # -----------------------------
    # 식품 관련 사이트
    # -----------------------------
    elif menu == "식품 관련 사이트":
        st.title("🔗 식품 관련 사이트")
        st.markdown("""
        - 식품안전나라  
        - 식품 표시 기준  
        - 영양성분 DB  
        """)

# -----------------------------
# 앱 진입점
# -----------------------------
def main():
    if st.session_state["user"] is None:
        show_login_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()

