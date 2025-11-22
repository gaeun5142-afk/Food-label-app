import streamlit as st
import requests
from PIL import Image
import io
import json

from supabase import create_client, Client

# -----------------------------
# Supabase 설정 (Streamlit Secrets 사용)
# -----------------------------
# Streamlit Cloud의 Secrets에 아래 키들이 있어야 함:
# SUPABASE_URL = "https://xxxx.supabase.co"
# SUPABASE_KEY = "supabase anon key"
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Flask 서버 주소 (Render)
# -----------------------------
FLASK_API_URL = "https://food-label-app-4.onrender.com"

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None   # 로그인 정보
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# -----------------------------
# 로그인 화면 함수
# -----------------------------
def show_login_page():
    st.title("🔐 바른식품표시 - 로그인")

    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if not email or not password:
            st.error("이메일과 비밀번호를 모두 입력해주세요.")
            return

        try:
            result = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["user"] = {
                "email": email,
                "access_token": result.session.access_token if result.session else None,
            }
            st.session_state["logged_in"] = True
            st.success("로그인 성공!")
            st.experimental_rerun()
        except Exception as e:
            st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")
            st.write(e)


# -----------------------------
# 상단 헤더 + 로그아웃 버튼
# -----------------------------
def show_header():
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown("### 🏷 바른식품표시 플랫폼")
    with cols[1]:
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.session_state["logged_in"] = False
            st.experimental_rerun()


# -----------------------------
# 메인 앱 (로그인 후)
# -----------------------------
def show_main_app():
    show_header()

    # 사이드바 메뉴
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"]
    )

    # -----------------------------
    # 1. 홈 화면
    # -----------------------------
    if menu == "홈":
        st.title("🏠 바른식품표시 플랫폼")
        st.write("식품 표시사항을 자동으로 변환하고, 오류를 검사하고, 식품 관련 사이트를 모아놓은 서비스입니다.")

        user = st.session_state.get("user")
        if user and user.get("email"):
            st.info(f"현재 로그인 계정: **{user['email']}**")

    # -----------------------------
    # 2. 자동 변환 화면 (QA 자료 → 자동 라벨 생성)
    # -----------------------------
    elif menu == "자동 변환":
        st.title("📄 자동 변환 (QA 기반 표시사항 생성)")

        uploaded_files = st.file_uploader(
            "QA 자료 업로드 (여러 파일 가능)",
            type=["pdf", "jpg", "png", "jpeg", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if st.button("결과 확인하기"):
            if not uploaded_files:
                st.error("파일을 업로드하세요.")
            else:
                # Flask 백엔드로 파일 보내기
                files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]

                try:
                    with st.spinner("AI가 분석 중입니다..."):
                        response = requests.post(
                            f"{FLASK_API_URL}/api/upload-qa",
                            files=files,
                            timeout=180
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("분석 완료!")
                        st.json(result)
                    else:
                        st.error(f"서버 오류 발생 (status: {response.status_code})")
                        st.write(response.text)
                except Exception as e:
                    st.error("요청 중 오류가 발생했습니다.")
                    st.write(e)

    # -----------------------------
    # 3. 오류 자동체크 화면
    # -----------------------------
    elif menu == "오류 자동체크":
        st.title("🔍 오류 자동체크 (기준 데이터 vs 디자인 검증)")

        standard_excel = st.file_uploader(
            "📘 기준데이터 (Excel 파일)",
            type=["xlsx", "xls"],
            key="standard_excel"
        )
        design_file = st.file_uploader(
            "🖼 디자인파일 (PDF/이미지)",
            type=["pdf", "jpg", "png", "jpeg"],
            key="design_file"
        )

        if st.button("결과 확인하기"):
            if not design_file:
                st.error("디자인 파일을 업로드하세요.")
            else:
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

                try:
                    with st.spinner("오류 검사 중..."):
                        response = requests.post(
                            f"{FLASK_API_URL}/api/verify-design",
                            files=files,
                            timeout=180
                        )

                    if response.status_code == 200:
                        st.success("검사 완료!")
                        result = response.json()

                        # 점수
                        score = result.get("score")
                        if score is not None:
                            st.markdown(f"### 🧮 검증 점수: **{score}점**")

                        # 법규 준수 정보
                        law_info = result.get("law_compliance", {})
                        if law_info:
                            st.markdown("### ⚖️ 법규 준수 상태")
                            st.json(law_info)

                        # 상세 이슈
                        issues = result.get("issues", [])
                        if issues:
                            st.markdown("### 📌 발견된 이슈 목록")
                            st.json(issues)
                        else:
                            st.info("특별한 이슈가 발견되지 않았습니다.")

                        # 원하면 전체 결과도 펼쳐보기
                        with st.expander("🔍 전체 결과 JSON 보기"):
                            st.json(result)
                    else:
                        st.error(f"서버 오류 발생 (status: {response.status_code})")
                        st.write(response.text)
                except Exception as e:
                    st.error("요청 중 오류가 발생했습니다.")
                    st.write(e)

    # -----------------------------
    # 4. 식품 관련 사이트
    # -----------------------------
    elif menu == "식품 관련 사이트":
        st.title("🔗 식품 관련 사이트 모음")

        st.markdown("""
        ### 📌 유용한 링크
        - **식약처 식품안전나라**  
          https://www.foodsafetykorea.go.kr  

        - **법제처사이트**  
          import streamlit as st
import requests
from PIL import Image
import io
import json

from supabase import create_client, Client

# -----------------------------
# Supabase 설정 (Streamlit Secrets 사용)
# -----------------------------
# Streamlit Cloud의 Secrets에 아래 키들이 있어야 함:
# SUPABASE_URL = "https://xxxx.supabase.co"
# SUPABASE_KEY = "supabase anon key"
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Flask 서버 주소 (Render)
# -----------------------------
FLASK_API_URL = "https://food-label-app-4.onrender.com"

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="바른식품표시",
    layout="wide"
)

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None   # 로그인 정보
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# -----------------------------
# 로그인 화면 함수
# -----------------------------
def show_login_page():
    st.title("🔐 바른식품표시 - 로그인")

    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if not email or not password:
            st.error("이메일과 비밀번호를 모두 입력해주세요.")
            return

        try:
            result = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["user"] = {
                "email": email,
                "access_token": result.session.access_token if result.session else None,
            }
            st.session_state["logged_in"] = True
            st.success("로그인 성공!")
            st.experimental_rerun()
        except Exception as e:
            st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")
            st.write(e)


# -----------------------------
# 상단 헤더 + 로그아웃 버튼
# -----------------------------
def show_header():
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown("### 🏷 바른식품표시 플랫폼")
    with cols[1]:
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.session_state["logged_in"] = False
            st.experimental_rerun()


# -----------------------------
# 메인 앱 (로그인 후)
# -----------------------------
def show_main_app():
    show_header()

    # 사이드바 메뉴
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"]
    )

    # -----------------------------
    # 1. 홈 화면
    # -----------------------------
    if menu == "홈":
        st.title("🏠 바른식품표시 플랫폼")
        st.write("식품 표시사항을 자동으로 변환하고, 오류를 검사하고, 식품 관련 사이트를 모아놓은 서비스입니다.")

        user = st.session_state.get("user")
        if user and user.get("email"):
            st.info(f"현재 로그인 계정: **{user['email']}**")

    # -----------------------------
    # 2. 자동 변환 화면 (QA 자료 → 자동 라벨 생성)
    # -----------------------------
    elif menu == "자동 변환":
        st.title("📄 자동 변환 (QA 기반 표시사항 생성)")

        uploaded_files = st.file_uploader(
            "QA 자료 업로드 (여러 파일 가능)",
            type=["pdf", "jpg", "png", "jpeg", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if st.button("결과 확인하기"):
            if not uploaded_files:
                st.error("파일을 업로드하세요.")
            else:
                # Flask 백엔드로 파일 보내기
                files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]

                try:
                    with st.spinner("AI가 분석 중입니다..."):
                        response = requests.post(
                            f"{FLASK_API_URL}/api/upload-qa",
                            files=files,
                            timeout=180
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("분석 완료!")
                        st.json(result)
                    else:
                        st.error(f"서버 오류 발생 (status: {response.status_code})")
                        st.write(response.text)
                except Exception as e:
                    st.error("요청 중 오류가 발생했습니다.")
                    st.write(e)

    # -----------------------------
    # 3. 오류 자동체크 화면
    # -----------------------------
    elif menu == "오류 자동체크":
        st.title("🔍 오류 자동체크 (기준 데이터 vs 디자인 검증)")

        standard_excel = st.file_uploader(
            "📘 기준데이터 (Excel 파일)",
            type=["xlsx", "xls"],
            key="standard_excel"
        )
        design_file = st.file_uploader(
            "🖼 디자인파일 (PDF/이미지)",
            type=["pdf", "jpg", "png", "jpeg"],
            key="design_file"
        )

        if st.button("결과 확인하기"):
            if not design_file:
                st.error("디자인 파일을 업로드하세요.")
            else:
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

                try:
                    with st.spinner("오류 검사 중..."):
                        response = requests.post(
                            f"{FLASK_API_URL}/api/verify-design",
                            files=files,
                            timeout=180
                        )

                    if response.status_code == 200:
                        st.success("검사 완료!")
                        result = response.json()

                        # 점수
                        score = result.get("score")
                        if score is not None:
                            st.markdown(f"### 🧮 검증 점수: **{score}점**")

                        # 법규 준수 정보
                        law_info = result.get("law_compliance", {})
                        if law_info:
                            st.markdown("### ⚖️ 법규 준수 상태")
                            st.json(law_info)

                        # 상세 이슈
                        issues = result.get("issues", [])
                        if issues:
                            st.markdown("### 📌 발견된 이슈 목록")
                            st.json(issues)
                        else:
                            st.info("특별한 이슈가 발견되지 않았습니다.")

                        # 원하면 전체 결과도 펼쳐보기
                        with st.expander("🔍 전체 결과 JSON 보기"):
                            st.json(result)
                    else:
                        st.error(f"서버 오류 발생 (status: {response.status_code})")
                        st.write(response.text)
                except Exception as e:
                    st.error("요청 중 오류가 발생했습니다.")
                    st.write(e)

    # -----------------------------
    # 4. 식품 관련 사이트
    # -----------------------------
    elif menu == "식품 관련 사이트":
        st.title("🔗 식품 관련 사이트 모음")

        st.markdown("""
        ### 📌 유용한 링크
        - **식약처 식품안전나라**  
          https://www.foodsafetykorea.go.kr  

        - **식품 영양성분 DB**  
          https://koreanfood.rda.go.kr/kfi/fct/fctList  

        - **부정불량식품 신고센터 (1399)**  
          https://www.mfds.go.kr
        """)


# -----------------------------
# 앱 실행 흐름
# -----------------------------
if not st.session_state["logged_in"]:
    # 로그인 안 됐으면 로그인 화면 먼저
    show_login_page()
else:
    # 로그인 후 메인 앱
    show_main_app()

        - **식품 영양성분 DB**  
          https://koreanfood.rda.go.kr/kfi/fct/fctList  

        - **부정불량식품 신고센터 (1399)**  
          https://www.mfds.go.kr
        """)


# -----------------------------
# 앱 실행 흐름
# -----------------------------
if not st.session_state["logged_in"]:
    # 로그인 안 됐으면 로그인 화면 먼저
    show_login_page()
else:
    # 로그인 후 메인 앱
    show_main_app()
