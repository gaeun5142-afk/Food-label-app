import streamlit as st
from supabase import create_client, Client

# 🔑 Streamlit Secrets 에 저장된 Supabase 정보 사용
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="식품표시 웹앱", layout="centered")


# ---------------------- 로그인 페이지 ----------------------
def login_page():
    st.title("식품표시 웹앱")
    st.subheader("로그인")

    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if not email or not password:
            st.error("이메일/비밀번호를 모두 입력해주세요.")
            return

        try:
            result = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user = getattr(result, "user", None)

            if user:
                st.session_state["user"] = {
                    "email": user.email,
                    "id": user.id,
                }
                st.success("로그인 성공! 잠시 후 이동합니다.")
                st.experimental_rerun()
            else:
                st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")
        except Exception:
            st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")


# ---------------------- 메인 앱 ----------------------
def main_app():
    user = st.session_state["user"]
    email = user["email"]

    st.title("식품표시 웹앱")
    st.caption(f"현재 로그인: {email}")

    # 로그아웃 버튼 (모든 탭 공통)
    if st.button("로그아웃", key="logout_top"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("---")

    tab_home, tab_auto, tab_error = st.tabs(["🏠 홈", "🔁 자동 변환", "⚠ 오류 자동체크"])

    # -------- 홈 탭 --------
    with tab_home:
        st.subheader("홈")
        st.write(
            """
            이 웹앱은 **식품 표시 라벨**을 가지고  
            - 자동 변환(분류/정리)  
            - 오류 자동 체크  

            를 할 수 있도록 만들고 있는 **초기 버전**입니다.

            현재 화면에서는 이미지 업로드와 결과 확인 흐름만 만들었고,  
            실제 분석 로직(OCR, 기준 검증 등)은 나중에 추가할 예정입니다.
            """
        )

    # -------- 자동 변환 탭 --------
    with tab_auto:
        st.subheader("자동 변환")

        auto_image = st.file_uploader(
            "자동 변환할 라벨/포장 이미지 업로드", type=["png", "jpg", "jpeg"], key="auto_image"
        )

        if st.button("결과 확인하기", key="auto_check_btn"):
            if auto_image is None:
                st.error("먼저 이미지를 업로드해주세요.")
            else:
                st.success("자동 변환 결과입니다. (현재는 예시 텍스트)")
                st.markdown("**1) 업로드한 이미지 미리보기**")
                st.image(auto_image, use_column_width=True)

                # 👉 여기 부분에 나중에 실제 자동 변환 로직(OCR, 카테고리 분류 등) 연결
                st.markdown("---")
                st.markdown("**2) 변환된 내용 (데모)**")
                st.write(
                    """
                    - 예시) 카테고리: 과자류  
                    - 예시) 브랜드/제품명: (이미지에서 인식 예정)  
                    - 예시) 내용량, 원재료명, 알레르기 등은  
                      나중에 OCR 결과를 기반으로 자동 채워질 예정입니다.
                    """
                )
        else:
            st.info("이미지를 업로드한 후 **결과 확인하기** 버튼을 눌러주세요.")

    # -------- 오류 자동체크 탭 --------
    with tab_error:
        st.subheader("오류 자동체크")

        error_image = st.file_uploader(
            "오류를 체크할 라벨/포장 이미지 업로드",
            type=["png", "jpg", "jpeg"],
            key="error_image",
        )

        if st.button("결과 확인하기", key="error_check_btn"):
            if error_image is None:
                st.error("먼저 이미지를 업로드해주세요.")
            else:
                st.success("오류 자동체크 결과입니다. (현재는 예시 텍스트)")
                st.markdown("**1) 업로드한 이미지 미리보기**")
                st.image(error_image, use_column_width=True)

                # 👉 여기 부분에 나중에 실제 규정 위반 체크 로직을 붙이면 됨
                st.markdown("---")
                st.markdown("**2) 자동 체크 결과 (데모)**")
                st.write(
                    """
                    - 예시) 필수 항목 누락 여부: (나중에 실제 규칙으로 체크)  
                    - 예시) 알레르기 표시 누락 여부: (예: 우유, 대두, 땅콩 등)  
                    - 예시) 유통기한/보관방법 표기 여부: (라벨에서 인식 예정)  

                    현재는 구조만 만들어 둔 상태이며,  
                    나중에 실제 법적 기준/규정을 연결해 자동으로 체크하도록 확장할 수 있습니다.
                    """
                )
        else:
            st.info("이미지를 업로드한 후 **결과 확인하기** 버튼을 눌러주세요.")


# ---------------------- 실행 진입점 ----------------------
if "user" not in st.session_state:
    login_page()
else:
    main_app()



