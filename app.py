import streamlit as st
from supabase import create_client, Client

# 🔑 Streamlit Secrets 에 저장된 Supabase 정보 사용
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# 앱 이름을 바른식품표시로 변경
st.set_page_config(page_title="바른식품표시", layout="centered")


# ---------------------- 로그인 페이지 ----------------------
def login_page():
    st.title("바른식품표시")
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

    st.title("바른식품표시")
    st.caption(f"현재 로그인: {email}")

    # 로그아웃 버튼 (모든 탭 공통)
    if st.button("로그아웃", key="logout_top"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("---")

    # 탭: 홈 / 자동 변환 / 오류 자동체크 / 식품 관련 사이트
    tab_home, tab_auto, tab_error, tab_links = st.tabs(
        ["🏠 홈", "🔁 자동 변환", "⚠ 오류 자동체크", "🔗 식품 관련 사이트"]
    )

    # -------- 홈 탭 --------
    with tab_home:
        st.subheader("홈")
        st.write(
            """
            **바른식품표시**는 식품 라벨 이미지를 기반으로

            - 자동 변환(분류/정리)
            - 오류 자동 체크
            - 관련 공공 사이트/자료로 바로 연결

            을 목표로 하는 웹앱입니다.  
            지금은 구조만 만들어 둔 **프로토타입** 단계이고,
            앞으로 실제 법령·가이드라인을 기반으로 기능을 확장할 수 있습니다.
            """
        )

    # -------- 자동 변환 탭 --------
    with tab_auto:
        st.subheader("자동 변환")

        auto_image = st.file_uploader(
            "자동 변환할 라벨/포장 이미지 업로드",
            type=["png", "jpg", "jpeg"],
            key="auto_image",
        )

        if st.button("결과 확인하기", key="auto_check_btn"):
            if auto_image is None:
                st.error("먼저 이미지를 업로드해주세요.")
            else:
                st.success("자동 변환 결과입니다. (현재는 예시 텍스트)")
                st.markdown("**1) 업로드한 이미지 미리보기**")
                st.image(auto_image, use_column_width=True)

                # 👉 나중에 실제 OCR/분류 로직을 여기에 연결
                st.markdown("---")
                st.markdown("**2) 변환된 내용 (데모)**")
                st.write(
                    """
                    - 예시) 카테고리: 과자류  
                    - 예시) 제품명/브랜드: (이미지에서 인식 예정)  
                    - 예시) 내용량, 원재료명, 알레르기 등은  
                      추후 자동 추출 기능을 통해 채워질 예정입니다.
                    """
                )
        else:
            st.info("이미지를 업로드한 뒤 **결과 확인하기** 버튼을 눌러주세요.")

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

                # 👉 나중에 실제 규정 위반 체크 로직을 붙이면 됨
                st.markdown("---")
                st.markdown("**2) 자동 체크 결과 (데모)**")
                st.write(
                    """
                    - 예시) 필수 표시항목 누락 여부  
                    - 예시) 알레르기 표시 누락 여부 (우유, 대두, 땅콩 등)  
                    - 예시) 유통기한·보관방법 표기 여부  

                    현재는 참고용 설명만 보여주고 있으며,
                    실제 법적 기준 검토는 별도로 필요합니다.
                    """
                )
        else:
            st.info("이미지를 업로드한 뒤 **결과 확인하기** 버튼을 눌러주세요.")

    # -------- 식품 관련 사이트 탭 --------
    with tab_links:
        st.subheader("식품 관련 사이트 모음")

        st.write("식품 표시·안전 관련해서 자주 참고하는 사이트들을 모아두는 공간입니다.")

        st.markdown("### 🏛 공공/기관 사이트")
        st.markdown(
            """
- 식품의약품안전처(MFDS):  
  - 홈페이지: https://www.mfds.go.kr  
  - **식품안전나라**: https://www.foodsafetykorea.go.kr  
- 국가법령정보센터(식품 관련 법령 검색):  
  - https://www.law.go.kr
            """
        )

        st.markdown("### 📚 가이드·자료 (나중에 링크 추가 가능)")
        st.write(
            "- 식품 표시 기준 요약 자료\n"
            "- 알레르기 표시 의무 품목 안내\n"
            "- 영양성분 표시 가이드\n"
            "\n(필요한 자료 링크를 천천히 더 추가하면 돼요.)"
        )


# ---------------------- 실행 진입점 ----------------------
if "user" not in st.session_state:
    login_page()
else:
    main_app()




