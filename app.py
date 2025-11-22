import streamlit as st
from supabase import create_client, Client

# 👉 Streamlit Cloud의 Secrets에서 불러오기 (코드에 직접 쓰지 않기!)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="식품표시 웹앱", layout="centered")


def show_login_page():
    st.title("식품표시 웹앱")
    st.subheader("로그인")

    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if not email or not password:
            st.error("이메일과 비밀번호를 모두 입력해주세요.")
            return

        try:
            # Supabase 이메일/비밀번호 로그인
            result = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            # result.user 가 없으면 실패로 처리
            if not result.user:
                st.error("로그인에 실패했습니다. 이메일/비밀번호를 확인해주세요.")
                return

            # 세션에 사용자 정보 저장
            st.session_state["user"] = {
                "id": result.user.id,
                "email": result.user.email,
            }

            st.success("로그인 성공! 잠시 후 대시보드로 이동합니다.")
            st.experimental_rerun()

        except Exception as e:
            st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")


def show_dashboard():
    user = st.session_state.get("user")
    email = user.get("email", "") if user else ""

    st.title("식품표시 웹앱 대시보드")
    st.write(f"👋 **{email}** 님, 안녕하세요!")

    # 로그아웃 버튼
    if st.button("로그아웃"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("식품표시 정보 입력 (틀만 먼저 만들기)")

    with st.form("food_form"):
        product_name = st.text_input("제품명")
        category = st.text_input("식품 유형 (예: 과자, 음료 등)")
        ingredients = st.text_area("원재료명 및 함량")
        allergy = st.text_input("알레르기 표시")
        expiration = st.text_input("유통기한 표시")

        submitted = st.form_submit_button("임시 저장")

    if submitted:
        # 아직 DB 저장은 안 하고, 입력값만 보여주기
        st.success("입력값이 임시로 제출되었습니다. (나중에 DB에 저장 예정)")
        st.write("**제품명:**", product_name)
        st.write("**유형:**", category)
        st.write("**원재료명:**", ingredients)
        st.write("**알레르기:**", allergy)
        st.write("**유통기한:**", expiration)


# 👉 메인 흐름: 로그인 여부에 따라 화면 분기
if "user" not in st.session_state:
    show_login_page()
else:
    show_dashboard()
