import streamlit as st
from supabase import create_client, Client

# 🔑 Streamlit Secrets 에 저장한 Supabase 정보 불러오기
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

    # 로그인 버튼 누르면 실행
    if login_btn:
        if not email or not password:
            st.error("이메일/비밀번호를 입력해주세요.")
            return

        try:
            result = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            # 유저 객체 가져오기 (버전마다 다름 → 둘 다 커버)
            user = getattr(result, "user", None)
            if user is None and isinstance(result, dict):
                user = result.get("user")

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

    tab1, tab2, tab3 = st.tabs(["🏠 홈", "📝 식품 등록", "👤 내 계정"])

    # -------- 홈 --------
    with tab1:
        st.header("식품표시 웹앱 대시보드")
        st.write(f"👋 {email}님 환영합니다!")
        st.write("아직 초기 버전입니다. 기능이 계속 추가될 예정입니다.")

    # -------- 식품 등록 --------
    with tab2:
        st.header("식품 표시사항 입력")

        with st.form("food_form"):
            name = st.text_input("제품명")
            category = st.text_input("식품 유형")
            volume = st.text_input("내용량")
            ingredients = st.text_area("원재료명")
            allergy = st.text_input("알레르기 표시")
            expiration = st.text_input("유통/품질유지기한")
            storage = st.text_input("보관방법")

            submit = st.form_submit_button("임시로 확인")

        if submit:
            st.success("입력한 내용입니다 (저장은 아직 X)")
            st.write("**제품명:**", name)
            st.write("**유형:**", category)
            st.write("**내용량:**", volume)
            st.write("**원재료명:**", ingredients)
            st.write("**알레르기:**", allergy)
            st.write("**유통기한:**", expiration)
            st.write("**보관방법:**", storage)

    # -------- 내 계정 --------
    with tab3:
        st.header("내 계정")
        st.write("이메일:", email)

        if st.button("로그아웃"):
            st.session_state.clear()
            st.experimental_rerun()


# ---------------------- 실행 ----------------------
if "user" not in st.session_state:
    login_page()
else:
    main_app()

