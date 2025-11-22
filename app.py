import streamlit as st
from supabase import create_client, Client

# 👉 키를 코드에 직접 쓰지 않고, Streamlit secrets에서 가져오기
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="식품표시 웹앱 - 로그인", layout="centered")

st.title("식품표시 웹앱")
st.subheader("로그인")

with st.form("login_form"):
    email = st.text_input("이메일")
    password = st.text_input("비밀번호", type="password")
    login_btn = st.form_submit_button("로그인")

if login_btn:
    try:
        result = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        st.success("로그인 성공!")
        st.write(result)   # 나중에 이 부분은 다른 페이지로 바꾸면 됨
    except Exception as e:
        st.error("로그인 실패: 이메일/비밀번호 확인하세요.")
