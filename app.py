import streamlit as st

st.set_page_config(page_title="식품표시 웹앱 - 로그인", layout="centered")

st.title("식품표시 웹앱")
st.subheader("로그인")

with st.form("login_form"):
    email = st.text_input("이메일")
    password = st.text_input("비밀번호", type="password")
    login_btn = st.form_submit_button("로그인")

if login_btn:
    st.info("여기에서 나중에 Supabase로 이메일/비밀번호 확인할 예정입니다.")
  
