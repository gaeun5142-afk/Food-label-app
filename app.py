import streamlit as st
from supabase import create_client

# ---------------------- Supabase 연결 ---------------------- #
import os
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------- 페이지 설정 ---------------------- #
st.set_page_config(page_title="바른식품표시", layout="wide")

# ---------------------- 로그인 상태 ---------------------- #
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------------- 스타일 ---------------------- #
st.markdown("""
<style>
    .step-box {
        background: #f1f6ff;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        border: 1px solid #d3e2ff;
    }
    .result-box {
        background: #fafafa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        min-height:120px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- 로그인 페이지 ---------------------- #
def login_page():
    st.title("🔐 바른식품표시 로그인")

    with st.form("login"):
        email = st.text_input("이메일")
        pw = st.text_input("비밀번호", type="password")
        btn = st.form_submit_button("로그인")

    if btn:
        try:
            supabase.auth.sign_in_with_password({"email": email, "password": pw})
            st.session_state.user = email
            st.success("로그인 성공! 페이지로 이동합니다.")
            st.experimental_rerun()
        except:
            st.error("로그인 실패! 이메일 또는 비밀번호 확인해주세요.")


# ---------------------- 홈 페이지 ---------------------- #
def home_page():
    st.title("🏡 바른식품표시 플랫폼")

    st.subheader("📌 식품 관련 사이트 모음")
    st.markdown("""
    - [식약처 식품안전나라](https://www.foodsafetykorea.go.kr)
    - [KATRI 시험연구원](https://www.katri.re.kr)
    - [식품의약품안전처](https://www.mfds.go.kr)
    """)


# ---------------------- 자동 변환 페이지 ---------------------- #
def auto_convert_page():
    st.title("⚙ 자동 변환")

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 1. 파일 업로드")

    uploaded_file = st.file_uploader("파일 선택 (엑셀, 이미지 등 가능)", type=["xlsx", "pdf", "png", "jpg"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 2. 기준 데이터 생성")

    if st.button("📄 기준 데이터 생성하기"):
        st.session_state["auto_convert_result"] = "✔ 기준 데이터가 정상적으로 생성되었습니다."

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 3. 결과 보기")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(st.session_state.get("auto_convert_result", "결과가 여기에 표시됩니다."))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------- 오류 자동 체크 페이지 ---------------------- #
def auto_check_page():
    st.title("🔍 오류 자동체크")

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 1. 최종 파일 업로드")
    uploaded = st.file_uploader("PDF 또는 이미지 업로드", type=["pdf", "png", "jpg"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 2. 오류 검증")

    if st.button("🔎 검증 시작"):
        st.session_state["auto_check_result"] = "⚠ 라벨 내 표시사항 일부 항목이 누락되었습니다."
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 3. 결과")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(st.session_state.get("auto_check_result", "결과가 여기에 표시됩니다."))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------- 라우팅 ---------------------- #
if st.session_state.user is None:
    login_page()
else:
    menu = st.sidebar.radio("메뉴", ["🏡 홈", "⚙ 자동 변환", "🔍 오류 자동체크"])

    if menu.startswith("🏡"):
        home_page()
    elif menu.startswith("⚙"):
        auto_convert_page()
    elif menu.startswith("🔍"):
        auto_check_page()




