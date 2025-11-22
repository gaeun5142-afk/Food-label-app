import streamlit as st
from supabase import create_client, Client

# ---------------------- Supabase 연결 ---------------------- #
# 🔑 Streamlit Secrets 에 아래 두 개가 들어있다고 가정:
# SUPABASE_URL, SUPABASE_ANON_KEY
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------------- 페이지 설정 ---------------------- #
st.set_page_config(page_title="바른식품표시", layout="wide")

# ---------------------- 로그인 상태 초기화 ---------------------- #
if "user" not in st.session_state:
    st.session_state["user"] = None

# ---------------------- 공통 스타일 ---------------------- #
st.markdown(
    """
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
            min-height: 120px;
        }
        .full-width {
            max-width: 900px;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- 로그인 페이지 ---------------------- #
def login_page():
    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.title("🔐 바른식품표시 로그인")

    with st.form("login_form"):
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        login_btn = st.form_submit_button("로그인")

    if login_btn:
        if not email or not password:
            st.error("이메일/비밀번호를 모두 입력해주세요.")
        else:
            try:
                supabase.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                st.session_state["user"] = email
                st.success("로그인 성공! 메인 페이지로 이동합니다.")
                st.experimental_rerun()
            except Exception:
                st.error("로그인 실패: 이메일/비밀번호를 확인해주세요.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- 홈 페이지 ---------------------- #
def home_page():
    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.title("🏡 바른식품표시 플랫폼")
    st.caption(f"현재 로그인: {st.session_state['user']}")

    if st.button("로그아웃", key="logout_home"):
        st.session_state["user"] = None
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("서비스 소개")

    st.write(
        """
        **바른식품표시**는 식품 라벨 자료를 업로드해서

        - ⚙ 자동 변환 (기준 데이터 만들기)
        - 🔍 오류 자동체크 (표시사항 누락 여부 확인)
        - 🔗 식품 관련 사이트 바로가기

        를 할 수 있도록 만드는 웹앱입니다.  
        지금은 구조와 화면을 먼저 만들어 두는 단계이고,
        나중에 실제 분석 로직(OCR, 법령 기준 체크 등)을 붙일 수 있습니다.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- 자동 변환 페이지 ---------------------- #
def auto_convert_page():
    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.title("⚙ 자동 변환")
    st.caption(f"현재 로그인: {st.session_state['user']}")

    if st.button("로그아웃", key="logout_auto"):
        st.session_state["user"] = None
        st.experimental_rerun()

    st.markdown("---")

    # STEP 1
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 1. 파일 업로드")
    uploaded_file = st.file_uploader(
        "기준 데이터를 만들 원본 파일을 업로드하세요. (엑셀, PDF, 이미지 등)",
        type=["xlsx", "xls", "csv", "pdf", "png", "jpg", "jpeg"],
        key="auto_upload",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 2
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 2. 기준 데이터 생성")

    if st.button("📄 기준 데이터 생성하기", key="auto_generate_btn"):
        if uploaded_file is None:
            st.error("먼저 파일을 업로드해주세요.")
        else:
            # 실제 변환 로직은 나중에 구현
            st.session_state["auto_convert_result"] = (
                "✔ 예시) 업로드한 파일을 기반으로 기준 데이터를 생성했습니다. "
                "추후 여기에서 실제 변환 결과(엑셀, 텍스트 등)를 보여줄 수 있습니다."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 3
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 3. 최종 결과")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(
        st.session_state.get(
            "auto_convert_result", "기준 데이터 생성 결과가 여기 표시됩니다."
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- 오류 자동체크 페이지 ---------------------- #
def auto_check_page():
    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.title("🔍 오류 자동체크")
    st.caption(f"현재 로그인: {st.session_state['user']}")

    if st.button("로그아웃", key="logout_check"):
        st.session_state["user"] = None
        st.experimental_rerun()

    st.markdown("---")

    # STEP 1
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 1. 최종 디자인 파일 업로드")
    uploaded_file = st.file_uploader(
        "검증할 최종 라벨 디자인(PDF 또는 이미지)을 업로드하세요.",
        type=["pdf", "png", "jpg", "jpeg"],
        key="check_upload",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 2
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 2. 검증 시작")

    if st.button("✅ 검증 시작", key="check_start_btn"):
        if uploaded_file is None:
            st.error("먼저 파일을 업로드해주세요.")
        else:
            # 실제 자동 체크 로직은 나중에 구현
            st.session_state["auto_check_result"] = (
                "⚠ 예시) 필수 표시항목 일부가 누락되어 있습니다.\n"
                "- 예시) 알레르기 표시 항목에 '우유' 누락\n"
                "- 예시) 보관방법 문구 미표기\n"
                "추후 실제 기준에 맞춘 상세 체크 결과를 여기 표시할 수 있습니다."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # STEP 3
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("STEP 3. 결과 확인")

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(
        st.session_state.get(
            "auto_check_result", "검증 결과가 여기 표시됩니다."
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- 식품 관련 사이트 페이지 ---------------------- #
def links_page():
    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.title("🔗 식품 관련 사이트")
    st.caption(f"현재 로그인: {st.session_state['user']}")

    if st.button("로그아웃", key="logout_links"):
        st.session_state["user"] = None
        st.experimental_rerun()

    st.markdown("---")

    st.subheader("🏛 공공/기관 사이트")
    st.markdown(
        """
- **식품의약품안전처(MFDS)**  
  - 홈페이지: https://www.mfds.go.kr  
  - 식품안전나라: https://www.foodsafetykorea.go.kr  

- **국가법령정보센터 (식품 관련 법령 검색)**  
  - https://www.law.go.kr  
        """
    )

    st.subheader("📚 참고 자료 (추가 예정)")
    st.write(
        "- 식품 표시 기준 요약 자료\n"
        "- 알레르기 표시 의무 품목 안내\n"
        "- 영양성분 표시 가이드\n"
        "\n필요한 사이트/자료가 생기면 여기 계속 추가하면 돼요."
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- 라우팅 ---------------------- #
if st.session_state["user"] is None:
    # 로그인 안 된 경우
    login_page()
else:
    # 로그인 된 경우: 사이드바 메뉴로 페이지 이동
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["🏡 홈", "⚙ 자동 변환", "🔍 오류 자동체크", "🔗 식품 관련 사이트"],
    )

    if menu.startswith("🏡"):
        home_page()
    elif menu.startswith("⚙"):
        auto_convert_page()
    elif menu.startswith("🔍"):
        auto_check_page()
    elif menu.startswith("🔗"):
        links_page()




