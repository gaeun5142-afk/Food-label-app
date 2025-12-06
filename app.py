import streamlit as st
import requests
from PIL import Image
import io
import json
import re
from supabase import create_client, Client

# ===== Supabase 설정 =====
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== Flask 백엔드 주소 =====
FLASK_API_URL = "https://food-label-app.onrender.com"


# ===== 유틸: 위반 문구 정리 =====
def clean_violation_text(violation_text: str):
    """
    - 불필요한 특수문자(, 전각괄호 등) 제거
    - '... 위반' 까지만 남기고 뒤는 잘라냄
    - 공백 정리
    """
    if not violation_text:
        return violation_text

    cleaned = str(violation_text)

    # 이상한 특수문자 블록 제거
    while True:
        new_cleaned = re.sub(r'\s*[^()]*', '', cleaned)
        new_cleaned = re.sub(r'\s*（[^）]*）', '', new_cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned

    # '위반' 이라는 단어까지 자르기
    match = re.search(r'위반', cleaned)
    if match:
        cleaned = cleaned[: match.end()].strip()

    # 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ===== 페이지 공통 설정 =====
st.set_page_config(page_title="바른식품표시", layout="wide")

if "user" not in st.session_state:
    st.session_state["user"] = None
if "login_error" not in st.session_state:
    st.session_state["login_error"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "login"


# ===== 로그인 페이지 =====
def show_login_page():
    st.title("🔒 바른식품표시 로그인")

    if st.session_state["login_error"]:
        st.error(st.session_state["login_error"])
        st.session_state["login_error"] = None

    email = st.text_input("이메일", key="login_email")
    password = st.text_input("비밀번호", type="password", key="login_password")

    if st.button("로그인"):
        if not email or not password:
            st.session_state["login_error"] = "이메일과 비밀번호를 모두 입력해 주세요."
            st.rerun()
            return
        try:
            res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user = getattr(res, "user", None)
            if user is None:
                st.session_state["login_error"] = "로그인 실패: 이메일/비밀번호를 확인해 주세요."
                st.rerun()
                return
            st.session_state["user"] = {"id": user.id, "email": user.email}
            st.session_state["login_error"] = None
            st.session_state["page"] = "main"
            st.rerun()
        except Exception as e:
            print("로그인 오류:", e)
            st.session_state["login_error"] = "로그인 실패: 이메일/비밀번호를 확인해 주세요."
            st.rerun()

    st.write("---")
    if st.button("➡️ 회원가입"):
        st.session_state["page"] = "signup"
        st.rerun()


# ===== 회원가입 페이지 =====
def show_signup_page():
    st.title("🆕 회원가입")

    email = st.text_input("이메일", key="signup_email")
    password = st.text_input("비밀번호", type="password", key="signup_password")

    if st.button("회원가입 완료하기"):
        if not email or not password:
            st.error("이메일과 비밀번호를 모두 입력해 주세요.")
        else:
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("회원가입이 완료되었습니다! 이제 로그인해 주세요.")
                st.session_state["page"] = "login"
                st.rerun()
            except Exception as e:
                st.error(f"회원가입 실패: {str(e)}")

    if st.button("⬅️ 로그인으로 돌아가기"):
        st.session_state["page"] = "login"
        st.rerun()


# ===== 상단 바 =====
def show_top_bar():
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("### 바른식품표시 플랫폼")
        if st.session_state["user"]:
            st.markdown(f"**로그인된 사용자:** {st.session_state['user']['email']}")
    with cols[1]:
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.session_state["login_error"] = None
            st.session_state["page"] = "login"
            st.rerun()


# ===== 메인 앱 =====
def show_main_app():
    show_top_bar()

    menu = st.sidebar.radio("메뉴 선택", ["홈", "자동 변환", "오류 자동체크", "식품 관련 사이트"])

    # --- 홈 ---
    if menu == "홈":
        st.title("🏠 바른식품표시 플랫폼")
        st.write(
            "이 웹앱은 식품 표시사항을 **자동으로 생성**하고,  "
            "**디자인과 기준데이터를 비교해 오류를 자동으로 검출**하는 플랫폼입니다."
        )

    # --- 자동 변환 ---
    elif menu == "자동 변환":
        st.title("📄 자동 변환 (QA 기반 표시사항 생성)")

        uploaded_files = st.file_uploader(
            "QA 자료 업로드 (여러 파일 가능)",
            type=["pdf", "jpg", "jpeg", "png", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        if st.button("결과 확인하기"):
            if not uploaded_files:
                st.error("파일을 업로드하세요.")
            else:
                files = [("qa_files", (f.name, f.read(), f.type)) for f in uploaded_files]
                with st.spinner("AI가 QA 자료를 분석 중입니다..."):
                    try:
                        response = requests.post(
                            f"{FLASK_API_URL}/api/upload-qa",
                            files=files,
                            timeout=600,
                        )
                    except Exception as e:
                        st.error(f"서버 연결 오류: {e}")
                    else:
                        if response.status_code == 200:
                            result = response.json()
                            st.success("분석 완료!")
                            st.subheader("📌 생성된 식품표시 기준 데이터 (JSON)")
                            st.json(result)
                        else:
                            st.error("서버에서 오류가 발생했습니다.")
                            st.write("상태 코드:", response.status_code)
                            st.write(response.text)

    # --- 오류 자동체크 ---
    elif menu == "오류 자동체크":
        st.title("🔍 오류 자동체크 ")

        standard_excel = st.file_uploader(
            "📘 기준데이터 (Excel / PDF)", type=["xlsx", "xls", "pdf"]
        )
        design_file = st.file_uploader(
            "🖼️ 디자인 파일 (PDF / 이미지)", type=["pdf", "jpg", "jpeg", "png"]
        )

        if st.button("결과 확인하기"):
            if not design_file:
                st.error("디자인 파일을 업로드하세요.")
            else:
                files = {
                    "design_file": (design_file.name, design_file.read(), design_file.type)
                }
                if standard_excel:
                    files["standard_excel"] = (
                        standard_excel.name,
                        standard_excel.read(),
                        standard_excel.type,
                    )

                with st.spinner("디자인과 기준 데이터를 비교 중입니다..."):
                    try:
                        response = requests.post(
                            f"{FLASK_API_URL}/api/verify-design",
                            files=files,
                            timeout=600,
                        )
                    except Exception as e:
                        st.error(f"서버 연결 오류: {e}")
                    else:
                        if response.status_code == 200:
                            result = response.json()

                            st.success("검사 완료!")

                            # ===== 1. 상단 하이라이트 뷰 =====
                            st.subheader("🔎 AI 정밀 분석 결과 (하이라이트)")
                            highlight_html = result.get("design_ocr_highlighted_html")
                            if highlight_html:
                                st.markdown(
                                    "<div style='font-size:13px; color:#555; margin-bottom:8px;'>"
                                    "* 붉은색으로 표시된 부분은 기준 정보와 다르거나 오타가 의심되는 곳입니다."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                                # 🔴 server.py 에서 만들어준 HTML 그대로 렌더
                                st.markdown(highlight_html, unsafe_allow_html=True)
                            else:
                                st.write("하이라이트 결과가 없습니다.")

                            st.markdown("---")

                            # ===== 2. 점수 + 법령 위반 리포트 =====
                            score = result.get("score", "N/A")
                            law = result.get("law_compliance", {}) or {}
                            status_raw = law.get("status", "")
                            violations_raw = law.get("violations", []) or []

                            # 🔧 violations 가 HTML 블록으로 올 수도 있으므로 정규화
                            violations = []
                            for v in violations_raw:
                                if not v:
                                    continue
                                v_str = str(v)

                                # 1) <li>...</li> 가 포함된 HTML 블록인 경우
                                if "<li" in v_str:
                                    # li 내용만 추출
                                    li_contents = re.findall(
                                        r'<li[^>]*>(.*?)</li>',
                                        v_str,
                                        flags=re.IGNORECASE | re.DOTALL,
                                    )
                                    for li in li_contents:
                                        # li 안의 태그 제거
                                        plain = re.sub(r"<[^>]+>", "", li)
                                        plain = clean_violation_text(plain)
                                        if plain:
                                            violations.append(plain)
                                else:
                                    # 2) 그냥 문자열인 경우
                                    plain = clean_violation_text(v_str)
                                    if plain:
                                        violations.append(plain)

                            # 뱃지 색상
                            if status_raw.lower() == "compliant":
                                badge_color = "#2e7d32"
                                badge_label = "법률 준수"
                                badge_icon = "✅"
                            elif status_raw.lower() == "violation":
                                badge_color = "#d32f2f"
                                badge_label = "법률 위반"
                                badge_icon = "⚠️"
                            else:
                                badge_color = "#546e7a"
                                badge_label = status_raw or "확인 필요"
                                badge_icon = "ℹ️"

                            violations_html = ""
                            if violations:
                                items = "".join(f"<li>{v}</li>" for v in violations)
                                violations_html = f"""
                                <div style="margin-top:12px;">
                                  <div style="font-weight:600; margin-bottom:4px;">위반 사항:</div>
                                  <ul style="margin-top:0; padding-left:20px; font-size:13px; color:#444;">
                                    {items}
                                  </ul>
                                </div>
                                """

                            report_html = f"""
                            <div style="background:#f5f7fb; padding:24px; border-radius:18px; margin-top:8px;">
                              <div style="font-weight:700; font-size:16px; margin-bottom:16px;">📊 검증 결과 리포트</div>
                              <div style="font-size:18px; margin-bottom:10px;">점수:
                                <span style="background:#2962ff; color:#ffffff; padding:6px 14px; border-radius:999px; font-weight:700;">{score}점</span>
                              </div>
                              <div style="margin-top:4px; font-size:14px;">법률 준수 상태:
                                <span style="background:{badge_color}1A; color:{badge_color}; padding:4px 12px; border-radius:999px; font-weight:600;">{badge_icon} {badge_label}</span>
                              </div>
                              {violations_html}
                            </div>
                            """
                            st.markdown(report_html, unsafe_allow_html=True)

                            st.markdown("---")

                            # ===== 3. 상세 문제 카드 =====
                            st.subheader("📌 상세 문제 목록")
                            issues = result.get("issues", []) or []
                            if not issues:
                                st.write("발견된 문제가 없습니다. 👍")
                            else:
                                for i, issue in enumerate(issues, start=1):
                                    issue = issue or {}
                                    title = issue.get("location") or "표시 항목"
                                    desc = issue.get("issue") or ""
                                    expected = issue.get("expected") or ""
                                    actual = issue.get("actual") or ""
                                    suggestion = issue.get("suggestion") or ""

                                    card_html = f"""
                                    <div style="background:#fff9e6; border-radius:14px; padding:16px 20px; margin-bottom:12px; border-left:6px solid #ffb300;">
                                      <div style="font-weight:700; margin-bottom:4px;">[문제 {i}] {title}</div>
                                      <div style="font-size:13px; color:#555; margin-bottom:8px;">{desc}</div>
                                      <div style="font-size:13px; margin-bottom:4px;"><b>정답:</b> {expected}</div>
                                      <div style="font-size:13px; margin-bottom:4px;"><b>실제:</b> {actual}</div>
                                      <div style="font-size:13px; color:#1565c0; margin-top:4px;"><b>수정 제안:</b> {suggestion}</div>
                                    </div>
                                    """
                                    st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.error("서버에서 오류가 발생했습니다.")
                            st.write("상태 코드:", response.status_code)
                            st.write(response.text)

    # --- 식품 관련 사이트 ---
    elif menu == "식품 관련 사이트":
        st.title("🔗 식품 관련 사이트 모음")
        st.markdown(
            """
        ### 📌 유용한 링크  
        - **식약처 식품안전나라**  
          https://www.foodsafetykorea.go.kr  
        - **식품 표시 기준 고시**  
          https://www.foodsafetykorea.go.kr/foodcode/04_03.jsp  
        - **식품 영양성분 DB**  
          https://koreanfood.rda.go.kr/kfi/fct/fctList  
        - **부정불량식품 신고센터 (1399)**  
          https://www.mfds.go.kr
        """
        )


# ===== 메인 엔트리 =====
def main():
    if st.session_state["user"] is None:
        if st.session_state["page"] == "signup":
            show_signup_page()
        else:
            show_login_page()
    else:
        show_main_app()


if __name__ == "__main__":
    main()
