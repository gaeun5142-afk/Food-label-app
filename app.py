import streamlit as st
from supabase import create_client, Client

# 🔑 Streamlit Secrets 에 저장된 Supabase 정보 사용
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.set_page_config(page_title="식품표시 웹앱", layout="centered")


# ---------------------- 카테고리 자동 변환 로직 ----------------------
CATEGORY_RULES = [
    ("과자", "과자류"),
    ("스낵", "과자류"),
    ("쿠키", "과자류"),
    ("초콜릿", "초콜릿류"),
    ("라면", "면류"),
    ("국수", "면류"),
    ("빵", "빵류"),
    ("케이크", "빵류"),
    ("주스", "음료류"),
    ("음료", "음료류"),
    ("커피", "커피류"),
    ("차", "차류"),
]


def auto_convert_category(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    for kw, cat in CATEGORY_RULES:
        if kw in text:
            return cat
    return "기타"


# ---------------------- 오류 자동 체크 로직 ----------------------
FIELD_LABELS = {
    "name": "제품명",
    "category_raw": "식품 유형(입력값)",
    "category_auto": "식품 유형(자동 분류)",
    "volume": "내용량",
    "ingredients": "원재료명",
    "allergy": "알레르기 표시",
    "expiration": "유통/품질유지기한",
}

ALLERGEN_KEYWORDS = [
    "우유",
    "대두",
    "땅콩",
    "밀",
    "계란",
    "돼지고기",
    "닭고기",
    "쇠고기",
    "새우",
    "고등어",
    "게",
    "오징어",
    "조개",
    "호두",
    "토마토",
]


def check_food_label_errors(data: dict):
    errors = []
    warnings = []

    # 필수값 비어있는지 체크
    required = ["name", "category_raw", "volume", "ingredients", "expiration"]
    for key in required:
        if not data.get(key):
            errors.append(f"✅ `{FIELD_LABELS[key]}` 을(를) 입력해주세요.")

    # 알레르기 자동 체크: 원재료에 있는데 알레르기 칸에 없는 경우
    ingredients = data.get("ingredients", "")
    allergy = data.get("allergy", "")
    found = [a for a in ALLERGEN_KEYWORDS if a in ingredients]
    missing = [a for a in found if a not in allergy]

    if missing:
        warnings.append(
            f"⚠ 원재료에 `{', '.join(missing)}` 가 포함되어 있지만, "
            f"`알레르기 표시` 항목에 빠져 있습니다."
        )

    # 카테고리 자동 분류와 입력값이 너무 다르면 참고 메시지
    if data.get("category_raw") and data.get("category_auto"):
        if data["category_auto"] == "기타":
            warnings.append(
                "ℹ 입력한 식품 유형으로 자동 분류가 어려워 `기타`로 처리했습니다. "
                "공식 분류명을 한 번 더 확인해주세요."
            )

    return errors, warnings


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
                st.success("로그인 성공! 잠시 후 대시보드로 이동합니다.")
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

    # ---- 홈 탭 ----
    with tab1:
        st.header("식품표시 웹앱 대시보드")
        st.write(f"👋 {email} 님 환영합니다!")
        st.markdown(
            """
            이 서비스는 **식품 표시사항**을 정리하고,  
            간단한 **자동 카테고리 분류 + 오류 체크**를 도와주는 도구입니다.

            현재 기능:
            - 로그인/로그아웃
            - 식품 표시사항 입력
            - 카테고리 자동 변환
            - 알레르기 표시 누락 자동 경고
            - 라벨 이미지 업로드 (저장은 추후 Supabase Storage로 확장 가능)
            """
        )

    # ---- 식품 등록 탭 ----
    with tab2:
        st.header("식품 표시사항 입력")

        with st.form("food_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("제품명")
                category_raw = st.text_input("식품 유형 (임의로 적어도 됨, 예: 과자, 초콜릿, 라면)")
                volume = st.text_input("내용량 / 중량")
            with col2:
                brand = st.text_input("브랜드명 (선택)")
                storage = st.text_input("보관방법 (예: 실온보관, 냉장보관)")
                expiration = st.text_input("유통기한 / 품질유지기한")

            ingredients = st.text_area("원재료명 및 함량")
            allergy = st.text_input("알레르기 표시")

            # 이미지 업로드
            label_image = st.file_uploader(
                "라벨 / 포장 사진 업로드 (jpg, png)", type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("자동 체크 실행")

        # 폼 제출 후 처리
        if submitted:
            # 카테고리 자동 변환
            category_auto = auto_convert_category(category_raw)

            data = {
                "name": name,
                "category_raw": category_raw,
                "category_auto": category_auto,
                "volume": volume,
                "ingredients": ingredients,
                "allergy": allergy,
                "expiration": expiration,
                "storage": storage,
                "brand": brand,
            }

            errors, warnings = check_food_label_errors(data)

            st.markdown("### ✅ 입력 요약")
            st.write("**제품명:**", name or "-")
            st.write("**식품 유형 (입력값):**", category_raw or "-")
            st.write("**식품 유형 (자동 분류):**", category_auto or "-")
            st.write("**브랜드:**", brand or "-")
            st.write("**내용량:**", volume or "-")
            st.write("**원재료명:**", ingredients or "-")
            st.write("**알레르기:**", allergy or "-")
            st.write("**유통기한:**", expiration or "-")
            st.write("**보관방법:**", storage or "-")

            # 이미지 미리보기
            if label_image is not None:
                st.markdown("**라벨 이미지 미리보기:**")
                st.image(label_image, use_column_width=True)
            else:
                st.info("라벨 이미지를 업로드하지 않았습니다.")

            st.markdown("---")
            st.markdown("### 🔍 자동 체크 결과")

            if errors:
                st.error("아래 항목들을 고쳐야 합니다:")
                for e in errors:
                    st.write("- ", e)
            else:
                st.success("필수 항목은 모두 입력되었습니다.")

            if warnings:
                st.warning("주의/권장 사항:")
                for w in warnings:
                    st.write("- ", w)
            else:
                st.info("추가로 발견된 경고는 없습니다.")

            st.caption("※ 이 체크는 간단한 참고용이며, 실제 법적 검토를 대체하지 않습니다.")

    # ---- 내 계정 탭 ----
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


