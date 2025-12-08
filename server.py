import os
import json
import io
import glob
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import PIL.ImageEnhance
import re
import unicodedata
import html as html_mod  # ✅ 하이라이트용 이스케이프

# --- 설정 및 초기화 ---
load_dotenv()


def normalize_text_strict(text):
    """엄격한 비교용 정규화 (공백/특수문자 유지)"""
    if not isinstance(text, str):
        text = str(text)
    # 유니코드만 정규화, 공백/특수문자는 유지
    return unicodedata.normalize('NFKC', text)


# ⭐ 문자 단위 비교 (strict)
def compare_texts_strict(standard_text, design_text):
    """문자 단위 정확 비교 (AI 없이)"""
    std_norm = normalize_text_strict(standard_text)
    des_norm = normalize_text_strict(design_text)

    issues = []
    max_len = max(len(std_norm), len(des_norm))

    for i in range(max_len):
        std_char = std_norm[i] if i < len(std_norm) else '(없음)'
        des_char = des_norm[i] if i < len(des_norm) else '(없음)'

        if std_char != des_char:
            issues.append({
                "position": i,
                "expected": std_char,
                "actual": des_char,
                "context_before": std_norm[max(0, i - 5):i],
                "context_after": std_norm[i + 1:min(len(std_norm), i + 6)]
            })

    return issues


def add_issue_positions(issues, full_text: str):
    """
    issues 리스트에 'position' 키를 채워넣는 함수.
    full_text: OCR로 뽑은 전체 텍스트 (design_ocr_text)

    - 기본적으로 issue["actual"] 전체 문자열 기준으로 위치를 찾는다.
    - 못 찾으면 issue["expected"]로 다시 시도한다.
    """
    if not full_text or not issues:
        return issues

    text = full_text

    for issue in issues:
        # 이미 position이 있으면 건너뛰기
        if isinstance(issue.get("position"), int):
            continue

        actual = (issue.get("actual") or "").strip()
        expected = (issue.get("expected") or "").strip()

        pos = -1

        # 1) 우선 actual(실제 표시된 문자열) 위치 찾기
        if actual:
            pos = text.find(actual)

        # 2) 못 찾으면 expected(정답) 위치라도 찾아보기
        if pos == -1 and expected:
            pos = text.find(expected)

        # 3) 찾으면 position 저장
        if pos != -1:
            issue["position"] = pos

    return issues


#5번
def ocr_with_voting(image_file, num_runs=5):
    """같은 이미지를 여러 번 OCR해서 가장 많이 나온 결과 선택"""
    from collections import Counter

    # ⭐ 함수 시작 시 파일 포인터 초기화
    image_file.seek(0)

    results = []
    print(f"🔄 OCR 안정화: {num_runs}회 실행 중...")

    for i in range(num_runs):
        try:
            image_file.seek(0)  # 파일 포인터 초기화
            parts = [
                PROMPT_EXTRACT_RAW_TEXT,
                process_file_to_part(image_file)
            ]

            model = genai.GenerativeModel(MODEL_NAME, generation_config={
                "temperature": 0.0,
                "top_k": 1,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            })

            response = model.generate_content(parts)
            # ⭐ finish_reason 체크 추가
            if response.candidates and response.candidates[0].finish_reason == 2:
                print(f"  ⚠️ {i+1}번째 OCR: 토큰 제한 초과, 재시도 중...")
                continue

            result_text = response.text.strip()

            # JSON 파싱
            if result_text.startswith("```"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text[3:-3]

            ocr_result = json.loads(result_text)
            extracted_text = ocr_result.get('raw_text', '')
            results.append(extracted_text)

            print(f"  {i + 1}/{num_runs} 완료: {len(extracted_text)}자")

        except Exception as e:
            print(f"  ⚠️ {i + 1}번째 OCR 실패: {e}")
            continue

    # ⭐ 함수 종료 전에도 리셋 (다음 사용을 위해)
    image_file.seek(0)

    if not results:
        raise Exception("모든 OCR 시도 실패")

    # 가장 많이 나온 결과 선택
    counter = Counter(results)
    most_common_text, count = counter.most_common(1)[0]

    print(f"📊 투표 결과:")
    for text, freq in counter.most_common():
        print(f"  - {freq}/{num_runs}회: {text[:50]}...")

    print(f"✅ 최종 선택: {count}/{num_runs}회 일치")

    return most_common_text


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# API 키 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("🚨 경고: .env 파일에 GOOGLE_API_KEY가 없습니다!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 설정 (기본값, 자동 감지로 덮어씌워질 수 있음)
MODEL_NAME = 'gemini-1.5-flash'

# ⭐ 모든 함수에서 동일하게 사용할 Generation Config
STABLE_GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "candidate_count": 1,
    "max_output_tokens": 32768,
    "response_mime_type": "application/json"
}

# ⭐ 이미지 전처리 고정 파라미터
IMAGE_TARGET_SIZE = 2400
IMAGE_DPI = 300
CLAHE_CLIP_LIMIT = 2.5
THRESHOLD_BLOCK_SIZE = 15
THRESHOLD_C = 3


# 모델 사용 가능 여부 확인 함수
def check_available_models():
    """사용 가능한 모델 목록을 확인하고 적절한 모델을 반환합니다."""
    global MODEL_NAME
    try:
        models = genai.list_models()
        available_models = []
        print("\n📋 사용 가능한 모델 목록:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                # 모델 이름에서 'models/' 접두사 제거
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
                print(f"   - {model_name}")

        # Flash 모델 우선 선택
        for model in available_models:
            if 'flash' in model.lower():
                MODEL_NAME = model
                print(f"\n✅ 추천 모델 선택: {MODEL_NAME}\n")
                return MODEL_NAME

        # Flash가 없으면 Pro 모델 선택
        for model in available_models:
            if 'pro' in model.lower():
                MODEL_NAME = model
                print(f"\n✅ Pro 모델 선택: {MODEL_NAME}\n")
                return MODEL_NAME

        # 둘 다 없으면 첫 번째 모델 사용
        if available_models:
            MODEL_NAME = available_models[0]
            print(f"\n✅ 첫 번째 모델 선택: {MODEL_NAME}\n")
            return MODEL_NAME

        print(f"\n⚠️ 사용 가능한 모델을 찾을 수 없습니다. 기본값 사용: {MODEL_NAME}\n")
        return None
    except Exception as e:
        print(f"⚠️ 모델 목록 확인 실패: {e}")
        print(f"⚠️ 기본 모델 사용: {MODEL_NAME}\n")
        return None


# 서버 시작 시 모델 확인 및 자동 설정
if GOOGLE_API_KEY:
    check_available_models()
else:
    print(f"⚠️ API 키가 없어 모델 확인을 건너뜁니다. 기본 모델 사용: {MODEL_NAME}\n")


# --- 법령 텍스트 로드 ---
def load_law_texts() -> str:
    """법령 .txt 파일들을 모두 읽어 하나의 큰 텍스트로 합칩니다."""
    print("📚 법령 파일들을 읽어오는 중...")
    # 프로젝트 루트와 현재 디렉토리 모두 확인
    law_files = glob.glob("law_text_*.txt") + glob.glob("../law_text_*.txt")

    if not law_files:
        print("⚠️ 법령 파일이 없습니다. 법률 검토 기능이 제한될 수 있습니다.")
        return ""

    all_law_text = ""
    for file_path in law_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_law_text += f"--- 법령 [{file_path}] 시작 ---\n\n"
                all_law_text += f.read()
                all_law_text += f"\n\n--- 법령 [{file_path}] 끝 ---\n\n"
            print(f"✅ 법령 파일 '{file_path}' 로드 완료")
        except Exception as e:
            print(f"❌ 법령 파일 '{file_path}' 읽기 실패: {e}")

    print(f"✅ 모든 법령 파일 로드 완료 (총 {len(all_law_text)}자)")
    return all_law_text


ALL_LAW_TEXT = load_law_texts()

# --- 프롬프트 (지시사항) ---

PROMPT_EXTRACT_INGREDIENT_INFO = """
당신은 한국 식품 라벨 OCR 전문가입니다.
이미지에서 원부재료 표시사항을 **정확하게** 추출하세요.
추측하거나 창의적으로 해석하지 말고, 보이는 텍스트만 정확히 추출하세요.

🚨 절대 규칙 🚨
🚨 절대 규칙 🚨
1. 이미지에 **보이는 글자만** 추출 (추론/보정 금지)
2. **특수문자(쉼표, 점, 괄호) 누락도 그대로** 추출
3. 오타, 띄어쓰기, 특수문자 모두 **정확히 그대로**
4. 문법적으로 틀려도 **이미지와 100% 동일**하게



[추출해야 할 정보]
1. **원재료명**: 원재료의 정확한 명칭 (오타 없이)
2. **복합원재료 내역**: 괄호 안의 하위 원재료 정보 (예: (탈지대두, 소맥))
3. **원산지 정보**: 원산지 표기 (예: 외국산, 국내산, 인도산 등)
4. **함량 정보**: 백분율(%) 표시
5. **알레르기 유발물질**: 알레르기 표시 정보
6. **식품첨가물**: 첨가물명과 용도 병기 여부

[출력 형식]
반드시 JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요:
{
  "ingredient_name": "원재료명",
  "content_percentage": "함량(%)",
  "sub_ingredients": "하위원재료 내역 (복합원재료인 경우)",
  "origin": "원산지 정보",
  "allergens": ["알레르기 유발물질 목록"],
  "additives": ["식품첨가물 목록"],
  "raw_ocr_text": "이미지에서 추출한 전체 텍스트 (원본 그대로)"
}
"""
PROMPT_EXTRACT_RAW_TEXT = """
당신은 OCR 전문가입니다. 이미지의 텍스트를 **기계적으로** 추출하세요.

🤖 기계 모드 활성화:
- 철자 교정기 OFF
- 문법 검사기 OFF
- 자동 완성 OFF
- 추론 엔진 OFF

출력 규칙:
1. 보이는 글자 → 그대로 출력
2. 틀린 글자 → 틀린 대로 출력
3. 빠진 쉼표 → 빠진 대로 출력
4. 이상한 숫자 → 이상한 대로 출력

예시:
- 이미지: "전반가공품" → 출력: "전반가공품" (전분가공품 아님!)
- 이미지: "대두 게" → 출력: "대두 게" (대두, 게 아님!)
- 이미지: "221%" → 출력: "221%" (2.21% 아님!)

JSON 형식으로만 응답:
{
  "raw_text": "있는 그대로의 텍스트"
}
"""

# 1. 기준 데이터 생성용 (엑셀 + 원재료 사진들 -> 정답지 생성)
PROMPT_CREATE_STANDARD = """
당신은 식품 규정 및 표시사항 전문가입니다.
제공된 [배합비 데이터(Excel)]와 [원재료 표시사항 사진들에서 추출한 정보]를 종합하여,
법적으로 완벽한 **'식품표시사항 기준 데이터(Standard)'**를 실제 라벨 형식으로 생성하세요.

[분석 단계]
1. **Excel 데이터 분석**: 배합비율(%)이 높은 순서대로 원재료 나열 순서를 결정하세요. (가장 중요)
2. **이미지 데이터 매핑**: Excel에 적힌 원재료명(예: '간장')에 해당하는 사진(원재료 라벨)을 찾아서 상세 정보(복합원재료 내역, 알레르기, 원산지)를 보강하세요.
    - 예: Excel엔 '간장'만 있지만, 사진에 '탈지대두(인도산), 소맥(밀)'이 있다면 이를 반영해야 함.
    - **중요**: 보관방법, 포장재질 등은 무시하고 원재료 관련 정보만 추출하세요.
3. **법률 검토**: 제공된 법령을 참고하여 표시사항이 법적으로 올바른지 확인하세요.
4. **최종 조합**: 품목제조보고서 기반의 비율과 원재료 라벨의 상세 내용을 합쳐 최종 표시 텍스트를 만드세요.

[출력 양식 - JSON]
반드시 아래 JSON 형식으로만 응답하세요. 실제 식품 라벨 형식처럼 구조화하세요.
{
    "product_info": {
        "product_name": "제품명",
        "food_type": "식품의 유형 (예: 어묵(유탕처리제품/비살균))",
        "net_weight": "내용량 (예: 1kg)",
        "expiration_date": "소비기한 (예: 전면 별도표시일까지)",
        "storage_method": "보관방법 (예: 0~10℃이하 냉장보관)",
        "packaging_material": "포장재질 (예: 폴리에틸렌(내면))",
        "item_report_number": "품목보고번호",
        "front_calories": "전면부 총열량/문구 (예: 1,291kcal / 연육70.6%, 당근4.41%)"
    },
    "ingredients": {
        "structured_list": [
            "냉동연육70.6%(외국산/어육살, 설탕, D-소비톨, 산도조절제)",
            "전분가공품1 [카사바전분(태국, 베트남산), 감자전분]",
            "혼합제제[인산이전분(타피오카), 덱스트린]",
            "당근(국내산)",
            "..."
        ],
        "continuous_text": "냉동연육70.6%(외국산/어육살, 설탕, D-소비톨, 산도조절제), 전분가공품1 [카사바전분(태국, 베트남산), 감자전분], 혼합제제[인산이전분(타피오카), 덱스트린], 당근(국내산), ..."
    },
    "allergens": {
        "contains": ["대두", "게"],
        "manufacturing_facility": "본 제품은 밀, 계란, 새우, 오징어, 고등어, 우유, 쇠고기, 토마토, 조개류(굴․전복,홍합 포함)를 사용한 제품과 같은 제조시설에서 제조하고 있습니다."
    },
    "nutrition_info": {
        "total_content": "1000 g",
        "per_100g": {
            "calories": "130 Kcal",
            "sodium": {"amount": "530 mg", "daily_value": "27%"},
            "fat": {"amount": "1.5 g", "daily_value": "3%"},
            "cholesterol": {"amount": "17 mg", "daily_value": "6%"},
            "carbohydrates": {"amount": "19 g", "daily_value": "6%"},
            "sugars": {"amount": "5 g", "daily_value": "5%"},
            "trans_fat": {"amount": "0 g", "daily_value": "0%"},
            "saturated_fat": {"amount": "0.3 g", "daily_value": "2%"},
            "protein": {"amount": "10 g", "daily_value": "18%"}
        },
        "disclaimer": "1일 영양성분 기준치에 대한 비율(%)은 2,000kcal 기준이므로 개인의 필요 열량에 따라 다를 수 있습니다."
    },
    "manufacturer": {
        "name": "삼진식품(주)",
        "address": "부산광역시 사하구 다대로 1066번길 51(장림동)"
    },
    "precautions": [
        "반드시 냉장보관하시고 개봉 후에는 빠른시일 내 섭취하시길 바랍니다.",
        "간혹 흑막이 발견될 수 있으나 생선 내부복막이오니 안심하고 드시기 바랍니다.",
        "반품 및 교환: 유통 중 변질 파손된 제품은 본사 및 구입처에서 교환해드립니다.",
        "본 제품은 공정거래위원회고시 소비자 분쟁해결기준에 의거 교환 또는 보상받을 수 있습니다.",
        "부정, 불량식품 신고는 국번없이 1399"
    ],
    "law_compliance": {
        "status": "compliant" | "needs_review",
        "issues": ["법률 위반 사항 목록 (있는 경우)"]
    },
    "details": [
        {"name": "원재료명", "ratio": "배합비율", "origin": "원산지", "sub_ingredients": "하위원료"}
    ]
}

**중요**: 
- Excel 데이터에서 추출 가능한 모든 정보를 포함하세요.
- 영양정보는 Excel에 있는 경우에만 포함하고, 없으면 빈 객체로 두세요.
- 원재료명은 배합비율 순서대로 정확히 나열하세요.
- 실제 라벨에 표시되는 형식 그대로 구조화하세요.
"""

# 2. 디자인 검증용 (정답지 vs 디자인PDF)

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사 AI입니다.
제공된 [Standard(기준서)]와 [Design(디자인)]을 1:1 정밀 대조하여, 아래 규칙에 따라 냉철하게 채점하세요.

=== 🚨 초중요: OCR 규칙 🚨 ===
**절대 금지 사항**:
❌ 맞춤법 자동 보정 금지 (틀린 글자도 그대로 추출)
❌ 오타 수정 금지 (전반 → 전분 수정 금지)
❌ 띄어쓰기 자동 보정 금지
❌ 숫자/단위 보정 금지 (900g과 900 g은 다름)
❌ 문장부호 보정 금지 (점, 쉼표 빠진 것도 그대로)

... (기존 PROMPT_VERIFY_DESIGN 내용 그대로, 생략 없이 유지) ...
"""


def check_image_quality(img):
    """이미지 품질을 확인하고 경고 반환"""
    width, height = img.size
    warnings = []

    if width < 800 or height < 800:
        warnings.append(f"⚠️ 이미지 해상도가 낮습니다 ({width}x{height}). 정확도가 떨어질 수 있습니다.")

    # 이미지가 너무 밝거나 어두운지 확인
    if img.mode in ('L', 'RGB'):
        if img.mode == 'RGB':
            img_gray = img.convert('L')
        else:
            img_gray = img
        pixels = list(img_gray.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        if avg_brightness < 50:
            warnings.append("⚠️ 이미지가 너무 어둡습니다.")
        elif avg_brightness > 200:
            warnings.append("⚠️ 이미지가 너무 밝습니다.")

    return warnings


# --- 파일 처리 함수들 ---

def process_file_to_part(file_storage):
    """파일을 Gemini가 이해할 수 있는 Part 객체로 변환"""
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)  # 포인터 초기화

    # 엑셀 파일은 텍스트(CSV)로 변환해서 주는게 AI가 더 잘 이해함
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 🔥 이미지 전처리 최소화 (안정성 향상)
    if mime_type.startswith('image/'):
        try:
            img = PIL.Image.open(io.BytesIO(file_data))

            # ✅ 최소한의 전처리만 수행
            # 1. 투명도 제거 (필수)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = PIL.Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background

            # 2. RGB 유지 (흑백 변환 제거)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. 해상도만 조정 (너무 작으면)
            width, height = img.size
            if width < 1200 or height < 1200:
                scale = max(1200 / width, 1200 / height)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, PIL.Image.LANCZOS)

            # ❌ 대비, 선명도, 밝기 조정 제거 (불안정성 원인)

            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG', dpi=(300, 300))
            byte_io.seek(0)

            return {"mime_type": "image/png", "data": byte_io.read()}
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패 (원본 사용): {e}")
            return {"mime_type": mime_type, "data": file_data}

    return {"mime_type": mime_type, "data": file_data}


def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출"""
    try:
        image_data = image_file.read()
        image_file.seek(0)

        img_pil = PIL.Image.open(io.BytesIO(image_data))

        generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "candidate_count": 1,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }

        model = genai.GenerativeModel(MODEL_NAME)

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        response = model.generate_content(parts)

        result_text = response.text.strip()
        # JSON 파싱
        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        elif result_text.startswith("```"):
            result_text = result_text.split("```")[1].strip()
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()

        return json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"원재료 정보 JSON 파싱 실패: {e}")
        print(f"응답 텍스트: {result_text[:500]}...")
        return None
    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        return None


def create_standard_excel(data):
    """기준 데이터를 엑셀 파일로 생성"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 제품 정보 시트
        if 'product_info' in data:
            product_df = pd.DataFrame([data['product_info']])
            product_df.to_excel(writer, sheet_name='제품정보', index=False)

        # 2. 원재료명 시트
        if 'ingredients' in data:
            ingredients_data = []
            if 'structured_list' in data['ingredients']:
                for idx, item in enumerate(data['ingredients']['structured_list'], 1):
                    ingredients_data.append({
                        '순번': idx,
                        '원재료명': item
                    })
            ingredients_df = pd.DataFrame(ingredients_data)
            if not ingredients_df.empty:
                ingredients_df.to_excel(writer, sheet_name='원재료명', index=False)

            # 연속 텍스트도 추가
            if 'continuous_text' in data['ingredients']:
                continuous_df = pd.DataFrame([{
                    '원재료명_연속텍스트': data['ingredients']['continuous_text']
                }])
                continuous_df.to_excel(writer, sheet_name='원재료명_연속텍스트', index=False)

        # 3. 알레르기 정보 시트
        if 'allergens' in data:
            allergens_data = []
            if 'contains' in data['allergens']:
                allergens_data.append({
                    '항목': '함유 알레르기 유발물질',
                    '내용': ', '.join(data['allergens']['contains'])
                })
            if 'manufacturing_facility' in data['allergens']:
                allergens_data.append({
                    '항목': '제조시설 안내',
                    '내용': data['allergens']['manufacturing_facility']
                })
            if allergens_data:
                allergens_df = pd.DataFrame(allergens_data)
                allergens_df.to_excel(writer, sheet_name='알레르기정보', index=False)

        # 4. 영양정보 시트
        if 'nutrition_info' in data and 'per_100g' in data['nutrition_info']:
            nutrition_data = []
            nut = data['nutrition_info']['per_100g']
            if 'calories' in nut:
                nutrition_data.append({
                    '영양성분': '총 열량',
                    '100g 당': nut['calories'],
                    '1일 영양성분 기준치에 대한 비율(%)': '-'
                })
            for key, value in nut.items():
                if key != 'calories' and isinstance(value, dict):
                    nutrition_data.append({
                        '영양성분': key,
                        '100g 당': value.get('amount', ''),
                        '1일 영양성분 기준치에 대한 비율(%)': value.get('daily_value', '')
                    })
            if nutrition_data:
                nutrition_df = pd.DataFrame(nutrition_data)
                nutrition_df.to_excel(writer, sheet_name='영양정보', index=False)

        # 5. 제조원 정보 시트
        if 'manufacturer' in data:
            manufacturer_df = pd.DataFrame([data['manufacturer']])
            manufacturer_df.to_excel(writer, sheet_name='제조원정보', index=False)

        # 6. 주의사항 시트
        if 'precautions' in data:
            precautions_df = pd.DataFrame([{'주의사항': item} for item in data['precautions']])
            precautions_df.to_excel(writer, sheet_name='주의사항', index=False)

        # 7. 상세 정보 시트 (원재료 상세)
        if 'details' in data and data['details']:
            details_df = pd.DataFrame(data['details'])
            details_df.to_excel(writer, sheet_name='원재료상세', index=False)

    output.seek(0)
    return output


# --- 라우트 ---

@app.route('/')
def index():
    return render_template('index.html')


# 1단계: 정답지 만들기 (엑셀 + 원재료 사진들 몽땅)
@app.route('/api/create-standard', methods=['POST'])
def create_standard():
    print("⚙️ 1단계: 기준 데이터 생성 시작...")

    # 1. 엑셀 파일 (배합비)
    excel_file = request.files.get('excel_file')

    # 2. 원재료 이미지들 (여러 개)
    raw_images = request.files.getlist('raw_images')

    if not excel_file:
        return jsonify({"error": "배합비 엑셀 파일이 필요합니다."}), 400

    # AI에게 보낼 데이터 꾸러미 만들기
    parts = []

    # (1) 프롬프트 + 법령 정보
    enhanced_prompt = PROMPT_CREATE_STANDARD
    if ALL_LAW_TEXT:
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)

    # (2) 엑셀 데이터
    excel_part = process_file_to_part(excel_file)
    if excel_part:
        parts.append(excel_part)

    # (3) 원재료 사진들 - 필요한 정보만 추출
    ingredient_info_list = []
    for img in raw_images[:15]:
        print(f"📷 원재료 이미지 처리 중: {img.filename}")
        ingredient_info = extract_ingredient_info_from_image(img)
        if ingredient_info:
            ingredient_info_list.append(ingredient_info)

    # 추출된 원재료 정보를 텍스트로 변환하여 추가
    if ingredient_info_list:
        ingredients_text = "--- [원재료 표시사항에서 추출한 정보] ---\n"
        for idx, info in enumerate(ingredient_info_list, 1):
            ingredients_text += f"\n[원재료 {idx}]\n"
            ingredients_text += json.dumps(info, ensure_ascii=False, indent=2)
            ingredients_text += "\n"
        ingredients_text += "--- [원재료 정보 끝] ---\n"
        parts.append({"text": ingredients_text})

    print(f"📂 처리 중: 엑셀 1개 + 원재료 이미지 {len(raw_images)}장 (정보 추출 완료)")

    try:
        generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "candidate_count": 1,
            "max_output_tokens": 32768,
            "response_mime_type": "application/json"
        }

        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config
        )

        response = model.generate_content(parts)

        # JSON 파싱
        result_text = response.text.strip()

        # JSON 코드 블록 제거
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            # ``` ... ``` 형식 처리
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]

        result_text = result_text.strip()

        # JSON 파싱 시도
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 1000자): {result_text[:1000]}")
            print(f"오류 위치: line {json_err.lineno}, column {json_err.colno}")
            # JSON 수정 시도 (마지막 쉼표 제거 등)
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 기준 데이터 엑셀 파일 다운로드
@app.route('/api/download-standard-excel', methods=['POST'])
def download_standard_excel():
    """기준 데이터를 엑셀 파일로 다운로드"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "기준 데이터가 없습니다."}), 400

        excel_buffer = create_standard_excel(data)
        product_name = data.get('product_info', {}).get('product_name', '기준데이터') or data.get('product_name', '기준데이터')
        filename = f"{product_name}_기준데이터.xlsx"

        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"❌ 엑셀 다운로드 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 엑셀 파일에서 기준 데이터 읽기
@app.route('/api/read-standard-excel', methods=['POST'])
def read_standard_excel():
    """엑셀 파일에서 기준 데이터를 읽어옴"""
    try:
        excel_file = request.files.get('excel_file')
        if not excel_file:
            return jsonify({"error": "엑셀 파일이 필요합니다."}), 400

        df_dict = pd.read_excel(
            io.BytesIO(excel_file.read()),
            sheet_name=None,
            engine='openpyxl',
            dtype=str,
            keep_default_na=False,
            na_filter=False
        )

        for sheet_name, df in df_dict.items():
            df_dict[sheet_name] = df.astype(str)

        result = {}

        if '제품정보' in df_dict:
            product_info = df_dict['제품정보'].to_dict('records')[0]
            result['product_info'] = product_info

        first_sheet_name = list(df_dict.keys())[0]
        first_sheet_df = df_dict[first_sheet_name]

        if '원재료명' in df_dict:
            ingredients_list = df_dict['원재료명']['원재료명'].dropna().tolist()
            result['ingredients'] = {
                'structured_list': ingredients_list,
                'continuous_text': ', '.join(ingredients_list)
            }
        elif '원재료명_연속텍스트' in df_dict:
            continuous_text = df_dict['원재료명_연속텍스트']['원재료명_연속텍스트'].iloc[0]
            result['ingredients'] = {
                'structured_list': continuous_text.split(', '),
                'continuous_text': continuous_text
            }
        elif not first_sheet_df.empty:
            first_column = first_sheet_df.columns[0]
            if '원재료명' in first_sheet_df.columns:
                ingredients_list = first_sheet_df['원재료명'].dropna().tolist()
            else:
                ingredients_list = first_sheet_df[first_column].dropna().astype(str).tolist()

            if ingredients_list:
                result['ingredients'] = {
                    'structured_list': ingredients_list,
                    'continuous_text': ', '.join(ingredients_list)
                }

        if '알레르기정보' in df_dict:
            allergens_df = df_dict['알레르기정보']
            result['allergens'] = {}
            for _, row in allergens_df.iterrows():
                if row['항목'] == '함유 알레르기 유발물질':
                    result['allergens']['contains'] = row['내용'].split(', ')
                elif row['항목'] == '제조시설 안내':
                    result['allergens']['manufacturing_facility'] = row['내용']

        if '영양정보' in df_dict:
            nutrition_df = df_dict['영양정보']
            per_100g = {}
            for _, row in nutrition_df.iterrows():
                if row['영양성분'] == '총 열량':
                    per_100g['calories'] = row['100g 당']
                else:
                    per_100g[row['영양성분']] = {
                        'amount': row['100g 당'],
                        'daily_value': row['1일 영양성분 기준치에 대한 비율(%)']
                    }
            result['nutrition_info'] = {'per_100g': per_100g}

        if '제조원정보' in df_dict:
            result['manufacturer'] = df_dict['제조원정보'].to_dict('records')[0]

        if '주의사항' in df_dict:
            result['precautions'] = df_dict['주의사항']['주의사항'].tolist()

        if '원재료상세' in df_dict:
            result['details'] = df_dict['원재료상세'].to_dict('records')

        return jsonify(result)
    except Exception as e:
        print(f"❌ 엑셀 읽기 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 2단계: 검증하기 (엑셀 파일 또는 JSON + 디자인 이미지)
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")

    try:
        # 1. 파일 받기
        design_file = request.files.get('design_file')
        standard_excel = request.files.get('standard_excel')
        standard_json = request.form.get('standard_data')

        if not design_file:
            return jsonify({"error": "디자인 파일이 필요합니다."}), 400

        design_file.seek(0)
        if standard_excel:
            standard_excel.seek(0)

        # 2. 기준 데이터 로딩 (엑셀 → JSON)
        if standard_excel:
            try:
                df_dict = pd.read_excel(
                    io.BytesIO(standard_excel.read()),
                    sheet_name=None,
                    engine='openpyxl',
                    dtype=str,
                    keep_default_na=False
                )

                first_sheet_name = list(df_dict.keys())[0]
                first_sheet_df = df_dict[first_sheet_name]
                standard_data = {}

                if not first_sheet_df.empty:
                    col = first_sheet_df.columns[0]
                    if '원재료명' in first_sheet_df.columns:
                        col = '원재료명'

                    ingredients_list = first_sheet_df[col].dropna().astype(str).tolist()
                    standard_data = {
                        'ingredients': {
                            'structured_list': ingredients_list,
                            'continuous_text': ', '.join(ingredients_list)
                        }
                    }

                standard_json = json.dumps(standard_data, ensure_ascii=False)

            except Exception as e:
                return jsonify({"error": f"엑셀 읽기 실패: {str(e)}"}), 400

        # 3. ✅ 법령 텍스트 로딩
        law_text = ""
        all_law_files = glob.glob('law_*.txt')
        print(f"📚 법령 파일 로딩 중: {len(all_law_files)}개 발견")

        for file_path in all_law_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    law_text += f"\n\n=== [참고 법령: {file_path}] ===\n{content}\n==========================\n"
            except Exception as e:
                print(f"⚠️ 법령 파일 읽기 실패 ({file_path}): {e}")

        # 4. ✅ OCR 안정화 (3회 재시도)
        forced_design_text = ""
        ocr_errors = []

        for attempt in range(1, 4):
            try:
                print(f"🔄 OCR 시도 {attempt}/3")

                design_file.seek(0)

                ocr_parts = [
                    PROMPT_EXTRACT_RAW_TEXT,
                    process_file_to_part(design_file)
                ]

                ocr_model = genai.GenerativeModel(
                    MODEL_NAME,
                    generation_config={
                        "temperature": 0.0,
                        "top_k": 1,
                        "top_p": 1.0,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 8192
                    }
                )

                ocr_response = ocr_model.generate_content(ocr_parts)
                raw_text = ocr_response.text.strip()

                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1].strip()
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()

                ocr_json = json.loads(raw_text)
                forced_design_text = ocr_json.get("raw_text", "").strip()

                if forced_design_text:
                    print(f"✅ OCR 성공 ({attempt}회)")
                    break
                else:
                    raise ValueError("raw_text 비어 있음")

            except Exception as e:
                ocr_errors.append(str(e))

        if not forced_design_text:
            forced_design_text = "[OCR 실패]"

        # 5. ✅ Gemini 프롬프트 구성 (📚 법령 반드시 포함)
        parts = [f"""
🚨🚨🚨 절대 규칙 🚨🚨🚨
- 띄어쓰기 중요: "16 g" ≠ "16g"
- 숫자 그대로: "221%" → "221%"
- 오타 그대로 유지
- 절대 추측 금지

{PROMPT_VERIFY_DESIGN}

[참고 법령]
{law_text[:60000]}

[기준 데이터]
{standard_json}

[디자인 OCR (강제 추출)]
{forced_design_text}
"""]

        parts.append(process_file_to_part(design_file))

        # 6. ✅ Gemini 호출
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(parts)

        result_text = response.text.strip()

        match = re.search(r"(\{.*\})", result_text, re.DOTALL)
        if not match:
            return jsonify({"error": "Gemini 응답에서 JSON을 찾지 못함"}), 500

        json_obj = json.loads(match.group(1))

        # ✅ OCR 원문 강제 삽입
        json_obj["design_ocr_text"] = forced_design_text

        # 7. ✅ 하이라이트 위치 계산 (단어 전체 기준)
        design_text = forced_design_text
        issues = json_obj.get("issues", [])

        # position 채우기
        issues = add_issue_positions(issues, design_text)
        json_obj["issues"] = issues

        # 하이라이트 HTML 생성
        highlight_html = design_text

        # 뒤에서 앞으로 처리해서 인덱스 꼬임 방지
        for issue in sorted(issues, key=lambda x: x.get("position", -1), reverse=True):
            pos = issue.get("position")
            if not isinstance(pos, int) or pos < 0 or pos >= len(highlight_html):
                continue

            actual = (issue.get("actual") or "").strip()
            expected = (issue.get("expected") or "").strip()

            # 실제 표시 문자열 전체 길이 기준
            if actual:
                length = len(actual)
            else:
                # fallback: 한 글자만
                length = 1

            start = pos
            end = pos + length

            # 범위 방어
            if start < 0:
                start = 0
            if end > len(highlight_html):
                end = len(highlight_html)

            target_text = highlight_html[start:end]

            span = (
                "<span style='background:#ffe6e6; "
                "color:#d32f2f; font-weight:bold;' "
                f"title='정답: {html_mod.escape(expected)}'>"
                f"{html_mod.escape(target_text)}</span>"
            )

            highlight_html = highlight_html[:start] + span + highlight_html[end:]

        # 줄바꿈을 <br>로 변환해서 브라우저에서 그대로 보이도록
        highlight_html = highlight_html.replace("\n", "<br>")
        json_obj["design_ocr_highlighted_html"] = highlight_html

        return jsonify(json_obj)

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify-design-strict', methods=['POST'])
def verify_design_strict():
    """Python으로 정확한 비교 (AI 없이)"""
    try:
        design_file = request.files.get('design_file')
        standard_json = request.form.get('standard_data')

        if not design_file or not standard_json:
            return jsonify({"error": "파일과 기준 데이터가 필요합니다"}), 400

        # ⭐ 파일 포인터 초기화 (반드시 처음으로!)
        design_file.seek(0)

        standard_data = json.loads(standard_json)

        # 1. OCR 수행 (Gemini)
        parts = [
            PROMPT_EXTRACT_RAW_TEXT,
            process_file_to_part(design_file)
        ]

        model = genai.GenerativeModel(MODEL_NAME, generation_config={
            "temperature": 0.0,
            "top_k": 1,
            "response_mime_type": "application/json"
        })
        response = model.generate_content(parts)

        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        elif result_text.startswith("```json"):
            result_text = result_text[3:-3]

        design_ocr = json.loads(result_text)

        # 2. Python으로 정확한 비교 (AI 없이!)
        all_issues = []

        # 원재료명 비교
        if 'ingredients' in standard_data:
            std_text = standard_data['ingredients']['continuous_text']
        else:
            std_text = ""
        des_text = design_ocr.get('raw_text', '')
        issues = compare_texts_strict(std_text, des_text)

        for issue in issues:
            all_issues.append({
                "type": "Critical" if issue['expected'] not in [' ', ',', '.'] else "Minor",
                "location": f"원재료명 (위치: {issue['position']})",
                "issue": f"'{issue['expected']}' → '{issue['actual']}'",
                "expected": std_text,
                "actual": des_text,
                "suggestion": f"위치 {issue['position']}의 '{issue['actual']}'을(를) '{issue['expected']}'(으)로 수정"
            })

        # 점수 계산
        critical_count = sum(1 for i in all_issues if i['type'] == 'Critical')
        minor_count = sum(1 for i in all_issues if i['type'] == 'Minor')
        score = max(0, 100 - critical_count * 5 - minor_count * 2)

        return jsonify({
            "design_ocr_text": design_ocr.get('raw_text', ''),
            "score": score,
            "issues": all_issues,
            "law_compliance": {"status": "compliant", "violations": []}
        })

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# QA 자료 업로드 및 식품표시사항 작성 API
@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    """QA 자료를 업로드하고 식품표시사항을 작성합니다."""
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")

    qa_files = request.files.getlist('qa_files')

    if not qa_files or len(qa_files) == 0:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    parts = []

    qa_prompt = """
당신은 식품표시사항 작성 전문가입니다.
제공된 QA 자료를 분석하여 법률을 준수하는 식품표시사항을 작성하세요.

[작업 단계]
1. QA 자료 분석: 엑셀, 이미지 등 모든 QA 자료를 종합적으로 분석하세요.
2. 법률 검토: 제공된 법령을 참고하여 필수 표시사항이 모두 포함되었는지 확인하세요.
3. 식품표시사항 작성: 법률을 준수하는 완전한 식품표시사항을 작성하세요.

[출력 양식 - JSON]
{
    "product_name": "제품명",
    "label_text": "작성된 식품표시사항 전체 텍스트",
    "law_compliance": {
        "status": "compliant" | "needs_review",
        "issues": ["법률 검토 사항 목록"]
    },
    "sections": {
        "ingredients": "원재료명",
        "nutrition": "영양정보",
        "allergens": "알레르기 유발물질",
        "storage": "보관방법",
        "manufacturer": "제조사 정보"
    }
}
"""

    if ALL_LAW_TEXT:
        qa_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"

    parts.append(qa_prompt)

    for qa_file in qa_files[:20]:
        file_part = process_file_to_part(qa_file)
        if file_part:
            parts.append(file_part)

    print(f"📂 QA 자료 처리 중: {len(qa_files)}개 파일")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(parts)

        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]

        result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 1000자): {result_text[:1000]}")
            print(f"오류 위치: line {json_err.lineno}, column {json_err.colno}")
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ QA 자료 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동")
    print("   - 원부재료 표시사항 스마트 추출")
    print("   - 법률 검토 기능 통합")
    print("   - QA 자료 업로드 지원")
    from waitress import serve

    serve(app, host='0.0.0.0', port=8080)
