import os
import io
import json
import glob
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
import PIL.Image
import re
import html
from io import BytesIO

# Optional OCR fallback libraries (install if available)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("🚨 경고: .env 파일에 GOOGLE_API_KEY가 없습니다!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'gemini-1.5-flash'

def check_available_models():
    global MODEL_NAME
    try:
        models = genai.list_models()
        available_models = []
        print("\n📋 사용 가능한 모델 목록:")
        for m in models:
            if hasattr(m, "supported_generation_methods") and 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
                print(f"   - {model_name}")
        for model in available_models:
            if 'flash' in model.lower():
                MODEL_NAME = model
                print(f"\n✅ 추천 모델 선택: {MODEL_NAME}\n")
                return MODEL_NAME
        for model in available_models:
            if 'pro' in model.lower():
                MODEL_NAME = model
                print(f"\n✅ Pro 모델 선택: {MODEL_NAME}\n")
                return MODEL_NAME
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

if GOOGLE_API_KEY:
    check_available_models()
else:
    print(f"⚠️ API 키가 없어 모델 확인을 건너뜁니다. 기본 모델 사용: {MODEL_NAME}\n")

# --- 법령 텍스트 로드 ---
def load_law_texts() -> str:
    print("📚 법령 파일들을 읽어오는 중...")
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

# --- PROMPTS: (원본 길이 그대로 유지) ---
PROMPT_EXTRACT_INGREDIENT_INFO = """
이 이미지는 원부재료 표시사항 사진입니다. 
**필수적으로 추출해야 할 정보만** 추출하세요.

[추출해야 할 정보]
1. **원재료명**: 원재료의 정확한 명칭
2. **복합원재료 내역**: 괄호 안의 하위 원재료 정보 (예: (탈지대두, 소맥))
3. **원산지 정보**: 원산지 표기 (예: 외국산, 국내산, 인도산 등)
4. **알레르기 유발물질**: 알레르기 표시 정보
5. **식품첨가물**: 첨가물명과 용도 병기 여부

[추출하지 말아야 할 정보]
- 보관방법 (예: 냉장보관, 실온보관 등)
- 포장재질 정보
- 분리배출 마크
- 바코드 번호
- 제조일자/유통기한
- 단순 홍보 문구
- 기타 표시사항과 무관한 정보

[출력 형식]
JSON 형식으로만 응답하세요:
{
    "ingredient_name": "원재료명",
    "sub_ingredients": "하위원재료 내역 (복합원재료인 경우)",
    "origin": "원산지 정보",
    "allergens": ["알레르기 유발물질 목록"],
    "additives": ["식품첨가물 목록"]
}

원재료명이 명확하지 않으면 "ingredient_name"을 빈 문자열로 두세요.
"""

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

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사관이자 법률 전문가입니다.
[기준 데이터(Standard)]와 [디자인 시안(Design)]을 비교하여 오류를 검출하세요.

[입력]
1. **Standard**: 앞서 생성된 완벽한 표시사항 정답지
2. **Design**: 검수할 실제 포장지 디자인 파일 (PDF/이미지)
3. **법령**: 식품 표시 관련 법령

[검증 원칙 - 매우 중요! 반드시 준수하세요!]
1. **오탈자 검출 중심**: Standard와 Design을 문자 단위로 정확히 비교하여 실제 오탈자만 검출하세요.
2. **함량 정보(%) 추가는 허용**: Standard에 없어도 Design에 함량 정보(%)가 추가된 것은 절대 문제로 보지 않습니다.
   ✅ 허용 예시: Standard "당근(국내산)" → Design "당근(국내산) 4.1%" (문제 없음)
   ✅ 허용 예시: Standard "양파" → Design "양파 2.2%" (문제 없음)
3. **비정상 값만 문제**: 함량이 100%를 초과하거나 말도 안되는 값인 경우만 문제로 표시합니다.
   ❌ 문제 예시: "양파221%" (소수점 누락으로 221%가 되어 비정상) → "양파2.21%"로 수정 필요
   ❌ 문제 예시: "당근999%" (말도 안되는 값) → 문제로 표시
4. **라벨명 누락은 무시**: 내용이나 수치는 있지만 라벨명(예: "전면부 총열량", "제조시설안내")만 없는 경우는 문제로 보지 않습니다.
   ✅ 허용: 영양정보에 127Kcal 수치가 있으면 "전면부 총열량" 라벨이 없어도 문제 없음
   ✅ 허용: 제조시설 내용이 있으면 "제조시설안내" 라벨이 없어도 문제 없음
5. **실제 오류만 검출**:
   ✅ 원재료명 오탈자: "전분가공품" → "전반가공품" (글자 오기)
   ✅ 원재료명 오탈자: "D-소비톨" → "D-솔비톨" (글자 오기)
   ✅ 숫자 오탈자: "130kcal" → "127kcal" (숫자 오기)
   ✅ 단위 오탈자: "10g" → "10mg" (단위 오기)
   ✅ 구두점 오탈자: "우유, 쇠고기, 토마토" → "우유 쇠고기 토마토" (쉼표 누락)
   ✅ 소수점 누락: "2.21%" → "221%" (비정상 값)
   ✅ 원산지 오기: "국내산" → "수입산" (내용 오기)
   ✅ 순서 위반: 배합비 순서와 다름
   ✅ 법률 위반: 첨가물 유형 누락 (예: "소브산칼륨" → "소브산칼륨(보존료)" 필수)

[검증하지 말아야 할 것들 - 절대 문제로 표시하지 마세요!]
❌ Standard에 없는 함량 정보(%)가 Design에 추가된 경우
❌ 라벨명은 없지만 내용이나 수치가 있는 경우
❌ 공백이나 포맷팅 차이만 있는 경우 (예: "태국, 베트남산" vs "태국,베트남산")
❌ Standard와 Design이 의미상 동일하지만 표현만 다른 경우

[검증 항목]
1. **원재료명 오탈자**: Standard의 원재료명과 Design의 원재료명을 문자 단위로 비교하여 오탈자 검출
2. **숫자/단위 오탈자**: 영양정보, 함량 등의 숫자나 단위 오기 확인
3. **구두점 오탈자**: 쉼표, 소수점 등 구두점 누락/오기 확인
4. **원산지 오기**: 원산지 정보가 Standard와 다른지 확인
5. **순서 위반**: 원재료 나열 순서가 Standard(배합비 순)와 다른지 확인
6. **법률 위반**: 법령에 명시된 의무사항(예: 첨가물 유형 표시)이 누락되었는지 확인
7. **비정상 값**: 함량이 100% 초과이거나 말도 안되는 값인지 확인

[출력 양식 - JSON]
{
    "design_ocr_text": "디자인 파일에서 인식한 텍스트",
    "score": 90,
    "law_compliance": {
        "status": "compliant" | "violation",
        "violations": ["법률 위반 사항 목록 - 법률 조항만 표시 (예: '식품 등의 표시ㆍ광고에 관한 법률 제4조제1항제1호다목 위반')"]
    },
    "issues": [
        {
            "type": "Critical" | "Minor" | "Law_Violation",
            "location": "위치 (예: 원재료명 3번째 줄, 후면부 영양정보)",
            "issue": "오류 내용 (간단명료하게)",
            "expected": "정답 내용 (Standard 기준)",
            "actual": "실제 내용 (Design에서 인식한 내용)",
            "suggestion": "수정 제안",
            "law_reference": "관련 법령 조항 (법률 위반인 경우만)"
        }
    ],
    "design_ocr_highlighted_html": "<div>하이라이트된 HTML</div>"
}
"""

# --- 유틸 함수들 ---
def clean_html_text(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'style\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'class\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'font-weight\s*:\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'margin[^;]*;?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'padding[^;]*;?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'color[^;]*;?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'font-size[^;]*;?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_ai_response(data):
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ['violations', 'issues'] and isinstance(value, list):
                cleaned[key] = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for k, v in item.items():
                            if isinstance(v, str):
                                cleaned_item[k] = clean_html_text(v)
                            else:
                                cleaned_item[k] = clean_ai_response(v)
                        cleaned[key].append(cleaned_item)
                    elif isinstance(item, str):
                        cleaned[key].append(clean_html_text(item))
                    else:
                        cleaned[key].append(clean_ai_response(item))
            elif isinstance(value, str):
                cleaned[key] = clean_html_text(value)
            else:
                cleaned[key] = clean_ai_response(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_ai_response(item) for item in data]
    elif isinstance(data, str):
        return clean_html_text(data)
    else:
        return data

# --- OCR 폴백 ---
def ocr_image_bytes(image_bytes: bytes) -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='kor+eng')
        return text
    except Exception as e:
        print("pytesseract OCR 실패:", e)
        return ""

# --- 파일 처리 (수정됨: 이미지 -> PIL.Image 반환) ---
def process_file_to_part(file_storage):
    mime_type = file_storage.mimetype or ""
    file_data = file_storage.read()
    file_storage.seek(0)

    # Excel -> CSV text
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # Image -> PIL.Image (for model OCR)
    if mime_type.startswith('image/'):
        try:
            img = PIL.Image.open(io.BytesIO(file_data)).convert("RGB")
            max_size = 1500
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                print(f"📉 이미지 리사이징: {new_size}")
            return img
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패, bytes로 반환: {e}")
            return {"mime_type": mime_type, "data": file_data}

    # PDF -> convert to image if possible
    if mime_type == 'application/pdf' and PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(file_data, dpi=200)
            if images:
                print(f"📄 PDF->이미지 변환: {len(images)} 페이지 (첫 페이지 사용)")
                return images[0].convert("RGB")
        except Exception as e:
            print("PDF->이미지 변환 실패:", e)
            return {"mime_type": mime_type, "data": file_data}

    return {"mime_type": mime_type, "data": file_data}

# --- 이미지 원재료 정보 추출 (기존 방식 유지) ---
def extract_ingredient_info_from_image(image_file):
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")
        model = genai.GenerativeModel(MODEL_NAME)
        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        response = model.generate_content(parts)

        print("---- extract_ingredient_info_from_image 모델 응답 시작 ----")
        try:
            print(getattr(response, "text", str(response))[:4000])
        except Exception as e:
            print("응답 출력 실패:", e)
        print("---- extract_ingredient_info_from_image 모델 응답 끝 ----")

        result_text = getattr(response, "text", "").strip()
        if not result_text and TESSERACT_AVAILABLE:
            ocr_text = ocr_image_bytes(image_data)
            if ocr_text:
                return {"ocr_fallback_text": ocr_text}
        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text.split("```")[1].strip() if "```" in result_text else result_text
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"원재료 정보 JSON 파싱 실패: {e}")
            print("응답 텍스트 일부:", result_text[:1000])
            return None
    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        traceback.print_exc()
        return None

# --- 엑셀 만들기 ---
def create_standard_excel(data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if 'product_info' in data:
            product_df = pd.DataFrame([data['product_info']])
            product_df.to_excel(writer, sheet_name='제품정보', index=False)
        if 'ingredients' in data:
            ingredients_data = []
            if 'structured_list' in data['ingredients']:
                for idx, item in enumerate(data['ingredients']['structured_list'], 1):
                    ingredients_data.append({'순번': idx, '원재료명': item})
            ingredients_df = pd.DataFrame(ingredients_data)
            if not ingredients_df.empty:
                ingredients_df.to_excel(writer, sheet_name='원재료명', index=False)
            if 'continuous_text' in data['ingredients']:
                continuous_df = pd.DataFrame([{'원재료명_연속텍스트': data['ingredients']['continuous_text']}])
                continuous_df.to_excel(writer, sheet_name='원재료명_연속텍스트', index=False)
        if 'allergens' in data:
            allergens_data = []
            if 'contains' in data['allergens']:
                allergens_data.append({'항목': '함유 알레르기 유발물질', '내용': ', '.join(data['allergens']['contains'])})
            if 'manufacturing_facility' in data['allergens']:
                allergens_data.append({'항목': '제조시설 안내', '내용': data['allergens']['manufacturing_facility']})
            if allergens_data:
                allergens_df = pd.DataFrame(allergens_data)
                allergens_df.to_excel(writer, sheet_name='알레르기정보', index=False)
        if 'nutrition_info' in data and 'per_100g' in data['nutrition_info']:
            nutrition_data = []
            nut = data['nutrition_info']['per_100g']
            if 'calories' in nut:
                nutrition_data.append({'영양성분': '총 열량', '100g 당': nut['calories'], '1일 영양성분 기준치에 대한 비율(%)': '-'})
            for key, value in nut.items():
                if key != 'calories' and isinstance(value, dict):
                    nutrition_data.append({'영양성분': key, '100g 당': value.get('amount', ''), '1일 영양성분 기준치에 대한 비율(%)': value.get('daily_value', '')})
            if nutrition_data:
                nutrition_df = pd.DataFrame(nutrition_data)
                nutrition_df.to_excel(writer, sheet_name='영양정보', index=False)
        if 'manufacturer' in data:
            manufacturer_df = pd.DataFrame([data['manufacturer']])
            manufacturer_df.to_excel(writer, sheet_name='제조원정보', index=False)
        if 'precautions' in data:
            precautions_df = pd.DataFrame([{'주의사항': item} for item in data['precautions']])
            precautions_df.to_excel(writer, sheet_name='주의사항', index=False)
        if 'details' in data and data['details']:
            details_df = pd.DataFrame(data['details'])
            details_df.to_excel(writer, sheet_name='원재료상세', index=False)
    output.seek(0)
    return output

# --- 라우트 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create-standard', methods=['POST'])
def create_standard():
    print("⚙️ 1단계: 기준 데이터 생성 시작...")
    excel_file = request.files.get('excel_file')
    raw_images = request.files.getlist('raw_images')
    if not excel_file:
        return jsonify({"error": "배합비 엑셀 파일이 필요합니다."}), 400

    parts = []
    enhanced_prompt = PROMPT_CREATE_STANDARD
    if ALL_LAW_TEXT:
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)

    excel_part = process_file_to_part(excel_file)
    if excel_part:
        if isinstance(excel_part, dict) and 'text' in excel_part:
            parts.append(excel_part['text'])
        else:
            parts.append(excel_part)

    ingredient_info_list = []
    for img in raw_images[:15]:
        print(f"📷 원재료 이미지 처리 중: {img.filename}")
        ingredient_info = extract_ingredient_info_from_image(img)
        if ingredient_info:
            ingredient_info_list.append(ingredient_info)

    if ingredient_info_list:
        ingredients_text = "--- [원재료 표시사항에서 추출한 정보] ---\n"
        for idx, info in enumerate(ingredient_info_list, 1):
            ingredients_text += f"\n[원재료 {idx}]\n"
            ingredients_text += json.dumps(info, ensure_ascii=False, indent=2)
            ingredients_text += "\n"
        ingredients_text += "--- [원재료 정보 끝] ---\n"
        parts.append(ingredients_text)

    print(f"📂 처리 중: 엑셀 1개 + 원재료 이미지 {len(raw_images)}장 (정보 추출 완료)")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(parts)

        print("---- 모델 응답(원문) 시작 ----")
        try:
            print(getattr(response, "text", str(response))[:4000])
        except Exception as e:
            print("응답 출력 실패:", e)
        print("---- 모델 응답(원문) 끝 ----")

        result_text = getattr(response, "text", "").strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 2000자): {result_text[:2000]}")
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception as e:
                print("최종 JSON 파싱 실패:", e)
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:400]}..."}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-standard-excel', methods=['POST'])
def download_standard_excel():
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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/read-standard-excel', methods=['POST'])
def read_standard_excel():
    try:
        excel_file = request.files.get('excel_file')
        if not excel_file:
            return jsonify({"error": "엑셀 파일이 필요합니다."}), 400
        df_dict = pd.read_excel(io.BytesIO(excel_file.read()), sheet_name=None, engine='openpyxl')
        result = {}
        if '제품정보' in df_dict:
            product_info = df_dict['제품정보'].to_dict('records')[0]
            result['product_info'] = product_info
        first_sheet_name = list(df_dict.keys())[0]
        first_sheet_df = df_dict[first_sheet_name]
        if '원재료명' in df_dict:
            ingredients_list = df_dict['원재료명']['원재료명'].dropna().tolist()
            result['ingredients'] = {'structured_list': ingredients_list, 'continuous_text': ', '.join(ingredients_list)}
        elif '원재료명_연속텍스트' in df_dict:
            continuous_text = df_dict['원재료명_연속텍스트']['원재료명_연속텍스트'].iloc[0]
            result['ingredients'] = {'structured_list': continuous_text.split(', '), 'continuous_text': continuous_text}
        elif not first_sheet_df.empty:
            first_column = first_sheet_df.columns[0]
            if '원재료명' in first_sheet_df.columns:
                ingredients_list = first_sheet_df['원재료명'].dropna().tolist()
            else:
                ingredients_list = first_sheet_df[first_column].dropna().astype(str).tolist()
            if ingredients_list:
                result['ingredients'] = {'structured_list': ingredients_list, 'continuous_text': ', '.join(ingredients_list)}
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
                    per_100g[row['영양성분']] = {'amount': row['100g 당'], 'daily_value': row['1일 영양성분 기준치에 대한 비율(%)']}
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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- verify_design 대체: 모델 + 폴백 OCR + 하이라이트 생성 ---
def simple_generate_highlight_html(ocr_text: str, standard_ingredients: list):
    if not ocr_text:
        return "<div>OCR로 텍스트를 추출하지 못했습니다.</div>"
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if not lines:
        lines = [ocr_text.strip()]
    std_lower = [s.lower() for s in standard_ingredients]
    html_lines = []
    for line in lines:
        line_html = html.escape(line)
        lowered = line.lower()
        matched = False
        for idx, std in enumerate(std_lower):
            if std in lowered:
                matched = True
                line_html = line_html.replace(html.escape(standard_ingredients[idx]), f"<span style='background:#e6f4ea;padding:2px 4px;border-radius:4px;'>{html.escape(standard_ingredients[idx])}</span>")
        if not matched:
            line_html = f"<span style='color:#ad2e2e; font-weight:600;'>{line_html}</span>"
        html_lines.append(f"<div style='margin-bottom:6px; font-family:monospace; white-space:pre-wrap;'>{line_html}</div>")
    result_html = "<div style='padding:10px; background:#fff; border-radius:8px;'>" + "".join(html_lines) + "</div>"
    return result_html

def extract_text_from_design_part(design_part):
    try:
        from PIL import Image
        pil_type = Image.Image
    except Exception:
        pil_type = None
    if pil_type and isinstance(design_part, pil_type):
        bio = BytesIO()
        design_part.save(bio, format='PNG')
        bio.seek(0)
        img_bytes = bio.read()
        return ocr_image_bytes(img_bytes)
    if isinstance(design_part, dict) and 'data' in design_part:
        img_bytes = design_part['data']
        return ocr_image_bytes(img_bytes)
    return ""

@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작 (폴백 OCR 포함)...")
    design_file = request.files.get('design_file')
    standard_excel = request.files.get('standard_excel')
    standard_json = request.form.get('standard_data')
    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400
    if not standard_excel and not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 파일 또는 JSON)가 필요합니다."}), 400

    if standard_excel:
        try:
            df_dict = pd.read_excel(io.BytesIO(standard_excel.read()), sheet_name=None, engine='openpyxl')
            if not df_dict:
                return jsonify({"error": "엑셀 파일이 비어있습니다."}), 400
            first_sheet_name = list(df_dict.keys())[0]
            first_sheet_df = df_dict[first_sheet_name]
            if not first_sheet_df.empty:
                first_column = first_sheet_df.columns[0]
                if '원재료명' in first_sheet_df.columns:
                    ingredients_list = first_sheet_df['원재료명'].dropna().astype(str).tolist()
                else:
                    ingredients_list = first_sheet_df[first_column].dropna().astype(str).tolist()
                standard_data = {'ingredients': {'structured_list': ingredients_list, 'continuous_text': ', '.join(ingredients_list)}}
                standard_json = json.dumps(standard_data, ensure_ascii=False)
            else:
                return jsonify({"error": "엑셀의 첫 시트가 비어있습니다."}), 400
        except Exception as e:
            print(f"❌ 엑셀 파일 읽기 오류: {e}")
            traceback.print_exc()
            return jsonify({"error": f"엑셀 파일 읽기 실패: {str(e)}"}), 400

    parts = []
    enhanced_prompt = PROMPT_VERIFY_DESIGN
    if ALL_LAW_TEXT:
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)
    parts.append(f"\n--- [기준 데이터(Standard)] ---\n{standard_json}")

    design_part = process_file_to_part(design_file)
    if design_part:
        parts.append(design_part)

    model = genai.GenerativeModel(MODEL_NAME)
    result_text = ""
    try:
        response = model.generate_content(parts)
        print("---- 모델 응답(원문) 시작 ----")
        try:
            print(getattr(response, "text", str(response))[:4000])
        except Exception as e:
            print("응답 출력 실패:", e)
        print("---- 모델 응답(원문) 끝 ----")
        result_text = getattr(response, "text", "").strip()
    except Exception as e:
        print("모델 호출 실패:", e)
        traceback.print_exc()
        result_text = ""

    result = None
    if result_text:
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result_text = result_text.strip()
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print("JSON 파싱 오류:", json_err)
            print("응답 텍스트(일부):", result_text[:1000])
            try:
                fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception as e:
                print("최종 JSON 파싱 실패:", e)
                result = None

    highlight_html = None
    if result and isinstance(result, dict):
        highlight_html = result.get("design_ocr_highlighted_html") or None

    if not highlight_html:
        print("모델에서 하이라이트를 제공하지 않음 -> 서버 폴백 OCR 시도")
        try:
            ocr_text = extract_text_from_design_part(design_part)
            if not ocr_text:
                try:
                    raw_bytes = design_file.read()
                    design_file.seek(0)
                    ocr_text = ocr_image_bytes(raw_bytes)
                except Exception:
                    ocr_text = ""
            std_ingredients = []
            try:
                std_obj = json.loads(standard_json)
                std_ingredients = std_obj.get('ingredients', {}).get('structured_list', [])
            except Exception:
                std_ingredients = []
            highlight_html = simple_generate_highlight_html(ocr_text, std_ingredients)
            if not result:
                result = {}
            result['design_ocr_highlighted_html'] = highlight_html
            result.setdefault('design_ocr_text', ocr_text)
        except Exception as e:
            print("폴백 OCR 처리 실패:", e)
            traceback.print_exc()
            if not result:
                result = {}
            result['design_ocr_highlighted_html'] = "<div>서버 폴백 OCR 처리 중 오류가 발생했습니다.</div>"
            result['design_ocr_text'] = ""

    if not result:
        result = {
            "design_ocr_text": "",
            "score": 0,
            "law_compliance": {"status": "needs_review", "violations": []},
            "issues": [],
            "design_ocr_highlighted_html": "<div>모델과 폴백 모두에서 OCR 결과를 얻지 못했습니다.</div>"
        }

    result = clean_ai_response(result)
    return jsonify(result)

@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
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
        if not file_part:
            continue
        if isinstance(file_part, dict) and 'text' in file_part:
            parts.append(file_part['text'])
        else:
            parts.append(file_part)

    print(f"📂 QA 자료 처리 중: {len(qa_files)}개 파일")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(parts)

        print("---- 모델 응답(원문) 시작 ----")
        try:
            print(getattr(response, "text", str(response))[:4000])
        except Exception as e:
            print("응답 출력 실패:", e)
        print("---- 모델 응답(원문) 끝 ----")

        result_text = getattr(response, "text", "").strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 2000자): {result_text[:2000]}")
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ QA 자료 처리 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동")
    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=4,
        channel_timeout=600
    )

