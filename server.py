import os
import json
import io
import glob
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import PIL.ImageEnhance
import re

def normalize_number(text):
    if not text:
        return ""
    # 숫자와 소수점만 남김
    return re.sub(r"[^0-9.]", "", str(text))

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# API 키 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("🚨 경고: .env 파일에 GOOGLE_API_KEY가 없습니다!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 설정 (기본값, 자동 감지로 덮어씌워질 수 있음)
MODEL_NAME = 'gemini-1.5-flash'

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

# 원재료 표시사항 이미지에서 필요한 부분만 추출하는 프롬프트
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
당신은 대한민국 최고의 [식품표시사항 정밀 감사 AI]이자 감정 없는 [자동 채점기]입니다.
제공된 [Standard(기준서)]와 [Design(디자인)]을 1:1 정밀 대조하여, 아래 규칙에 따라 냉철하게 채점하세요.

[감점 기준표 (총점 100점에서 시작)]
기본 100점에서 아래 오류가 발견될 때마다 점수를 차감하세요. (최하 0점)

1. **원재료명 오류 (-3점/건)**:
   - Standard(엑셀)에 있는 원재료가 Design(이미지)에 없거나 순서가 다름.
   - 함량(%) 숫자가 0.1%라도 다름. (예: 70.6% vs 70.5%)
2. **영양성분 오류 (-3점/건)**:
   - 나트륨, 탄수화물, 당류 등의 수치 또는 단위(g, mg) 불일치.
   - 비율(%) 숫자가 다름.
3. **법적 의무 문구 누락 (-5점/건)**:
   - "소비기한" (유통기한 아님) 표기 여부.
   - "부정 불량식품 신고는 국번없이 1399" 표기 여부.
   - 알레르기 유발물질 별도 표시란 유무.
   - 포장재질 및 분리배출 마크 유무.
4. **단순 오타 (-1점/건)**:
   - 띄어쓰기, 괄호 위치 등 경미한 차이.

[분석 프로세스 - 단계별 수행]

1. **구조화 (Structuring)**:
   - Standard 데이터(엑셀)를 [제품명, 식품유형, 내용량, 원재료명, 영양정보, 보관방법, 포장재질, 품목보고번호] 항목별로 분류하세요.
   - Design 이미지(OCR)에서도 동일한 항목들을 찾아내어 1:1 매칭 준비를 하세요.

2. **정밀 대조 (Cross-Checking)**:
   - **(1) 원재료명 검증 (가장 중요)**: 
     Standard의 원재료 목록 순서와 함량(%)이 Design에 정확히 기재되었는지 확인하세요.
     * 띄어쓰기, 괄호 위치, 특수문자 하나라도 다르면 '오류'입니다.
   - **(2) 영양정보 숫자 검증**: 
     나트륨, 탄수화물, 당류 등 모든 수치와 단위(g, mg, %)가 일치하는지 확인하세요.
   - **(3) 법적 의무사항 검증**: 
     알레르기 유발물질 표시, "소비기한" 문구, 분리배출 마크 등이 법규대로 있는지 확인하세요.

3. **핀셋 오류 지적 (Pinpoint Reporting)**:
   - "원재료명이 다릅니다" 같이 뭉뚱그리지 마세요.
   - **오류가 있는 '단어' 또는 '숫자'만 정확히 잘라내어 `actual` 필드에 넣으세요.**
   - 예: "L-글루탐산나트륨"이 빠졌다면, 그 위치 주변 텍스트를 `actual`로 잡아 하이라이트 하세요.

    - 모든 issue에는 반드시 "law_reference" 필드를 포함할 것
    - 법령은 반드시 "식품등의 표시기준 제몇조" 형식으로 쓸 것
    - 추정 금지, 실제 근거 조항만 작성할 것


[출력 양식 - JSON Only]
- Markdown 포맷 없이 오직 JSON 데이터만 출력하세요.
{
  "design_ocr_text": "디자인 전체 텍스트...",
  "score": (100점에서 차감된 최종 점수),
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": ["식품등의 표시기준 제X조 위반..."]
  },
  "issues": [
  {
    "type": "Critical" | "Minor" | "Law_Violation" | "Law_Review",
    "location": "...",
    "issue": "...",
    "expected": "...",
    "actual": "...",
    "suggestion": "...",
    "law_reference": "식품등의 표시기준 제8조 제2항 (영양성분 표시방법 위반)"  # ← 이런 형태로
  }
]

}
[법령 매핑 규칙 - 매우 중요]

- 각 issue마다 반드시 "law_reference" 필드를 포함해야 합니다.
- law_reference는 아래 형식의 한국어 텍스트로만 작성합니다.
  예) "식품등의 표시기준 제8조 제2항 (영양성분 표시방법 위반)"
      "식품위생법 제13조 제1항 (허위표시 금지 위반)"
      "알레르기 유발 물질 표시기준 제4조 (알레르기 표시 누락)"

- 사용할 수 있는 법령 이름은 다음 중에서만 선택합니다.
  * "식품위생법"
  * "식품등의 표시기준"
  * "알레르기 유발 물질 표시기준"
  * "원산지표시법"
  * "건강기능식품의 표시기준"
  (법령 텍스트에 실제로 존재하는 조문만 사용하세요.)

- 절대 금지:
  * "관련 법령 종합 적용" 같이 모호한 표현
  * "법령 전반", "여러 법령" 등의 뭉뚱그린 표현
  * 제공된 법령 텍스트에 존재하지 않는 조문을 추측해서 작성하는 것

- 만약 명확하게 어떤 조문인지 판단할 수 없다면:
  * 해당 issue의 type을 "Law_Review"로 설정하고,
  * law_reference는 "관계 법령 검토 필요 (조문 특정 불가)" 로 설정합니다.

"""

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

    # 이미지나 PDF는 그대로 전달
    # Gemini는 image/jpeg, image/png, application/pdf 등을 지원함
    return {"mime_type": mime_type, "data": file_data}

def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출 (✅ 3회 투표 방식 적용)"""
    try:
        image_data = image_file.read()
        image_file.seek(0)

        img_pil = PIL.Image.open(io.BytesIO(image_data))
        model = genai.GenerativeModel(MODEL_NAME)

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]

        # ✅ ✅ ✅ 1. AI 호출 3번 수행
        results = []
        for _ in range(3):
            response = model.generate_content(parts)
            result_text = response.text.strip()

            # JSON 코드블럭 제거
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].strip()
                if result_text.startswith("json"):
                    result_text = result_text[4:].strip()

            results.append(result_text)

        # ✅ ✅ ✅ 2. 가장 많이 등장한 결과 선택 (다수결)
        final_result_text = max(set(results), key=results.count)

        # ✅ ✅ ✅ 3. JSON 파싱
        return json.loads(final_result_text)

    except json.JSONDecodeError as e:
        print(f"❌ 원재료 정보 JSON 파싱 실패: {e}")
        print(f"❌ 최종 응답 텍스트: {final_result_text[:500]}...")
        return None

    except Exception as e:
        print(f"❌ 원재료 정보 추출 실패: {e}")
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
                allergens_df.to_excel(writer, sheet_name='알레르리정보', index=False)
        
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

# 🔴 하이라이트 HTML 생성 헬퍼 함수 추가
def make_highlighted_html(design_text: str, issues: list) -> str:
    """
    디자인 전체 텍스트(design_text) 안에서
    issues[*]["actual"] 에 해당하는 부분만 빨간색으로 하이라이트해서
    HTML 문자열로 돌려준다.
    """
    if not design_text:
        return ""

    highlight_ranges = []

    # 1) 각 이슈의 actual 문자열 위치 찾기
    for issue in issues or []:
        actual = (issue or {}).get("actual")
        if not actual:
            continue

        idx = design_text.find(actual)
        if idx == -1:
            continue  # 못 찾으면 스킵

        highlight_ranges.append((idx, idx + len(actual)))

    if not highlight_ranges:
        # 하이라이트할 게 없으면 그냥 <br> 만 바꿔서 반환
        return design_text.replace("\n", "<br>")

    # 2) 겹치는 구간 정리
    highlight_ranges.sort()
    merged = []
    cur_start, cur_end = highlight_ranges[0]
    for start, end in highlight_ranges[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # 3) HTML 조립
    parts = []
    last_idx = 0
    for start, end in merged:
        # 일반 텍스트
        if start > last_idx:
            parts.append(design_text[last_idx:start])
        # 하이라이트 텍스트
        highlight_text = design_text[start:end]
        parts.append(
            f'<span style="color:#e53935; font-weight:bold;">{highlight_text}</span>'
        )
        last_idx = end

    # 마지막 꼬리 부분
    if last_idx < len(design_text):
        parts.append(design_text[last_idx:])

    html = "".join(parts)
    # 줄바꿈을 <br> 로 변환
    html = html.replace("\n", "<br>")
    # 전체 블록 스타일
    return f'<div style="line-height:1.6; font-size:14px;">{html}</div>'

# --- 라우트 ---

@app.route('/')
def index():
    return "Food Label API is running"

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
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT[:12000]}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)

    # (2) 엑셀 데이터
    excel_part = process_file_to_part(excel_file)
    if excel_part: parts.append(excel_part)

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
        # 창의성(Temperature) 0으로 설정해서 로봇처럼 만들기
        generation_config = {"temperature": 0.0}
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)

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
                # 마지막 쉼표 제거 시도
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except:
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

# 2단계: 검증하기 (엑셀 파일 또는 JSON + 디자인 이미지)
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")

    try:
        # -----------------------------
        # 1. 파일 받기
        # -----------------------------
        design_file = request.files.get('design_file')
        standard_excel = request.files.get('standard_excel')
        standard_json = request.form.get('standard_data')

        if not design_file:
            return jsonify({"error": "디자인 파일이 필요합니다. (design_file)"}), 400

        # -----------------------------
        # 2. 기준 데이터 로딩 (엑셀 -> JSON)
        # -----------------------------
        if standard_excel:
            df_dict = pd.read_excel(
                io.BytesIO(standard_excel.read()),
                sheet_name=None,
                engine='openpyxl'
            )

            first_sheet_df = list(df_dict.values())[0]

            ingredients_list = []
            if '원재료명' in first_sheet_df.columns:
                ingredients_list = (
                    first_sheet_df['원재료명']
                    .dropna()
                    .astype(str)
                    .tolist()
                )

            standard_data = {
                'ingredients': {
                    'structured_list': ingredients_list,
                    'continuous_text': ', '.join(ingredients_list)
                }
            }

            standard_json = json.dumps(standard_data, ensure_ascii=False)

        # -----------------------------
        # 3. 프롬프트 조합
        # -----------------------------
        full_prompt = f"""
{PROMPT_VERIFY_DESIGN}
[절대 규칙]
- 추측 금지
- 보이는 텍스트만 근거로 판단
- 수치·문장·특수문자 하나라도 불명확하면 “불일치”로 처리
- 기준 데이터에 없는 정보는 절대 추가하지 말 것
- 동일 입력에 대해 항상 동일한 JSON 구조로만 출력

[기준 데이터(JSON)]
{standard_json}
"""

        parts = [full_prompt]

        design_file.stream.seek(0)
        design_part = process_file_to_part(design_file)
        if design_part:
            parts.append(design_part)
        else:
            return jsonify({"error": "디자인 파일 처리 실패"}), 400

        # -----------------------------
        # 4. Gemini 호출
        # -----------------------------
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config={"temperature": 0.0}
        )

        response = model.generate_content(parts)
        result_text = response.text.strip()

        # -----------------------------
        # 5. ✅ JSON 안전 파싱 + 502 방지
        # -----------------------------
        try:
            json_match = re.search(r"(\{.*\})", result_text, re.DOTALL)

            if json_match:
                clean_json = json_match.group(1)
            else:
                clean_json = result_text.replace("```", "").strip()

            clean_json = clean_json.replace(",\n}", "\n}").replace(",\n]", "\n]")

            result = json.loads(clean_json)
            cleaned_issues = []
            for issue in result.get("issues", []):
                expected = issue.get("expected")
                actual = issue.get("actual")

                if expected and actual:
                    if normalize_number(expected) == normalize_number(actual):
                        continue 
                        
                    cleaned_issues.append(issue)

                result["issues"] = cleaned_issues

        except Exception as e:
            print("❌ JSON 파싱 실패:", e)
            print("❌ 원본 응답:", result_text[:1000])
            return jsonify({
                "error": "AI JSON 파싱 실패",
                "raw_ai_text": result_text[:1000]
            }), 200

        # -----------------------------
        # ✅ ✅ ✅ 위반 상세 HTML 완전 제거
        # -----------------------------
        if "law_compliance" in result:
            result["law_compliance"]["violations"] = []

        # -----------------------------
        # ✅ ✅ ✅ 하이라이트 HTML 생성
        # -----------------------------
        design_text = result.get("design_ocr_text", "")
        issues = result.get("issues", [])
        highlighted_html = make_highlighted_html(design_text, issues)
        result["design_ocr_highlighted_html"] = highlighted_html

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"서버 내부 오류: {str(e)}"
        }), 500


# QA 자료 업로드 및 식품표시사항 작성 API
@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    """QA 자료를 업로드하고 식품표시사항을 작성합니다."""
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")
    
    # QA 자료 파일들 (엑셀, 이미지 등)
    qa_files = request.files.getlist('qa_files')
    
    if not qa_files or len(qa_files) == 0:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    # AI에게 보낼 데이터 꾸러미 만들기
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
    
    # 법령 정보 추가
    if ALL_LAW_TEXT:
        qa_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT[:12000]}\n--- [법령 끝] ---\n"
    
    parts.append(qa_prompt)
    
    # QA 파일들 처리
    for qa_file in qa_files[:20]:  # 최대 20개 파일
        file_part = process_file_to_part(qa_file)
        if file_part:
            parts.append(file_part)
    
    print(f"📂 QA 자료 처리 중: {len(qa_files)}개 파일")
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(parts)
        
        # JSON 파싱
        result_text = response.text.strip()
        
        # JSON 코드 블록 제거
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
        
        # JSON 파싱 시도
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 1000자): {result_text[:1000]}")
            print(f"오류 위치: line {json_err.lineno}, column {json_err.colno}")
            # JSON 수정 시도
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except:
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
