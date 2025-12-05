# server.py (OpenAI/ChatGPT API 버전)

import os
import json
import io
import glob
import re
import base64
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import PIL.Image
import PIL.ImageEnhance
from html import unescape

# === OpenAI 클라이언트 ===
from openai import OpenAI

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# API 키 설정 (환경변수: CHATGPT_API_KEY)
CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')
if not CHATGPT_API_KEY:
    print("🚨 경고: .env 파일에 CHATGPT_API_KEY가 없습니다!")
client = OpenAI(api_key=CHATGPT_API_KEY)

# OpenAI 모델 (필요 시 아래 목록 로직으로 자동 대체)
MODEL_NAME = "gpt-4o-mini"  # 속도/비용 최적. 정밀도 우선이면 "gpt-4o"


def check_available_models():
    """사용 가능한 모델 목록을 확인하고 적절한 모델을 반환합니다. (OpenAI)"""
    global MODEL_NAME
    try:
        models = list(client.models.list())
        names = [m.id for m in models]
        print("\n📋 사용 가능한 모델 목록:")
        for n in names:
            print(f" - {n}")

        preferred = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
        for p in preferred:
            if p in names:
                MODEL_NAME = p
                print(f"\n✅ 선택된 모델: {MODEL_NAME}\n")
                return MODEL_NAME

        if names:
            MODEL_NAME = names[0]
            print(f"\n✅ 첫 번째 모델 선택: {MODEL_NAME}\n")
            return MODEL_NAME

        print(f"\n⚠️ 모델 목록이 비어 있습니다. 기본값 사용: {MODEL_NAME}\n")
        return None
    except Exception as e:
        print(f"⚠️ 모델 목록 확인 실패: {e}")
        print(f"⚠️ 기본 모델 사용: {MODEL_NAME}\n")
        return None


if CHATGPT_API_KEY:
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
이 이미지는 원부재료 표시사항 사진입니다. **필수적으로 추출해야 할 정보만** 추출하세요.

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
JSON 형식으로만 응답하세요(코드블록 금지):
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
당신은 식품 규정 및 표시사항 전문가입니다. 제공된 [배합비 데이터(Excel)]와 [원재료 표시사항 사진들에서 추출한 정보]를 종합하여, 법적으로 완벽한 **'식품표시사항 기준 데이터(Standard)'**를 실제 라벨 형식으로 생성하세요.

[분석 단계]
1. **Excel 데이터 분석**: 배합비율(%)이 높은 순서대로 원재료 나열 순서를 결정하세요. (가장 중요)
2. **이미지 데이터 매핑**: Excel에 적힌 원재료명(예: '간장')에 해당하는 사진(원재료 라벨)을 찾아서 상세 정보(복합원재료 내역, 알레르기, 원산지)를 보강하세요.
  - 예: Excel엔 '간장'만 있지만, 사진에 '탈지대두(인도산), 소맥(밀)'이 있다면 이를 반영해야 함.
  - **중요**: 보관방법, 포장재질 등은 무시하고 원재료 관련 정보만 추출하세요.
3. **법률 검토**: 제공된 법령을 참고하여 표시사항이 법적으로 올바른지 확인하세요.
4. **최종 조합**: 품목제조보고서 기반의 비율과 원재료 라벨의 상세 내용을 합쳐 최종 표시 텍스트를 만드세요.

[출력 양식 - JSON만 응답(코드블록 금지)]
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
**중요**
- Excel 데이터에서 추출 가능한 모든 정보를 포함하세요.
- 영양정보는 Excel에 있는 경우에만 포함하고, 없으면 빈 객체로 두세요.
- 원재료명은 배합비율 순서대로 정확히 나열하세요.
- 실제 라벨에 표시되는 형식 그대로 구조화하세요.
"""

PROMPT_VERIFY_DESIGN = """
당신은 대한민국 최고의 [식품표시사항 정밀 감사 AI]이자 감정 없는 [자동 채점기]입니다. 제공된 [Standard(기준서)]와 [Design(디자인 이미지 - 식품표시사항 영역만 크롭됨)]을 1:1 정밀 대조하여, 아래 규칙에 따라 냉철하게 채점하세요.

**중요**: Design 이미지는 이미 식품표시사항 영역만 크롭되어 제공됩니다. 브랜드 로고, 제품 사진, 조리법 등은 이미 제거되었으므로, 식품표시사항 텍스트에만 집중하세요.

[감점 기준표 (총점 100점에서 시작)]
기본 100점에서 아래 오류가 발견될 때마다 점수를 차감하세요. (최하 0점)
1. **원재료명 오류 (-5점/건)**:
  - Standard(엑셀)에 있는 원재료가 Design(이미지)에 없거나 순서가 다름.
  - 함량(%) 숫자가 0.1%라도 다름. (예: 70.6% vs 70.5%)
2. **영양성분 오류 (-5점/건)**:
  - 나트륨, 탄수화물, 당류 등의 수치 또는 단위(g, mg) 불일치.
  - 비율(%) 숫자가 다름.
3. **법적 의무 문구 누락 (-10점/건)**:
  - "소비기한" (유통기한 아님) 표기 여부.
  - "부정 불량식품 신고는 국번없이 1399" 표기 여부.
  - 알레르기 유발물질 별도 표시란 유무.
  - 포장재질 및 분리배출 마크 유무.
4. **비현실적 수치 오류 (-5점/건)**:
  - 함량이 100%를 초과하는 경우 (예: "221%", "150%")
  - 비현실적으로 큰 수치 (예: "나트륨 50000mg")
  - 날짜 형식 오류 (예: "13월", "32일")
5. **디자인/표기 오탈자 (-3점/건)**:
  - 명백한 철자 오류 (예: "제조벙법" → "제조방법")
  - 단위 표기 오류 (예: "10Kg" → "10 kg", 단위 누락)
  - 부자연스러운 공백 (예: "보관방 법" → "보관방법")
6. **단순 오타 (-2점/건)**:
  - 괄호 위치 등 경미한 차이.

[분석 프로세스 - 단계별 수행]
1. **구조화 (Structuring)**:
  - Standard 데이터(엑셀)를 [제품명, 식품유형, 내용량, 원재료명, 영양정보, 보관방법, 포장재질, 품목보고번호] 항목별로 분류하세요.
  - Design 이미지는 이미 식품표시사항 영역만 크롭되어 제공되므로, 이 영역의 텍스트만 OCR하여 동일한 항목들을 찾아내어 1:1 매칭 준비를 하세요.
  - **무시할 것**: 브랜드 로고, 제품 사진, 조리법, 홍보 문구는 이미 제거되었으므로 신경쓰지 마세요.
2. **정밀 대조 (Cross-Checking)**:
  - **(1) 원재료명 검증 (가장 중요)**: Standard의 원재료 목록 순서와 함량(%)이 Design에 정확히 기재되었는지 확인하세요. * 띄어쓰기, 괄호 위치, 특수문자 하나라도 다르면 '오류'입니다.
  - **(2) 영양정보 숫자 검증**: 나트륨, 탄수화물, 당류 등 모든 수치와 단위(g, mg, %)가 일치하는지 확인하세요.
  - **(3) 법적 의무사항 검증**: 알레르기 유발물질 표시, "소비기한" 문구, 분리배출 마크 등이 법규대로 있는지 확인하세요. **중요**: 법률 위반 사항을 발견하면 반드시 관련 법령 조항을 명시하세요. 예: "식품등의 표시·광고에 관한 법률 제5조 제1항", "식품등의 표시기준 제3조 제2항" 등
3. **Step 3: Verdict (판단) - 3가지 오류 유형 모두 적극 감지**:
  **3-1. 법령 위반 감지 (Legal Compliance)** … (중략)

[출력 양식 - JSON Only (코드블록 금지)]
{
  "design_ocr_text": "디자인 전체 텍스트...",
  "score": (100점에서 차감된 최종 점수),
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": [
      {
        "violation": "위반 내용 상세 설명 …",
        "law_reference": "관련 법령 조항 번호만"
      }
    ]
  },
  "issues": [
    {
      "type": "Critical" | "Minor" | "Law_Violation" | "Logical_Error" | "Spelling_Error",
      "location": "항목명 (예: 영양정보)",
      "issue": "오류 상세 설명",
      "expected": "기준서 데이터",
      "actual": "디자인에서 발견된 틀린 텍스트 (하이라이트용)",
      "suggestion": "수정 제안",
      "law_reference": "관련 법령 조항 (법률 위반인 경우 필수)"
    }
  ]
}
"""


# --- OpenAI 호환 모델 래퍼 (기존 genai.GenerativeModel 대체) ---

class OpenAICompatResponse:
    def __init__(self, text: str):
        self.text = text or ""


class OpenAICompatModel:
    def __init__(self, model_name: str, generation_config: dict | None = None):
        self.model = model_name
        self.temperature = 0.0
        if generation_config and "temperature" in generation_config:
            self.temperature = generation_config["temperature"]

    def _filepart_to_image_content(part: dict) -> dict | None:
    try:
        mime = part.get("mime_type") or "image/png"
        data = part.get("data")
        if not data:
            return None
        
        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        # ===== 핵심 수정 =====
        return {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }
        # ====================

    except Exception:
        return None


  def _pil_to_image_content(self, pil_img) -> dict | None:
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        }

    except Exception:
        return None


    def generate_content(self, parts: list) -> OpenAICompatResponse:
        """
        parts 요소:
          - 문자열(str)
          - {"text": "..."}
          - {"mime_type": "...", "data": b"..."} (이미지/파일)
          - PIL.Image
        """
        content = []
        for p in parts:
            if isinstance(p, str):
                content.append({"type": "text", "text": p})
                continue
            if isinstance(p, dict) and p.get("text"):
                content.append({"type": "text", "text": p["text"]})
                continue
            if isinstance(p, dict) and p.get("mime_type") and p.get("data"):
                imgc = self._filepart_to_image_content(p)
                if imgc:
                    content.append(imgc)
                continue
            try:
                from PIL.Image import Image as PILImage
                if isinstance(p, PILImage):
                    imgc = self._pil_to_image_content(p)
                    if imgc:
                        content.append(imgc)
                    continue
            except Exception:
                pass

        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return OpenAICompatResponse(text)


# --- 파일 처리 함수들 ---
def process_file_to_part(file_storage):
    """파일을 OpenAI가 이해할 수 있는 part로 변환(텍스트/이미지)"""
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)  # 포인터 초기화

    # 엑셀은 CSV 텍스트로 변환
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 이미지: OCR 정확도 UP 전처리
    if mime_type.startswith('image/'):
        try:
            img = PIL.Image.open(io.BytesIO(file_data))
            img = img.convert('L')
            enhancer = PIL.ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            enhancer = PIL.ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)

            byte_io = io.BytesIO()
            fmt = img.format if img.format else 'PNG'
            img.save(byte_io, format=fmt)
            byte_io.seek(0)
            return {"mime_type": mime_type, "data": byte_io.read()}
        except Exception as e:
            print(f"⚠️ 이미지 보정 실패 (원본 사용): {e}")
            return {"mime_type": mime_type, "data": file_data}

    # PDF 등 기타 파일은 그대로 전달(※ 여기서는 텍스트 변환 없이 사용)
    return {"mime_type": mime_type, "data": file_data}


def clean_html_text(text):
    """HTML 태그와 HTML 코드를 완전히 제거하고 텍스트 내용만 유지"""
    if not isinstance(text, str):
        return text
    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'<div[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<ul[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</ul>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<li[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()


def detect_label_area(image_file):
    """이미지에서 식품표시사항 영역을 자동 감지 후 크롭"""
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data))
        original_size = img_pil.size

        model = OpenAICompatModel(MODEL_NAME)
        detection_prompt = """
이 이미지는 식품 포장지 디자인입니다. 이미지에서 **식품표시사항 영역**만 찾아 JSON으로 bbox를 주세요.
반환 형식(코드블록 금지):
{
  "found": true/false,
  "bbox": { "x1": 0, "y1": 0, "x2": 100, "y2": 100 },
  "description": "..."
}
식품표시사항 영역에는 제품명/식품유형/내용량/원재료명/영양정보/알레르기/제조원/주의사항 텍스트가 포함됩니다.
로고/제품사진/홍보문구는 무시하세요.
"""
        response = model.generate_content([detection_prompt, img_pil])
        result_text = response.text.strip()

        # 코드블록 정리
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

        detection_result = json.loads(result_text)

        if detection_result.get("found", False) and "bbox" in detection_result:
            bbox = detection_result["bbox"]
            x1 = max(0, int(bbox.get("x1", 0)))
            y1 = max(0, int(bbox.get("y1", 0)))
            x2 = min(original_size[0], int(bbox.get("x2", original_size[0])))
            y2 = min(original_size[1], int(bbox.get("y2", original_size[1])))

            cropped_img = img_pil.crop((x1, y1, x2, y2))
            print(f"✅ 식품표시사항 영역 감지: ({x1}, {y1}) ~ ({x2}, {y2}), 크기: {cropped_img.size}")

            output = io.BytesIO()
            cropped_img.save(output, format='PNG')
            output.seek(0)
            return output, True
        else:
            print("⚠️ 식품표시사항 영역을 찾을 수 없어 전체 이미지를 사용합니다.")
            image_file.seek(0)
            return image_file, False
    except Exception as e:
        print(f"❌ 영역 감지 실패: {e}, 전체 이미지 사용")
        image_file.seek(0)
        return image_file, False


def clean_ai_response(data):
    """AI 응답에서 HTML 태그를 제거하고 정리"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key == 'violations' and isinstance(value, list):
                cleaned[key] = [clean_ai_response(item) for item in value]
            elif key == 'issues' and isinstance(value, list):
                cleaned[key] = [clean_ai_response(item) for item in value]
            elif isinstance(value, str):
                cleaned[key] = clean_html_text(value)
            elif isinstance(value, (dict, list)):
                cleaned[key] = clean_ai_response(value)
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        return [clean_ai_response(item) for item in data]
    else:
        return clean_html_text(data) if isinstance(data, str) else data


def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출"""
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data))
        model = OpenAICompatModel(MODEL_NAME)

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        response = model.generate_content(parts)
        result_text = response.text.strip()

        # 코드블록 처리
        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text.strip("`")

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
                    ingredients_data.append({'순번': idx, '원재료명': item})
            ingredients_df = pd.DataFrame(ingredients_data)
            if not ingredients_df.empty:
                ingredients_df.to_excel(writer, sheet_name='원재료명', index=False)

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


# 1단계: 정답지 만들기
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
        parts.append({"text": ingredients_text})

    print(f"📂 처리 중: 엑셀 1개 + 원재료 이미지 {len(raw_images)}장 (정보 추출 완료)")

    try:
        model = OpenAICompatModel(MODEL_NAME, generation_config={"temperature": 0.0})
        response = model.generate_content(parts)
        result_text = response.text.strip()

        # 코드블록 제거
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
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception as e2:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 일부: {result_text[:200]}... / 보정오류: {e2}"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 기준 데이터 엑셀 파일 다운로드
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 엑셀 파일에서 기준 데이터 읽기
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


# 2단계: 검증하기
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")

    design_file = request.files.get('design_file')
    standard_excel = request.files.get('standard_excel')
    standard_json = request.form.get('standard_data')

    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400

    if standard_excel:
        try:
            df_dict = pd.read_excel(io.BytesIO(standard_excel.read()), sheet_name=None, engine='openpyxl')
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

    # 법령 파일 로딩
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

    parts = [f"""
{PROMPT_VERIFY_DESIGN}

[참고 법령]
{law_text[:60000]}

[기준 데이터]
{standard_json}
"""]

    if design_file:
        print("🔍 식품표시사항 영역 자동 감지 중...")
        cropped_image, is_cropped = detect_label_area(design_file)

        if is_cropped:
            print("✂️ 식품표시사항 영역만 크롭하여 사용합니다.")
            cropped_image.seek(0)
            cropped_pil = PIL.Image.open(cropped_image)
            parts.append(cropped_pil)
        else:
            print("📄 전체 이미지를 사용합니다.")
            parts.append(process_file_to_part(design_file))

    try:
        model = OpenAICompatModel(MODEL_NAME, generation_config={"temperature": 0.0})
        response = model.generate_content(parts)
        result_text = response.text.strip()

        # JSON만 추출
        json_match = re.search(r"(\{.*\})", result_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1)
            clean_json = clean_json.replace(",\n}", "\n}").replace(",\n]", "\n]")
            result = json.loads(clean_json)
            result = clean_ai_response(result)
            return jsonify(result)
        else:
            clean_json = result_text.replace("```", "").strip()
            result = json.loads(clean_json)
            result = clean_ai_response(result)
            return jsonify(result)

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# QA 자료 업로드 및 식품표시사항 작성
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

[출력 양식 - JSON만 응답(코드블록 금지)]
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
        model = OpenAICompatModel(MODEL_NAME)
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
            except:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ QA 자료 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동 (OpenAI 버전)")
    print(" - 원부재료 표시사항 스마트 추출")
    print(" - 법률 검토 기능 통합")
    print(" - QA 자료 업로드 지원")
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
