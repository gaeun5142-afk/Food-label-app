import os
import json
import io
import glob
import traceback
import base64
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import PIL.Image
import re
import html

# Optional OCR fallback (if installed)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Optional PDF->Image (if installed)
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
CORS(app)

# ✅ OpenAI API 설정 (무조건 ChatGPT만 사용)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다!")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ChatGPT 멀티모달 모델
MODEL_NAME = "gpt-4.1-mini"   # 텍스트+이미지 모두 지원


# --- 공통 OpenAI 호출 헬퍼 (Gemini 대체) ---

def to_image_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """이미지 바이너리를 data URL(base64)로 변환"""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_openai_from_parts(parts, json_mode=True) -> str:
    """
    Gemini의 model.generate_content(parts)를 대체하는 OpenAI 호출.
    - parts: 문자열, PIL.Image.Image 섞여 있는 리스트 (기존 코드 그대로 사용)
    - json_mode: True면 "JSON만 출력"이라고 시스템 지시를 앞에 붙임
    - 반환값: ChatGPT가 반환한 텍스트 전체 (string)
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    content = []

    if json_mode:
        content.append({
            "type": "input_text",
            "text": (
                "항상 유효한 JSON만 출력하세요. "
                "설명, 마크다운, 코드블록, 자연어 문장은 절대 포함하지 마세요."
            ),
        })

    for p in parts:
        if isinstance(p, str):
            content.append({"type": "input_text", "text": p})
        elif isinstance(p, PIL.Image.Image):
            buf = io.BytesIO()
            fmt = p.format if p.format else "PNG"
            p.save(buf, format=fmt)
            buf.seek(0)
            data_url = to_image_data_url(buf.getvalue(), mime_type=f"image/{fmt.lower()}")
            content.append({
                "type": "input_image",
                "image_url": data_url,
            })
        else:
            # dict 등 다른 타입은 필요시 여기서 처리 (현재는 무시)
            pass

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=0.0,
        max_output_tokens=32768,
    )

    # text 결과만 모으기 (Responses API output 구조 기준)
    result_chunks = []
    for out in getattr(resp, "output", []):
        for c in getattr(out, "content", []):
            if hasattr(c, "text") and c.text:
                result_chunks.append(c.text)
    result_text = "".join(result_chunks).strip()
    return result_text


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

# --- 프롬프트 (지시사항) ---
PROMPT_EXTRACT_INGREDIENT_INFO = """
이 이미지는 원부재료 표시사항 사진입니다. 
**필수적으로 추출해야 할 정보만** 추출하세요.

[추출해야 할 정보]
1. **원재료명**: 원재료의 정확한 명칭
2. **복합원재료 내역**: 괄호 안의 하위 원재료 정보 (예: (탈지대두, 소맥))
3. **원산지 정보**: 원산지 표기 (예: 외국산, 국내산, 인도산 등)
4. **알레르기 유발물질**: 알레르기 표시 정보
5. **식품첨가물**: 첨가물명과 용도 병기 여부

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
제공된 [배합비 데이터(Excel)]와 [원재료 표시사항에서 추출된 정보]를 종합하여,
법적으로 완벽한 '식품표시사항 기준 데이터(Standard)'를 실제 라벨 형식으로 생성하세요.

[출력 양식 - JSON만 출력]
{
  "product_info": {
    "product_name": "제품명",
    "food_type": "식품의 유형",
    "net_weight": "내용량",
    "expiration_date": "소비기한",
    "storage_method": "보관방법",
    "packaging_material": "포장재질",
    "item_report_number": "품목보고번호",
    "front_calories": "전면부 총열량/문구"
  },
  "ingredients": {
    "structured_list": ["..."],
    "continuous_text": "원재료명, 원재료명2, ..."
  },
  "allergens": {
    "contains": ["대두", "게"],
    "manufacturing_facility": "제조시설 안내 문구"
  },
  "nutrition_info": {
    "total_content": "1000 g",
    "per_100g": {
      "calories": "130 Kcal"
    },
    "disclaimer": "영양정보 주의 문구 등"
  },
  "manufacturer": {
    "name": "제조업체명",
    "address": "주소"
  },
  "precautions": ["주의사항1", "주의사항2"],
  "law_compliance": {
    "status": "compliant" | "needs_review",
    "issues": ["법률 위반 사항 목록 (있는 경우)"]
  },
  "details": [
    {"name": "원재료명", "ratio": "배합비율", "origin": "원산지", "sub_ingredients": "하위원료"}
  ]
}
"""

PROMPT_VERIFY_DESIGN = """
당신은 대한민국 최고의 식품표시사항 정밀 감사 AI이자 자동 채점기입니다.
제공된 [Standard(기준서)]와 [Design OCR(raw_text 또는 이미지)]를 1:1 정밀 대조하여 채점하세요.

[절대 규칙]
- Standard와 디자인에 있는 텍스트를 절대로 임의로 수정하거나 보정하지 마세요.
- 맞춤법, 띄어쓰기, 숫자, 단위, 특수문자 차이를 그대로 기반으로만 비교하세요.

[감점 기준표 (총점 100점에서 시작)]
1. 원재료명 오류 (-5점/건)
2. 영양성분 오류 (-5점/건)
3. 법적 의무 문구 누락 (-10점/건)
4. 단순 오타 (-2점/건)

[출력 형식 - JSON만 출력]
{
  "design_ocr_text": "디자인 전체 텍스트(raw_text 또는 OCR 결과)",
  "score": 100,
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": ["식품등의 표시기준 제X조 위반..."]
  },
  "issues": [
    {
      "type": "Critical" | "Minor" | "Law_Violation",
      "location": "항목명 (예: 영양정보)",
      "issue": "오류 상세 설명",
      "expected": "기준서 데이터",
      "actual": "디자인에서 발견된 틀린 텍스트 (원문 일부)",
      "suggestion": "수정 제안"
    }
  ]
}
"""

# --- 텍스트/HTML 정리 함수 ---
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

# --- OCR 폴백 (선택적) ---
def ocr_bytes_to_text(image_bytes):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='kor+eng')
        return text
    except Exception as e:
        print("OCR 폴백 실패:", e)
        return ""

# --- 파일 처리 함수 ---
def process_file_to_part(file_storage):
    """
    파일을 모델 파트로 변환.
    - 엑셀: 텍스트(CSV) 스트링 반환
    - 이미지: PIL.Image 객체 반환 (모델 입력용)
    - PDF: 첫 페이지 이미지를 PIL.Image로 변환 (가능한 경우)
    - 기타: {'mime_type','data'} 반환
    """
    mime_type = file_storage.mimetype or ""
    file_data = file_storage.read()
    file_storage.seek(0)

    # 엑셀 -> CSV 텍스트
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 이미지 -> PIL.Image 객체 반환
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

    # PDF -> 이미지(첫 페이지)
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


# --- 이미지에서 원재료 정보 추출 (ChatGPT + OCR 폴백 결합) ---
def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출 (우선 ChatGPT, 실패 시 pytesseract OCR 폴백)"""
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        result_text = call_openai_from_parts(parts, json_mode=True)

        print("---- extract_ingredient_info_from_image 응답(원문 일부) ----")
        print(result_text[:4000])
        print("--------------------------------------------------")

        # ChatGPT 응답이 완전 비었으면 바로 OCR 폴백
        if (not result_text) and TESSERACT_AVAILABLE:
            ocr_text = ocr_bytes_to_text(image_data)
            if ocr_text:
                return {"ocr_fallback_text": ocr_text}
            return None

        # ```json ... ``` 제거
        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text.split("```", 1)[1].strip() if "```" in result_text else result_text
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()
        result_text = result_text.strip()

        # JSON 파싱 시도
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"원재료 정보 JSON 파싱 실패: {e}")
            print("응답 텍스트 일부:", result_text[:1000])
            # JSON이 망가졌을 때도 OCR 폴백 한 번 더 시도
            if TESSERACT_AVAILABLE:
                ocr_text = ocr_bytes_to_text(image_data)
                if ocr_text:
                    return {"ocr_fallback_text": ocr_text}
            return None

    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        traceback.print_exc()
        return None


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
    # 원재료 이미지들에 대해 ChatGPT + OCR 추출
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
        result_text = call_openai_from_parts(parts, json_mode=True)

        print("---- create-standard 응답(원문 일부) ----")
        print(result_text[:4000])
        print("------------------------------------")

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

@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")
    design_file = request.files.get('design_file')
    standard_excel = request.files.get('standard_excel')
    standard_json = request.form.get('standard_data')
    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400
    if not standard_excel and not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 파일 또는 JSON)가 필요합니다."}), 400

    if standard_excel and not standard_json:
        try:
            df_dict = pd.read_excel(io.BytesIO(standard_excel.read()), sheet_name=None, engine='openpyxl')
            if not df_dict:
                return jsonify({"error": "엑셀 파일이 비어있습니다."}), 400
            first_sheet_name = list(df_dict.keys())[0]
            first_sheet_df = df_dict[first_sheet_name]
            standard_data = {}
            if not first_sheet_df.empty:
                first_column = first_sheet_df.columns[0]
                if '원재료명' in first_sheet_df.columns:
                    ingredients_list = first_sheet_df['원재료명'].dropna().tolist()
                elif first_column:
                    ingredients_list = first_sheet_df[first_column].dropna().astype(str).tolist()
                else:
                    ingredients_list = first_sheet_df.iloc[:, 0].dropna().astype(str).tolist()
                if ingredients_list:
                    standard_data = {'ingredients': {'structured_list': ingredients_list, 'continuous_text': ', '.join(ingredients_list)}}
                else:
                    return jsonify({"error": "엑셀 파일의 첫 번째 시트에 데이터가 없습니다."}), 400
            else:
                return jsonify({"error": "엑셀 파일의 첫 번째 시트가 비어있습니다."}), 400
            standard_json = json.dumps(standard_data, ensure_ascii=False)
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

    try:
        result_text = call_openai_from_parts(parts, json_mode=True)

        print("---- verify-design 응답(원문 일부) ----")
        print(result_text[:4000])
        print("----------------------------------")

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

        result = clean_ai_response(result)
        return jsonify(result)

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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

[출력 양식 - JSON만 출력]
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
        result_text = call_openai_from_parts(parts, json_mode=True)

        print("---- upload-qa 응답(원문 일부) ----")
        print(result_text[:4000])
        print("--------------------------------")

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
        print(f"❌ QA 자료 처리 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 (ChatGPT+OCR 버전) 가동")
    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=4,
        channel_timeout=600
    )


