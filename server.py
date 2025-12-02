import os
import json
import io
import glob
import base64
import time

import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import PIL.Image

from openai import OpenAI

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
CORS(app)

# === OpenAI 설정 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다! OpenAI API 호출이 실패할 수 있습니다.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# 텍스트/비전 모델 분리 (원하면 .env에서 덮어쓰기)
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.1-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")


# --- 공통 OpenAI 호출 유틸리티 ---

def call_openai_response(model, input_data, *, response_format=None, max_retries=3):
    """
    OpenAI Responses API 호출 + 간단 Retry 래퍼.
    - model: TEXT_MODEL / VISION_MODEL
    - input_data: 문자열 또는 Responses input(JSON)
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "input": input_data,
            }
            if response_format:
                kwargs["response_format"] = response_format

            # 필요하면 timeout 인자 추가 가능 (예: timeout=600)
            response = client.responses.create(**kwargs)
            return response
        except Exception as e:
            last_err = e
            print(f"⚠️ OpenAI 호출 실패 {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                # 간단한 지수형 backoff
                time.sleep(2 * attempt)
    # 여기까지 오면 전부 실패
    raise last_err


def extract_output_text_from_response(response):
    """
    OpenAI Responses API 응답에서 text 부분만 꺼내는 헬퍼.
    - response_format={"type": "json_object"} 를 쓰면, JSON 문자열이 들어있다고 가정.
    """
    try:
        # Python SDK 객체 형태일 때
        output_items = getattr(response, "output", None)
        if output_items:
            texts = []
            for item in output_items:
                contents = getattr(item, "content", None) or []
                for c in contents:
                    # output_text 타입일 때
                    if getattr(c, "type", None) == "output_text":
                        texts.append(getattr(c, "text", ""))
            if texts:
                return "\n".join(texts).strip()
    except Exception as e:
        print(f"⚠️ 응답 파싱 중 예외: {e}")

    # dict 형태로 들어온 경우 (안전장치)
    if isinstance(response, dict):
        output_items = response.get("output", [])
        if output_items:
            contents = output_items[0].get("content", [])
            if contents and contents[0].get("type") == "output_text":
                return contents[0].get("text", "")

    # 최후의 수단
    return str(response)


def combine_parts_to_prompt(parts):
    """
    기존 Gemini의 "parts" 리스트를 단일 텍스트 프롬프트로 합치는 함수.
    - 문자열이면 그대로
    - {"text": "..."} 형태면 text만 추출
    - 기타는 무시
    """
    chunks = []
    for p in parts:
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, dict) and "text" in p:
            chunks.append(str(p["text"]))
    return "\n\n".join(chunks)


def resize_image_bytes(image_bytes, max_size=1500):
    """
    메모리 절약 + OCR 성능 유지용 이미지 리사이즈 헬퍼.
    - 긴 변이 max_size를 넘으면 비율 유지하며 리사이즈
    - JPEG(또는 원본 포맷)로 재저장
    """
    img = PIL.Image.open(io.BytesIO(image_bytes))

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
        print(f"📉 이미지 리사이징: {img.size}")
    else:
        print(f"✅ 리사이징 불필요: {img.size}")

    fmt = img.format if img.format else "JPEG"
    buf = io.BytesIO()
    # 품질 85 정도로 살짝 압축 (메모리/트래픽 절약)
    img.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return buf.read(), fmt


# --- 법령 텍스트 로드 ---
def load_law_texts() -> str:
    """법령 .txt 파일들을 모두 읽어 하나의 큰 텍스트로 합칩니다."""
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
# (여기 PROMPT_EXTRACT_INGREDIENT_INFO / PROMPT_CREATE_STANDARD / PROMPT_VERIFY_DESIGN
#  는 질문에서 준 그대로 사용, 내용은 동일하므로 생략하지 않고 그대로 둡니다.)

PROMPT_EXTRACT_INGREDIENT_INFO = """
이 이미지는 원부재료 표시사항 사진입니다. 
**필수적으로 추출해야 할 정보만** 추출하세요.
...
(생략 없이 기존 그대로 사용)
"""  # 👉 실제 구현 시에는 질문에 주신 전문을 그대로 넣으세요

PROMPT_CREATE_STANDARD = """
당신은 식품 규정 및 표시사항 전문가입니다.
...
(생략 없이 기존 그대로 사용)
"""

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사관이자 법률 전문가입니다.
...
(생략 없이 기존 그대로 사용)
"""


# --- 텍스트 정리 유틸리티 ---

def clean_html_text(text):
    """HTML 태그와 엔티티를 완전히 제거하여 순수 텍스트만 반환"""
    if not text:
        return ""

    import re
    import html

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
    """AI 응답의 모든 문자열 값에서 HTML 태그 제거 (재귀적)"""
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


# --- 파일 처리 함수들 ---

def process_file_to_part(file_storage):
    """
    (텍스트 기반으로만) 파일을 모델에 줄 수 있는 형태로 변환.
    - Excel: CSV 텍스트
    - 이미지/PDF: 여기서는 단순히 설명 텍스트만 제공 (실제 이미지 분석은 Vision API에서 별도 처리)
    """
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)

    # 엑셀 → CSV 텍스트
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 이미지 / PDF / 기타는 현재 버전에서는 내용 자체를 여기서 분석하지 않고,
    # 단순한 설명만 텍스트로 넘김 (실제 내용 분석은 Vision/별도 OCR에서 처리)
    return {
        "text": f"[파일] 이름: {file_storage.filename}, MIME: {mime_type}, 크기: {len(file_data)} bytes"
    }


def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출 (OpenAI Vision 사용)"""
    try:
        image_data = image_file.read()
        image_file.seek(0)

        # 메모리 절약용 리사이징
        resized_bytes, fmt = resize_image_bytes(image_data)
        mime_type = image_file.mimetype or f"image/{fmt.lower()}"

        b64_image = base64.b64encode(resized_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PROMPT_EXTRACT_INGREDIENT_INFO.strip()
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ]

        response = call_openai_response(
            VISION_MODEL,
            input_items,
            response_format={"type": "json_object"}  # JSON 강제
        )

        result_text = extract_output_text_from_response(response).strip()

        # 만약 모델이 ```json 코드블록으로 감싸서 보내면 제거
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
        # (이 아래 엑셀 생성 로직은 기존 코드 그대로 유지)
        # ...
        if 'product_info' in data:
            product_df = pd.DataFrame([data['product_info']])
            product_df.to_excel(writer, sheet_name='제품정보', index=False)

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

            if 'continuous_text' in data['ingredients']:
                continuous_df = pd.DataFrame([{
                    '원재료명_연속텍스트': data['ingredients']['continuous_text']
                }])
                continuous_df.to_excel(writer, sheet_name='원재료명_연속텍스트', index=False)

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


# 1단계: 정답지 만들기 (엑셀 + 원재료 사진들 몽땅)
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

    # (1) 엑셀 → 텍스트
    excel_part = process_file_to_part(excel_file)
    if excel_part:
        parts.append(excel_part)

    # (2) 원재료 이미지들 Vision으로 먼저 분석 → JSON만 텍스트로 붙임
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
        prompt_text = combine_parts_to_prompt(parts)

        response = call_openai_response(
            TEXT_MODEL,
            prompt_text,
            response_format={"type": "json_object"}
        )

        result_text = extract_output_text_from_response(response).strip()

        # JSON 코드블록 제거
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


@app.route('/api/read-standard-excel', methods=['POST'])
def read_standard_excel():
    """엑셀 파일에서 기준 데이터를 읽어옴"""
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


# 2단계: 검증하기 (엑셀 파일 또는 JSON + 디자인 이미지)
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

    if standard_excel:
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
                    standard_data = {
                        'ingredients': {
                            'structured_list': ingredients_list,
                            'continuous_text': ', '.join(ingredients_list)
                        }
                    }
                else:
                    return jsonify({"error": "엑셀 파일의 첫 번째 시트에 데이터가 없습니다."}), 400
            else:
                return jsonify({"error": "엑셀 파일의 첫 번째 시트가 비어있습니다."}), 400

            standard_json = json.dumps(standard_data, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 엑셀 파일 읽기 오류: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"엑셀 파일 읽기 실패: {str(e)}"}), 400

    parts = []

    enhanced_prompt = PROMPT_VERIFY_DESIGN
    if ALL_LAW_TEXT:
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)

    parts.append(f"\n--- [기준 데이터(Standard)] ---\n{standard_json}")

    prompt_text = combine_parts_to_prompt(parts)

    try:
        mime_type = design_file.mimetype or ""
        input_data = None

        if mime_type.startswith("image/"):
            # 이미지인 경우 Vision 사용
            img_bytes = design_file.read()
            design_file.seek(0)

            resized_bytes, fmt = resize_image_bytes(img_bytes)
            real_mime = mime_type or f"image/{fmt.lower()}"
            b64_image = base64.b64encode(resized_bytes).decode("utf-8")
            data_url = f"data:{real_mime};base64,{b64_image}"

            input_data = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt_text
                        },
                        {
                            "type": "input_image",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ]

            response = call_openai_response(
                VISION_MODEL,
                input_data,
                response_format={"type": "json_object"}
            )
        else:
            # 이미지가 아니면 일단 텍스트만 기반으로 검증 (PDF는 별도 OCR 전처리 추가 가능)
            input_data = prompt_text
            response = call_openai_response(
                TEXT_MODEL,
                input_data,
                response_format={"type": "json_object"}
            )

        result_text = extract_output_text_from_response(response).strip()

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
            try:
                result_text_fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."}), 500

        result = clean_ai_response(result)

        return jsonify(result)

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
...
(질문에 있던 프롬프트 전문 그대로)
"""

    if ALL_LAW_TEXT:
        qa_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"

    parts.append(qa_prompt)

    # QA 파일들 처리 (현재는 엑셀/텍스트 위주로 사용, 이미지는 별도 전처리 필요)
    for qa_file in qa_files[:20]:
        file_part = process_file_to_part(qa_file)
        if file_part:
            parts.append(file_part)

    print(f"📂 QA 자료 처리 중: {len(qa_files)}개 파일")

    try:
        prompt_text = combine_parts_to_prompt(parts)

        response = call_openai_response(
            TEXT_MODEL,
            prompt_text,
            response_format={"type": "json_object"}
        )

        result_text = extract_output_text_from_response(response).strip()

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
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 (OpenAI 버전) 가동")
    print("   - 원부재료 표시사항 스마트 추출 (OpenAI Vision)")
    print("   - 법률 검토 기능 통합")
    print("   - QA 자료 업로드 지원")
    from waitress import serve

    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=4,
        channel_timeout=600  # 600초(10분) 동안 응답 없어도 연결 유지
    )
