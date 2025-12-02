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
import openai
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

# ✅ OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다!")
    client = None
else:
    openai.api_key = OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4"          # 텍스트용 모델
OCR_MODEL_NAME = "gpt-4o-mini"  # 이미지 OCR용 모델 (비전 지원)

def call_openai_chat(messages, temperature=0.4):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI 호출 실패: {e}")
        return ""

# 텍스트 정리 유틸
def clean_html_text(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(r'<[^>]+>', '', text)
    # 🔧 여기 3줄만 수정
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
        return {k: clean_ai_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_ai_response(item) for item in data]
    elif isinstance(data, str):
        return clean_html_text(data)
    return data

# -----------------------
#   ChatGPT OCR 헬퍼
# -----------------------

def _ocr_via_openai(image_bytes, mime_type="image/png"):
    """OpenAI 비전 모델을 사용해 OCR 수행 (가능하면 이 결과 사용)."""
    if client is None:
        return ""

    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        resp = client.chat.completions.create(
            model=OCR_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "이미지에 보이는 모든 글자를 한 글자도 빼지 말고 그대로 적어 주세요. "
                                "맞춤법/띄어쓰기/숫자/단위/기호를 고치지 말고, 줄바꿈도 최대한 유지해 주세요."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            temperature=0.0,
        )

        message_content = resp.choices[0].message.content
        # SDK 버전에 따라 content가 str 또는 list일 수 있음
        if isinstance(message_content, str):
            return message_content.strip()
        else:
            chunks = []
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(part.get("text", ""))
            return "".join(chunks).strip()
    except Exception as e:
        print(f"⚠️ OpenAI OCR 실패: {e}")
        return ""

# 이미지 OCR 처리 (ChatGPT 우선, 실패 시 Tesseract 폴백)
def ocr_bytes_to_text(image_bytes, mime_type="image/png"):
    # 1) OpenAI 비전으로 시도
    text = _ocr_via_openai(image_bytes, mime_type=mime_type)
    if text:
        return text

    # 2) 실패 시 Tesseract 폴백 (설치된 경우)
    if not TESSERACT_AVAILABLE:
        return ""

    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("L")  # 그레이스케일

        # 🔧 라벨 OCR에 유리하도록 살짝 선명하게 / 이진화
        img = img.point(lambda x: 0 if x < 160 else 255, '1')  # 단순 임계값

        # 🔧 Tesseract 설정
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(
            img,
            lang='kor+eng',
            config=config
        )
        return text
    except Exception as e:
        print("OCR 폴백 실패:", e)
        return ""

# 파일을 모델 파트로 변환
def process_file_to_part(file_storage):
    mime_type = file_storage.mimetype or ""
    file_data = file_storage.read()
    file_storage.seek(0)

    # 엑셀 파일
    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 이미지 파일
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
            print(f"⚠️ 이미지 처리 실패: {e}")
            return {"mime_type": mime_type, "data": file_data}

    # PDF 파일
    if mime_type == 'application/pdf' and PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(file_data, dpi=200)
            if images:
                print(f"📄 PDF→이미지 변환 완료 (총 {len(images)} 페이지)")
                return images[0].convert("RGB")
        except Exception as e:
            print("PDF->이미지 변환 실패:", e)
            return {"mime_type": mime_type, "data": file_data}

    return {"mime_type": mime_type, "data": file_data}

# 원재료 정보 추출 (OCR + ChatGPT 조합)
def extract_ingredient_info_from_image(image_file):
    try:
        image_data = image_file.read()
        image_file.seek(0)

        # ChatGPT OCR (필요 시 Tesseract 폴백)
        ocr_text = ocr_bytes_to_text(
            image_data,
            mime_type=image_file.mimetype or "image/png"
        )

        messages = [
            {"role": "system", "content": "당신은 식품 표시사항 전문가입니다."},
            {"role": "user", "content": f"{PROMPT_EXTRACT_INGREDIENT_INFO}\n\n{ocr_text}"}
        ]
        result_text = call_openai_chat(messages)

        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text.split("```")[1].strip()

        return json.loads(result_text)

    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/create-standard', methods=['POST'])
def create_standard():
    print("⚙️ 기준 데이터 생성 시작...")
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
        ingredients_text += "\n--- [원재료 정보 끝] ---"
        parts.append(ingredients_text)

    try:
        result_text = call_model_with_parts(enhanced_prompt, parts[1:])
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result = json.loads(result_text)
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

@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️ 디자인 검증 시작...")
    design_file = request.files.get('design_file')
    standard_json = request.form.get('standard_data')

    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400
    if not standard_json:
        return jsonify({"error": "기준 데이터가 필요합니다."}), 400

    enhanced_prompt = PROMPT_VERIFY_DESIGN
    if ALL_LAW_TEXT:
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"

    parts = [
        enhanced_prompt,
        f"\n--- [기준 데이터] ---\n{standard_json}"
    ]

    design_part = process_file_to_part(design_file)
    if design_part:
        if isinstance(design_part, dict) and 'text' in design_part:
            parts.append(design_part['text'])

    try:
        result_text = call_model_with_parts(enhanced_prompt, parts[1:])
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result = json.loads(result_text)
        result = clean_ai_response(result)
        return jsonify(result)
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    print("📋 QA 자료 업로드 시작...")
    qa_files = request.files.getlist('qa_files')
    if not qa_files:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    qa_prompt = """
당신은 식품표시사항 작성 전문가입니다.
제공된 QA 자료를 분석하여 법률을 준수하는 식품표시사항을 작성하세요.
(중략)
"""
    if ALL_LAW_TEXT:
        qa_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"

    parts = [qa_prompt]
    for file in qa_files[:20]:
        part = process_file_to_part(file)
        if part:
            if isinstance(part, dict) and 'text' in part:
                parts.append(part['text'])

    try:
        result_text = call_model_with_parts(qa_prompt, parts[1:])
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result = json.loads(result_text)
        return jsonify(result)
    except Exception as e:
        print(f"❌ QA 처리 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- 서버 실행부 ---
if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 (ChatGPT API 버전) 가동")
    from waitress import serve
    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=4,
        channel_timeout=600
    )

