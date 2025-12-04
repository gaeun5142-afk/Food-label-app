import os
import json
import io
import glob
import traceback
import base64
import re
import html
import unicodedata

import pandas as pd 
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import PIL.Image

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


# =========================
# 기본 설정
# =========================
load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다!")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ChatGPT 멀티모달 모델
MODEL_NAME = "gpt-4.1-mini"


# =========================
# 유틸 함수들
# =========================

def normalize_text_strict(text):
    """엄격한 비교용 정규화 (공백/특수문자 유지)"""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize('NFKC', text)


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


def to_image_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_openai_from_parts(parts, json_mode: bool = True) -> str:
    """
    OpenAI Responses API 호출.
    - parts: 문자열, PIL.Image.Image 섞여 있는 리스트
    - json_mode: True면 "JSON만 출력" 시스템 지시 추가
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    content = []

    if json_mode:
        content.append({
            "type": "input_text",
            "text": (
                "항상 유효한 JSON만 출력하세요. "
                "마크다운, 코드블록, 설명 문장은 절대 포함하지 마세요."
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
                "image_url": {"url": data_url},
            })
        else:
            # 필요시 확장
            pass

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=0.0,
        max_output_tokens=32768,
    )

    result_chunks = []
    for out in getattr(resp, "output", []):
        for c in getattr(out, "content", []):
            if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                result_chunks.append(c.text)
    return "".join(result_chunks).strip()


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


def clean_law_violation_text(text: str) -> str:
    """
    AI가 생성한 법령 위반 문구를 사람이 읽기 좋게 정리한다.
    예: '식품등의_표시기준 III. 1. 버. 2) 타) (1) 위반'
        → '식품등의 표시기준 III.1.버.2)타)(1) 위반'
    """
    if not text:
        return text

    # 1) _ 제거
    text = text.replace("_", "")

    # 2) 공백 정리
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\s+', '.', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3) 통일 (여기서는 그냥 그대로, 필요하면 추가 규칙 가능)
    text = text.replace("식품등의 표시기준", "식품등의 표시기준")

    return text


def highlight_ocr_errors(ocr_text: str, issues: list) -> str:
    """
    OCR 텍스트에서 오류 부분을 하이라이트.
    - issues[*]['actual'] 에 들어있는 문자열들을 찾아서 span으로 감싼다.
    """
    if not ocr_text:
        return ""

    import html as html_mod

    highlighted_text = str(ocr_text)

    if not issues:
        highlighted_text = html_mod.escape(highlighted_text)
        highlighted_text = highlighted_text.replace("\n", "<br>")
        return highlighted_text

    # 하이라이트할 텍스트 수집 (중복 제거, 긴 문자열 우선)
    highlight_texts = []
    seen = set()
    for issue in issues:
        actual = issue.get("actual", "")
        if actual:
            actual_clean = str(actual).strip()
            if actual_clean and actual_clean not in seen:
                highlight_texts.append(actual_clean)
                seen.add(actual_clean)
                print(f"🔴 하이라이트 대상: '{actual_clean}'")

    if not highlight_texts:
        highlighted_text = html_mod.escape(highlighted_text)
        highlighted_text = highlighted_text.replace("\n", "<br>")
        return highlighted_text

    highlight_texts.sort(key=len, reverse=True)

    highlight_positions = []
    for highlight_text in highlight_texts:
        start = 0
        while True:
            pos = highlighted_text.find(highlight_text, start)
            if pos == -1:
                break
            overlap = False
            for existing_pos in highlight_positions:
                if not (pos + len(highlight_text) <= existing_pos[0] or pos >= existing_pos[1]):
                    overlap = True
                    break
            if not overlap:
                highlight_positions.append((pos, pos + len(highlight_text), highlight_text))
            start = pos + 1

    highlight_positions.sort(reverse=True)

    for start, end, highlight_text in highlight_positions:
        escaped_text = html_mod.escape(highlight_text)
        highlighted = (
            '<span style="background-color:#ffcccc;'
            ' color:#cc0000; font-weight:bold; padding:2px 4px;'
            ' border-radius:3px;">'
            f'{escaped_text}</span>'
        )
        highlighted_text = highlighted_text[:start] + highlighted + highlighted_text[end:]
        print(f"✅ 하이라이트 적용: '{highlight_text}' (위치: {start}-{end})")

    # 하이라이트 span 외부 텍스트 이스케이프
    parts = re.split(r'(<span[^>]*>.*?</span>)', highlighted_text)
    result_parts = []
    for part in parts:
        if part.startswith('<span'):
            result_parts.append(part)
        else:
            result_parts.append(html_mod.escape(part))
    highlighted_text = ''.join(result_parts)

    highlighted_text = highlighted_text.replace("\n", "<br>")
    return highlighted_text


def process_file_to_part(file_storage):
    """
    파일을 OpenAI Vision이 이해할 수 있는 형태로 변환
    - 엑셀: 텍스트(CSV) 스트링
    - 이미지: PIL.Image
    - PDF: 첫 페이지 이미지를 PIL.Image (가능하면)
    """
    mime_type = file_storage.mimetype or ""
    file_data = file_storage.read()
    file_storage.seek(0)

    # 엑셀 -> CSV 텍스트
    if mime_type in [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
    ]:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # PDF -> 이미지
    if mime_type == 'application/pdf' and PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(file_data, dpi=200)
            if images:
                img = images[0].convert("RGB")
                return img
        except Exception as e:
            print(f"PDF->이미지 변환 실패: {e}")
            return None

    # 이미지
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
            return None

    # 그 외 타입은 텍스트로 해석 불가 -> 생데이터로 넘기지 않고 None
    return None


# =========================
# 프롬프트
# =========================

PROMPT_EXTRACT_INGREDIENT_INFO = """
당신은 한국 식품 라벨 OCR 전문가입니다.
이미지에서 원부재료 표시사항을 **정확하게** 추출하세요.
추측하거나 창의적으로 해석하지 말고, 보이는 텍스트만 정확히 추출하세요.

🚨 절대 규칙 🚨
1. 이미지에 **보이는 글자만** 추출 (추론/보정 금지)
2. **특수문자(쉼표, 점, 괄호) 누락도 그대로** 추출
3. 오타, 띄어쓰기, 특수문자 모두 **정확히 그대로**
4. 문법적으로 틀려도 **이미지와 100% 동일**하게

[추출해야 할 정보]
1. 원재료명
2. 복합원재료 내역
3. 원산지 정보
4. 함량 정보
5. 알레르기 유발물질
6. 식품첨가물

JSON 형식으로만 응답하세요:
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

🤖 기계 모드:
- 철자 교정기 OFF
- 문법 검사기 OFF
- 자동 완성 OFF
- 추론 엔진 OFF

출력 규칙:
1. 보이는 글자 → 그대로
2. 틀린 글자 → 틀린 대로
3. 빠진 쉼표 → 빠진 대로
4. 이상한 숫자 → 이상한 대로

JSON 형식으로만 응답:
{
  "raw_text": "있는 그대로의 텍스트"
}
"""

PROMPT_CREATE_STANDARD = """
당신은 식품 규정 및 표시사항 전문가입니다.
제공된 [배합비 데이터(Excel)]와 [원재료 표시사항 사진들에서 추출한 정보]를 종합하여,
법적으로 완벽한 '식품표시사항 기준 데이터(Standard)'를 생성하세요.

출력은 반드시 JSON 하나만:
{
    "product_info": {...},
    "ingredients": {...},
    "allergens": {...},
    "nutrition_info": {...},
    "manufacturer": {...},
    "precautions": [...],
    "law_compliance": {...},
    "details": [...]
}
"""

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사 AI입니다.
제공된 [Standard(기준서)]와 [Design OCR 텍스트]를 1:1 정밀 대조하여 채점하세요.

- 맞춤법 자동 보정 금지
- 오타 수정 금지
- 숫자/단위 보정 금지 (900g ≠ 900 g)
- 문장부호 보정 금지

항상 JSON으로만 응답:
{
  "design_ocr_text": "디자인 전체 텍스트",
  "score": 100,
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": ["식품등의 표시기준 ... 위반", ...]
  },
  "issues": [
    {
      "type": "Critical" | "Minor" | "Law_Violation",
      "location": "항목명",
      "issue": "오류 설명",
      "expected": "기준서 텍스트",
      "actual": "디자인에서 발견된 오류 텍스트 (하이라이트 대상)",
      "suggestion": "수정 제안"
    }
  ]
}
"""


# =========================
# OCR 함수
# =========================

def ocr_image_bytes_with_chatgpt(image_bytes: bytes) -> str:
    if client is None:
        return ""

    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        max_size = 1600
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            print(f"📉 OCR용 이미지 리사이즈: {new_size}")

        ocr_prompt = """
이 이미지는 식품 포장지/라벨 사진입니다.
이미지 안에 보이는 모든 글자를 그대로 적어 주세요.

- 줄바꿈, 공백, 숫자, 기호를 최대한 원문 그대로 유지
- 요약/설명 금지
- 순수 텍스트만
"""
        parts = [ocr_prompt, img]
        text = call_openai_from_parts(parts, json_mode=False).strip()

        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if text:
            print("✅ ChatGPT OCR 성공")
            return text
        else:
            print("⚠️ ChatGPT OCR 결과 비어 있음")
            return ""
    except Exception as e:
        print("❌ ChatGPT OCR 실패:", e)
        return ""


def ocr_bytes_to_text(image_bytes: bytes) -> str:
    text = ocr_image_bytes_with_chatgpt(image_bytes)
    if text:
        return text

    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='kor+eng').strip()
        if text:
            print("✅ pytesseract OCR 성공 (폴백)")
        else:
            print("⚠️ pytesseract OCR 결과 비어 있음")
        return text
    except Exception as e:
        print("❌ pytesseract OCR 실패:", e)
        return ""


def extract_ingredient_info_from_image(image_file):
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        result_text = call_openai_from_parts(parts, json_mode=True).strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("```", 1)
            if len(lines) > 1:
                result_text = lines[1].strip()
        result_text = result_text.strip()

        return json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"원재료 정보 JSON 파싱 실패: {e}")
        print(f"응답 텍스트 일부: {result_text[:500]}...")
        return None
    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        traceback.print_exc()
        return None


# =========================
# 엑셀 생성/읽기
# =========================

def create_standard_excel(data):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 제품 정보
        if 'product_info' in data:
            product_df = pd.DataFrame([data['product_info']])
            product_df.to_excel(writer, sheet_name='제품정보', index=False)

        # 원재료명
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

        # 알레르기
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

        # 영양정보
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

        # 제조원
        if 'manufacturer' in data:
            manufacturer_df = pd.DataFrame([data['manufacturer']])
            manufacturer_df.to_excel(writer, sheet_name='제조원정보', index=False)

        # 주의사항
        if 'precautions' in data:
            precautions_df = pd.DataFrame([{'주의사항': item} for item in data['precautions']])
            precautions_df.to_excel(writer, sheet_name='주의사항', index=False)

        # 원재료 상세
        if 'details' in data and data['details']:
            details_df = pd.DataFrame(data['details'])
            details_df.to_excel(writer, sheet_name='원재료상세', index=False)

    output.seek(0)
    return output


# =========================
# 라우트
# =========================

@app.route('/')
def index():
    return render_template('index.html')


# 1단계: 기준 데이터 생성
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
    if excel_part and isinstance(excel_part, dict) and "text" in excel_part:
        parts.append(excel_part["text"])

    ingredient_info_list = []
    for img in raw_images[:15]:
        print(f"📷 원재료 이미지 처리 중: {img.filename}")
        info = extract_ingredient_info_from_image(img)
        if info:
            ingredient_info_list.append(info)

    if ingredient_info_list:
        ingredients_text = "--- [원재료 표시사항에서 추출한 정보] ---\n"
        for idx, info in enumerate(ingredient_info_list, 1):
            ingredients_text += f"\n[원재료 {idx}]\n"
            ingredients_text += json.dumps(info, ensure_ascii=False, indent=2)
            ingredients_text += "\n"
        ingredients_text += "--- [원재료 정보 끝] ---\n"
        parts.append(ingredients_text)

    print(f"📂 처리 중: 엑셀 1개 + 원재료 이미지 {len(raw_images)}장")

    try:
        result_text = call_openai_from_parts(parts, json_mode=True).strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                lines = "\n".join(lines[1:])
            if lines.endswith("```"):
                lines = lines[:-3]
            result_text = lines.strip()

        result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 1000자): {result_text[:1000]}")
            try:
                fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}"}), 500

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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 2단계: 디자인 검증 (OpenAI + 하이라이트 + 법령정리)
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")

    design_file = request.files.get('design_file')
    standard_excel = request.files.get('standard_excel')
    standard_json = request.form.get('standard_data')

    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400

    # 기준 데이터 처리
    if standard_excel and not standard_json:
        try:
            standard_excel.seek(0)
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

    if not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 또는 JSON)가 필요합니다."}), 400

    # 디자인 파일 -> 이미지 bytes
    try:
        design_data = design_file.read()
        design_file.seek(0)

        if design_file.mimetype == 'application/pdf' and PDF2IMAGE_AVAILABLE:
            images = convert_from_bytes(design_data, dpi=200)
            if not images:
                return jsonify({"error": "PDF에서 이미지를 추출할 수 없습니다."}), 400
            img_io = io.BytesIO()
            images[0].save(img_io, format='PNG')
            design_image_bytes = img_io.getvalue()
        elif design_file.mimetype.startswith('image/'):
            design_image_bytes = design_data
        else:
            return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400

        # OCR
        ocr_text = ocr_bytes_to_text(design_image_bytes)
        if not ocr_text:
            return jsonify({"error": "OCR 실행에 실패했습니다."}), 500

        # AI 프롬프트 조립
        law_text = ALL_LAW_TEXT or ""
        parts = [
            PROMPT_VERIFY_DESIGN
            + "\n\n[참고 법령]\n"
            + law_text
            + "\n\n[기준 데이터(Standard)]\n"
            + standard_json
            + "\n\n[디자인 OCR 텍스트]\n"
            + ocr_text
            + "\n[디자인 OCR 텍스트 끝]\n"
        ]

        result_text = call_openai_from_parts(parts, json_mode=True).strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            result_text = "\n".join(lines).strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            print("❌ 검증 JSON 파싱 실패:", e)
            print("응답 일부:", result_text[:1000])
            return jsonify({"error": "검증 결과 JSON 파싱 실패"}), 500

        # design_ocr_text 없으면 우리가 사용한 OCR 텍스트 넣기
        if not result.get("design_ocr_text"):
            result["design_ocr_text"] = ocr_text

        # 법령 위반 문구 정리
        law_obj = result.get("law_compliance", {})
        violations = law_obj.get("violations", [])
        if isinstance(violations, list):
            cleaned = [clean_law_violation_text(v) for v in violations]
            result.setdefault("law_compliance", {})
            result["law_compliance"]["violations"] = cleaned

        # 하이라이트 HTML 생성
        issues = result.get("issues", []) or []
        highlighted_html = highlight_ocr_errors(result["design_ocr_text"], issues)
        result["design_ocr_highlighted_html"] = highlighted_html

        # score 없으면 issues 개수로 단순 계산
        if "score" not in result:
            score = max(0, 100 - 5 * len(issues))
            result["score"] = score

        return jsonify(result)

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify-design-strict', methods=['POST'])
def verify_design_strict():
    """
    Python으로 문자 단위 정확 비교 (AI 없이, OCR만 OpenAI 사용)
    """
    try:
        design_file = request.files.get('design_file')
        standard_json = request.form.get('standard_data')

        if not design_file or not standard_json:
            return jsonify({"error": "파일과 기준 데이터가 필요합니다"}), 400

        design_file.seek(0)
        standard_data = json.loads(standard_json)

        # 기준 텍스트
        if 'ingredients' in standard_data:
            std_text = standard_data['ingredients']['continuous_text']
        else:
            return jsonify({"error": "standard_data에 ingredients.continuous_text가 필요합니다"}), 400

        # 디자인 OCR
        design_data = design_file.read()
        design_file.seek(0)

        if design_file.mimetype == 'application/pdf' and PDF2IMAGE_AVAILABLE:
            images = convert_from_bytes(design_data, dpi=200)
            if not images:
                return jsonify({"error": "PDF에서 이미지를 추출할 수 없습니다."}), 400
            img_io = io.BytesIO()
            images[0].save(img_io, format='PNG')
            design_image_bytes = img_io.getvalue()
        elif design_file.mimetype.startswith('image/'):
            design_image_bytes = design_data
        else:
            return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400

        des_text = ocr_bytes_to_text(design_image_bytes)
        if not des_text:
            return jsonify({"error": "OCR 실패"}), 500

        # 문자 단위 비교
        raw_issues = compare_texts_strict(std_text, des_text)

        all_issues = []
        for issue in raw_issues:
            pos = issue['position']
            expected_char = issue['expected']
            actual_char = issue['actual']
            ctx_before = issue['context_before']
            ctx_after = issue['context_after']

            expected_snippet = ctx_before + expected_char + ctx_after
            actual_snippet = ctx_before + actual_char + ctx_after

            issue_type = "Critical" if expected_char not in [' ', ',', '.', '·'] else "Minor"

            all_issues.append({
                "type": issue_type,
                "location": f"원재료명 (위치: {pos})",
                "issue": f"'{expected_char}' → '{actual_char}'",
                "expected": expected_snippet,
                "actual": actual_snippet,
                "suggestion": f"위치 {pos}의 '{actual_char}'을(를) '{expected_char}'(으)로 수정"
            })

        critical_count = sum(1 for i in all_issues if i['type'] == 'Critical')
        minor_count = sum(1 for i in all_issues if i['type'] == 'Minor')
        score = max(0, 100 - critical_count * 5 - minor_count * 2)

        highlighted_html = highlight_ocr_errors(des_text, all_issues)

        return jsonify({
            "design_ocr_text": des_text,
            "design_ocr_highlighted_html": highlighted_html,
            "score": score,
            "issues": all_issues,
            "law_compliance": {"status": "compliant", "violations": []}
        })

    except Exception as e:
        print(f"❌ verify_design_strict 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    """QA 자료 업로드 후 식품표시사항 작성 (OpenAI 기반)"""
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")

    qa_files = request.files.getlist('qa_files')
    if not qa_files:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    parts = []
    qa_prompt = """
당신은 식품표시사항 작성 전문가입니다.
제공된 QA 자료(엑셀, 이미지)를 분석하여 법률을 준수하는 식품표시사항을 작성하세요.

JSON 형식으로만 응답:
{
  "product_name": "제품명",
  "label_text": "작성된 식품표시사항 전체 텍스트",
  "law_compliance": {
    "status": "compliant" | "needs_review",
    "issues": ["법률 검토 사항"]
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
        part = process_file_to_part(qa_file)
        if isinstance(part, PIL.Image.Image):
            parts.append(part)
        elif isinstance(part, dict) and "text" in part:
            parts.append(part["text"])

    try:
        result_text = call_openai_from_parts(parts, json_mode=True).strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            result_text = "\n".join(lines).strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 1000자): {result_text[:1000]}")
            try:
                fixed = result_text.replace(',\n}', '\n}').replace(',\n]', '\n]')
                result = json.loads(fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify({"error": f"JSON 파싱 실패: {str(json_err)}"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"❌ QA 자료 처리 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 (OpenAI 통합 버전) 가동")
    from waitress import serve

    serve(app, host='0.0.0.0', port=8080)
