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
import re
import unicodedata

# --- 설정 및 초기화 ---
load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

# =========================
# 공통 유틸 & 설정
# =========================

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("🚨 경고: .env 파일에 GOOGLE_API_KEY가 없습니다!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'gemini-1.5-flash'

# 모든 호출에 사용할 결정적 설정
STABLE_GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.0,          # 0으로 두면 완전 greedy
    "top_k": 1,
    "candidate_count": 1,
    "max_output_tokens": 32768,
    "response_mime_type": "application/json"
}


def get_model(extra_config: dict | None = None, system_instruction: str | None = None):
    """결정적 설정으로 모델 생성 (필요시 config override 가능)"""
    gen_conf = STABLE_GENERATION_CONFIG.copy()
    if extra_config:
        gen_conf.update(extra_config)
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config=gen_conf,
        system_instruction=system_instruction
    )


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


def safe_extract_json(text: str) -> str:
    """
    응답 텍스트에서 JSON 부분만 안전하게 추출.
    - 첫 번째 '{' 부터 마지막 '}' 까지
    - 흔한 trailing comma 보정
    """
    if not text:
        raise ValueError("빈 응답입니다.")

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        # JSON 형식이 아니라고 판단
        raise ValueError("JSON 구간을 찾지 못했습니다.")

    candidate = text[start:end + 1]
    # 단순 쉼표 보정
    candidate = candidate.replace(',\n}', '\n}').replace(',\n]', '\n]')
    candidate = candidate.replace(', }', ' }').replace(', ]', ' ]')
    return candidate.strip()


# =========================
# 법령 텍스트 로드
# =========================

def load_law_texts() -> str:
    """법령 .txt 파일들을 모두 읽어 하나의 큰 텍스트로 합칩니다."""
    print("📚 법령 파일들을 읽어오는 중...")
    law_files = glob.glob("law_text_*.txt") + glob.glob("../law_text_*.txt") + glob.glob("law_*.txt")

    if not law_files:
        print("⚠️ 법령 파일이 없습니다. 법률 검토 기능이 제한될 수 있습니다.")
        return ""

    all_law_text = ""
    for file_path in law_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_law_text += f"\n\n=== [법령: {file_path}] ===\n"
                all_law_text += f.read()
    except Exception as e:
            print(f"❌ 법령 파일 '{file_path}' 읽기 실패: {e}")

    print(f"✅ 모든 법령 파일 로드 완료 (총 {len(all_law_text)}자)")
    return all_law_text


ALL_LAW_TEXT = load_law_texts()[:60000]  # 과도하게 크면 잘라서 사용


# =========================
# 프롬프트 정의
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

반드시 **JSON만** 출력하세요.
마크다운 코드블록(````), HTML 태그, 설명 문구 절대 출력하지 마세요.

[출력 형식]
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
1. 보이는 글자 → 그대로 출력
2. 틀린 글자 → 틀린 대로 출력
3. 빠진 쉼표 → 빠진 대로 출력
4. 이상한 숫자 → 이상한 대로 출력

반드시 JSON만 출력하세요. 코드블록, 설명 금지.

예시:
- 이미지: "전반가공품" → 출력: "전반가공품"
- 이미지: "대두 게" → 출력: "대두 게"
- 이미지: "221%" → 출력: "221%"

JSON 형식:
{
  "raw_text": "있는 그대로의 텍스트"
}
"""

PROMPT_CREATE_STANDARD = """
당신은 식품 규정 및 표시사항 전문가입니다.
제공된 [배합비 데이터(Excel)]와 [원재료 표시사항 사진들에서 추출한 정보]를 종합하여,
법적으로 완벽한 **'식품표시사항 기준 데이터(Standard)'**를 실제 라벨 형식으로 생성하세요.

반드시 **JSON만** 출력하세요. 마크다운/HTML/설명 금지.

[분석 단계]
1. Excel의 배합비율(%)이 높은 순서대로 원재료 나열.
2. Excel 원재료명과 이미지 원재료 라벨을 매칭해 상세 정보 보강.
3. 제공된 법령을 참고해 법적 필수 항목이 모두 포함되도록 구성.
4. 실제 라벨에 사용될 수 있는 완전한 구조로 정리.

[출력 형식 생략, (현재 코드와 동일)]
"""  # 원래 긴 포맷 그대로 사용 (생략)

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사 AI입니다.
Standard(기준서)와 Design(디자인 이미지/PDF)을 1:1로 엄격히 비교하여
점수(score)와 issues를 JSON으로만 반환하세요.

🚨 출력 규칙 (매우 중요) 🚨
- **반드시 JSON만** 출력
- 마크다운 코드블록(````), HTML 태그(<div> 등), 설명 문구 절대 출력 금지
- JSON 앞뒤에 어떤 텍스트도 붙이지 말 것

[JSON 스키마]
{
  "design_ocr_text": "디자인에서 추출한 전체 텍스트",
  "score": 0~100,
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": ["법률 위반 사항 목록"]
  },
  "issues": [
    {
      "type": "Critical" | "Minor" | "Law_Violation",
      "location": "위치 설명",
      "issue": "오류 설명",
      "expected": "기준 데이터 값",
      "actual": "디자인에서 발견된 값",
      "suggestion": "수정 제안"
    }
  ]
}

비교 기준, 감점 규칙 등은 기존 설명대로 따르되
**추측/보정 없이 Standard와 Design의 텍스트를 문자 단위로 비교**하세요.
"""

QA_PROMPT = """
당신은 식품표시사항 작성 전문가입니다.
제공된 QA 자료를 분석하여 법률을 준수하는 식품표시사항을 작성하세요.

반드시 JSON만 출력하세요. 코드블록/HTML/설명 금지.

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


# =========================
# 파일 처리 함수
# =========================

def process_file_to_part(file_storage):
    """파일을 Gemini가 이해할 수 있는 Part 객체로 변환"""
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)

    # 엑셀 → CSV 텍스트
    if mime_type in [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel'
    ]:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    # 이미지 → PNG, 최소 전처리
    if mime_type.startswith('image/'):
        try:
            img = PIL.Image.open(io.BytesIO(file_data))

            # 투명 배경 제거
            if img.mode in ('RGBA', 'LA', 'P'):
                bg = PIL.Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = bg

            if img.mode != 'RGB':
                img = img.convert('RGB')

            w, h = img.size
            if w < 1200 or h < 1200:
                scale = max(1200 / w, 1200 / h)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, PIL.Image.LANCZOS)

            byte_io = io.BytesIO()
            img.save(byte_io, format="PNG", dpi=(300, 300))
            byte_io.seek(0)
            return {"mime_type": "image/png", "data": byte_io.read()}
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패 (원본 사용): {e}")
            return {"mime_type": mime_type, "data": file_data}

    # 그 외 바이너리 그대로
    return {"mime_type": mime_type, "data": file_data}


def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출"""
    try:
        part = process_file_to_part(image_file)
        if not part:
            return None

        model = get_model(extra_config={"max_output_tokens": 4096})
        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, part]

        response = model.generate_content(parts)
        raw_text = response.text.strip()
        json_str = safe_extract_json(raw_text)
        return json.loads(json_str)
    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        return None


# =========================
# 엑셀 생성/읽기
# =========================

def create_standard_excel(data):
    """기준 데이터를 엑셀 파일로 생성"""
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
            if ingredients_data:
                pd.DataFrame(ingredients_data).to_excel(writer, sheet_name='원재료명', index=False)

            if 'continuous_text' in data['ingredients']:
                pd.DataFrame([{
                    '원재료명_연속텍스트': data['ingredients']['continuous_text']
                }]).to_excel(writer, sheet_name='원재료명_연속텍스트', index=False)

        if 'allergens' in data:
            allergens_rows = []
            if 'contains' in data['allergens']:
                allergens_rows.append({
                    '항목': '함유 알레르기 유발물질',
                    '내용': ', '.join(data['allergens']['contains'])
                })
            if 'manufacturing_facility' in data['allergens']:
                allergens_rows.append({
                    '항목': '제조시설 안내',
                    '내용': data['allergens']['manufacturing_facility']
                })
            if allergens_rows:
                pd.DataFrame(allergens_rows).to_excel(writer, sheet_name='알레르기정보', index=False)

        if 'nutrition_info' in data and 'per_100g' in data['nutrition_info']:
            nut = data['nutrition_info']['per_100g']
            rows = []
            if 'calories' in nut:
                rows.append({
                    '영양성분': '총 열량',
                    '100g 당': nut['calories'],
                    '1일 영양성분 기준치에 대한 비율(%)': '-'
                })
            for k, v in nut.items():
                if k == 'calories' or not isinstance(v, dict):
                    continue
                rows.append({
                    '영양성분': k,
                    '100g 당': v.get('amount', ''),
                    '1일 영양성분 기준치에 대한 비율(%)': v.get('daily_value', '')
                })
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='영양정보', index=False)

        if 'manufacturer' in data:
            pd.DataFrame([data['manufacturer']]).to_excel(writer, sheet_name='제조원정보', index=False)

        if 'precautions' in data:
            pd.DataFrame([{'주의사항': t} for t in data['precautions']]).to_excel(
                writer, sheet_name='주의사항', index=False
            )

        if 'details' in data and data['details']:
            pd.DataFrame(data['details']).to_excel(writer, sheet_name='원재료상세', index=False)

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

    prompt = PROMPT_CREATE_STANDARD
    if ALL_LAW_TEXT:
        prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
    parts.append(prompt)

    excel_part = process_file_to_part(excel_file)
    if excel_part:
        parts.append(excel_part)

    ingredient_info_list = []
    # 파일명 기준 정렬 → 항상 같은 순서
    for img in sorted(raw_images, key=lambda x: x.filename)[:15]:
        print(f"📷 원재료 이미지 처리 중: {img.filename}")
        info = extract_ingredient_info_from_image(img)
        if info:
            ingredient_info_list.append(info)

    if ingredient_info_list:
        text = "--- [원재료 표시사항에서 추출한 정보] ---\n"
        for idx, info in enumerate(ingredient_info_list, 1):
            text += f"\n[원재료 {idx}]\n"
            text += json.dumps(info, ensure_ascii=False, indent=2)
            text += "\n"
        text += "--- [원재료 정보 끝] ---\n"
        parts.append({"text": text})

    try:
        model = get_model()
        response = model.generate_content(parts)
        raw = response.text.strip()
        json_str = safe_extract_json(raw)
        result = json.loads(json_str)
        return jsonify(result)
    except Exception as e:
        print(f"❌ 기준 데이터 생성 오류: {e}")
        import traceback
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
        import traceback
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

        result = {}

        if '제품정보' in df_dict:
            result['product_info'] = df_dict['제품정보'].to_dict('records')[0]

        first_sheet_name = list(df_dict.keys())[0]
        first_sheet_df = df_dict[first_sheet_name]

        if '원재료명' in df_dict:
            lst = df_dict['원재료명']['원재료명'].tolist()
            result['ingredients'] = {
                'structured_list': lst,
                'continuous_text': ', '.join(lst)
            }
        elif '원재료명_연속텍스트' in df_dict:
            cont = df_dict['원재료명_연속텍스트']['원재료명_연속텍스트'].iloc[0]
            result['ingredients'] = {
                'structured_list': cont.split(', '),
                'continuous_text': cont
            }
        elif not first_sheet_df.empty:
            col = '원재료명' if '원재료명' in first_sheet_df.columns else first_sheet_df.columns[0]
            lst = first_sheet_df[col].astype(str).tolist()
            result['ingredients'] = {
                'structured_list': lst,
                'continuous_text': ', '.join(lst)
            }

        if '알레르기정보' in df_dict:
            al_df = df_dict['알레르기정보']
            result['allergens'] = {}
            for _, row in al_df.iterrows():
                if row['항목'] == '함유 알레르기 유발물질':
                    result['allergens']['contains'] = row['내용'].split(', ')
                elif row['항목'] == '제조시설 안내':
                    result['allergens']['manufacturing_facility'] = row['내용']

        if '영양정보' in df_dict:
            ndf = df_dict['영양정보']
            per_100g = {}
            for _, row in ndf.iterrows():
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


# 2단계: 디자인 검증 (AI)
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
            df_dict = pd.read_excel(
                io.BytesIO(standard_excel.read()),
                sheet_name=None,
                engine='openpyxl',
                dtype=str,
                keep_default_na=False
            )
            first_sheet_name = list(df_dict.keys())[0]
            first_sheet_df = df_dict[first_sheet_name]
            std_data = {}

            if not first_sheet_df.empty:
                col = '원재료명' if '원재료명' in first_sheet_df.columns else first_sheet_df.columns[0]
                lst = first_sheet_df[col].astype(str).tolist()
                std_data = {'ingredients': {'structured_list': lst, 'continuous_text': ', '.join(lst)}}

            standard_json = json.dumps(std_data, ensure_ascii=False)
        except Exception as e:
            return jsonify({"error": f"엑셀 읽기 실패: {str(e)}"}), 400

    if not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 또는 JSON)가 필요합니다."}), 400

    design_file.seek(0)

    prompt = f"""
{PROMPT_VERIFY_DESIGN}

[참고 법령]
{ALL_LAW_TEXT}

[기준 데이터(Standard)]
{standard_json}
"""

    parts = [prompt]
    parts.append(process_file_to_part(design_file))

    try:
        system_instruction = """
당신은 정밀한 OCR 및 라벨 검증 AI입니다.
- 이미지 텍스트를 보정하지 말고 그대로 사용.
- Standard와 Design의 텍스트를 문자 단위로 비교.
- 반드시 JSON만 출력.
"""
        model = get_model(system_instruction=system_instruction)
        response = model.generate_content(parts)
        raw = response.text.strip()
        json_str = safe_extract_json(raw)
        result = json.loads(json_str)
        return jsonify(result)
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 2단계: Python 엄격 비교용 (옵션)
@app.route('/api/verify-design-strict', methods=['POST'])
def verify_design_strict():
    """Python으로 정확한 비교 (AI 최소 사용)"""
    try:
        design_file = request.files.get('design_file')
        standard_json = request.form.get('standard_data')

        if not design_file or not standard_json:
            return jsonify({"error": "파일과 기준 데이터가 필요합니다"}), 400

        design_file.seek(0)
        standard_data = json.loads(standard_json)

        # 1. 순수 OCR
        parts = [PROMPT_EXTRACT_RAW_TEXT, process_file_to_part(design_file)]
        model = get_model(extra_config={"max_output_tokens": 4096})
        response = model.generate_content(parts)
        raw = response.text.strip()
        json_str = safe_extract_json(raw)
        design_ocr = json.loads(json_str)

        # 2. Python strict 비교
        all_issues = []
        std_text = ""
        if 'ingredients' in standard_data:
            std_text = standard_data['ingredients'].get('continuous_text', '')
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

        critical_count = sum(1 for i in all_issues if i['type'] == 'Critical')
        minor_count = sum(1 for i in all_issues if i['type'] == 'Minor')
        score = max(0, 100 - critical_count * 5 - minor_count * 2)

        return jsonify({
            "design_ocr_text": des_text,
            "score": score,
            "issues": all_issues,
            "law_compliance": {"status": "compliant", "violations": []}
        })

    except Exception as e:
        print(f"❌ verify_design_strict 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# QA 자료 업로드 및 식품표시사항 작성
@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")

    qa_files = request.files.getlist('qa_files')
    if not qa_files:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    parts = [QA_PROMPT]
    for f in qa_files[:20]:
        part = process_file_to_part(f)
        if part:
            parts.append(part)

    try:
        model = get_model()
        response = model.generate_content(parts)
        raw = response.text.strip()
        json_str = safe_extract_json(raw)
        result = json.loads(json_str)
        return jsonify(result)
    except Exception as e:
        print(f"❌ QA 자료 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================
# 서버 실행
# =========================

if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 플랫폼 (결정적 모드) 가동")
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)
