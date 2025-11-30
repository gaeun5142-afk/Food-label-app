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

# --- 설정 및 초기화 ---
load_dotenv()
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
CORS(app)

# API 키 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("🚨 경고: .env 파일에 GOOGLE_API_KEY가 없습니다!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 기본값
MODEL_NAME = 'gemini-1.5-flash'

# 결과를 최대한 고정시키기 위한 생성 설정 (결정론적에 가깝게)
GENERATION_CONFIG = {
    "temperature": 0.0,   # 랜덤성 최소화
    "top_p": 1.0,
    "top_k": 1,
    "max_output_tokens": 4096,
}

MODEL = None  # 전역 모델 핸들


# 모델 사용 가능 여부 확인 함수
def check_available_models():
    """사용 가능한 모델 목록을 확인하고 적절한 모델을 반환합니다."""
    global MODEL_NAME, MODEL
    try:
        models = genai.list_models()
        available_models = []
        print("\n📋 사용 가능한 모델 목록:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
                print(f"   - {model_name}")
        
        for model in available_models:
            if 'flash' in model.lower():
                MODEL_NAME = model
                break
        else:
            for model in available_models:
                if 'pro' in model.lower():
                    MODEL_NAME = model
                    break
            else:
                if available_models:
                    MODEL_NAME = available_models[0]

        print(f"\n✅ 선택된 모델: {MODEL_NAME}\n")
    except Exception as e:
        print(f"⚠️ 모델 목록 확인 실패: {e}")
        print(f"⚠️ 기본 모델 사용: {MODEL_NAME}\n")

    # 전역 MODEL 인스턴스 생성 (한 번만)
    try:
        MODEL = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=GENERATION_CONFIG,
        )
        print("✅ GenerativeModel 인스턴스 생성 완료")
    except Exception as e:
        print(f"❌ 모델 인스턴스 생성 실패: {e}")


# 서버 시작 시 모델 확인 및 자동 설정
if GOOGLE_API_KEY:
    check_available_models()
else:
    print(f"⚠️ API 키가 없어 모델 확인을 건너뜁니다. 기본 모델 사용: {MODEL_NAME}\n")


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

PROMPT_CREATE_STANDARD = """(중략)"""  # 질문에 있던 긴 프롬프트 그대로 사용
PROMPT_VERIFY_DESIGN = """(중략)"""   # 질문에 있던 긴 프롬프트 그대로 사용
# 실제 코드에서는 위 두 프롬프트 부분을 질문에 작성하신 전체 텍스트로 넣어 주세요.


# --- 유틸 함수들 ---

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


def process_file_to_part(file_storage):
    """파일을 Gemini가 이해할 수 있는 Part 객체로 변환"""
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)

    if mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     'application/vnd.ms-excel']:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    if mime_type.startswith('image/'):
        try:
            img = PIL.Image.open(io.BytesIO(file_data))
            max_size = 1500
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                print(f"📉 이미지 리사이징: {new_size}")
            byte_io = io.BytesIO()
            fmt = img.format if img.format else 'JPEG'
            img.save(byte_io, format=fmt, quality=95)
            byte_io.seek(0)
            return {"mime_type": mime_type, "data": byte_io.read()}
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패 (원본 사용): {e}")
            return {"mime_type": mime_type, "data": file_data}

    return {"mime_type": mime_type, "data": file_data}


def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출 (결정론 설정 사용)"""
    if MODEL is None:
        print("❌ MODEL 이 초기화되지 않았습니다.")
        return None
    try:
        image_data = image_file.read()
        image_file.seek(0)
        
        img_pil = PIL.Image.open(io.BytesIO(image_data))
        
        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        response = MODEL.generate_content(parts)

        result_text = response.text.strip()
        if result_text.startswith("```
            result_text = result_text[7:-3]
        elif result_text.startswith("```"):
            result_text = result_text.split("```
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


@app.route('/api/create-standard', methods=['POST'])
def create_standard():
    print("⚙️ 1단계: 기준 데이터 생성 시작...")

    if MODEL is None:
        return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500

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
    # 속도 문제를 줄이기 위해 이미지 수를 10장으로 더 줄임
    for img in raw_images[:10]:
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
        response = MODEL.generate_content(parts)

        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```
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


@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")

    if MODEL is None:
        return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500

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

    design_part = process_file_to_part(design_file)
    if design_part:
        parts.append(design_part)

    try:
        response = MODEL.generate_content(parts)

        result_text = response.text.strip()
        if result_text.startswith("```
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```
            lines = result_text.split("\n")
            if lines.startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```
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


@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")

    if MODEL is None:
        return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500
    
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
        response = MODEL.generate_content(parts)
        
        result_text = response.text.strip()
        
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines[0].startswith("```
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
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동")
    print("   - 원부재료 표시사항 스마트 추출")
    print("   - 법률 검토 기능 통합")
    print("   - QA 자료 업로드 지원")
    from waitress import serve

    serve(
        app, 
        host='0.0.0.0', 
        port=8080,
        threads=4,
        channel_timeout=600  # 600초(10분)
    )

