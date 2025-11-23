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
# 👉 한 번에 모델에 넘길 법령 텍스트 최대 길이 (필요하면 숫자 조절)
MAX_LAW_CHARS = 30000

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
(중략, 그대로)
"""

# 2. 디자인 검증용 (정답지 vs 디자인PDF)
PROMPT_VERIFY_DESIGN = """
(중략, 그대로)
"""

# --- 파일 처리 함수들 ---
def process_file_to_part(file_storage):
    ...
    # (이 부분은 전부 기존 그대로, 생략)

def extract_ingredient_info_from_image(image_file):
    ...
    # (기존 그대로)

def create_standard_excel(data):
    ...
    # (기존 그대로)

# 🔴 하이라이트 HTML 생성 헬퍼 함수 (기존 그대로)
def make_highlighted_html(design_text: str, issues: list) -> str:
    ...
    # (기존 그대로)

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
    # ✅ 여기만 수정: 법령 전체가 아니라 앞부분만 사용
    enhanced_prompt = PROMPT_CREATE_STANDARD
    if ALL_LAW_TEXT:
        law_snippet = ALL_LAW_TEXT[:MAX_LAW_CHARS]
        enhanced_prompt += f"\n\n--- [참고 법령] ---\n{law_snippet}\n--- [법령 끝] ---\n"
    parts.append(enhanced_prompt)

    # 이하 create_standard 나머지 코드는 네가 올린 그대로
    ...
    return jsonify(result)

@app.route('/api/download-standard-excel', methods=['POST'])
def download_standard_excel():
    ...
    # (기존 그대로)

@app.route('/api/read-standard-excel', methods=['POST'])
def read_standard_excel():
    ...
    # (기존 그대로)

# 2단계: 검증하기 (엑셀 파일 또는 JSON + 디자인 이미지)
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작...")
    try:
        design_file = request.files.get('design_file')
        standard_excel = request.files.get('standard_excel')
        standard_json = request.form.get('standard_data')

        if not design_file:
            return jsonify({"error": "디자인 파일이 필요합니다. (design_file)"}), 400

        # 엑셀 → JSON 부분은 그대로
        if standard_excel:
            ...
            # (기존 그대로)

        # ✅ 3. 법령 텍스트: 파일 다시 읽지 말고, 미리 로드한 것 일부만 사용
        law_text = (ALL_LAW_TEXT or "")[:MAX_LAW_CHARS]

        # 4. 프롬프트 조합
        full_prompt = f"""

        {PROMPT_VERIFY_DESIGN}

        [참고 법령]

        {law_text}

        [기준 데이터(JSON)]

        {standard_json}

        """

        parts = [full_prompt]

        design_file.stream.seek(0)
        design_part = process_file_to_part(design_file)
        if design_part:
            parts.append(design_part)
        else:
            return jsonify({"error": "디자인 파일을 처리할 수 없습니다."}), 400

        # 이하 Gemini 호출/파싱 부분은 그대로
        ...
        return jsonify(result)

    except Exception as e:
        ...
        return jsonify({"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}), 500

# QA 자료 업로드 및 식품표시사항 작성 API
@app.route('/api/upload-qa', methods=['POST'])
def upload_qa():
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")
    qa_files = request.files.getlist('qa_files')
    if not qa_files or len(qa_files) == 0:
        return jsonify({"error": "QA 자료 파일이 필요합니다."}), 400

    parts = []

    qa_prompt = """
    (기존 프롬프트 내용 그대로)
    """

    # ✅ 여기도 일부만 사용
    if ALL_LAW_TEXT:
        law_snippet = ALL_LAW_TEXT[:MAX_LAW_CHARS]
        qa_prompt += f"\n\n--- [참고 법령] ---\n{law_snippet}\n--- [법령 끝] ---\n"

    parts.append(qa_prompt)

    # 이하 upload_qa 나머지 코드는 그대로
    ...
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동")
    print("   - 원부재료 표시사항 스마트 추출")
    print("   - 법률 검토 기능 통합")
    print("   - QA 자료 업로드 지원")
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
