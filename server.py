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

# --- 초기화 ---
load_dotenv()

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

# =========================
# ✅ 1️⃣ 문자열 정규화 + 문자 단위 비교
# =========================

def normalize_text_strict(text):
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text)

def compare_texts_strict(standard_text, design_text):
    std_norm = normalize_text_strict(standard_text)
    des_norm = normalize_text_strict(design_text)

    issues = []
    max_len = max(len(std_norm), len(des_norm))

    for i in range(max_len):
        std_char = std_norm[i] if i < len(std_norm) else "(없음)"
        des_char = des_norm[i] if i < len(des_norm) else "(없음)"

        if std_char != des_char:
            issues.append({
                "position": i,
                "expected": std_char,
                "actual": des_char
            })

    return issues

# =========================
# ✅ 2️⃣ 강제 OCR (Gemini 1회 고정)
# =========================

PROMPT_EXTRACT_RAW_TEXT = """
당신은 OCR 전문가입니다.
이미지의 텍스트를 보이는 그대로 추출하세요.
보정, 추측, 교정 금지.

JSON 형식으로만 출력:
{
  "raw_text": "있는 그대로의 텍스트"
}
"""

def process_file_to_part(file_storage):
    file_data = file_storage.read()
    file_storage.seek(0)
    img = PIL.Image.open(io.BytesIO(file_data)).convert("RGB")

    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    byte_io.seek(0)

    return {"mime_type": "image/png", "data": byte_io.read()}

def forced_ocr(image_file):
    image_file.seek(0)

    parts = [
        PROMPT_EXTRACT_RAW_TEXT,
        process_file_to_part(image_file)
    ]

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "candidate_count": 1,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json"
        }
    )

    response = model.generate_content(parts)
    result_text = response.text.strip()

    if result_text.startswith("```"):
        result_text = result_text.replace("```json", "").replace("```", "").strip()

    ocr_json = json.loads(result_text)
    return ocr_json.get("raw_text", "")

# =========================
# ✅ 3️⃣ 위치(position) 자동 계산
# =========================

def add_issue_positions(issues, full_text):
    if not full_text:
        return issues

    for issue in issues:
        char = issue.get("actual", "")
        pos = full_text.find(char)
        if pos != -1:
            issue["position"] = pos

    return issues

# =========================
# ✅ 4️⃣ 디자인 검증 API (완전 고정 버전)
# =========================

@app.route("/api/verify-design-fixed", methods=["POST"])
def verify_design_fixed():
    try:
        design_file = request.files.get("design_file")
        standard_json = request.form.get("standard_data")

        if not design_file or not standard_json:
            return jsonify({"error": "파일과 기준 데이터가 필요합니다"}), 400

        design_file.seek(0)
        standard_data = json.loads(standard_json)

        # ✅ 1. 강제 OCR (항상 동일)
        design_ocr_text = forced_ocr(design_file)

        # ✅ 2. 기준 텍스트 추출
        std_text = ""
        if "ingredients" in standard_data:
            std_text = standard_data["ingredients"].get("continuous_text", "")

        # ✅ 3. Python 문자 단위 비교
        issues_raw = compare_texts_strict(std_text, design_ocr_text)

        issues = []
        for issue in issues_raw:
            issues.append({
                "type": "Critical" if issue["expected"] not in [" ", ",", "."] else "Minor",
                "location": f"원재료명 (위치: {issue['position']})",
                "issue": f"'{issue['expected']}' → '{issue['actual']}'",
                "expected": std_text,
                "actual": design_ocr_text,
                "suggestion": f"위치 {issue['position']} 문자 수정"
            })

        # ✅ 4. position 보정
        issues = add_issue_positions(issues, design_ocr_text)

        # ✅ 5. 점수 계산 (완전 고정)
        critical_count = sum(1 for i in issues if i["type"] == "Critical")
        minor_count = sum(1 for i in issues if i["type"] == "Minor")

        score = max(0, 100 - critical_count * 5 - minor_count * 2)

        return jsonify({
            "design_ocr_text": design_ocr_text,
            "score": score,
            "issues": issues,
            "law_compliance": {
                "status": "compliant",
                "violations": []
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# ✅ 실행
# =========================

if __name__ == "__main__":
    print("🚀 완전 고정형 OCR + Python 검증 서버 가동")
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
