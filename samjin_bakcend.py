# --- 1. '부품' 가져오기 (import) ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import io
import base64
import PIL.Image

from dotenv import load_dotenv

# === OpenAI (ChatGPT) ===
from openai import OpenAI


# --- 2. ChatGPT API 키 준비 ---
load_dotenv()
CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')

if CHATGPT_API_KEY:
    client = OpenAI(api_key=CHATGPT_API_KEY)
    print("서버: ✅ 'OpenAI API Key' 로딩 완료 (GPT Vision OCR 가능)")
else:
    raise RuntimeError("❌ CHATGPT_API_KEY 누락! .env 파일에 설정하세요.")


# --- 3. Flask 서버 생성 ---
app = Flask(__name__)
CORS(app)


# --- 4. OpenAI Vision OCR 함수 ---
def get_ocr_text_from_image(image_file):
    """
    OpenAI GPT-Vision 모델을 사용하여 이미지 속 텍스트(OCR)를 추출합니다.
    정확도 높음! Gemini 코드 완전 제거됨!
    """

    try:
        # 파일을 PIL 이미지로 변환
        img = PIL.Image.open(image_file.stream)

        # 이미지 → base64 인코딩
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Vision OCR 요청 메시지
        prompt = (
            "이 이미지에 포함된 모든 텍스트를 가능한 한 정확하게 OCR 해줘.\n"
            "- 줄바꿈 유지\n"
            "- 글씨가 흐리거나 겹쳐도 최대한 복원\n"
            "- 괄호, %, 숫자, 기호 그대로 보존"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",   # 더 정확하게 하고 싶으면 gpt-4o 로 변경
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content.strip()
        return text

    except Exception as e:
        print(f"❌ Vision OCR 실패: {e}")
        return f"오류: Vision OCR 중 문제가 발생했습니다. ({e})"


# --- 5. /analyze 라우트 ---
@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    HTML 페이지에서 파일을 받아
    OpenAI Vision OCR 결과를 JSON 형태로 반환합니다.
    """

    print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 파일 체크
    if "file" not in request.files:
        return jsonify({"error": "file 필드가 비어 있습니다."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    print(f"서버: 파일 '{file.filename}' 수신 완료")
    print("서버: 🤖 OpenAI Vision OCR 시작...")

    # OCR 실행
    ocr_text = get_ocr_text_from_image(file)

    print(f"서버: OCR 완료! (텍스트 길이: {len(ocr_text)}글자)")

    # 결과 JSON 반환
    result = {
        "status": "OCR 완료 (OpenAI Vision)",
        "typos": 0,             # 이후 백엔드 검증 기능과 연동 가능
        "violations": 0,
        "ocrText": ocr_text,
        "aiAnalysis": [
            {
                "type": "info",
                "text": "이미지 텍스트 추출 완료 (AI Vision OCR)",
            }
        ],
    }

    print("서버: 분석 결과 전송 완료 ✔️")
    return jsonify(result)


# --- 6. 서버 실행 ---
if __name__ == "__main__":
    print("-----------------------------------------------------")
    print(" 삼진식품 원재료 법령 점검 플랫폼 - OCR 서버 (OpenAI 전용 Ver.)")
    print(" Gemini 코드 완전 제거 완료 ✓")
    print(" 이미지 OCR은 gpt-4o-mini Vision 모델 기반")
    print("-----------------------------------------------------")
    app.run(debug=True, port=5000)
