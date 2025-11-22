# --- 1. '부품' 가져오기 (import) ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import os  # '금고'를 열기 위해 필요
import time  # (간단한 로그용)
import io  # 이미지를 메모리에서 다루기 위해 필요
import PIL.Image  # 이미지를 열어보기 위해 필요 (Pillow 라이브러리)

from dotenv import load_dotenv  # '금고'(.env)를 여는 도구
import google.generativeai as genai  # '진짜 OCR 기계' (Gemini)

# --- 2. '금고'(.env) 열어서 '비밀 키' 준비하기 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("서버: ✅ 'Google AI 비밀 키'를 금고에서 성공적으로 불러왔습니다.")
else:
    print("서버: ❌ 'Google AI 비밀 키'를 찾을 수 없습니다! .env 파일을 확인하세요.")

# --- 3. '주방' 설정하기 (Flask 앱 생성) ---
app = Flask(__name__)
CORS(app)  # '매장'과 '주방'이 자유롭게 통신하도록 허용


# --- 4. '진짜 OCR 기계' 호출 함수 (신규) ---
def get_ocr_text_from_image(image_file):
    """Google Gemini AI를 호출하여 이미지에서 텍스트를 추출합니다."""

    if not GOOGLE_API_KEY:
        return "오류: 서버에 Google AI API 키가 설정되지 않았습니다."

    try:
        # 1. 이미지를 '사진 뷰어(PIL)'로 엽니다.
        #    (Gemini가 알아볼 수 있는 형식으로 변환)
        img = PIL.Image.open(image_file.stream)

        # 2. '진짜 OCR 기계'(Gemini) 모델을 선택합니다.
        #    (gemini-2.5-flash-preview-09-2025는 이미지와 텍스트를 함께 잘 이해합니다)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-09-2025")

        # 3. 'OCR 기계'에게 명령을 내립니다.
        prompt = "이 이미지에 보이는 모든 텍스트를 순서대로 정확하게 추출해줘."

        # 4. 이미지와 명령을 함께 전송!
        response = model.generate_content([prompt, img])

        # 5. 'OCR 기계'가 보내준 '결과물(텍스트)'을 반환합니다.
        return response.text

    except Exception as e:
        print(f"서버: ❌ Google AI 호출 중 심각한 오류 발생: {e}")
        return f"오류: AI가 이미지를 분석하는 데 실패했습니다. (서버 오류: {e})"


# --- 5. '주문' 받는 창구 (/analyze) 업그레이드 ---
@app.route("/analyze", methods=["POST"])
def analyze_image():
    """ '매장'(홈페이지)에서 파일을 받아 '진짜 OCR'을 수행합니다. """

    print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. '매장'에서 파일(주문)이 왔는지 확인
    if 'file' not in request.files:
        print("서버: ❌ 'file'이 없는 잘못된 요청입니다.")
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']

    if file.filename == '':
        print("서버: ❌ 파일 이름이 없습니다.")
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    print(f"서버: ✅ '매장'으로부터 파일 '{file.filename}' (을)를 받았습니다.")
    print("서버: 🤖 '진짜 OCR 기계'(Google AI) 호출을 시작합니다...")

    # 2. (신규!) '진짜 OCR 기계'에게 이미지 전달하고 텍스트 받기
    ocr_result_text = get_ocr_text_from_image(file)

    print(f"서버: 🤖 '진짜 OCR' 완료! (추출된 글자 수: {len(ocr_result_text)}자)")

    # 3. (임시) 법령 분석 및 AI 2차 분석
    #    (이 부분은 3단계에서 '진짜' 법령 데이터와 연결하겠습니다.)
    ai_analysis_mock = [
        {"type": "info", "text": "AI 분석 (2단계 - 작업 예정)", "description": "추출된 OCR 텍스트를 기반으로 법령 비교 및 2차 AI 분석이 진행될 예정입니다.",
         "reference": "작업 대기 중"}
    ]

    # 4. '진짜' OCR 결과를 '매장'에 돌려주기
    response_data = {
        "status": "분석 완료 (진짜 OCR)",  # 상태 변경
        "typos": 0,  # (아직 가짜)
        "violations": 0,  # (아직 가짜)
        "ocrText": ocr_result_text,  # <<<< ✨ 여기가 '진짜' 텍스트로 바뀜! ✨
        "aiAnalysis": ai_analysis_mock
    }

    print("서버: ✅ '매장'에 '진짜 OCR' 결과를 성공적으로 전송했습니다.")
    return jsonify(response_data)


# --- 6. '주방' 문 열기 (서버 실행) ---
if __name__ == '__main__':
    print("-----------------------------------------------------")
    print("  삼진식품 원재료 법령 점검 시스템 - '주방' 서버")
    print("  (버전 2.0: '진짜 Google AI OCR' 탑재 완료)")
    print("  '매장'(html)의 주문을 기다리는 중...")
    print("-----------------------------------------------------")
    app.run(debug=True, port=5000)