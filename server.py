import os
import json
import io
import glob
import pandas as pd
import PIL.Image
import google.generativeai as genai

# ------------------------------
# 🔵 환경 설정
# ------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-1.5-flash"


# ------------------------------
# 🔵 법령 로드 함수
# ------------------------------

def load_law_texts():
    law_files = glob.glob("law_text_*.txt")
    combined = ""

    for f in law_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                combined += f"\n--- [{f}] ---\n"
                combined += file.read()
        except:
            pass

    return combined


ALL_LAW_TEXT = load_law_texts()


# ------------------------------
# 🔵 파일을 Gemini가 읽을 수 있게 변환
# ------------------------------

def process_file_to_part(file):
    mime = file.type
    data = file.read()
    file.seek(0)

    if mime in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]:
        df = pd.read_excel(io.BytesIO(data))
        csv_text = df.to_csv(index=False)
        return {"text": f"[EXCEL DATA]\n{csv_text}"}

    return {"mime_type": mime, "data": data}


# ------------------------------
# 🔵 원재료 사진에서 정보 추출
# ------------------------------

def extract_ingredient_info(image_file, prompt):
    data = image_file.read()
    image_file.seek(0)
    img = PIL.Image.open(io.BytesIO(data))

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([prompt, img])

    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:-3]

    return json.loads(text)


# ------------------------------
# 🔵 기준 데이터 생성 (엑셀 + 이미지들)
# ------------------------------

def create_standard(excel_file, images, prompt, law_text):
    parts = [prompt + "\n\n" + law_text]

    excel_part = process_file_to_part(excel_file)
    if excel_part:
        parts.append(excel_part)

    # 이미지 추출 결과 리스트
    extracted_info = []
    for img in images:
        try:
            info = extract_ingredient_info(img, prompt)
            extracted_info.append(info)
        except:
            pass

    if extracted_info:
        text = "[원재료 표시사항 추출 결과]\n"
        for i, info in enumerate(extracted_info, 1):
            text += f"\n# Raw Ingredient {i}\n"
            text += json.dumps(info, ensure_ascii=False, indent=2)
        parts.append({"text": text})

    model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0})
    response = model.generate_content(parts)

    result_text = response.text.strip()
    if result_text.startswith("```json"):
        result_text = result_text[7:-3]

    return json.loads(result_text)


# ------------------------------
# 🔵 디자인 검증 기능
# ------------------------------

def verify_design(design_file, standard_data, prompt, law_text):
    design_part = process_file_to_part(design_file)

    parts = [
        prompt,
        "\n\n---[LAW]---\n" + law_text,
        "\n\n---[STANDARD]---\n" + json.dumps(standard_data, ensure_ascii=False, indent=2),
        design_part,
    ]

    model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0})
    response = model.generate_content(parts)

    text = response.text.strip()

    # JSON만 추출
    import re
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    return {"error": "JSON not detected", "raw": text}
