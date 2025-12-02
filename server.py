import os
import io
import json
import glob
import traceback
import time
import base64
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import PIL.Image
import re
import html
import difflib  # 🔹 OCR 의심 판별용

from openai import OpenAI

# Optional OCR fallback libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False


# =======================
#  기본 설정
# =======================

load_dotenv()
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다!")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.1-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")


# =======================
#  OpenAI 유틸 함수
# =======================

def call_openai_response(model: str, input_data, *, response_format=None, max_retries: int = 3):
    """
    OpenAI Responses API 호출 + 간단 Retry
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

            resp = client.responses.create(**kwargs)
            return resp
        except Exception as e:
            last_err = e
            print(f"⚠️ OpenAI 호출 실패 {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise last_err


def extract_output_text_from_response(response) -> str:
    """
    OpenAI Responses API 응답에서 text 부분만 추출
    """
    try:
        output_items = getattr(response, "output", None)
        if output_items:
            texts = []
            for item in output_items:
                contents = getattr(item, "content", None) or []
                for c in contents:
                    if getattr(c, "type", None) == "output_text":
                        texts.append(getattr(c, "text", ""))
            if texts:
                return "\n".join(texts).strip()
    except Exception as e:
        print(f"⚠️ 응답 텍스트 추출 중 예외: {e}")

    if isinstance(response, dict):
        output_items = response.get("output", [])
        if output_items:
            contents = output_items[0].get("content", [])
            if contents and contents[0].get("type") == "output_text":
                return contents[0].get("text", "")

    return str(response)


def resize_image_bytes(image_bytes: bytes, max_size: int = 1500) -> tuple[bytes, str]:
    """
    메모리 절약 + OCR 성능 유지용 이미지 리사이즈
    - 긴 변이 max_size를 넘으면 비율 유지하며 리사이즈
    - JPEG(또는 원본 포맷)로 재저장 (quality=85로 가볍게)
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
    img.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return buf.read(), fmt


def combine_parts_to_prompt(parts) -> str:
    """
    parts 리스트를 하나의 텍스트 프롬프트로 합치기
    - 문자열: 그대로
    - {"text": "..."}: text 필드 사용
    """
    chunks = []
    for p in parts:
        if isinstance(p, str):
            chunks.append(p)
        elif isinstance(p, dict) and "text" in p:
            chunks.append(str(p["text"]))
    return "\n\n".join(chunks)


# =======================
#  법령 텍스트 로드
# =======================

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


# =======================
#  프롬프트들
# =======================

PROMPT_EXTRACT_INGREDIENT_INFO = """
이 이미지는 원부재료 표시사항 사진입니다. 
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

PROMPT_VERIFY_DESIGN = """
당신은 식품표시사항 감사관이자 법률 전문가입니다.
[기준 데이터(Standard)]와 [디자인 시안(Design)]을 비교하여 오류를 검출하세요.

[입력]
1. **Standard**: 앞서 생성된 완벽한 표시사항 정답지
2. **Design OCR 텍스트**: 실제 포장지 디자인 파일에서 OCR로 추출한 순수 텍스트
3. **법령**: 식품 표시 관련 법령

[검증 원칙 - 매우 중요! 반드시 준수하세요!]
1. **오탈자 검출 중심**: Standard와 Design을 문자 단위로 정확히 비교하여 실제 오탈자만 검출하세요.
2. **함량 정보(%) 추가는 허용**: Standard에 없어도 Design에 함량 정보(%)가 추가된 것은 절대 문제로 보지 않습니다.
   ✅ 허용 예시: Standard "당근(국내산)" → Design "당근(국내산) 4.1%" (문제 없음)
   ✅ 허용 예시: Standard "양파" → Design "양파 2.2%" (문제 없음)
3. **비정상 값만 문제**: 함량이 100%를 초과하거나 말도 안되는 값인 경우만 문제로 표시합니다.
   ❌ 문제 예시: "양파221%" (소수점 누락으로 221%가 되어 비정상) → "양파2.21%"로 수정 필요
   ❌ 문제 예시: "당근999%" (말도 안되는 값) → 문제로 표시
4. **라벨명 누락은 무시**: 내용이나 수치는 있지만 라벨명(예: "전면부 총열량", "제조시설안내")만 없는 경우는 문제로 보지 않습니다.
   ✅ 허용: 영양정보에 127Kcal 수치가 있으면 "전면부 총열량" 라벨이 없어도 문제 없음
   ✅ 허용: 제조시설 내용이 있으면 "제조시설안내" 라벨이 없어도 문제 없음
5. **실제 오류만 검출**:
   ✅ 원재료명 오탈자: "전분가공품" → "전반가공품" (글자 오기)
   ✅ 원재료명 오탈자: "D-소비톨" → "D-솔비톨" (글자 오기)
   ✅ 숫자 오탈자: "130kcal" → "127kcal" (숫자 오기)
   ✅ 단위 오탈자: "10g" → "10mg" (단위 오기)
   ✅ 구두점 오탈자: "우유, 쇠고기, 토마토" → "우유 쇠고기 토마토" (쉼표 누락)
   ✅ 소수점 누락: "2.21%" → "221%" (비정상 값)
   ✅ 원산지 오기: "국내산" → "수입산" (내용 오기)
   ✅ 순서 위반: 배합비 순서와 다름
   ✅ 법률 위반: 첨가물 유형 누락 (예: "소브산칼륨" → "소브산칼륨(보존료)" 필수)

[검증하지 말아야 할 것들 - 절대 문제로 표시하지 마세요!]
❌ Standard에 없는 함량 정보(%)가 Design에 추가된 경우
❌ 라벨명은 없지만 내용이나 수치가 있는 경우
❌ 공백이나 포맷팅 차이만 있는 경우 (예: "태국, 베트남산" vs "태국,베트남산")
❌ Standard와 Design이 의미상 동일하지만 표현만 다른 경우

[중요 규칙 - hallucination 방지]
- "expected" 값은 반드시 Standard JSON 텍스트에서 실제로 존재하는 문자열을 그대로 복사해서 사용해야 합니다.
- "actual" 값은 반드시 디자인 OCR 텍스트(design_ocr_text)에서 실제로 존재하는 문자열을 그대로 복사해서 사용해야 합니다.
- Standard나 디자인 OCR 텍스트에 존재하지 않는 숫자, 단위, 문구를 상상해서 만들면 안 됩니다.
- 존재하지 않는 500g, 존재하지 않는 오타 등을 상상으로 만들지 마세요.

[검증 항목]
1. **원재료명 오탈자**
2. **숫자/단위 오탈자**
3. **구두점 오탈자**
4. **원산지 오기**
5. **순서 위반**
6. **법률 위반**
7. **비정상 값**

[출력 양식 - JSON]
{
    "design_ocr_text": "디자인 파일에서 인식한 텍스트 (입력으로 받은 OCR 텍스트를 그대로 사용)",
    "score": 90,
    "law_compliance": {
        "status": "compliant" | "violation",
        "violations": ["법률 위반 사항 목록 - 법률 조항만 표시"]
    },
    "issues": [
        {
            "type": "Critical" | "Minor" | "Law_Violation",
            "location": "위치 설명",
            "issue": "오류 유형",
            "expected": "정답 내용 (Standard 기준, 반드시 Standard에서 실제 있는 텍스트)",
            "actual": "실제 내용 (Design OCR에서 실제 있는 텍스트)",
            "suggestion": "수정 제안",
            "law_reference": "관련 법령 조항 (법률 위반인 경우만)"
        }
    ]
}
"""


# =======================
#  텍스트/HTML 정리
# =======================

def clean_html_text(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r'style\s*=\s*["\'][^"\']*["\']', "", text, flags=re.IGNORECASE)
    text = re.sub(r'class\s*=\s*["\'][^"\']*["\']', "", text, flags=re.IGNORECASE)
    text = re.sub(r"font-weight\s*:\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"margin[^;]*;?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"padding[^;]*;?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"color[^;]*;?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"font-size[^;]*;?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_ai_response(data):
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ["violations", "issues"] and isinstance(value, list):
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


# =======================
#  OCR
# =======================

def ocr_image_bytes(image_bytes: bytes) -> str:
    """
    이미지에서 텍스트를 추출하는 OCR 함수
    1순위: OpenAI Vision
    2순위: pytesseract (설치되어 있는 경우)
    """
    # 1) OpenAI Vision 기반 OCR
    try:
        resized_bytes, fmt = resize_image_bytes(image_bytes, max_size=1600)
        mime_type = f"image/{fmt.lower()}"
        b64_image = base64.b64encode(resized_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        ocr_prompt = """
이 이미지는 식품 포장지/라벨 등의 사진입니다.
이미지 안에 보이는 모든 글자를 **그대로** 인식해서 적어 주세요.

[중요]
- 줄바꿈, 공백, 숫자, 기호를 최대한 원문 그대로 유지하세요.
- 의미를 요약하거나 설명하지 말고, 순수 텍스트만 출력하세요.
- 한국어는 한국어로, 영어/숫자는 있는 그대로 적어 주세요.
"""

        input_items = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ocr_prompt.strip()},
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = call_openai_response(VISION_MODEL, input_items)
        text = extract_output_text_from_response(resp).strip()

        # 코드블록 제거
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if text:
            print("✅ OpenAI Vision OCR 성공")
            return text
        else:
            print("⚠️ OpenAI Vision OCR 결과가 비어 있습니다.")
    except Exception as e:
        print("❌ OpenAI Vision OCR 실패:", e)

    # 2) pytesseract 폴백
    if TESSERACT_AVAILABLE:
        try:
            img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text = pytesseract.image_to_string(img, lang="kor+eng")
            text = text.strip()
            if text:
                print("✅ pytesseract OCR 성공 (폴백)")
            else:
                print("⚠️ pytesseract OCR 결과가 비어 있습니다.")
            return text
        except Exception as e:
            print("pytesseract OCR 실패:", e)

    print("⚠️ OCR 결과를 얻지 못했습니다.")
    return ""


# =======================
#  파일 처리
# =======================

def process_file_to_part(file_storage):
    """
    파일을 모델에 줄 수 있는 형태로 변환
    - Excel: CSV 텍스트
    - 이미지: bytes (OCR용)
    - 기타: 간단 설명 텍스트
    """
    mime_type = file_storage.mimetype
    file_data = file_storage.read()
    file_storage.seek(0)

    if mime_type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]:
        try:
            df = pd.read_excel(io.BytesIO(file_data))
            csv_text = df.to_csv(index=False)
            return {"text": f"--- [Excel 배합비 데이터] ---\n{csv_text}"}
        except Exception as e:
            print(f"엑셀 변환 실패: {e}")
            return None

    if mime_type.startswith("image/"):
        return {"mime_type": mime_type, "data": file_data}

    return {
        "text": f"[파일] 이름: {file_storage.filename}, MIME: {mime_type}, 크기: {len(file_data)} bytes"
    }


def extract_ingredient_info_from_image(image_file):
    """
    원재료 표시사항 이미지에서 필요한 정보만 추출
    1순위: OpenAI Vision + JSON
    2순위: 단순 OCR 텍스트
    """
    try:
        image_data = image_file.read()
        image_file.seek(0)

        resized_bytes, fmt = resize_image_bytes(image_data, max_size=1500)
        mime_type = image_file.mimetype or f"image/{fmt.lower()}"
        b64_image = base64.b64encode(resized_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": PROMPT_EXTRACT_INGREDIENT_INFO.strip(),
                    },
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }
        ]

        resp = call_openai_response(
            VISION_MODEL,
            input_items,
            response_format={"type": "json_object"},
        )

        result_text = extract_output_text_from_response(resp).strip()
        print("---- extract_ingredient_info_from_image 응답 ----")
        print(result_text[:1000])

        if not result_text:
            ocr_text = ocr_image_bytes(image_data)
            if ocr_text:
                return {"ocr_fallback_text": ocr_text}
            return None

        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            blocks = result_text.split("```")
            if len(blocks) > 1:
                result_text = blocks[1].strip()

        return json.loads(result_text)
    except json.JSONDecodeError as e:
        print("원재료 JSON 파싱 실패:", e)
        print("응답 일부:", result_text[:500])
        return None
    except Exception as e:
        print("원재료 정보 추출 실패:", e)
        return None


# =======================
#  하이라이트 HTML / 기타 유틸
# =======================

def simple_generate_highlight_html(ocr_text: str, standard_ingredients: list[str]) -> str:
    lines = ocr_text.splitlines()
    std_lower = [s.lower() for s in standard_ingredients]
    html_lines = []
    for line in lines:
        lowered = line.lower()
        line_html = html.escape(line)
        matched = False
        for idx, std in enumerate(std_lower):
            if std in lowered:
                matched = True
                line_html = line_html.replace(
                    html.escape(standard_ingredients[idx]),
                    f"<span style='background:#e6f4ea;padding:2px 4px;border-radius:4px;'>{html.escape(standard_ingredients[idx])}</span>",
                )
        if not matched:
            line_html = (
                f"<span style='color:#ad2e2e; font-weight:600;'>{line_html}</span>"
            )
        html_lines.append(
            f"<div style='margin-bottom:6px; font-family:monospace; white-space:pre-wrap;'>{line_html}</div>"
        )
    result_html = (
        "<div style='padding:10px; background:#fff; border-radius:8px;'>"
        + "".join(html_lines)
        + "</div>"
    )
    return result_html


def extract_text_from_design_part(design_part):
    try:
        from PIL import Image
        pil_type = Image.Image
    except Exception:
        pil_type = None
    if pil_type and isinstance(design_part, pil_type):
        bio = BytesIO()
        design_part.save(bio, format="PNG")
        bio.seek(0)
        img_bytes = bio.read()
        return ocr_image_bytes(img_bytes)
    if isinstance(design_part, dict) and "data" in design_part:
        img_bytes = design_part["data"]
        return ocr_image_bytes(img_bytes)
    return ""


def filter_issues_by_text_evidence(result, standard_json: str, ocr_text: str):
    """
    LLM hallucination 방지 필터:
    - expected가 Standard에 실제 존재하는지
    - actual이 OCR 텍스트에 실제 존재하는지
    확인 후, 둘 중 하나라도 없으면 이슈에서 제거
    """
    if not isinstance(result, dict):
        return result

    # Standard 텍스트 펼치기
    try:
        std_obj = json.loads(standard_json) if standard_json else {}
        std_text = json.dumps(std_obj, ensure_ascii=False)
    except Exception:
        std_text = standard_json or ""

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        return result

    filtered = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        expected = str(issue.get("expected", "") or "")
        actual = str(issue.get("actual", "") or "")

        ok_expected = (expected == "") or (expected in std_text)
        ok_actual = (actual == "") or (actual in ocr_text)

        if ok_expected and ok_actual:
            filtered.append(issue)
        else:
            print("🚫 hallucination 의심 이슈 제거:", {"expected": expected, "actual": actual})

    result["issues"] = filtered
    return result


def mark_possible_ocr_error_issues(result, max_edit_distance: int = 2):
    """
    expected / actual 간 문자 차이가 너무 작으면
    -> 'OCR 오류 가능성' 플래그를 달고, 심각도를 한 단계 낮춘다.

    max_edit_distance: 허용할 최대 편집 거리 (1~2 정도 추천)
    """
    if not isinstance(result, dict):
        return result

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        return result

    def approx_distance(a: str, b: str) -> int:
        """Levenshtein 대신 SequenceMatcher로 근사 거리 계산"""
        if not a or not b:
            return 999
        s = difflib.SequenceMatcher(None, a, b)
        return int(round((1.0 - s.ratio()) * max(len(a), len(b))))

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        expected = str(issue.get("expected", "") or "").strip()
        actual = str(issue.get("actual", "") or "").strip()

        if not expected or not actual:
            continue

        dist = approx_distance(expected, actual)
        min_len = min(len(expected), len(actual))

        # 글자 길이가 너무 짧으면 노이즈라서 제외, 최소 3자 이상만 판단
        if min_len >= 3 and dist <= max_edit_distance:
            # OCR 오류 가능성 높음
            flags = issue.setdefault("flags", [])
            if "possible_ocr_error" not in flags:
                flags.append("possible_ocr_error")

            # 심각도 조정: Law_Violation → Minor
            old_type = issue.get("type", "")
            if old_type == "Law_Violation":
                issue["type"] = "Minor"

            # 설명에 한 줄 추가
            desc = issue.get("issue", "")
            if "OCR 오류 가능성" not in desc:
                issue["issue"] = (desc + " (OCR 오류 가능성 있음)").strip()

            print("🟡 OCR 의심 이슈:", {
                "expected": expected,
                "actual": actual,
                "distance": dist
            })

    return result


# =======================
#  라우트
# =======================

@app.route("/")
def index():
    return render_template("index.html")


# ---- 디자인 검증 ----

@app.route("/api/verify-design", methods=["POST"])
def verify_design():
    print("🕵️‍♂️ 2단계: 디자인 검증 시작 (OCR + hallucination 필터)...")
    design_file = request.files.get("design_file")
    standard_excel = request.files.get("standard_excel")
    standard_json = request.form.get("standard_data")

    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400
    if not standard_excel and not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 파일 또는 JSON)가 필요합니다."}), 400

    # 1) 기준 데이터 Excel → JSON 변환 (옵션)
    if standard_excel:
        try:
            df_dict = pd.read_excel(
                io.BytesIO(standard_excel.read()),
                sheet_name=None,
                engine="openpyxl",
            )
            if not df_dict:
                return jsonify({"error": "엑셀 파일이 비어있습니다."}), 400
            first_sheet_name = list(df_dict.keys())[0]
            first_sheet_df = df_dict[first_sheet_name]
            if not first_sheet_df.empty:
                first_column = first_sheet_df.columns[0]
                if "원재료명" in first_sheet_df.columns:
                    ingredients_list = (
                        first_sheet_df["원재료명"].dropna().astype(str).tolist()
                    )
                else:
                    ingredients_list = (
                        first_sheet_df[first_column].dropna().astype(str).tolist()
                    )
                standard_data = {
                    "ingredients": {
                        "structured_list": ingredients_list,
                        "continuous_text": ", ".join(ingredients_list),
                    }
                }
                standard_json = json.dumps(standard_data, ensure_ascii=False)
            else:
                return jsonify({"error": "엑셀의 첫 시트가 비어있습니다."}), 400
        except Exception as e:
            print(f"❌ 엑셀 파일 읽기 오류: {e}")
            traceback.print_exc()
            return jsonify({"error": f"엑셀 파일 읽기 실패: {str(e)}"}), 400

    # 2) 디자인 파일 OCR 수행
    design_part = process_file_to_part(design_file)
    ocr_text = ""
    try:
        ocr_text = extract_text_from_design_part(design_part)
        if not ocr_text:
            raw_bytes = design_file.read()
            design_file.seek(0)
            ocr_text = ocr_image_bytes(raw_bytes)
    except Exception as e:
        print("디자인 OCR 실패:", e)
        traceback.print_exc()
        ocr_text = ""

    print("===== DESIGN OCR TEXT (first 1000 chars) =====")
    print((ocr_text or "")[:1000])
    print("==============================================")

    # 3) ChatGPT에 검증 요청 (텍스트 기반)
    parts = []
    enhanced_prompt = PROMPT_VERIFY_DESIGN
    if ALL_LAW_TEXT:
        enhanced_prompt += (
            f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
        )
    parts.append(enhanced_prompt)
    parts.append(f"\n--- [기준 데이터(Standard)] ---\n{standard_json}\n")
    parts.append(
        f"\n--- [디자인 OCR 텍스트] ---\n{ocr_text}\n--- [디자인 OCR 텍스트 끝] ---\n"
    )

    prompt_text = combine_parts_to_prompt(parts)

    result_text = ""
    result = None

    try:
        resp = call_openai_response(
            TEXT_MODEL,
            prompt_text,
            response_format={"type": "json_object"},
        )
        result_text = extract_output_text_from_response(resp).strip()
        print("---- 모델 응답(원문) 시작 ----")
        print(result_text[:4000])
        print("---- 모델 응답(원문) 끝 ----")
    except Exception as e:
        print("모델 호출 실패:", e)
        traceback.print_exc()
        result_text = ""

    if result_text:
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result_text = result_text.strip()
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print("JSON 파싱 오류:", json_err)
            print("응답 텍스트(일부):", result_text[:1000])
            try:
                fixed = (
                    result_text.replace(",\n}", "\n}")
                    .replace(",\n]", "\n]")
                    .replace(", }", " }")
                    .replace(", ]", " ]")
                )
                result = json.loads(fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception as e:
                print("최종 JSON 파싱 실패:", e)
                result = None

    # 4) 모델이 하이라이트 안 주면 서버에서 생성
    highlight_html = None
    if result and isinstance(result, dict):
        highlight_html = result.get("design_ocr_highlighted_html") or None

    if not highlight_html:
        print("모델에서 하이라이트를 제공하지 않음 -> 서버 폴백 하이라이트 생성")
        try:
            std_ingredients = []
            try:
                std_obj = json.loads(standard_json)
                std_ingredients = std_obj.get("ingredients", {}).get(
                    "structured_list", []
                )
            except Exception:
                std_ingredients = []
            highlight_html = simple_generate_highlight_html(ocr_text or "", std_ingredients)
            if not result:
                result = {}
            result["design_ocr_highlighted_html"] = highlight_html
            result.setdefault("design_ocr_text", ocr_text)
        except Exception as e:
            print("폴백 OCR 처리 실패:", e)
            traceback.print_exc()
            if not result:
                result = {}
            result[
                "design_ocr_highlighted_html"
            ] = "<div>서버 폴백 OCR 처리 중 오류가 발생했습니다.</div>"
            result["design_ocr_text"] = ocr_text or ""

    if not result:
        result = {
            "design_ocr_text": ocr_text,
            "score": 0,
            "law_compliance": {"status": "needs_review", "violations": []},
            "issues": [],
            "design_ocr_highlighted_html": "<div>모델과 폴백 모두에서 OCR 결과를 얻지 못했습니다.</div>",
        }

    # 5) hallucination 필터 적용 (expected/actual이 실제 텍스트에 있는지 검증)
    result = filter_issues_by_text_evidence(result, standard_json or "", ocr_text or "")

    # 6) OCR 의심 이슈 표시 (expected/actual 차이가 매우 작은 경우)
    result = mark_possible_ocr_error_issues(result, max_edit_distance=2)

    # 7) HTML 태그 정리
    result = clean_ai_response(result)

    return jsonify(result)


# ---- QA 자료 업로드 & 표시사항 작성 ----

@app.route("/api/upload-qa", methods=["POST"])
def upload_qa():
    print("📋 QA 자료 업로드 및 식품표시사항 작성 시작...")
    qa_files = request.files.getlist("qa_files")
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
        qa_prompt += (
            f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"
        )
    parts.append(qa_prompt)

    for qa_file in qa_files[:20]:
        file_part = process_file_to_part(qa_file)
        if not file_part:
            continue
        if isinstance(file_part, dict) and "text" in file_part:
            parts.append(file_part["text"])
        else:
            parts.append(str(file_part))

    print(f"📂 QA 자료 처리 중: {len(qa_files)}개 파일")

    try:
        prompt_text = combine_parts_to_prompt(parts)
        resp = call_openai_response(
            TEXT_MODEL,
            prompt_text,
            response_format={"type": "json_object"},
        )

        result_text = extract_output_text_from_response(resp).strip()

        print("---- QA 모델 응답(원문) 시작 ----")
        print(result_text[:4000])
        print("---- QA 모델 응답(원문) 끝 ----")

        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                result_text = "\n".join(lines[1:])
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        result_text = result_text.strip()

        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON 파싱 오류: {json_err}")
            print(f"응답 텍스트 (처음 2000자): {result_text[:2000]}")
            try:
                result_text_fixed = (
                    result_text.replace(",\n}", "\n}")
                    .replace(",\n]", "\n]")
                    .replace(", }", " }")
                    .replace(", ]", " ]")
                )
                result = json.loads(result_text_fixed)
                print("✅ JSON 수정 후 파싱 성공")
            except Exception:
                return jsonify(
                    {
                        "error": f"JSON 파싱 실패: {str(json_err)}. 응답의 일부: {result_text[:200]}..."
                    }
                ), 500

        result = clean_ai_response(result)
        return jsonify(result)

    except Exception as e:
        print("❌ QA 자료 처리 오류:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =======================
#  메인
# =======================

if __name__ == "__main__":
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 (OpenAI 버전) 가동")
    from waitress import serve

    serve(
        app,
        host="0.0.0.0",
        port=8080,
        threads=4,
        channel_timeout=600,
    )
