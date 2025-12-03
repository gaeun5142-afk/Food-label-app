import os
import json
import io
import glob
import traceback
import base64
import difflib

import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
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

# ✅ OpenAI API 설정 (무조건 ChatGPT만 사용)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("🚨 경고: .env 파일에 OPENAI_API_KEY가 없습니다!")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ChatGPT 멀티모달 모델
MODEL_NAME = "gpt-4.1-mini"   # 텍스트+이미지 모두 지원


# --- 공통 OpenAI 호출 헬퍼 ---

def to_image_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """이미지 바이너리를 data URL(base64)로 변환"""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_openai_from_parts(parts, json_mode: bool = True) -> str:
    """
    OpenAI Responses API 호출.
    - parts: 문자열, PIL.Image.Image 섞여 있는 리스트
    - json_mode: True면 "JSON만 출력"이라고 시스템 지시를 앞에 붙임
    - 반환값: ChatGPT가 반환한 텍스트 전체 (string)
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    content = []

    if json_mode:
        # JSON 강제 지시
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
            # dict 등 기타 타입은 필요시 확장
            pass

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        temperature=0.0,
        max_output_tokens=32768,
    )

    # text 결과만 모으기 (Responses API output 구조 기준)
    result_chunks = []
    for out in getattr(resp, "output", []):
        for c in getattr(out, "content", []):
            if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                result_chunks.append(c.text)
    result_text = "".join(result_chunks).strip()
    return result_text


# --- 법령 텍스트 로드 ---
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

PROMPT_CREATE_STANDARD = """
당신은 식품 규정 및 표시사항 전문가입니다.
제공된 [배합비 데이터(Excel)]와 [원재료 표시사항에서 추출된 정보]를 종합하여,
법적으로 완벽한 '식품표시사항 기준 데이터(Standard)'를 실제 라벨 형식으로 생성하세요.

[출력 양식 - JSON만 출력]
{
  "product_info": {
    "product_name": "제품명",
    "food_type": "식품의 유형",
    "net_weight": "내용량",
    "expiration_date": "소비기한",
    "storage_method": "보관방법",
    "packaging_material": "포장재질",
    "item_report_number": "품목보고번호",
    "front_calories": "전면부 총열량/문구"
  },
  "ingredients": {
    "structured_list": ["..."],
    "continuous_text": "원재료명, 원재료명2, ..."
  },
  "allergens": {
    "contains": ["대두", "게"],
    "manufacturing_facility": "제조시설 안내 문구"
  },
  "nutrition_info": {
    "total_content": "1000 g",
    "per_100g": {
      "calories": "130 Kcal"
    },
    "disclaimer": "영양정보 주의 문구 등"
  },
  "manufacturer": {
    "name": "제조업체명",
    "address": "주소"
  },
  "precautions": ["주의사항1", "주의사항2"],
  "law_compliance": {
    "status": "compliant" | "needs_review",
    "issues": ["법률 위반 사항 목록 (있는 경우)"]
  },
  "details": [
    {"name": "원재료명", "ratio": "배합비율", "origin": "원산지", "sub_ingredients": "하위원료"}
  ]
}
"""

PROMPT_VERIFY_DESIGN = """
당신은 대한민국 최고의 식품표시사항 정밀 감사 AI이자 자동 채점기입니다.
제공된 [Standard(기준서)]와 [Design OCR(raw_text)]를 1:1 정밀 대조하여 채점하세요.

[입력]
1) Standard: JSON 형식의 기준 데이터
2) Design OCR 텍스트: 서버에서 미리 추출한 순수 텍스트 (이미지 OCR 결과)

[절대 규칙]
- Standard와 디자인 OCR 텍스트에 **실제로 존재하는 내용만** 사용하세요.
- 맞춤법, 띄어쓰기, 숫자, 단위, 특수문자 차이를 그대로 기반으로만 비교하세요.
- 존재하지 않는 “500g”, “솔비톨” 등의 값은 상상해서 만들지 마세요.
- "expected" 값은 반드시 Standard에서 실제로 존재하는 문자열을 그대로 복사해서 사용해야 합니다.
- "actual" 값은 반드시 디자인 OCR 텍스트(design_ocr_text)에서 실제로 존재하는 문자열을 그대로 복사해서 사용해야 합니다.

[감점 기준표 (총점 100점에서 시작)]
1. 원재료명 오류 (-5점/건)
2. 영양성분 오류 (-5점/건)
3. 법적 의무 문구 누락 (-10점/건)
4. 단순 오타 (-2점/건)

[출력 형식 - JSON만 출력]
{
  "design_ocr_text": "디자인 전체 텍스트(raw_text 또는 OCR 결과) 그대로",
  "score": 100,
  "law_compliance": {
    "status": "compliant" | "violation",
    "violations": ["식품등의 표시기준 제X조 위반..."]
  },
  "issues": [
    {
      "type": "Critical" | "Minor" | "Law_Violation",
      "location": "항목명 (예: 영양정보)",
      "issue": "오류 상세 설명",
      "expected": "기준서 데이터에서 실제 발췌한 텍스트",
      "actual": "디자인 OCR에서 실제 발췌한 틀린 텍스트",
      "suggestion": "수정 제안"
    }
  ]
}
"""

# --- 텍스트/HTML 정리 함수 ---
def clean_html_text(text):
    if not text:
        return ""
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


# --- ChatGPT Vision OCR (우선 사용) ---
def ocr_image_bytes_with_chatgpt(image_bytes: bytes) -> str:
    """
    ChatGPT 멀티모달로 OCR만 수행 (텍스트만 그대로 달라고 강하게 지시).
    실패하면 빈 문자열 반환.
    """
    if client is None:
        return ""

    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 너무 크면 약간 줄이기
        max_size = 1600
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
            print(f"📉 OCR용 이미지 리사이즈: {new_size}")

        ocr_prompt = """
이 이미지는 식품 포장지/라벨 사진입니다.
**이미지 안에 보이는 모든 글자를 그대로 적어 주세요.**

[중요]
- 줄바꿈, 공백, 숫자, 기호를 최대한 원문 그대로 유지하세요.
- 의미를 요약하거나 설명하지 말고, 순수 텍스트만 출력하세요.
- 한국어는 한국어로, 영어/숫자는 있는 그대로 적어 주세요.
"""
        parts = [ocr_prompt, img]
        text = call_openai_from_parts(parts, json_mode=False).strip()

        # 혹시 코드블록으로 오면 제거
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if text:
            print("✅ ChatGPT OCR 성공 (vision)")
            return text
        else:
            print("⚠️ ChatGPT OCR 결과가 비어 있음")
            return ""
    except Exception as e:
        print("❌ ChatGPT OCR 실패:", e)
        return ""


# --- OCR 폴백 ---
def ocr_bytes_to_text(image_bytes: bytes) -> str:
    """
    1순위: ChatGPT Vision OCR
    2순위: pytesseract (설치된 경우)
    """
    # 1) ChatGPT Vision
    text = ocr_image_bytes_with_chatgpt(image_bytes)
    if text:
        return text

    # 2) pytesseract
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang='kor+eng')
        text = text.strip()
        if text:
            print("✅ pytesseract OCR 성공 (폴백)")
        else:
            print("⚠️ pytesseract OCR 결과가 비어 있음")
        return text
    except Exception as e:
        print("OCR 폴백 실패:", e)
        return ""


# --- OCR 3회 실행 및 결과 비교 ---
def ocr_multiple_times(image_bytes: bytes, num_runs: int = 3) -> list:
    """
    OCR을 여러 번 실행하여 결과 리스트 반환
    """
    results = []
    for i in range(num_runs):
        print(f"🔄 OCR 실행 {i+1}/{num_runs}...")
        text = ocr_bytes_to_text(image_bytes)
        if text:
            results.append(text)
        else:
            print(f"⚠️ OCR 실행 {i+1} 실패")
    return results


def verify_with_ocr(ocr_text: str, standard_json: str) -> dict:
    """
    OCR 텍스트와 Standard를 비교하여 검증 결과 반환
    (한 번의 OCR 결과에 대해 독립적으로 검증)
    """
    if not ocr_text:
        return {"issues": [], "design_ocr_text": ""}

    try:
        enhanced_prompt = PROMPT_VERIFY_DESIGN
        if ALL_LAW_TEXT:
            enhanced_prompt += f"\n\n--- [참고 법령] ---\n{ALL_LAW_TEXT}\n--- [법령 끝] ---\n"

        parts = [
            enhanced_prompt,
            f"\n--- [기준 데이터(Standard)] ---\n{standard_json}",
            f"\n--- [디자인 OCR 텍스트] ---\n{ocr_text}\n--- [디자인 OCR 텍스트 끝] ---\n",
        ]

        result_text = call_openai_from_parts(parts, json_mode=True).strip()

        # JSON 파싱 전 코드블럭 제거
        if result_text.startswith("```json"):
            result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
        elif result_text.startswith("```"):
            lines = result_text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            result_text = "\n".join(lines).strip()

        result = json.loads(result_text)
        result = clean_ai_response(result)

        # design_ocr_text가 없으면 우리가 사용한 OCR 텍스트를 그대로 넣는다.
        result.setdefault("design_ocr_text", ocr_text)
        return result
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        traceback.print_exc()
        return {"issues": [], "design_ocr_text": ocr_text}


def find_common_errors(ocr_results: list, standard_json: str) -> dict:
    """
    3번의 OCR 결과를 비교하여 2번 이상 일치하는 오류만 반환.

    - OCR 결과마다 verify_with_ocr 로 독립 검증.
    - 같은 (location, expected)를 가진 issue 를 ‘같은 오류’로 인식.
    - 같은 오류가 3번 중 2번 이상 등장하면 최종 issues 에 포함.
    - 1번만 등장한 오류는 모두 제외.
    """
    if not ocr_results:
        return {"ocr_text": "", "issues": [], "design_ocr_text": ""}

    # 각 OCR 결과에 대해 독립 검증 수행
    all_verifications = []
    for i, ocr_text in enumerate(ocr_results):
        print(f"🔍 OCR 결과 {i+1}/{len(ocr_results)} 검증 중...")
        result = verify_with_ocr(ocr_text, standard_json)
        issues = result.get("issues", []) or []
        print(f"   → {len(issues)}개 오류 발견")
        all_verifications.append({
            "ocr_text": ocr_text,
            "design_ocr_text": result.get("design_ocr_text", ocr_text),
            "issues": issues,
        })

    # location+expected 를 key 로 해서 등장 횟수 카운트
    issue_counts = {}
    issue_repr = {}  # 대표 issue 저장

    def norm(s):
        return (s or "").strip()

    for ver in all_verifications:
        for issue in ver["issues"]:
            loc = norm(issue.get("location"))
            exp = norm(issue.get("expected"))

            # location과 expected 둘 다 없으면 키로 사용하지 않음
            if not loc and not exp:
                continue

            key = (loc, exp)
            issue_counts[key] = issue_counts.get(key, 0) + 1
            # 가장 먼저 본 issue를 대표로 사용
            if key not in issue_repr:
                issue_repr[key] = issue

    # 2번 이상 등장한 오류만 남김
    common_issues = []
    for key, count in issue_counts.items():
        if count >= 2:
            issue = issue_repr[key]
            common_issues.append(issue)
            print(f"✅ 공통 오류 ({count}/{len(ocr_results)}회): "
                  f"[{key[0]}] expected='{key[1]}'")
        else:
            print(f"❌ 단일 발견 오류 제외 (1/{len(ocr_results)}회): "
                  f"[{key[0]}] expected='{key[1]}'")

    print(f"📊 총 {sum(len(v['issues']) for v in all_verifications)}개 오류 중 "
          f"{len(common_issues)}개가 2회 이상 일치하여 최종 선택됨")

    # 첫 번째 OCR 결과를 대표 텍스트로 사용
    main_ocr = all_verifications[0]["ocr_text"]
    main_design_ocr = all_verifications[0]["design_ocr_text"]

    return {
        "ocr_text": main_ocr,
        "design_ocr_text": main_design_ocr,
        "issues": common_issues,
    }


def filter_issues_by_text_evidence(result, standard_json: str, ocr_text: str):
    """
    LLM 헛소리 방지 필터:

    1) expected(정답)는 반드시 Standard JSON 텍스트 안에 실제 존재해야 함
    2) actual(실제)는 반드시 OCR 텍스트 안에 실제 존재해야 함

    둘 중 하나라도 없으면 그 issue 는 제거.
    """
    if not isinstance(result, dict):
        return result

    try:
        std_obj = json.loads(standard_json) if standard_json else {}
        std_text = json.dumps(std_obj, ensure_ascii=False)
    except Exception:
        std_text = standard_json or ""

    ocr_text = ocr_text or ""

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        return result

    filtered = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue

        expected = str(issue.get("expected", "") or "")
        actual = str(issue.get("actual", "") or "")

        if expected and expected not in std_text:
            print("🚫 expected 가 Standard 안에 없음 → 이슈 제거:", expected)
            continue
        if actual and actual not in ocr_text:
            print("🚫 actual 이 OCR 텍스트 안에 없음 → 이슈 제거:", actual)
            continue

        filtered.append(issue)

    result["issues"] = filtered
    return result


def mark_possible_ocr_error_issues(result, hard_drop_distance: int = 1, soft_drop_distance: int = 2):
    """
    expected / actual 간 차이가 너무 작으면 OCR 노이즈로 처리.
    (여기서는 일단 그대로 패스. 필요하면 추가 규칙 넣으면 됨.)
    """
    return result


def highlight_ocr_errors(ocr_text: str, issues: list) -> str:
    """
    OCR 텍스트에서 오류 부분을 빨간색으로 하이라이트 처리
    """
    if not ocr_text or not issues:
        return ocr_text

    import html as html_mod
    highlighted_text = html_mod.escape(ocr_text)

    for issue in issues:
        actual = issue.get("actual", "")
        if not actual:
            continue
        escaped_actual = html_mod.escape(actual)
        if escaped_actual in highlighted_text:
            highlighted = (
                "<span style=\"background-color:#ffcccc;"
                " color:#cc0000; font-weight:bold; padding:2px 4px;"
                " border-radius:3px;\">"
                f"{escaped_actual}</span>"
            )
            # 한 번만 교체
            highlighted_text = highlighted_text.replace(escaped_actual, highlighted, 1)

    highlighted_text = highlighted_text.replace("\n", "<br>")
    return highlighted_text


# --- 파일 처리 함수 ---
def process_file_to_part(file_storage):
    """
    파일을 모델 파트로 변환.
    - 엑셀: 텍스트(CSV) 스트링 반환
    - 이미지: PIL.Image 객체 반환 (모델 입력용)
    - PDF: 첫 페이지 이미지를 PIL.Image로 변환 (가능한 경우)
    - 기타: {'mime_type','data'} 반환
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

    # 이미지 -> PIL.Image 객체 반환
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
            print(f"⚠️ 이미지 처리 실패, bytes로 반환: {e}")
            return {"mime_type": mime_type, "data": file_data}

    # PDF -> 이미지(첫 페이지)
    if mime_type == 'application/pdf' and PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(file_data, dpi=200)
            if images:
                print(f"📄 PDF->이미지 변환: {len(images)} 페이지 (첫 페이지 사용)")
                return images[0].convert("RGB")
        except Exception as e:
            print("PDF->이미지 변환 실패:", e)
            return {"mime_type": mime_type, "data": file_data}

    return {"mime_type": mime_type, "data": file_data}


# --- 이미지에서 원재료 정보 추출 (ChatGPT + OCR 폴백 결합) ---
def extract_ingredient_info_from_image(image_file):
    """원재료 표시사항 이미지에서 필요한 정보만 추출 (우선 ChatGPT, 실패 시 OCR 폴백)"""
    try:
        image_data = image_file.read()
        image_file.seek(0)
        img_pil = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")

        parts = [PROMPT_EXTRACT_INGREDIENT_INFO, img_pil]
        result_text = call_openai_from_parts(parts, json_mode=True)

        print("---- extract_ingredient_info_from_image 응답(원문 일부) ----")
        print(result_text[:4000])
        print("--------------------------------------------------")

        # ChatGPT 응답이 완전 비었으면 바로 OCR 폴백
        if not result_text:
            ocr_text = ocr_bytes_to_text(image_data)
            if ocr_text:
                return {"ocr_fallback_text": ocr_text}
            return None

        # ```json ... ``` 제거
        if result_text.startswith("```json"):
            result_text = result_text[7:-3] if result_text.endswith("```") else result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text.split("```", 1)[1].strip() if "```" in result_text else result_text
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()
        result_text = result_text.strip()

        # JSON 파싱 시도
        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"원재료 정보 JSON 파싱 실패: {e}")
            print("응답 텍스트 일부:", result_text[:1000])
            # JSON이 망가졌을 때도 OCR 폴백 한 번 더 시도
            ocr_text = ocr_bytes_to_text(image_data)
            if ocr_text:
                return {"ocr_fallback_text": ocr_text}
            return None

    except Exception as e:
        print(f"원재료 정보 추출 실패: {e}")
        traceback.print_exc()
        return None


# --- API 엔드포인트: 디자인 검증 ---
@app.route('/api/verify-design', methods=['POST'])
def verify_design():
    """
    디자인 검증 API
    - OCR을 3번 실행
    - 각 결과에 대해 독립적으로 검증 수행
    - 같은 (location, expected)를 가진 issue는 같은 오류로 인식
    - 3번 중 2번 이상 일치하는 오류만 최종 결과에 포함
    - 단일 발견 오류는 제외
    """
    print("🕵️‍♂️ 디자인 검증 시작...")

    # 1. 디자인 파일 (PDF or 이미지)
    design_file = request.files.get('design_file')

    # 2. 기준 데이터 (엑셀 파일 또는 JSON 문자열)
    standard_excel = request.files.get('standard_excel')
    standard_json = request.form.get('standard_data')

    if not design_file:
        return jsonify({"error": "디자인 파일이 필요합니다."}), 400

    if not standard_excel and not standard_json:
        return jsonify({"error": "기준 데이터(엑셀 파일 또는 JSON)가 필요합니다."}), 400

    # 기준 데이터 처리 (엑셀 → 간단 JSON)
    if standard_excel:
        try:
            df_dict = pd.read_excel(
                io.BytesIO(standard_excel.read()),
                sheet_name=None,
                engine='openpyxl'
            )
            if not df_dict:
                return jsonify({"error": "엑셀 파일이 비어있습니다."}), 400

            first_sheet_name = list(df_dict.keys())[0]
            first_sheet_df = df_dict[first_sheet_name]

            if first_sheet_df.empty:
                return jsonify({"error": "엑셀 파일의 첫 번째 시트가 비어있습니다."}), 400

            first_column = first_sheet_df.columns[0]
            if '원재료명' in first_sheet_df.columns:
                ingredients_list = first_sheet_df['원재료명'].dropna().astype(str).tolist()
            else:
                ingredients_list = first_sheet_df[first_column].dropna().astype(str).tolist()

            if not ingredients_list:
                return jsonify({"error": "엑셀 파일의 첫 번째 시트에 데이터가 없습니다."}), 400

            standard_data = {
                'ingredients': {
                    'structured_list': ingredients_list,
                    'continuous_text': ', '.join(ingredients_list)
                }
            }
            standard_json = json.dumps(standard_data, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 엑셀 파일 읽기 오류: {e}")
            traceback.print_exc()
            return jsonify({"error": f"엑셀 파일 읽기 실패: {str(e)}"}), 400

    # 디자인 파일 → 이미지 bytes
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

        # OCR을 3번 실행
        print("🔄 OCR을 3번 실행합니다...")
        ocr_results = ocr_multiple_times(design_image_bytes, num_runs=3)
        if not ocr_results:
            return jsonify({"error": "OCR 실행에 실패했습니다."}), 500

        # 3번 결과를 비교해 공통 오류만 추출
        print("🔍 3번의 OCR 결과를 비교하여 공통 오류를 찾는 중...")
        common_result = find_common_errors(ocr_results, standard_json)

        # 헛소리 필터 적용 (expected ∈ Standard, actual ∈ OCR)
        common_result = filter_issues_by_text_evidence(
            {"issues": common_result.get("issues", [])},
            standard_json,
            common_result.get("ocr_text", "")
        )
        issues_filtered = common_result.get("issues", [])

        # 하이라이트 HTML 생성
        highlighted_html = highlight_ocr_errors(
            common_result.get("design_ocr_text", common_result.get("ocr_text", "")),
            issues_filtered
        )

        # 점수는 매우 단순하게: 오류당 -5점 (최소 0)
        score = max(0, 100 - 5 * len(issues_filtered))

        final_result = {
            "design_ocr_text": common_result.get("design_ocr_text", common_result.get("ocr_text", "")),
            "design_ocr_highlighted_html": highlighted_html,
            "score": score,
            "law_compliance": {
                "status": "compliant" if len(issues_filtered) == 0 else "violation",
                "violations": []
            },
            "issues": issues_filtered
        }

        return jsonify(final_result)

    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 삼진어묵 식품표시사항 완성 플랫폼 V3.0 가동")
    print("   - OCR 3회 + 다수결 오류 검증")
    from waitress import serve

    serve(
        app,
        host='0.0.0.0',
        port=8080,
        threads=4,
        channel_timeout=600
    )
