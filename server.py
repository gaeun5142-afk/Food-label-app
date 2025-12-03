import os
import json
import io
import glob
import traceback
import base64
import difflib

import pandas as pd
from flask import Flask, request, jsonify
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
PR
