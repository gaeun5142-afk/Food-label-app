if not design_file:
return jsonify({"error": "디자인 파일이 필요합니다."}), 400

    # ⭐ 파일 포인터 초기화
    # 파일 포인터 초기화
design_file.seek(0)
if standard_excel:
standard_excel.seek(0)

    # 2. 기준 데이터 로딩 (엑셀 -> JSON)
    # -------------------------------------------------
    # 1) 기준 데이터 준비 (JSON 우선, 없으면 엑셀에서 생성)
    # -------------------------------------------------
if standard_excel:
try:
df_dict = pd.read_excel(
io.BytesIO(standard_excel.read()),
sheet_name=None,
                engine="openpyxl",
                engine='openpyxl',
dtype=str,
keep_default_na=False,
                na_filter=False
)

            # 🔹 시트 이름 목록 중 첫 번째 시트 선택
            sheet_names = list(df_dict.keys())          # 예: ['제품정보', '원재료명', ...]
            first_sheet_name = sheet_names[0]           # 문자열 하나
            first_sheet_df = df_dict[first_sheet_name]  # DataFrame 하나
            sheet_names = list(df_dict.keys())
            first_sheet_df = df_dict[sheet_names[0]]

standard_data = {}

if not first_sheet_df.empty:
                # 기본은 첫 번째 컬럼 사용
                col = first_sheet_df.columns[0]

                # '원재료명' 컬럼이 있으면 그걸 우선 사용
                if "원재료명" in first_sheet_df.columns:
                    col = "원재료명"

                ingredients_list = (
                    first_sheet_df[col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                col = "원재료명" if "원재료명" in first_sheet_df.columns else first_sheet_df.columns[0]
                ingredients_list = first_sheet_df[col].dropna().astype(str).tolist()
standard_data = {
"ingredients": {
"structured_list": ingredients_list,
                        "continuous_text": ", ".join(ingredients_list),
                        "continuous_text": ", ".join(ingredients_list)
}
}

standard_json = json.dumps(standard_data, ensure_ascii=False)

except Exception as e:
            print("❌ 표준 엑셀 읽기 실패:", e)
return jsonify({"error": f"엑셀 읽기 실패: {str(e)}"}), 400

    # 3. 법령 파일 읽기
    if not standard_json:
        return jsonify({"error": "기준 데이터가 없습니다(standard_data / standard_excel)."}), 400

    # -------------------------------------------------
    # 2) 법령 텍스트 읽기
    # -------------------------------------------------
law_text = ""
    all_law_files = glob.glob('law_*.txt')
    all_law_files = glob.glob("law_*.txt")
print(f"📚 법령 파일 로딩 중: {len(all_law_files)}개 발견")

for file_path in all_law_files:
try:
            with open(file_path, 'r', encoding='utf-8') as f:
            with open(file_path, "r", encoding="utf-8") as f:
content = f.read()
law_text += f"\n\n=== [참고 법령: {file_path}] ===\n{content}\n==========================\n"
except Exception as e:
print(f"⚠️ 법령 파일 읽기 실패 ({file_path}): {e}")

    # 4. 메인 검증 AI 호출 준비
        prompt = f"""
    {PROMPT_VERIFY_DESIGN}
    
    [참고 법령]
    {law_text[:60000]}
    
    [기준 데이터]
    {standard_json}
    """
        parts = [prompt]
    
        if design_file:
            parts.append(process_file_to_part(design_file))
    # -------------------------------------------------
    # 3) 메인 AI 호출 (검증 + OCR 같이 수행)
    # -------------------------------------------------
    prompt = f"""
{PROMPT_VERIFY_DESIGN}

[참고 법령]
{law_text[:60000]}

[기준 데이터(JSON)]
{standard_json}
"""
    parts = [prompt, process_file_to_part(design_file)]

    result_json = {}

    # 5. AI 호출 및 결과 처리
try:
generation_config = {
"temperature": 0.0,
"top_p": 1.0,
"top_k": 1,
"candidate_count": 1,
"max_output_tokens": 32768,
            "response_mime_type": "application/json"
            "response_mime_type": "application/json",
}

system_instruction = """
@@ -1208,73 +1200,142 @@ def verify_design():
       4. 보이지 않는 내용은 절대 출력 금지
       """

        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config, system_instruction=system_instruction)
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )

response = model.generate_content(parts)
result_text = get_safe_response_text(response)
        result_text = strip_code_fence(result_text)

        # JSON 파싱
        # JSON 부분만 잘라서 파싱
json_match = re.search(r"(\{.*\})", result_text, re.DOTALL)
if json_match:
clean_json = json_match.group(1)
            clean_json = clean_json.replace(",\n}", "\n}").replace(",\n]", "\n]")
            result_json = json.loads(clean_json)
else:
            clean_json = result_text.replace("```json", "").replace("```", "")
            clean_json = clean_json.strip()
            result_json = json.loads(clean_json)
            clean_json = (
                result_text.replace("```json", "")
                .replace("```", "")
                .strip()
            )

        clean_json = clean_json.replace(",\n}", "\n}").replace(",\n]", "\n]")
        result_json = json.loads(clean_json)

except Exception as e:
print(f"❌ 메인 검증 실패 (일단 진행): {e}")
traceback.print_exc()
result_json = {"score": 0, "issues": [], "design_ocr_text": ""}

    # ---------------------------------------------------------
    # [안전장치] design_ocr_text가 비어있으면 백업 OCR 실행
    # ---------------------------------------------------------
    # -------------------------------------------------
    # 4) design_ocr_text가 없으면 -> 백업 OCR로 채우기
    # -------------------------------------------------
if not result_json.get("design_ocr_text"):
        print("⚠️ 검증 결과에 OCR 텍스트가 누락됨. 백업 OCR 수행 중...")
try:
            design_file.seek(0) # 파일 포인터 초기화
            print("⚠️ design_ocr_text 없음 → 백업 OCR 실행")
            design_file.seek(0)

            # [중요] 백업 OCR용 설정 (토큰 제한 넉넉하게)
ocr_config = {
"temperature": 0.0,
"max_output_tokens": 32768,
                "response_mime_type": "application/json"
                "response_mime_type": "application/json",
}
            
            # [중요] OCR 전용 프롬프트 (텍스트만 추출하라고 지시)
            PROMPT_EXTRACT_ONLY = """

            ocr_prompt = """
           Extract all text from the image exactly as it appears.
            Do not summarize. Output JSON: { "text": "extracted text..." }
            Do not summarize.
            Output ONLY JSON: { "raw_text": "extracted text..." }
           """
            
            ocr_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=ocr_config)
            ocr_response = ocr_model.generate_content([PROMPT_EXTRACT_ONLY, process_file_to_part(design_file)])
            

            ocr_model = genai.GenerativeModel(
                "gemini-1.5-flash", generation_config=ocr_config
            )
            ocr_response = ocr_model.generate_content(
                [ocr_prompt, process_file_to_part(design_file)]
            )

ocr_text_raw = get_safe_response_text(ocr_response)
ocr_text_raw = strip_code_fence(ocr_text_raw)
            
            try:
                ocr_data = json.loads(ocr_text_raw)
            except json.JSONDecodeError as e:
                print("❌ 백업 OCR JSON 파싱 실패:", e)
                print("↪ 응답 일부:", ocr_text_raw[:300])
                raise
            
            extracted_text = ocr_data.get("text") or ocr_data.get("raw_text", "")

            ocr_match = re.search(r"(\{.*\})", ocr_text_raw, re.DOTALL)
            if ocr_match:
                ocr_json = json.loads(ocr_match.group(1))
            else:
                ocr_json = json.loads(ocr_text_raw)

            extracted_text = ocr_json.get("raw_text") or ocr_json.get("text", "")
result_json["design_ocr_text"] = extracted_text
print(f"✅ 백업 OCR 완료 (길이: {len(extracted_text)})")
            

except Exception as e:
print(f"❌ 백업 OCR 실패: {e}")
            result_json["design_ocr_text"] = "OCR 텍스트를 불러올 수 없습니다. (서버 오류)"
            traceback.print_exc()
            # 실패해도 key는 있어야 프론트에서 에러 문구 안뜸
            if not result_json.get("design_ocr_text"):
                result_json["design_ocr_text"] = ""

    # -------------------------------------------------
    # 5) issue 타입/position 보정  → 색상 & 하이라이트용
    # -------------------------------------------------
    try:
        full_text = result_json.get("design_ocr_text") or ""
        issues = result_json.get("issues") or []

        import string as _s

        def norm_no_ws_punct(s: str) -> str:
            return "".join(
                ch for ch in str(s) if ch not in _s.whitespace + _s.punctuation
            )

        for issue in issues:
            raw_type = (issue.get("type") or "").lower()
            loc = str(issue.get("location", ""))
            desc = str(issue.get("issue", ""))
            expected = str(issue.get("expected", ""))
            actual = str(issue.get("actual", ""))

            # 5-1) 법률 위반 추정 → Law_Violation
            if (
                "law_violation" in raw_type
                or "위반" in raw_type
                or "법" in raw_type
                or any(k in loc for k in ["법률", "소비기한", "1399", "포장재질"])
                or any(k in desc for k in ["누락", "미표기", "미기재"])
            ):
                issue["type"] = "Law_Violation"
            else:
                # 5-2) 공백/문장부호만 다른 경우 → Minor
                if expected and actual and norm_no_ws_punct(expected) == norm_no_ws_punct(actual):
                    issue["type"] = "Minor"
                else:
                    issue["type"] = "Critical"

            # 5-3) position 없으면 직접 계산
            pos = issue.get("position")
            if not isinstance(pos, int) or pos < 0:
                pos = -1
                if full_text and actual:
                    pos = full_text.find(actual)
                if pos == -1 and full_text and expected:
                    pos = full_text.find(expected)
                if pos < 0:
                    pos = 0
                issue["position"] = pos

        result_json["issues"] = issues

    except Exception as e:
        print("⚠️ issue 후처리 중 오류:", e)
        traceback.print_exc()

return jsonify(result_json)




@app.route('/api/verify-design-strict', methods=['POST'])
def verify_design_strict():
"""Python으로 정확한 비교 (AI 없이)"""
