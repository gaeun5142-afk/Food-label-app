import os
import re
import json
import requests
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple

# ===== 사용자 설정 =====
# (사장님의 PyCharm 환경에 환경변수(os.getenv)가 설정되어 있지 않을 수 있으므로,
# 코드에 직접 ID를 입력하는 것으로 수정했습니다.)
OC = "gaeun5142"  # 국가법령정보 공동활용 ID (사장님이 주신 코드 기준)

# ===== API 설정 =====
SEARCH_URL = "http://www.law.go.kr/DRF/lawSearch.do"
CONTENT_URL = "http://www.law.go.kr/DRF/lawService.do"
HEADERS = {"User-Agent": "law-crawler/5.0"}
TIMEOUT = 20
DEBUG = False  # (디버그 로그는 끔)


# -----------------------------
# 유틸
# -----------------------------
def _norm(s: str) -> str:
    """한/영/숫자만 남기고 소문자화."""
    return re.sub(r"[^0-9A-Za-z\uAC00-\uD7A3]", "", s or "").lower()


def _as_list(val):
    if not val:
        return []
    return val if isinstance(val, list) else [val]


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    if DEBUG:
        print(f"[DEBUG] GET {r.url}\n→ {r.text[:400]}")
    r.raise_for_status()
    try:
        # (JSON 변환 오류에 대비해, r.text를 먼저 확인하는 로직 추가)
        if not r.text or r.text.strip() == "":
            raise ValueError("API 응답이 비어있습니다.")
        return r.json()
    except ValueError:
        print(f"서버: ❌ JSON 변환 오류! API가 HTML이나 비정상 텍스트를 반환했습니다.\n(응답: {r.text[:400]}...)")
        # (회의록 'JSON 변환 오류' 언급에 따라 예외 처리 강화)
        # 빈 딕셔너리 대신, 오류를 확실히 알리기 위해 None 반환 또는 예외 발생
        raise ValueError(f"JSON 변환 실패: {r.text}")


# -----------------------------
# 검색 (법령 + 행정규칙)
# -----------------------------
def find_id_by_name_any(law_name_ko: str, oc: str) -> Optional[Tuple[str, str, str]]:
    """
    주어진 명칭으로 먼저 법령(target=law)을 찾고,
    실패하면 행정규칙(target=admrul)을 찾는다.
    return: (kind, id, display_name)
      - kind: "law" | "admrul"
      - id  : law => MST(법령일련번호), admrul => ID(행정규칙일련번호)
      - display_name: 검색에서 선택된 실제 명칭
    """
    # 1) 법령 검색
    try:
        resp = get_json(SEARCH_URL, {
            "OC": oc, "target": "law", "type": "JSON",
            "search": 1, "query": law_name_ko, "display": 100
        })
        items = resp.get("LawSearch", {}).get("law", [])
        if isinstance(items, dict):
            items = [items]
    except Exception as e:
        print(f"서버: ❌ '법령' 검색 API 호출 실패 ({law_name_ko}): {e}")
        items = []

    best = None
    best_score = 0.0
    best_name = ""
    for it in items:
        nm = it.get("법령명한글") or it.get("법령명") or ""
        score = _similar(law_name_ko, nm)
        if score > best_score:
            best_score = score
            best = ("law", str(it.get("법령일련번호", "")))
            best_name = nm

    if best and best_score >= 0.75 and best[1]:
        return best[0], best[1], best_name

    # 2) 행정규칙(고시) 검색
    try:
        resp2 = get_json(SEARCH_URL, {
            "OC": oc, "target": "admrul", "type": "JSON",
            "query": law_name_ko, "display": 100
        })
        items2 = resp2.get("AdmrulSearch", {}).get("admrul", [])
        if isinstance(items2, dict):
            items2 = [items2]
    except Exception as e:
        print(f"서버: ❌ '행정규칙' 검색 API 호출 실패 ({law_name_ko}): {e}")
        items2 = []

    best = None
    best_score = 0.0
    best_name = ""
    for it in items2:
        nm = it.get("행정규칙명") or ""
        score = _similar(law_name_ko, nm)
        if score > best_score:
            best_score = score
            best = ("admrul", str(it.get("행정규칙일련번호", "")))
            best_name = nm

    if best and best_score >= 0.75 and best[1]:
        return best[0], best[1], best_name

    return None


# -----------------------------
# 본문 조회 (법령/행정규칙 분기)
# -----------------------------
def fetch_any_by_id(kind: str, id_: str, oc: str) -> Dict[str, Any]:
    params = {"OC": oc, "type": "JSON"}
    if kind == "law":
        params.update({"target": "law", "MST": id_})
    else:  # admrul
        params.update({"target": "admrul", "ID": id_})
    return get_json(CONTENT_URL, params)


# -----------------------------
# 조문 추출 (법령 구조 위주)
# -----------------------------
def _join_article_text(art: Dict[str, Any]) -> str:
    parts: List[str] = []
    if art.get("조문내용"):
        parts.append(art["조문내용"])

    for para in _as_list(art.get("항")):
        # 항
        if isinstance(para, dict):
            if para.get("항내용"):
                parts.append(para["항내용"])
            # 호
            for item in _as_list(para.get("호")):
                if isinstance(item, dict) and item.get("호내용"):
                    parts.append(item["호내용"])
                # 목
                for mok in _as_list(item.get("목") if isinstance(item, dict) else []):
                    if isinstance(mok, dict) and mok.get("목내용"):
                        parts.append(mok["목내용"])
    return "\n".join([p for p in parts if p])


def _walk_collect_articles(node: Any) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        if "조문번호" in node:
            res.append(node)
        for v in node.values():
            res += _walk_collect_articles(v)
    elif isinstance(node, list):
        for it in node:
            res += _walk_collect_articles(it)
    return res


def extract_articles(doc: Dict[str, Any]) -> pd.DataFrame:
    """
    법령(LAW) JSON에서 조문 테이블을 최대한 추출한다.
    행정규칙은 구조가 들쑥날쑥하므로 실패할 수 있음.
    """
    articles = _walk_collect_articles(doc)
    rows: List[Dict[str, Any]] = []
    for a in articles:
        rows.append({
            "조문번호": a.get("조문번호"),
            "조문제목": a.get("조문제목"),
            "본문(통합)": _join_article_text(a)
        })
    return pd.DataFrame(rows)


# -----------------------------
# 평문/원문 텍스트 백업 (행정규칙 대비)
# -----------------------------
def extract_fallback_text(doc: Dict[str, Any]) -> str:
    """
    조문 구조가 없을 때 사용할 백업 텍스트.
    - 문서 내 문자열 필드를 긁어모으되, 너무 방대하면 JSON pretty를 저장.
    """
    # 문자열 후보를 수집
    texts: List[str] = []

    def walk(n: Any):
        if isinstance(n, dict):
            for k, v in n.items():
                if isinstance(v, (dict, list)):
                    walk(v)
                elif isinstance(v, str):
                    # 너무 짧은 것은 제외
                    if len(v.strip()) >= 5:
                        texts.append(v.strip())
        elif isinstance(n, list):
            for it in n:
                walk(it)

    walk(doc)
    uniq = []
    seen = set()
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    big = "\n\n".join(uniq)
    # 너무 비어있으면 JSON 원본으로 대체 (회의록 내용 반영 - 전체 텍스트 단위)
    if len(big.strip()) < 50:
        print("서버: ℹ️  조문 구조가 거의 없어, JSON 원본으로 대체합니다.")
        return json.dumps(doc, ensure_ascii=False, indent=2)
    return big


# -----------------------------
# 메인 처리
# -----------------------------
def process_law_or_rule(name: str):
    print(f"\n=== {name} '법령 수확' 시작 ===")
    found = find_id_by_name_any(name, OC)
    if not found:
        print(f"⚠️ {name} → ID를 찾을 수 없습니다. (법령/행정규칙 모두 검색)")
        return

    kind, id_, display_name = found
    print(f"서버: [INFO] kind={kind}, ID={id_}, matched='{display_name}'")
    data = fetch_any_by_id(kind, id_, OC)

    # 조문 테이블 시도
    df = pd.DataFrame()
    try:
        df = extract_articles(data)
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] extract_articles error: {e}")

    safe_name = re.sub(r'[^0-9A-Za-z가-힣]', '_', name)
    out_csv = f"articles_main_{safe_name}.csv"
    out_txt = f"law_text_{safe_name}.txt"  # << 우리가 사용할 파일!

    if not df.empty and "본문(통합)" in df.columns and df["본문(통합)"].str.len().sum() > 100:
        # 본문 통합 칼럼이 비어있으면 제거
        df["본문(통합)"] = df["본문(통합)"].fillna("")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # 텍스트 파일에는 본문만 합쳐 저장 (회의록: 조항 단위 구분 없이 전체 텍스트)
        full_law_text = "\n\n".join([x for x in df["본문(통합)"].tolist() if isinstance(x, str) and x.strip()])
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(full_law_text)

        print(f"✅ '조문' 기반 수확 완료 → {out_csv}, {out_txt}")
    else:
        # 조문형식이 아니라면 평문/원문 백업 저장 (회의록: 전체 텍스트 단위)
        fallback = extract_fallback_text(data)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(fallback)
        print(f"ℹ️ '조문' 테이블 없음 → '전체 텍스트'로 수확 완료: {out_txt}")


# -----------------------------
# 실행부
# -----------------------------
if __name__ == "__main__":
    print("==========================================")
    print("   법령 텍스트 수확기 (Crawler) v1.0")
    print("   국가법령정보 API에 연결을 시도합니다...")
    print(f"   (사용자 ID: {OC})")
    print("==========================================")

    # 주의: 공식 명칭 업데이트
    LAW_LIST = [
        "자원의 절약과 재활용 촉진에 관한 법률",  # 법령
        "농수산물의 원산지 표시 등에 관한 법률",  # (정식 명칭) 법령
        "식품 등의 표시·광고에 관한 법률",  # 법령
        "식품등의 표시기준",  # 행정규칙(고시) - 식약처
    ]

    for title in LAW_LIST:
        try:
            process_law_or_rule(title)
        except Exception as e:
            print(f"❌ {title} 처리 중 심각한 오류 발생: {e}")

    print("\n==========================================")
    print("   모든 법령 수확 완료!")
    print("   프로젝트 폴더에 .txt 파일이 생성되었는지 확인해주세요.")
    print("==========================================")