# CRF-specific extractor (layout-aware + GPT/rule hybrid)
# -*- coding: utf-8 -*-
import os
import json
import re
from typing import List, Dict, Any, Optional
from src.utils.text_normalizer import standardize_symbols


OPENAI_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4o-mini")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))



# 선택적: 참조 PDF 레이아웃용 유틸(없으면 내부 폴백)
try:
    from src.utils.pdf_processor import iter_pdf_layout
except Exception:
    iter_pdf_layout = None

class MedicalCRFQuestionExtractor:
    """의료 CRF 특화 문항 추출기 (GPT 기반 + 규칙 기반 + 레이아웃 인지 보강)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(" GPT API 연결 완료 (의료 CRF 문항 추출용)")
            except Exception as e:
                self.client = None
                print(f" GPT API 초기화 오류: {e}")

        # 섹션 앵커 키워드(필요시 docs/CRF_STRUCTURE.md로 옮겨 관리 가능)
        self.anchors = {
            "동의 취득": ["동의", "서면 동의", "연구대상자 동의", "동의서"],
            "선정기준": ["선정기준", "Eligibility", "포함 기준"],
            "제외기준": ["제외기준", "Exclusion", "배제 기준"],
            "인구학적특성": ["인구학적", "일반인구학적", "학력", "직업", "수입"],
            "질병력": ["질병력", "진단여부", "병력", "진단 여부"],
            "가족력": ["가족력"],
            "우울증선별": ["PHQ-9", "우울증", "선별도구"],
            "평가": ["평가 결과", "참여 가능", "연구대상자 평가"],
        }


    # ---------------------------------------------------------------------
    # 1) 텍스트 기반 추출 (기존: GPT → 실패 시 규칙)
    # ---------------------------------------------------------------------
    def extract_questions(self, text: str, page_num: int = 1) -> List[Dict]:
        """페이지 텍스트에서 의료 CRF 문항 추출 (GPT 우선, 실패 시 규칙 기반)"""
        if not self.client:
            return self._rule_based_extraction(text, page_num)

        try:
            prompt = f"""
다음은 의료 증례기록서(CRF) {page_num}페이지의 텍스트입니다.
이 텍스트에서 모든 "문항"과 "데이터 수집 항목"을 추출해주세요.

[문항의 정의 - 의료 CRF 특화]
1. 질문 형태의 문장
2. 체크박스가 있는 모든 항목 (□ 예 □ 아니오, Yes □ No □ 등)
3. 입력란이 있는 항목 (날짜, 숫자, 텍스트: ___ 또는 |___| 형태)
4. 선택지가 있는 항목 (①②③, 1)2)3) 등)
5. 표 형태의 데이터 수집 항목 (예: 질병명과 진단여부가 세트인 경우)
6. 평가 척도 항목 (0-1일, 2-6일 등의 스케일)
7. 진단명/병력 체크 항목

[중요 - 표 구조 인식]
- 질병명/진단여부 표는 각 행을 독립 문항으로.
- 예: "고혈압 ① ② ⑨" → "고혈압 진단여부"로 하나의 문항.

[출력 형식]
JSON 배열:
{{
  "question_id": "예: 2-1-hypertension, 3a",
  "question_text": "문항 텍스트",
  "question_category": "예: 질병력, 가족력, 인구학적특성",
  "question_type": "single_choice|multiple_choice|text_input|date_input|yes_no|scale|table_row",
  "parent_question": "상위 문항 ID 또는 null",
  "has_options": true/false,
  "options": ["선택지1","선택지2"] 또는 null,
  "is_table_item": true/false
}}

[텍스트]
{text}

JSON 배열만 출력:
"""
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 의료 CRF 문서 분석 전문가입니다. 표 구조와 계층적 문항을 정확히 인식합니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )

            result = response.choices[0].message.content or ""

            try:
                if "```" in result:
                    # ```json ... ``` 형태 제거
                    result = result.split("```")[1].replace("json", "").strip()
                parsed = json.loads(result)
                questions = parsed["questions"] if isinstance(parsed, dict) and "questions" in parsed else parsed
                if not isinstance(questions, list):
                    questions = []
                for q in questions:
                    q["page_number"] = page_num
                return questions
            except json.JSONDecodeError:
                print(" JSON 파싱 실패, 규칙 기반 추출로 전환")
                return self._rule_based_extraction(text, page_num)

        except Exception as e:
            print(f"  GPT 문항 추출 오류: {e}")
            return self._rule_based_extraction(text, page_num)

    # ---------------------------------------------------------------------
    # 2) 레이아웃 인지: 참조 PDF(원본)용
    # ---------------------------------------------------------------------
    def extract_questions_with_layout(self, pdf_path: str) -> List[Dict]:
        """
        참조 PDF 전체를 레이아웃 기반으로 돌려 섹션/표 행을 자동 그룹화하여 문항화.
        - iter_pdf_layout(pdf_path)가 사용 가능할 때만 동작
        - 실패/부재 시에는 호출측에서 텍스트 기반으로 폴백
        """
        if iter_pdf_layout is None:
            raise RuntimeError("iter_pdf_layout이 제공되지 않습니다. pdf_processor를 확인하세요.")

        out: List[Dict] = []
        for page_num, blocks in iter_pdf_layout(pdf_path):
            # 1) 섹션 앵커 탐색
            sec_blocks = self._find_section_anchors(blocks)
            # 2) y-클러스터로 행(row) 구성
            rows = self._group_rows_by_y(blocks, tol=8)
            # 3) 각 행을 가장 가까운 섹션에 귀속 → 문항화
            for row in rows:
                category = self._nearest_section(row, sec_blocks)
                text = self._concat_row_text(row)
                qs = self._rule_based_extraction(text, page_num)
                for q in qs:
                    q.setdefault("question_category", category)
                    if category in ("질병력", "가족력"):
                        q["is_table_item"] = True
                    q["source"] = "reference"
                out.extend(qs)
        return out

    # ---------------------------------------------------------------------
    # 3) 레이아웃 인지: OCR blocks(스캔본)용
    # ---------------------------------------------------------------------
    def extract_from_blocks(self, blocks: List[Dict], page_num: int) -> List[Dict]:
        """
        OCR 응답의 텍스트 블록(좌표 포함)을 입력받아 행(row) 단위로 묶어 문항화.
        """
        out: List[Dict] = []
        # 1) 섹션 앵커 탐색
        sec_blocks = self._find_section_anchors(blocks)
        # 2) y-클러스터로 행 구성
        rows = self._group_rows_by_y(blocks, tol=14)  # 표 행 묶임 강화
        # 3) 각 행을 가장 가까운 섹션에 귀속 → 문항화
        for row in rows:
            category = self._nearest_section(row, sec_blocks)
            text = self._concat_row_text(row)
            qs = self._rule_based_extraction(text, page_num)
            for q in qs:
                q.setdefault("question_category", category)
                if category in ("질병력", "가족력"):
                    q["is_table_item"] = True
                q["source"] = "ocr"
            out.extend(qs)
        return out

    # ---------------------------------------------------------------------
    # 규칙 기반 추출 (기존)
    # ---------------------------------------------------------------------
    def _rule_based_extraction(self, text: str, page_num: int = 1) -> List[Dict]:
        """규칙 기반 의료 CRF 문항 추출"""
        questions: List[Dict] = []
        lines = [ln for ln in (text or "").splitlines() if ln and ln.strip()]

        patterns = {
            "disease_with_options": r"(고혈압|당뇨병|고지혈증|뇌졸중|협심증|폐결핵|갑상선|위염|위궤양|십이지장|폴립|간질환|지방간|만성간염|담석증|담낭염|기관지염|폐쇄성|천식|알레르기|통풍|관절염|골다공증|백내장|녹내장|우울증|치주병|전립선|악성종양|골절).*[①②⑨]",
            "checkbox_yn": r"□\s*(예|아니오|Yes|No)",
            "numbered": r"^(\d+[\)\.]\s*.+)",
            "lettered": r"^([a-zA-Z][\)\.]\s*.+)",
            "input_field": r"[_|]{3,}|만\s*___\s*세",
            "scale": r"(0-1일|2-6일|7일|거의\s*매일)",
            "selection": r"[①②③④⑤⑥⑦⑧⑨]",
            "date_field": r"\|___\|___\|.*/(yy|mm|dd)",
            "question_mark": r".*\?$",
            "criterion_header": r"(선정기준|제외기준)\s*[:：]?",
            "criterion_item": r"^(\d+[\)\.]|\[•\-–\])\s*.+",
            "evaluation_line": r"(임상연구에\s*참여할\s*수\s*있습니다|없습니다|적합|부적합)",
            "yes_no": r"(예|아니오|Yes|No)",
        }

        current_category: Optional[str] = None
        question_counter = 0

        for raw in lines:
            line = standardize_symbols((raw or "").strip())
            if len(line) < MIN_QUESTION_LENGTH:
                continue

            # 카테고리 감지
            if "일반인구학적 특성" in line or "인구학적" in line:
                current_category = "인구학적특성"
            elif "질병력" in line and "가족력" not in line:
                current_category = "질병력"
            elif "가족력" in line:
                current_category = "가족력"
            elif "우울증 선별도구" in line or "PHQ-9" in line or "우울증" in line:
                current_category = "우울증선별"
            elif "Eligibility" in line or "선정기준" in line:
                current_category = "선정기준"
            elif "제외기준" in line or "Exclusion" in line:
                current_category = "제외기준"
            elif "평가" in line and "결과" in line:
                # 평가 결과/참여 가능 등
                current_category = "평가"

            is_question = False
            q_type = "text_input"
            is_table = False

            # 질병 표 행
            if re.search(patterns["disease_with_options"], line):
                is_question = True
                q_type = "table_row"
                is_table = True
                question_counter += 1

                disease_match = re.search(r"([\w가-힣]+)\s*[①②⑨]", line)
                if disease_match:
                    disease_name = disease_match.group(1)
                    questions.append(
                        {
                            "question_id": f"{current_category}_{question_counter}_{disease_name}",
                            "question_text": f"{disease_name} 진단여부",
                            "question_category": current_category,
                            "question_type": "multiple_choice",
                            "has_options": True,
                            "options": ["아니오", "예", "모름"],
                            "is_table_item": True,
                            "page_number": page_num,
                        }
                    )
                continue

            if re.search(patterns["checkbox_yn"], line):
                is_question = True
                q_type = "yes_no"
            elif re.search(patterns["numbered"], line) or re.search(patterns["lettered"], line):
                is_question = True
                if "①" in line or "②" in line:
                    q_type = "single_choice"
            elif re.search(patterns["input_field"], line):
                is_question = True
                q_type = "date_input" if ("날짜" in line or "yy/mm/dd" in line) else "text_input"
            elif re.search(patterns["scale"], line):
                is_question = True
                q_type = "scale"
            elif re.search(patterns["question_mark"], line):
                is_question = True
                q_type = "yes_no"

            if is_question and not is_table:
                question_counter += 1
                questions.append(
                    {
                        "question_id": f"{current_category or 'auto'}_{question_counter}",
                        "question_text": line,
                        "question_category": current_category,
                        "question_type": q_type,
                        "has_options": ("□" in line or "①" in line),
                        "options": None,
                        "is_table_item": False,
                        "page_number": page_num,
                    }
                )

        return questions

    # ---------------------------------------------------------------------
    # 레이아웃 보조 유틸
    # ---------------------------------------------------------------------
    def _find_section_anchors(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """블록 중 섹션 앵커(헤더) 후보를 추출하여 [(section, bbox), ...] 반환"""
        secs: List[Dict[str, Any]] = []
        for b in blocks:
            t = (b.get("text") or "").strip()
            if not t:
                continue
            for sec, keys in self.anchors.items():
                if any(k in t for k in keys):
                    secs.append({"section": sec, "bbox": b.get("bbox")})
                    break
        return secs

    def _group_rows_by_y(self, blocks: List[Dict], tol: int = 20) -> List[List[Dict]]:
        """y좌표가 가까운 블록들을 같은 행(row)으로 그룹화 (tol 상향)"""
        items = [b for b in blocks if b.get("bbox") and (b.get("text") or "").strip()]
        items.sort(key=lambda x: x["bbox"][1])  # y0
        rows: List[List[Dict]] = []
        for it in items:
            placed = False
            for r in rows:
                ry = sum(e["bbox"][1] for e in r) / len(r)
                if abs(it["bbox"][1] - ry) <= tol:
                    r.append(it); placed = True; break
            if not placed:
                rows.append([it])
        return rows

    def _nearest_section(self, row: List[Dict[str, Any]], sec_blocks: List[Dict[str, Any]]) -> Optional[str]:
        """row의 평균 y와 가장 가까운 섹션 헤더를 찾아 섹션명 반환"""
        if not sec_blocks:
            return None
        ry = sum(it["bbox"][1] for it in row) / len(row)
        best, bestd = None, 1e9
        for sb in sec_blocks:
            bbox = sb.get("bbox")
            if not bbox:
                continue
            y = bbox[1]
            d = abs(ry - y)
            if d < bestd:
                bestd = d
                best = sb.get("section")
        return best

    def _concat_row_text(self, row: List[Dict]) -> str:
        row_sorted = sorted(row, key=lambda x: x["bbox"][0])  # x0
        parts = []
        for it in row_sorted:
            t = (it.get("text") or "").strip()
            if len(t) >= 2:
                parts.append(t)
        text = " ".join(parts).strip()
        return standardize_symbols(text)  # ← ★ ①②/□/|___| 표준 토큰화
