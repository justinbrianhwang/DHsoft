# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
import numpy as np

class EnhancedQuestionMatcher:
    def __init__(self, similarity_threshold: float = 0.60):
        self.similarity_threshold = similarity_threshold

    def match_questions(self, ref_questions: List[Dict], ocr_questions: List[Dict]) -> List[Dict]:
        """
        같은 '페이지'의 ref_questions, ocr_questions가 들어온다고 가정.
        → 내부에서 추가 그룹핑하지 않고 전부 한 번에 전역 최적 배정.
        """
        matches: List[Dict] = []
        if not ref_questions and not ocr_questions:
            return matches
        if not ocr_questions:
            for r in ref_questions:
                matches.append({
                    "reference": r, "ocr": None, "similarity": 0.0,
                    "matched": False, "category": r.get("question_category","unknown")
                })
            return matches

        # 비용행렬(-score) 구성
        M = np.zeros((len(ref_questions), len(ocr_questions)), dtype=float)
        cache: Dict[Tuple[int,int], float] = {}
        for i, r in enumerate(ref_questions):
            for j, o in enumerate(ocr_questions):
                s = self._combo_score(r, o)
                cache[(i,j)] = s
                M[i, j] = -s  # 최대합 → 최소비용

        ri, oj = linear_sum_assignment(M)
        used_ocr = set()
        for i, j in zip(ri, oj):
            s = max(0.0, min(1.0, cache[(i,j)]))
            if s >= self._dynamic_threshold(ref_questions[i]):
                used_ocr.add(j)
                matches.append({
                    "reference": ref_questions[i],
                    "ocr": ocr_questions[j],
                    "similarity": s,
                    "matched": True,
                    "category": ref_questions[i].get("question_category","unknown")
                })
            else:
                matches.append({
                    "reference": ref_questions[i],
                    "ocr": None,
                    "similarity": 0.0,
                    "matched": False,
                    "category": ref_questions[i].get("question_category","unknown")
                })
        # 남은 OCR만
        for j, o in enumerate(ocr_questions):
            if j not in used_ocr:
                matches.append({
                    "reference": None, "ocr": o, "similarity": 0.0,
                    "matched": False, "ocr_only": True,
                    "category": o.get("question_category","unknown")
                })
        return matches

    # ----------------- 스코어 구성 -----------------
    def _combo_score(self, r: Dict, o: Dict) -> float:
        rt, ot = r.get("question_text",""), o.get("question_text","")
        qtype = r.get("question_type","text_input")

        sm = SequenceMatcher(None, self._norm(rt), self._norm(ot)).ratio()
        rf = fuzz.QRatio(self._norm(rt), self._norm(ot)) / 100.0

        # 토큰 Jaccard
        r_tok, o_tok = set(self._tokenize(rt)), set(self._tokenize(ot))
        jacc = len(r_tok & o_tok) / max(1, len(r_tok | o_tok))

        # 도메인 보너스
        kw_bonus = self._keyword_bonus(rt, ot, [
            "고혈압","당뇨","고지혈증","협심증","심근경색","우울","PHQ","진단","가족력","나이","연령","세","년","월","일"
        ])
        num_bonus = self._number_bonus(rt, ot)
        date_bonus = self._date_bonus(rt, ot)
        yn_bonus = 0.05 if (("예" in rt or "아니오" in rt) and ("예" in ot or "아니오" in ot)) else 0.0

        # 카테고리 일치 보너스(그룹 경계 X)
        rc, oc = r.get("question_category"), o.get("question_category")
        cat_bonus = 0.06 if (rc and oc and rc == oc) else 0.0

        # 유형별 가중치
        w = {
            "table_row":        (0.45, 0.25, 0.10, 0.08, 0.07, 0.05),  # sm, jacc, kw, num, date, yn
            "yes_no":           (0.55, 0.15, 0.10, 0.05, 0.05, 0.10),
            "date_input":       (0.45, 0.10, 0.05, 0.05, 0.30, 0.05),
            "text_input":       (0.60, 0.20, 0.10, 0.05, 0.03, 0.02),
            "single_choice":    (0.55, 0.20, 0.15, 0.05, 0.03, 0.02),
            "multiple_choice":  (0.55, 0.20, 0.15, 0.05, 0.03, 0.02),
            "scale":            (0.50, 0.15, 0.10, 0.05, 0.05, 0.15),
        }.get(qtype, (0.55, 0.20, 0.10, 0.05, 0.05, 0.05))

        score = (
            w[0]*sm + w[1]*jacc + w[2]*kw_bonus + w[3]*num_bonus + w[4]*date_bonus + w[5]*yn_bonus
        )
        # rapidfuzz 보정 + 카테고리 보너스
        score = 0.7*score + 0.3*rf + cat_bonus
        return max(0.0, min(1.0, score))

    def _dynamic_threshold(self, ref_q: Dict) -> float:
        base = 0.60
        if ref_q.get("is_table_item"): base -= 0.05
        t = ref_q.get("question_type","")
        if t in ("date_input","yes_no","scale"): base -= 0.05
        c = ref_q.get("question_category","")
        if c in ("질병력","가족력"): base -= 0.03
        return max(0.50, base)

    # ----------------- 유틸 -----------------
    def _norm(self, s: str) -> str:
        s = s or ""
        s = re.sub(r"[□☐■✓✔①②③④⑤⑥⑦⑧⑨\(\)\[\]\.:;,\-_/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _tokenize(self, s: str):
        s = self._norm(s)
        s = re.sub(r"(\d{2,4}[./-]\d{1,2}[./-]\d{1,2})", r" \1 ", s)
        return s.split()

    def _keyword_bonus(self, ref, hyp, kws):
        r = set([k for k in kws if k in ref])
        h = set([k for k in kws if k in hyp])
        return min(1.0, len(r & h) * 0.15)

    def _number_bonus(self, ref, hyp):
        rn = re.findall(r"\d+", ref); hn = re.findall(r"\d+", hyp)
        return min(0.3, len(set(rn) & set(hn)) * 0.1)

    def _date_bonus(self, ref, hyp):
        pat = r"\b(19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}\b"
        rdt = re.findall(pat, ref); hdt = re.findall(pat, hyp)
        return 0.3 if (rdt and hdt) else 0.0
