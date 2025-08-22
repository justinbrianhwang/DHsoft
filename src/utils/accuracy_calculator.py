# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os

# ---------- 기존 통계 함수들 (변경 없음/소폭 확장) ----------
def _is_ref(match: Dict) -> bool:
    return bool(match.get("reference")) or (match.get("reference") is not None)

def calc_overall_statistics(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len([m for m in matches if not m.get("ocr_only")])
    matched = len([m for m in matches if m["matched"]])

    sims = [m["similarity"] for m in matches if m["matched"]]
    cers = [m.get("cer") for m in matches if m["matched"] and m.get("cer") is not None]
    wers = [m.get("wer") for m in matches if m["matched"] and m.get("wer") is not None]
    accs = [m.get("accuracy") for m in matches if m["matched"] and m.get("accuracy") is not None]
    gpts = [m.get("gpt_confidence") for m in matches if m["matched"] and m.get("gpt_confidence") is not None]

    return {
        "total_questions": total,
        "matched_questions": matched,
        "match_rate": (matched / total * 100) if total > 0 else 0.0,
        "avg_similarity": float(np.mean(sims)) if sims else 0.0,
        "min_similarity": float(np.min(sims)) if sims else 0.0,
        "max_similarity": float(np.max(sims)) if sims else 0.0,

        # 추가: 3가지 전통 지표 + GPT 의미 유사 신뢰도 평균
        "avg_cer": float(np.mean(cers)) if cers else 0.0,
        "avg_wer": float(np.mean(wers)) if wers else 0.0,
        "avg_accuracy": float(np.mean(accs)) if accs else 0.0,
        "avg_gpt_confidence": float(np.mean(gpts)) if gpts else 0.0,

        "ocr_only_questions": len([m for m in matches if m.get("ocr_only")]),
    }

def summarize_by_category(matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cat_stats: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        cat = m.get("category", "unknown")
        s = cat_stats.setdefault(cat, {
            "total": 0,
            "matched": 0,
            "similarities": [],
            "table_items": 0,
            "table_matched": 0,
            "cers": [], "wers": [], "accs": [], "gpts": []
        })
        if not m.get("ocr_only"):
            s["total"] += 1
            if m.get("reference", {}).get("is_table_item"):
                s["table_items"] += 1
                if m["matched"]:
                    s["table_matched"] += 1
            if m["matched"]:
                s["matched"] += 1
                s["similarities"].append(m["similarity"])
                if m.get("cer") is not None: s["cers"].append(m["cer"])
                if m.get("wer") is not None: s["wers"].append(m["wer"])
                if m.get("accuracy") is not None: s["accs"].append(m["accuracy"])
                if m.get("gpt_confidence") is not None: s["gpts"].append(m["gpt_confidence"])

    for cat, s in cat_stats.items():
        s["avg_similarity"] = float(np.mean(s["similarities"])) if s["similarities"] else 0.0
        s["avg_cer"] = float(np.mean(s["cers"])) if s["cers"] else 0.0
        s["avg_wer"] = float(np.mean(s["wers"])) if s["wers"] else 0.0
        s["avg_accuracy"] = float(np.mean(s["accs"])) if s["accs"] else 0.0
        s["avg_gpt_confidence"] = float(np.mean(s["gpts"])) if s["gpts"] else 0.0
    return cat_stats

def summarize_by_page(matches: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    page_stats: Dict[int, Dict[str, Any]] = {}
    for m in matches:
        p = m["page_number"]
        s = page_stats.setdefault(p, {"total": 0, "matched": 0, "similarities": [], "cers": [], "wers": [], "accs": [], "gpts": []})
        if not m.get("ocr_only"):
            s["total"] += 1
            if m["matched"]:
                s["matched"] += 1
                s["similarities"].append(m["similarity"])
                if m.get("cer") is not None: s["cers"].append(m["cer"])
                if m.get("wer") is not None: s["wers"].append(m["wer"])
                if m.get("accuracy") is not None: s["accs"].append(m["accuracy"])
                if m.get("gpt_confidence") is not None: s["gpts"].append(m["gpt_confidence"])
    return page_stats

def summarize_by_type(matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tstats: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        if m.get("reference"):
            qtype = m["reference"].get("question_type", "unknown")
            s = tstats.setdefault(qtype, {"total": 0, "matched": 0, "avg_similarity": 0.0, "cers": [], "wers": [], "accs": [], "gpts": []})
            s["total"] += 1
            if m["matched"]:
                s["matched"] += 1
                if m.get("cer") is not None: s["cers"].append(m["cer"])
                if m.get("wer") is not None: s["wers"].append(m["wer"])
                if m.get("accuracy") is not None: s["accs"].append(m["accuracy"])
                if m.get("gpt_confidence") is not None: s["gpts"].append(m["gpt_confidence"])
    return tstats

def serialize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in matches:
        sm = {
            "page_number": m["page_number"],
            "category": m.get("category", ""),
            "matched": m["matched"],
            "similarity": float(m["similarity"]),
            "ocr_only": m.get("ocr_only", False),
        }
        # 추가 저장: 세 지표 + GPT
        if m.get("cer") is not None: sm["cer"] = float(m["cer"])
        if m.get("wer") is not None: sm["wer"] = float(m["wer"])
        if m.get("accuracy") is not None: sm["accuracy"] = float(m["accuracy"])
        if m.get("gpt_confidence") is not None: sm["gpt_confidence"] = float(m["gpt_confidence"])

        if m.get("reference"):
            sm["reference"] = {
                "question_id": m["reference"].get("question_id"),
                "question_text": m["reference"]["question_text"],
                "question_type": m["reference"].get("question_type"),
                "is_table_item": m["reference"].get("is_table_item", False),
            }
        if m.get("ocr"):
            sm["ocr"] = {
                "question_text": m["ocr"]["question_text"],
                "ocr_confidence": m["ocr"].get("ocr_confidence", 0.0),
            }
        out.append(sm)
    return out

# ---------- 추가: 문자열 기반 정량 지표 3종 ----------
def _levenshtein_distance(a: list, b: list) -> int:
    # editdistance 라이브러리 없이도 동작하도록 백업(단어 단위용)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): dp[i][0] = i
    for j in range(len(b) + 1): dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[-1][-1]

def cer(ref: str, hyp: str) -> float:
    try:
        import editdistance
        return editdistance.eval(ref, hyp) / len(ref) if len(ref) > 0 else 0.0
    except Exception:
        # 간단 백업(문자 단위)
        return _levenshtein_distance(list(ref), list(hyp)) / len(ref) if len(ref) > 0 else 0.0

def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0
    try:
        import editdistance
        return editdistance.eval(ref_words, hyp_words) / len(ref_words)
    except Exception:
        return _levenshtein_distance(ref_words, hyp_words) / len(ref_words)

def accuracy(ref: str, hyp: str) -> float:
    # 문자 단위 일치율
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    matches = sum(1 for r, h in zip(ref, hyp) if r == h)
    return matches / max(len(ref), len(hyp))

# ---------- 추가: GPT 기반 의미 유사 신뢰도 ----------
def gpt_semantic_confidence(ref: str, hyp: str, model: Optional[str] = None) -> Optional[float]:
    """
    예산/샘플링/토큰 상한 가드레일을 적용한 GPT 의미 신뢰도 평가.
    - .env:
      ENABLE_GPT_CONFIDENCE, GPT_CONFIDENCE_SAMPLE_RATE,
      GPT_MAX_TOKENS, PROMPT_TRUNCATE_CHARS
    """
    if os.getenv("ENABLE_GPT_CONFIDENCE", "true").lower() not in ("1", "true", "yes"):
        return None

    import random
    sample_rate = float(os.getenv("GPT_CONFIDENCE_SAMPLE_RATE", "0.2"))
    if random.random() > sample_rate:
        return None  # 샘플링으로 건너뜀

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # 프롬프트 길이 제한
    max_chars = int(os.getenv("PROMPT_TRUNCATE_CHARS", "700"))
    def _truncate(s: str, n: int) -> str:
        return s if len(s) <= n else (s[: n//2] + " ... " + s[-(n//2):])

    ref = _truncate(ref, max_chars//2)
    hyp = _truncate(hyp, max_chars//2)

    # 토큰 상한
    max_tokens = int(os.getenv("GPT_MAX_TOKENS", "150"))
    model = model or os.getenv("GPT_MODEL_FOR_CONFIDENCE", "gpt-4o-mini")

    # 대략 토큰 추정 (문자수/4): 러프 가드레일
    est_input_toks = (len(ref) + len(hyp) + 600) // 4  # 지시문 포함 여유분
    est_output_toks = max_tokens

    # 예산 가드레일
    try:
        from src.utils.budget_manager import BudgetManager
        bm = BudgetManager()
        est_cost = bm.estimate_cost_usd(est_input_toks, est_output_toks)
        if not bm.can_spend(est_cost):
            # 예산 부족 → GPT 스킵
            return None
    except Exception:
        pass  # 가드레일 실패시에도 실행은 계속

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""
You evaluate semantic equivalence of OCR to reference on a 0.0–1.0 confidence.
Return JSON only with keys: "confidence", "reason" (Korean, short).
Be tolerant to spacing/punctuation; penalize missing medical terms or negation errors.

REF: {ref}
HYP: {hyp}
"""
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are a precise semantic evaluator for medical CRF OCR outputs."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        data = json.loads(content)
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        # 실제 비용 커밋 (가능하면 usage에서 가져오되, 없으면 추정치 커밋)
        try:
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", est_input_toks) or est_input_toks
            out_tok = getattr(usage, "completion_tokens", est_output_toks) or est_output_toks
            from src.utils.budget_manager import BudgetManager
            bm = BudgetManager()
            bm.commit(bm.estimate_cost_usd(in_tok, out_tok))
        except Exception:
            pass

        return conf
    except Exception:
        return None
