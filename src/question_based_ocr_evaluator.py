# Main evaluation system
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
의료 CRF 문항 기반 OCR 정확도 평가 (오케스트레이션)
- PDF 텍스트/이미지 추출 → 문항 추출 → 문항 매칭 → 통계/리포팅 저장
- 레이아웃 인지(좌표 활용) / 예산 가드레일 / 안전장치(.env) 지원
"""

import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.utils.accuracy_calculator import cer, wer, accuracy, gpt_semantic_confidence
from src.utils.pdf_processor import iter_pdf_pages_text, iter_pdf_pages_as_images
from src.ocr.naver_clova_client import NaverOCRClient
from src.extractors.medical_crf_extractor import MedicalCRFQuestionExtractor
from src.matchers.enhanced_matcher import EnhancedQuestionMatcher
from src.utils.budget_manager import BudgetManager  # <- 이 클래스를 쓰고 있다면 필요
from src.utils.accuracy_calculator import (
    cer, wer, accuracy,
    calc_overall_statistics,
    summarize_by_category, summarize_by_page, summarize_by_type,
    serialize_matches,
)

# 외부 모듈
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(" .env 파일 로드 완료")
except Exception:
    print("  python-dotenv 없음 - 시스템 환경변수 사용")

# 내부 모듈
from src.extractors.medical_crf_extractor import MedicalCRFQuestionExtractor
from src.matchers.enhanced_matcher import EnhancedQuestionMatcher
from src.ocr.naver_clova_client import NaverOCRClient
from src.utils.pdf_processor import (
    iter_pdf_pages_text,
    iter_pdf_pages_as_images,
    # 선택: 레이아웃이 필요할 때만 사용 (없으면 폴백)
    # iter_pdf_layout  # 추출기 내부에서 사용 권장
)

from src.utils.accuracy_calculator import (
    calc_overall_statistics,
    summarize_by_category,
    summarize_by_page,
    summarize_by_type,
    serialize_matches,
    cer, wer, accuracy,                 # ← 추가
    gpt_semantic_confidence,            # ← 선택: 의미 유사 신뢰도 쓰면 유지
)
from src.utils.text_normalizer import safe_mkdir

# 선택: 예산 매니저가 있으면 사용(없으면 무시)
_BM = None
try:
    from src.utils.budget_manager import BudgetManager  # optional
    _BM = BudgetManager()
except Exception:
    _BM = None

# ------------------------------------------
# 환경설정 플래그/한도
# ------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y")

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

ENABLE_LAYOUT_AWARE = _env_bool("ENABLE_LAYOUT_AWARE", True)
MAX_PAGES_PER_RUN = _env_int("MAX_PAGES_PER_RUN", 10_000)          # 사실상 무제한(안전장치)
MAX_QUESTIONS_PER_RUN = _env_int("MAX_QUESTIONS_PER_RUN", 1_000_000)

RESULTS_DIR = Path("crf_evaluation_results")
RESULTS_DIR.mkdir(exist_ok=True)

class MedicalCRFOCREvaluator:
    """의료 CRF 문항 기반 OCR 평가 시스템 (조립/실행 담당)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # 참조 PDF 추출기 (원본용)
        self.ref_extractor = MedicalCRFQuestionExtractor(api_key=self.api_key)

        # 나머지 초기화 코드...
        self.matcher = EnhancedQuestionMatcher()
        self.budget_manager = BudgetManager()
        
        self.results_dir = Path("crf_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

        # ★ OCR 클라이언트 (폴백/스캔 처리에서 사용)
        self.ocr_client = NaverOCRClient()
        self.extractor = self.ref_extractor

    def evaluate(self, reference_pdf: str, scanned_pdf: str):
        print("\n" + "=" * 80)
        print(" 의료 CRF 문항 기반 OCR 정확도 평가")
        print("=" * 80)

        try:
            # 1) 레이아웃 기반 추출 (표/행에 강함)
            print("\n 1단계: 참조 PDF에서 문항 추출")
            ref_questions = self.ref_extractor.extract_questions_with_layout(reference_pdf)
        except Exception as e:
            print("  레이아웃 인지 추출 실패 → 텍스트/OCR 폴백:", e)
            ref_questions = []
            for page_num, text in self._get_reference_pages_with_fallback(reference_pdf):
                print(f"  {page_num}페이지 문항 추출 중...")
                ref_qs = self.ref_extractor.extract_questions(text, page_num=page_num)
                print(f"    {len(ref_qs)}개 문항 발견")
                ref_questions.extend(ref_qs)
        
        # 2) 스캔 PDF OCR → 문항 추출
        print("\n 2단계: 스캔 PDF OCR 및 문항 추출")
        ocr_questions = self._extract_ocr_questions(scanned_pdf)

        # 3) 문항 매칭
        print("\n 3단계: 문항 매칭 및 비교")
        all_matches = self._match_by_page(ref_questions, ocr_questions)
        
        self._inject_text_metrics(all_matches)  # ★ 이 줄 추가

        # 안전장치: 문항 상한
        if len(all_matches) > MAX_QUESTIONS_PER_RUN:
            all_matches = all_matches[:MAX_QUESTIONS_PER_RUN]

        # 4) 통계
        print("\n4단계: 통계 계산")
        overall = calc_overall_statistics(all_matches)
        by_category = summarize_by_category(all_matches)
        by_page = summarize_by_page(all_matches)
        by_type = summarize_by_type(all_matches)

        stats = {
            **overall,
            "category_statistics": by_category,
            "page_statistics": by_page,
            "type_statistics": by_type,
        }

        # 5) 출력/저장
        self._print_results(stats, all_matches)
        self._save_results(stats, all_matches, ref_questions, ocr_questions)

        # 선택: 예산 잔여 출력
        if _BM:
            try:
                print(f"\n 예산 잔여(일): ${_BM.remaining_today():.4f} / (월): ${_BM.remaining_month():.4f}")
            except Exception:
                pass

        return {
            "statistics": stats,
            "matches": all_matches,
            "reference_questions": ref_questions,
            "ocr_questions": ocr_questions,
        }

    # ---------------------------
    # (1) 참조 PDF 추출
    # ---------------------------
    def _extract_reference_questions(self, reference_pdf: str):
        ref_questions = []
        page_count = 0

        # 레이아웃 인지 모드가 가능하면 추출기 메서드 사용
        if ENABLE_LAYOUT_AWARE and hasattr(self.extractor, "extract_questions_with_layout"):
            try:
                # 전체 문서를 한 번에 레이아웃 인지 추출 (extractor 내부에서 페이지/카테고리 태깅)
                qlist = self.extractor.extract_questions_with_layout(reference_pdf) or []
                for q in qlist:
                    q["source"] = "reference"
                ref_questions.extend(qlist)

                # 페이지별 로그(간략화: 직접 집계)
                page_groups = {}
                for q in qlist:
                    p = q.get("page_number", -1)
                    page_groups.setdefault(p, []).append(q)
                # 상한 적용
                for p in sorted(page_groups):
                    page_count += 1
                    if page_count > MAX_PAGES_PER_RUN:
                        break
                    page_questions = page_groups[p]
                    cat_count = {}
                    for q in page_questions:
                        cat = q.get("question_category", "unknown")
                        cat_count[cat] = cat_count.get(cat, 0) + 1
                    print(f"   {p}페이지 문항 추출 중...")
                    print(f"     {len(page_questions)}개 문항 발견")
                    for cat, c in cat_count.items():
                        if cat and cat != "unknown":
                            print(f"       - {cat}: {c}개")
                return ref_questions
            except Exception as e:
                print(f"  레이아웃 인지 추출 실패 → 텍스트 기반으로 폴백: {e}")

        # 폴백: 텍스트 기반 페이지 반복
        for page_num, text in self._get_reference_pages_with_fallback(reference_pdf):
            page_count += 1
            if page_count > MAX_PAGES_PER_RUN:
                break
            print(f"   {page_num}페이지 문항 추출 중...")
            page_questions = self.extractor.extract_questions(text, page_num)
            for q in page_questions:
                q["source"] = "reference"
            ref_questions.extend(page_questions)

            # 카테고리 집계
            cat_count = {}
            for q in page_questions:
                cat = q.get("question_category", "unknown")
                cat_count[cat] = cat_count.get(cat, 0) + 1
            print(f"     {len(page_questions)}개 문항 발견")
            for cat, c in cat_count.items():
                if cat and cat != "unknown":
                    print(f"       - {cat}: {c}개")

        return ref_questions

    # ---------------------------
    # (2) 스캔 PDF(OCR) 추출
    # ---------------------------
    def _extract_ocr_questions(self, scanned_pdf: str):
        ocr_questions = []
        page_count = 0

        for page_num, image_path in iter_pdf_pages_as_images(scanned_pdf, out_dir=self.results_dir):
            page_count += 1
            if page_count > MAX_PAGES_PER_RUN:
                try:
                    image_path.unlink(missing_ok=True)
                except Exception:
                    pass
                break

            print(f"  {page_num}페이지 OCR 처리 중...")
            ocr_result = self.ocr_client.extract_text_from_image(str(image_path))

            if ocr_result.get("success"):
                # 좌표 블록이 있고, 추출기가 이를 처리할 수 있으면 레이아웃-aware 추출
                blocks = ocr_result.get("blocks") or []
                if ENABLE_LAYOUT_AWARE and blocks and hasattr(self.ref_extractor, "extract_from_blocks"):
                    try:
                        page_questions = self.ref_extractor.extract_from_blocks(blocks, page_num=page_num)
                    except Exception:
                        page_questions = self.ref_extractor.extract_questions(ocr_result["text"], page_num)
                else:
                    page_questions = self.ref_extractor.extract_questions(ocr_result["text"], page_num)

                for q in page_questions:
                    q["source"] = "ocr"
                    q["ocr_confidence"] = ocr_result.get("confidence", 0.0)
                ocr_questions.extend(page_questions)

                print(f"   OCR 신뢰도: {ocr_result.get('confidence', 0.0):.2f}, {len(page_questions)}개 문항 추출")

            # 임시 이미지 제거
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass

        return ocr_questions

    # ---------------------------
    # (3) 페이지 단위 매칭
    # ---------------------------
    def _match_by_page(self, ref_questions, ocr_questions):
        all_matches = []
        ref_pages = sorted(set(q.get("page_number", -1) for q in ref_questions))
        ocr_pages = sorted(set(q.get("page_number", -1) for q in ocr_questions))
        common_pages = [p for p in ref_pages if p in ocr_pages]  # ★

        for page_num in common_pages:
            ref_page_q = [q for q in ref_questions if q.get("page_number", -1) == page_num]
            ocr_page_q = [q for q in ocr_questions if q.get("page_number", -1) == page_num]
            print(f" {page_num}페이지: 참조 {len(ref_page_q)}개, OCR {len(ocr_page_q)}개 문항")
            page_matches = self.matcher.match_questions(ref_page_q, ocr_page_q)
            for m in page_matches:
                m["page_number"] = page_num
                all_matches.append(m)
        return all_matches

    # ---------------------------
    # (4) 출력/저장
    # ---------------------------
    def _print_results(self, stats, matches):
        print("\n" + "=" * 80)
        print("📈 의료 CRF 문항 기반 평가 결과")
        print("=" * 80)

        print(f"\n📊 전체 통계:")
        print(f"  • 총 문항 수: {stats['total_questions']}")
        print(f"  • 매칭된 문항: {stats['matched_questions']}")
        print(f"  • 전체 매칭률: {stats['match_rate']:.1f}%")
        print(f"  • 평균 유사도(문자열): {stats['avg_similarity']:.3f}")
        print(f"  • 평균 CER: {stats['avg_cer']:.3f}")
        print(f"  • 평균 WER: {stats['avg_wer']:.3f}")
        print(f"  • 평균 Accuracy: {stats['avg_accuracy']:.3f}")
        print(f"  • 평균 GPT 신뢰도: {stats['avg_gpt_confidence']:.3f}")
        print(f"  • 유사도 범위: {stats['min_similarity']:.3f} ~ {stats['max_similarity']:.3f}")
        print(f"  • OCR에만 있는 문항: {stats['ocr_only_questions']}")

        print(f"\n카테고리별 통계:")
        for category, cat_stat in stats["category_statistics"].items():
            if cat_stat["total"] > 0:
                rate = cat_stat["matched"] / cat_stat["total"] * 100
                print(f"\n  [{category or '미분류'}]")
                print(f"    • 전체: {cat_stat['matched']}/{cat_stat['total']} ({rate:.1f}%)")
                print(f"    • 평균 유사도: {cat_stat.get('avg_similarity', 0.0):.3f}")
                if cat_stat["table_items"] > 0:
                    tr = cat_stat["table_matched"] / cat_stat["table_items"] * 100
                    print(f"    • 표 항목: {cat_stat['table_matched']}/{cat_stat['table_items']} ({tr:.1f}%)")

        print(f"\n페이지별 통계:")
        for page, page_stat in sorted(stats["page_statistics"].items()):
            if page_stat["total"] > 0:
                rate = page_stat["matched"] / page_stat["total"] * 100
                avg_sim = (
                    sum(page_stat["similarities"]) / len(page_stat["similarities"])
                    if page_stat["similarities"]
                    else 0.0
                )
                print(f"  • {page}페이지: {page_stat['matched']}/{page_stat['total']} "
                      f"({rate:.1f}%), 평균 유사도: {avg_sim:.3f}")

        print(f"\n️ 문항 유형별 통계:")
        type_name_map = {
            "table_row": "표 항목",
            "multiple_choice": "다중선택",
            "single_choice": "단일선택",
            "yes_no": "예/아니오",
            "text_input": "텍스트입력",
            "date_input": "날짜입력",
            "scale": "척도",
        }
        for q_type, tstat in stats["type_statistics"].items():
            if tstat["total"] > 0:
                rate = tstat["matched"] / tstat["total"] * 100
                print(f"  • {type_name_map.get(q_type, q_type)}: "
                      f"{tstat['matched']}/{tstat['total']} ({rate:.1f}%)")

        # 매칭 실패 샘플
        print(f"\n  매칭 실패 주요 문항 (카테고리별):")
        failed_by_category = {}
        for m in matches:
            if not m["matched"] and m.get("reference"):
                cat = m.get("category", "미분류")
                failed_by_category.setdefault(cat, []).append(m)

        for cat, ms in failed_by_category.items():
            if ms:
                print(f"\n  [{cat}] - {len(ms)}개 실패")
                for m in ms[:3]:
                    ref_text = m["reference"]["question_text"]
                    if len(ref_text) > 50:
                        ref_text = ref_text[:50] + "..."
                    print(f"    • P{m['page_number']}: {ref_text}")

    def _save_results(self, stats, matches, ref_questions, ocr_questions):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # JSON
        json_path = self.results_dir / f"crf_evaluation_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(
                {
                    "timestamp": ts,
                    "statistics": stats,
                    "matches": serialize_matches(matches),
                    "reference_question_count": len(ref_questions),
                    "ocr_question_count": len(ocr_questions),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n 상세 결과: {json_path}")

        # 매칭 결과 CSV
        rows = []
        for m in matches:
            rows.append(
                {
                    "page": m["page_number"],
                    "category": m.get("category", ""),
                    "matched": m["matched"],
                    "similarity": m["similarity"],
                    "cer": m.get("cer"),
                    "wer": m.get("wer"),
                    "accuracy": m.get("accuracy"),
                    "gpt_confidence": m.get("gpt_confidence"),
                    "ref_question": m["reference"]["question_text"] if m.get("reference") else "",
                    "ocr_question": m["ocr"]["question_text"] if m.get("ocr") else "",
                    "question_type": m["reference"].get("question_type", "") if m.get("reference") else "",
                    "is_table_item": m["reference"].get("is_table_item", False) if m.get("reference") else False,
                    "ocr_only": m.get("ocr_only", False),
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.results_dir / f"crf_matching_{ts}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"매칭 결과 CSV: {csv_path}")

        # 카테고리 요약 CSV
        cat_rows = []
        for cat, cstat in stats["category_statistics"].items():
            if cstat["total"] > 0:
                cat_rows.append(
                    {
                        "category": cat or "미분류",
                        "total": cstat["total"],
                        "matched": cstat["matched"],
                        "match_rate": cstat["matched"] / cstat["total"] * 100,
                        "avg_similarity": cstat.get("avg_similarity", 0.0),
                        "table_items": cstat["table_items"],
                        "table_matched": cstat["table_matched"],
                        "table_match_rate": (cstat["table_matched"] / cstat["table_items"] * 100)
                        if cstat["table_items"] > 0 else 0,
                    }
                )
        if cat_rows:
            df_cat = pd.DataFrame(cat_rows)
            cat_path = self.results_dir / f"crf_category_summary_{ts}.csv"
            df_cat.to_csv(cat_path, index=False, encoding="utf-8-sig")
            print(f"카테고리 요약: {cat_path}")

        # 실패 문항 상세
        failed_rows = []
        for m in matches:
            if not m["matched"] and m.get("reference"):
                failed_rows.append(
                    {
                        "page": m["page_number"],
                        "category": m.get("category", "미분류"),
                        "question_type": m["reference"].get("question_type", ""),
                        "is_table_item": m["reference"].get("is_table_item", False),
                        "reference_text": m["reference"]["question_text"],
                        "question_id": m["reference"].get("question_id", ""),
                    }
                )
        if failed_rows:
            df_failed = pd.DataFrame(failed_rows)
            failed_path = self.results_dir / f"crf_failed_items_{ts}.csv"
            df_failed.to_csv(failed_path, index=False, encoding="utf-8-sig")
            print(f"실패 문항 분석: {failed_path}")
            
            
    def _inject_text_metrics(self, matches):
        for m in matches:
            if not m.get("matched"):
                continue
            ref = (m.get("reference") or {}).get("question_text") or ""
            hyp = (m.get("ocr") or {}).get("question_text") or ""
            if not ref and not hyp:
                m["cer"], m["wer"], m["accuracy"] = 0.0, 0.0, 1.0
                continue
            m["cer"] = cer(ref, hyp)
            m["wer"] = wer(ref, hyp)
            m["accuracy"] = accuracy(ref, hyp)
        
            try:
                conf = gpt_semantic_confidence(ref, hyp)
                if conf is not None:
                    m["gpt_confidence"] = conf
            except Exception:
                pass
    
    # ---------------------------
    # (5) 참조 PDF 페이지 텍스트를 추출하되, 비어있거나 너무 짧으면 OCR로 폴백.
    # ---------------------------
                
    def _get_reference_pages_with_fallback(self, reference_pdf: str, min_len: int = 10, zoom: float = 2.5):
        """
        참조 PDF 페이지 텍스트를 추출하되, 비어있거나 너무 짧으면 OCR로 폴백.
        min_len: 이 길이 미만이면 OCR 수행
        zoom: OCR 품질을 위한 렌더 배율
        """
        ref_texts = {p: t for p, t in iter_pdf_pages_text(reference_pdf, max_pages=MAX_PAGES_PER_RUN)}
        ocr_client = getattr(self, "ocr_client", None) or NaverOCRClient()

        # 이미지 OCR로 빈/짧은 페이지만 채우기 (전체 페이지)
        for p, img in iter_pdf_pages_as_images(reference_pdf, out_dir=self.results_dir, zoom=zoom, max_pages=MAX_PAGES_PER_RUN):
            if (not ref_texts.get(p)) or len((ref_texts[p] or "").strip()) < min_len:
                ocr_res = ocr_client.extract_text_from_image(str(img))
                ref_texts[p] = (ocr_res.get("text") or "").strip()
            try:
                img.unlink(missing_ok=True)
            except Exception:
                pass
        return [(p, ref_texts[p]) for p in sorted(ref_texts.keys())]


def main():
    print("의료 CRF 문항 기반 OCR 정확도 평가 시스템")
    print("=" * 80)

    # 환경변수 확인
    env_check = {
        "OpenAI API": bool(os.getenv("OPENAI_API_KEY")),
        "Naver OCR URL": bool(os.getenv("NAVER_OCR_API_URL")),
        "Naver Secret": bool(os.getenv("NAVER_OCR_SECRET_KEY")),
    }
    print("환경 설정 상태:")
    for k, ok in env_check.items():
        print(f"  {'✅' if ok else '❌'} {k}")
    if not all(env_check.values()):
        print("\n  일부 API 설정이 누락되었습니다.")
        print("   .env 파일에 다음 내용을 설정하세요:")
        print("   - OPENAI_API_KEY")
        print("   - NAVER_OCR_API_URL")
        print("   - NAVER_OCR_SECRET_KEY")

    reference_pdf = "data/input/3_WSCH표준권고증례기록서(CRF)양식Ver_3_0.pdf"
    scanned_pdf = "data/input/Original Scan.pdf"

    # 경로 확인
    if not Path(reference_pdf).exists():
        print(f"\n참조 PDF 없음: {reference_pdf}")
        return
    if not Path(scanned_pdf).exists():
        print(f"\n스캔 PDF 없음: {scanned_pdf}")
        return

    evaluator = MedicalCRFOCREvaluator()

    try:
        start = time.time()
        print(f"\n  평가 시작")
        print(f"   참조: {reference_pdf}")
        print(f"   스캔: {scanned_pdf}")
        results = evaluator.evaluate(reference_pdf, scanned_pdf)
        elapsed = time.time() - start
        print(f"\n 평가 완료 (소요시간: {elapsed:.1f}초)")
        print(f"   총 {results['statistics']['total_questions']}개 문항 평가")
        print(f"   매칭률: {results['statistics']['match_rate']:.1f}%")
    except KeyboardInterrupt:
        print("\n 사용자 중단")
    except Exception as e:
        print(f"\n 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
