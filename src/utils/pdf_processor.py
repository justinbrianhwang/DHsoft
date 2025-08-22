# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any
import fitz  # PyMuPDF
import math

def iter_pdf_pages_text(pdf_path: str, max_pages: int | None = None):
    import fitz
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if max_pages and i > max_pages:
                break
            txt = page.get_text("text") or ""
            if len(txt.strip()) < 10:
                txt = page.get_text("raw") or ""
            if len(txt.strip()) < 10:
                blocks = page.get_text("blocks") or []
                blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
                blocks.sort(key=lambda b: (b[1], b[0]))
                txt = "\n".join(b[4].strip() for b in blocks)
            yield i, (txt or "")

def iter_pdf_pages_as_images(pdf_path: str, out_dir: Path, zoom: float = 2.5, max_pages: int | None = None):
    import fitz
    out_dir.mkdir(exist_ok=True)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if max_pages and i > max_pages:
                break
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = out_dir / f"page_{i:02d}.png"
            pix.save(img_path.as_posix())
            yield i, img_path

def iter_pdf_layout(pdf_path: str) -> Iterator[Tuple[int, List[Dict]]]:
    """
    pdfplumber를 사용해 페이지별 텍스트 블록(좌표 포함)을 반환.
    반환 형식:
      for page_num, blocks in iter_pdf_layout(...):
          # page_num: 1-based
          # blocks: List[{"text": str, "bbox": (x0, y0, x1, y1)}]
    좌표계:
      - bbox는 (x0, y0, x1, y1)  (pdf 좌하단 원점)
      - medical_crf_extractor._group_rows_by_y 에 바로 사용 가능
    """
    import pdfplumber

    def _merge_words_to_lines(words: List[Dict], y_tol: float = 3.0) -> List[Dict]:
        """
        word-level을 y기반으로 line으로 합치고, line bbox계산.
        pdfplumber word dict 예시:
          {"text": "고혈압", "x0":..., "x1":..., "top":..., "bottom":...}
        """
        if not words:
            return []
        # y0(=top) 기준 정렬
        words_sorted = sorted(words, key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
        lines: List[List[Dict]] = []
        for w in words_sorted:
            placed = False
            wy = float(w.get("top", 0.0))
            for line in lines:
                # 라인 평균 top 과의 차이가 tol 이내면 같은 라인으로
                ly = sum(float(x.get("top", 0.0)) for x in line) / len(line)
                if abs(wy - ly) <= y_tol:
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])

        # 라인별 텍스트/바운딩박스 구성
        blocks: List[Dict] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x.get("x0", 0.0))
            text = " ".join((x.get("text") or "").strip() for x in line_sorted if (x.get("text") or "").strip())
            if not text:
                continue
            x0 = min(float(x.get("x0", 0.0)) for x in line_sorted)
            x1 = max(float(x.get("x1", 0.0)) for x in line_sorted)
            top = min(float(x.get("top", 0.0)) for x in line_sorted)
            bottom = max(float(x.get("bottom", 0.0)) for x in line_sorted)
            blocks.append({"text": text, "bbox": (x0, top, x1, bottom)})
        return blocks

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1) 가능한 한 풍부한 단위로 추출
            #    words(추천) → 없으면 chars → 최후 lines 사용
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,
                extra_attrs=["size", "fontname"]
            )
            blocks: List[Dict]

            if words:
                blocks = _merge_words_to_lines(words, y_tol=3.0)
            else:
                # words가 비면 chars로 라인 구성
                chars = page.chars or []
                if chars:
                    # chars를 word-like로 변환
                    fake_words = [{
                        "text": c.get("text", ""),
                        "x0": float(c.get("x0", 0.0)),
                        "x1": float(c.get("x1", 0.0)),
                        "top": float(c.get("top", 0.0)),
                        "bottom": float(c.get("bottom", 0.0)),
                    } for c in chars if (c.get("text") or "").strip()]
                    blocks = _merge_words_to_lines(fake_words, y_tol=3.0)
                else:
                    # 마지막 폴백: lines 사용
                    lines = page.extract_text_lines() or []
                    blocks = []
                    for ln in lines:
                        t = (ln.get("text") or "").strip()
                        if not t:
                            continue
                        x0 = float(ln.get("x0", 0.0))
                        x1 = float(ln.get("x1", x0 + max(1.0, len(t)*5)))
                        top = float(ln.get("top", 0.0))
                        bottom = float(ln.get("bottom", top + 10.0))
                        blocks.append({"text": t, "bbox": (x0, top, x1, bottom)})

            # 빈 블록이면 빈 리스트라도 반환 (상위 로직에서 OCR 폴백 가능)
            yield page_idx, blocks
