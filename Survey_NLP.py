#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr
import pytesseract
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification

# 프로젝트 유틸
from eojeol_tokenization import eojeol_BertTokenizer
import file_utils

# ─── 설정 ─────────────────────────────────────────────────────────────────────
PAGE_DPI       = 200
INPUT_DIR      = "input_pdfs"
OUTPUT_DIR     = "output_results"
WIDE_FILENAME  = "survey_wide.xlsx"
ERROR_CSV      = "errors.csv"

MIN_AREA           = 1000
MAX_AREA_RATIO     = 0.5        # 전체 이미지 대비 영역 비율
OCR_CONF_THRESHOLD = 0.5        # EasyOCR confidence 임계
UNDERSCORE_LEN     = 15         # 언더스코어 병합 길이

YES_NO_MAP = {
    "예": True, "네": True, "y": True, "yes": True,
    "아니오": False, "ㄴ": False, "n": False, "no": False
}
# ①②③…⑳ 매핑
CIRCLED_NUM = {chr(code): idx+1 for idx, code in enumerate(range(9312, 9332))}

# ─── OCR 리더 초기화 ─────────────────────────────────────────────────────────────
READER = easyocr.Reader(['ko','en'], gpu=False)

# ─── NLP 모델 및 토크나이저 로드 ─────────────────────────────────────────────────
BASE_PATH      = Path(__file__).parent
config_path    = BASE_PATH / "config.json"
model_bin      = BASE_PATH / "pytorch_model.bin"
vocab_file     = BASE_PATH / "vocab.korean.rawtext.list"
label_map_file = BASE_PATH / "label_map.json"


def load_nlp_model():
    config = BertConfig.from_json_file(str(config_path))
    model  = BertForSequenceClassification(config)
    # .pth 우선, 없으면 .bin
    pth = next(BASE_PATH.glob("*.pth"), None)
    if pth:
        ckpt = torch.load(str(pth), map_location="cpu")
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model = ckpt
    elif model_bin.exists():
        state = torch.load(str(model_bin), map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError("No NLP checkpoint (.pth or .bin) found")
    model.eval()
    return model

with open(label_map_file, 'r', encoding='utf-8') as f:
    LABEL_MAP = json.load(f)

tokenizer_nlp = eojeol_BertTokenizer(str(vocab_file))
tokenizer_nlp.cls_token_id = tokenizer_nlp.vocab.get("[CLS]")
tokenizer_nlp.sep_token_id = tokenizer_nlp.vocab.get("[SEP]")
nlp_model = load_nlp_model()


def classify_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"pred_label": None, "score": None}
    tokens = tokenizer_nlp.tokenize(text)
    ids    = tokenizer_nlp.convert_tokens_to_ids(tokens)
    input_ids      = [tokenizer_nlp.cls_token_id] + ids + [tokenizer_nlp.sep_token_id]
    attention_mask = [1] * len(input_ids)
    input_ids      = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    with torch.no_grad():
        outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    idx   = int(np.argmax(probs))
    label = LABEL_MAP.get(str(idx), idx)
    return {"pred_label": label, "score": probs[idx]}


def list_pdfs(dir_path: str) -> List[str]:
    return [str(p) for p in Path(dir_path).rglob("*.pdf")]

def ocr_text(img: np.ndarray) -> str:
    if img is None or img.size == 0:
        return ""
    res = READER.readtext(img)
    texts = [r[1] for r in res if r[2] >= OCR_CONF_THRESHOLD]
    if texts:
        return " ".join(texts).strip()
    try:
        return pytesseract.image_to_string(img, lang='kor+eng').strip()
    except:
        return ""


def detect_checkbox_any(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=gray.shape[0]/2, param1=50, param2=15,
        minRadius=int(min(gray.shape[:2])*0.1),
        maxRadius=int(min(gray.shape[:2])*0.4)
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            if cv2.countNonZero(cv2.bitwise_and(gray, mask)) / (np.pi*r*r) > 0.4:
                return True
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > MIN_AREA:
            x,y,w,h = cv2.boundingRect(approx)
            ratio   = cv2.contourArea(cnt) / (w*h)
            if 0.6 < ratio < 0.9:
                return True
    return False


def extract_label(full_img: np.ndarray, x:int,y:int,w:int,h:int, pad:int=30) -> str:
    x0, y0 = max(0, x-pad), max(0, y-pad*3)
    x1     = min(full_img.shape[1], x+w+pad)
    y1     = min(full_img.shape[0], y+pad)
    crop   = full_img[y0:y1, x0:x1]
    return ocr_text(crop)


def cast_value(raw: str) -> (Any, str):
    low = raw.lower().strip()
    if low in YES_NO_MAP:
        return YES_NO_MAP[low], "bool"
    if raw in CIRCLED_NUM:
        return CIRCLED_NUM[raw], "int"
    if re.fullmatch(r"-?\d+", low):
        return int(low), "int"
    if re.fullmatch(r"-?\d+\.\d+", low):
        return float(low), "float"
    return raw, "str"


def extract_question_id(label: str) -> Any:
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else None


def process_pdf(pdf_path: str) -> List[Dict[str,Any]]:
    recs = []
    pages = convert_from_path(pdf_path, dpi=PAGE_DPI, thread_count=4)
    for page_num, pil in enumerate(pages, start=1):
        img  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        und_ker  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, UNDERSCORE_LEN))
        merged   = cv2.dilate(bw, und_ker, iterations=1)
        clean    = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, und_ker, iterations=1)
        combined = cv2.bitwise_or(clean, merged)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area    = w * h
            if area < MIN_AREA or area > img.shape[0]*img.shape[1]*MAX_AREA_RATIO:
                continue

            crop = img[y:y+h, x:x+w]
            if detect_checkbox_any(crop):
                ftype, raw, value, vtype = "checkbox", "True", True, "bool"
                nlp_meta = {"pred_label": None, "score": None}
            else:
                raw = ocr_text(crop)
                value, vtype = cast_value(raw)
                ftype = "text_field"
                nlp_meta = classify_text(raw) if vtype == "str" else {"pred_label": None, "score": None}

            # expected type 자동 추론
            if ftype == "checkbox":
                expected = "bool"
            elif re.fullmatch(r"-?\d+", raw):
                expected = "int"
            elif re.fullmatch(r"-?\d+\.\d+", raw):
                expected = "float"
            else:
                expected = "str"

            error = (vtype != expected)
            err_msg = f"Expected={expected}, got={vtype} ('{raw}')" if error else None

            label = extract_label(img, x, y, w, h)
            qid   = extract_question_id(label)

            recs.append({
                "pdf_path":      os.path.basename(pdf_path),
                "page":          page_num,
                "question_id":   qid,
                "field_type":    ftype,
                "label":         label,
                "raw":           raw,
                "value":         value,
                "value_type":    vtype,
                **nlp_meta,
                "is_subjective": (vtype=="str" and not detect_checkbox_any(crop)),
                "expected_type": expected,
                "error":         error,
                "error_message": err_msg
            })
    return recs


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdfs = list_pdfs(INPUT_DIR)
    print(f"Found {len(pdfs)} PDF(s) in '{INPUT_DIR}'")
    all_recs = []
    for pdf in pdfs:
        print(f"Processing: {os.path.basename(pdf)}")
        all_recs.extend(process_pdf(pdf))

    df = pd.DataFrame(all_recs)

    # 오류 레코드만 CSV로 저장
    errs = df[df["error"]]
    if not errs.empty:
        errs.to_csv(Path(OUTPUT_DIR)/ERROR_CSV, index=False)
        print(f"⚠️  {len(errs)} type errors → {ERROR_CSV}")

    # Long 포맷 엑셀
    long_xlsx = Path(OUTPUT_DIR)/"survey_results_long.xlsx"
    df.to_excel(long_xlsx, index=False)

    # Wide 포맷 + 문항 순서 자동 정렬
    wide = (
        df.pivot_table(
            index=["pdf_path","page"],
            columns="question_id",
            values="value",
            aggfunc="first"
        )
        .rename_axis(columns=None)
        .reset_index()
    )
    question_cols = [c for c in wide.columns if re.match(r'^Q?\d+$', str(c))]
    sorted_cols   = sorted(question_cols, key=lambda c: int(re.sub(r'\D','', str(c))))
    wide = wide[["pdf_path","page"] + sorted_cols]

    wide_xlsx = Path(OUTPUT_DIR)/WIDE_FILENAME
    wide.to_excel(wide_xlsx, index=False)
    print(f"Wide format → {WIDE_FILENAME}")

    # 콘솔 리포트
    print("\n=== Sample (first 10 rows) ===")
    print(df.head(10).to_string(index=False))
    print("\n=== Field Type Counts ===")
    print(df["field_type"].value_counts().to_string())
    print("\n=== Value Type Counts ===")
    print(df["value_type"].value_counts().to_string())
    if "pred_label" in df.columns:
        print("\n=== NLP Anomaly Counts ===")
        print(df["pred_label"].value_counts(dropna=False).to_string())
    print(f"\nTotal errors: {len(errs)}")

    # report.txt
    rpt = Path(OUTPUT_DIR)/"report.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"Processed {len(pdfs)} files, extracted {len(df)} fields\n")
        for t, grp in df.groupby("field_type"):
            f.write(f"- {t}: {len(grp)} items\n")
        if "pred_label" in df.columns:
            anomalies = int((df['pred_label']==LABEL_MAP.get('1',1)).sum())
            f.write(f"- anomalies detected: {anomalies}\n")
        f.write(f"- type errors: {len(errs)}\n")
    print(f"Done. → {long_xlsx}, report.txt")

if __name__ == "__main__":
    main()
