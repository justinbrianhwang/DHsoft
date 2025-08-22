# Text normalization utilities
# -*- coding: utf-8 -*-
import re
from pathlib import Path

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def remove_marks(text: str) -> str:
    return re.sub(r"[□☐■✓✔①②③④⑤⑥⑦⑧⑨]", "", text or "")

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


CIRCLED_MAP = str.maketrans("①②③④⑤⑥⑦⑧⑨⓪", "1234567890")
def standardize_symbols(text: str) -> str:
    t = (text or "").translate(CIRCLED_MAP)
    t = re.sub(r"[□☐■✓✔]", " ", t)
    t = re.sub(r"(\|?_+|\|_+\|)+", " <BLANK> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
