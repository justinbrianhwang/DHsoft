# Naver Clova OCR client
# -*- coding: utf-8 -*-
import os
import json
import time
import base64
import uuid
import requests
import numpy as np
from typing import Dict, List, Any


class NaverOCRClient:
    """네이버 클로바 OCR V2 클라이언트"""

    def __init__(self):
        self.api_url = os.getenv("NAVER_OCR_API_URL")
        self.secret_key = os.getenv("NAVER_OCR_SECRET_KEY")
        self.enabled = bool(self.api_url and self.secret_key)
        if self.enabled:
            print("네이버 클로바 OCR API 연결 설정 완료")
        else:
            print("네이버 OCR API 정보 없음")

    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """이미지 경로를 입력받아 텍스트/신뢰도/좌표 블록을 반환"""
        if not self.enabled:
            return {"text": "", "blocks": [], "confidence": 0.0, "success": False}
        try:
            with open(image_path, "rb") as f:
                file_data = f.read()

            headers = {
                "X-OCR-SECRET": self.secret_key,
                "Content-Type": "application/json; charset=UTF-8",
            }
            req = {
                "images": [
                    {"format": "png", "name": "page", "data": base64.b64encode(file_data).decode()}
                ],
                "requestId": str(uuid.uuid4()),
                "version": "V2",
                "timestamp": int(round(time.time() * 1000)),
            }

            resp = requests.post(self.api_url, data=json.dumps(req), headers=headers, timeout=30)
            if resp.status_code == 200:
                return self._parse_response(resp.json())
            else:
                print(f" OCR HTTP 오류: status={resp.status_code}, body={resp.text[:200]}")
                return {"text": "", "blocks": [], "confidence": 0.0, "success": False}

        except Exception as e:
            print(f" OCR 처리 오류: {e}")
            return {"text": "", "blocks": [], "confidence": 0.0, "success": False}

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """클로바 V2 응답을 파싱하여 텍스트/좌표 블록/평균 신뢰도를 반환"""
        text_lines: List[str] = []
        confidences: List[float] = []
        blocks: List[Dict[str, Any]] = []

        try:
            images = response.get("images", []) if isinstance(response, dict) else []
            for image in images:
                fields = image.get("fields", [])
                for field in fields:
                    t = field.get("inferText", "") or ""
                    if t:
                        text_lines.append(t)
                    conf = field.get("inferConfidence", None)
                    if conf is not None:
                        try:
                            confidences.append(float(conf))
                        except Exception:
                            pass

                    # 좌표(boundingPoly → bbox) 파싱
                    poly = field.get("boundingPoly") or {}
                    verts = poly.get("vertices") or []
                    if verts:
                        try:
                            xs = [v.get("x", 0) for v in verts if isinstance(v, dict)]
                            ys = [v.get("y", 0) for v in verts if isinstance(v, dict)]
                            if xs and ys:
                                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                                blocks.append({"text": t, "bbox": (x0, y0, x1, y1)})
                        except Exception:
                            # 좌표 파싱 실패는 무시하고 진행
                            pass
        except Exception as e:
            print(f" OCR 응답 파싱 오류: {e}")

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return {
            "text": "\n".join(text_lines).strip(),
            "blocks": blocks,          # 좌표 블록 (없으면 빈 리스트)
            "confidence": avg_conf,
            "success": True,
        }
