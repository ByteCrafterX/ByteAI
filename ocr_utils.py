"""
ocr_utils.py  –  OCR para etiquetas usando *somente* PaddleOCR
"""
from __future__ import annotations
import os, json, logging, threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR                  # ← única dependência de OCR

# ─────────────── infra de metadados (mantida p/ app.py) ────────────────
_META = "ocr_metadata.json"
_lock = threading.Lock()

def carrega_ocr_metadata() -> dict:
    if os.path.isfile(_META):
        try:
            return json.load(open(_META, encoding="utf-8"))
        except Exception:
            log.warning("JSON inválido – recriando")
    return {}

def salva_ocr_metadata(d: dict):
    try:
        json.dump(d, open(_META, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Erro ao gravar JSON: %s", e)

# ─────────────── logger ────────────────────────────────────────────────
log = logging.getLogger("ocr")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:ocr:%(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ─────────────── PaddleOCR singleton ───────────────────────────────────
from functools import lru_cache
@lru_cache(maxsize=1)
def _get_paddle() -> PaddleOCR:
    log.info("Carregando PaddleOCR (CPU)…")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
    log.info("PaddleOCR pronto ✔")
    return ocr

# ─────────────── localizar etiqueta e melhorar contraste ───────────────
def _extrai_roi_etiqueta(img_bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    bw = cv2.adaptiveThreshold(clahe.apply(gray), 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    best, best_area = None, 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:                 # precisa ser retângulo
            continue
        area = cv2.contourArea(approx)
        if area < 0.01 * w * h:
            continue
        x, y, cw, ch = cv2.boundingRect(approx)
        ar = cw / float(ch)
        if 0.8 < ar < 4.0 and area > best_area:
            if cv2.mean(gray, mask=cv2.drawContours(
                np.zeros_like(gray), [approx], -1, 255, -1))[0] > 150:
                best, best_area = approx, area

    if best is None:                         # nada encontrado → usa tudo
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        pts = best.reshape(4, 2).astype(np.float32)
        s = pts.sum(1); diff = np.diff(pts, axis=1)
        ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                            pts[np.argmax(s)], pts[np.argmax(diff)]])
        (tl, tr, br, bl) = ordered
        dst_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        dst_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        M = cv2.getPerspectiveTransform(
            ordered, np.array([[0, 0], [dst_w-1, 0],
                               [dst_w-1, dst_h-1], [0, dst_h-1]], dtype="float32"))
        warp = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h))
        pil = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))

    return ImageEnhance.Contrast(pil).enhance(1.8)   # leve boost contraste

# ─────────────── função pública chamada pelo app ───────────────────────
def extrai_texto_e_indexa(path_img: str) -> str:
    abs_path = os.path.abspath(path_img)
    img_bgr = cv2.imread(abs_path)
    if img_bgr is None:
        log.warning("cv2.imread falhou em %s", abs_path)
        return ""

    pil_roi = _extrai_roi_etiqueta(img_bgr)
    ocr_res = _get_paddle().ocr(np.array(pil_roi), cls=True)

    # Desempacota se veio embrulhado em lista extra
    if len(ocr_res) == 1 and isinstance(ocr_res[0], list):
        ocr_res = ocr_res[0]

    textos = []
    for item in ocr_res:
        if len(item) >= 2 and isinstance(item[1], (list, tuple)):
            txt = item[1][0]
            if isinstance(txt, str):
                textos.append(txt)

    texto = " ".join(textos).strip()
    log.info("PaddleOCR → %d chars", len(texto))

    with _lock:
        meta = carrega_ocr_metadata()
        meta[abs_path] = texto
        salva_ocr_metadata(meta)

    log.info("%s – %d chars final", Path(abs_path).name, len(texto))
    return texto
