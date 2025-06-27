# ──────────────────────────────────────────────────────────────
#  ByteAI • container Flask + CLIP/FAISS + OCR (CPU-only)
# ──────────────────────────────────────────────────────────────
FROM python:3.8-slim

# ——— 1. pacotes de sistema (ImageMagick + ExifTool + Tesseract) ———
#      • tesseract-ocr-ENG/ITA/POR  → treinar etiquetas em 🇬🇧 🇮🇹 🇧🇷
#      • libtesseract-dev          → bindings C (requisito de pytesseract)
#      • libgl1 + libglib2.0-0     → dependências do OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev libssl-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    exiftool imagemagick \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-ita tesseract-ocr-por \
    libtesseract-dev \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# (opcional) tornar o path do tessdata explícito
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# ——— 2. diretório de trabalho ———
WORKDIR /app

# ——— 3. dependências Python (cache-friendly) ———
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ——— 4. cópia da aplicação ———
COPY . /app/

# ——— 5. porta exposta ———
EXPOSE 5000

# ——— 6. comando default ———
CMD ["python", "app.py"]
