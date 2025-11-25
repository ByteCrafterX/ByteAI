# ==========================================================
#   ByteAI – Flask + CLIP/FAISS + OCR (CPU-only, amd64)
#   Imagem otimizada para NAS / deploys leves
#   - PyTorch CPU-only
#   - PaddleOCR / PaddlePaddle
#   - ImageMagick p/ conversor seguro
# ==========================================================

# ---------- Etapa 1: builder (instala dependências Python) ----------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

WORKDIR /app

# Dependências de build (somente no builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      swig \
      libffi-dev libssl-dev \
      libxml2-dev libxslt1-dev zlib1g-dev \
      # deps comuns que também ajudam wheel build (Pillow, etc.)
      libjpeg62-turbo-dev libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (melhor uso de cache)
COPY requirements.txt .

# Atualiza o pip e instala deps (CPU-only)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ---------- Etapa 2: runtime (apenas o necessário para rodar) ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    # ===== Conversor: defaults seguros (pode sobrescrever no docker run) =====
    CONV_WORKERS_DEFAULT=1 \
    CONV_WORKERS_MAX=1 \
    CONV_SAFE_LONG_SIDE=2400 \
    CONV_HUGE_PIXELS=60000000 \
    CONV_HUGE_BYTES=150000000 \
    CONV_IM_BIN=magick \
    CONV_IM_LIMIT_MEM=512MiB \
    CONV_IM_LIMIT_MAP=1GiB \
    CONV_IM_LIMIT_DISK=2GiB

WORKDIR /app

# Somente libs de runtime (sem toolchain). Inclui ImageMagick + Tesseract + libs gráficas básicas.
RUN apt-get update && apt-get install -y --no-install-recommends \
      # libs básicas que Pillow/OpenCV/PyMuPDF geralmente precisam
      libgl1 libglib2.0-0 \
      libjpeg62-turbo libpng16-16 \
      libopenblas0 libgomp1 \
      # OCR/EXIF
      libimage-exiftool-perl \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-ita tesseract-ocr-por \
      # Fallback pesado e seguro para PSD/PSB/TIFF no conversor
      imagemagick \
      # (opcional, mas ajuda IM a ler formatos: heif/jp2/tiff/webp)
      libheif1 libde265-0 libopenjp2-7 libtiff6 libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Copia dependências Python já instaladas do builder (Python 3.11!)
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código do projeto
COPY . .

# Porta do Flask
EXPOSE 5000

# Comando padrão
CMD ["python", "app.py"]
