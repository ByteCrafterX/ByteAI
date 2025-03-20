# Usar uma imagem base com Python 3.8
FROM python:3.8-slim

# Instalar dependências do sistema e ImageMagick
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    exiftool \
    imagemagick \
 && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho
WORKDIR /app

# Copiar apenas o requirements.txt primeiro para aproveitar o cache do Docker
COPY requirements.txt /app/

# Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos da aplicação
COPY . /app/

# Dar permissão de execução ao script convert.sh (caso esteja em /app)
#RUN  chmod +x /app/convert.sh

# Expor a porta 5000
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["python", "app.py"]

