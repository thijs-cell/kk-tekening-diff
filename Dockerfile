FROM python:3.11-slim

# Systeemafhankelijkheden: poppler voor pdf2image, libgl voor OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies installeren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Applicatiecode kopiëren
COPY app/ ./app/

# Niet als root draaien
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Uvicorn starten met ruime timeout voor grote PDF's
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
