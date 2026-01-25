FROM python:3.11-slim

WORKDIR /app

# النظام الأساسي
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir --upgrade pip

# الحزم الثقيلة أولاً
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# الحزم المتوسطة
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    Pillow==10.1.0

# الحزم الخفيفة
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0

# نسخ التطبيق
COPY *.py .
COPY utils/ ./utils/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]