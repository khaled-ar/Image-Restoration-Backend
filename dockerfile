FROM python:3.11-slim

WORKDIR /app

# تحديث النظام
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# نسخ requirements
COPY requirements.txt .

# تثبيت pip وتحديثه
RUN pip install --upgrade pip setuptools wheel

# تثبيت torch أولاً (نسخة CPU خفيفة)
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# تثبيت باقي الحزم
RUN pip install -r requirements.txt --no-cache-dir

# نسخ ملفات التطبيق
COPY *.py .
COPY utils/ ./utils/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]