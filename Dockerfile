FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements-inference.txt /app/requirements-inference.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-inference.txt

COPY speech_digit /app/speech_digit
COPY outputs/model /app/outputs/model
COPY outputs/labels.json /app/outputs/labels.json
COPY app.py /app/app.py

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]