FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements-inference.txt /app/requirements-inference.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements-inference.txt

COPY speech_digit /app/speech_digit
COPY scripts/predict_audio.py /app/scripts/predict_audio.py
COPY outputs/model /app/outputs/model
COPY outputs/labels.json /app/outputs/labels.json

ENTRYPOINT ["python", "scripts/predict_audio.py"]