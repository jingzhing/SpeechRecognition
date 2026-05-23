import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from speech_digit.config import LABELS_PATH, MODEL_DIR

app = FastAPI(title="Spoken Digit Recognition API")

model = tf.keras.models.load_model(MODEL_DIR)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)


def get_spectrogram(waveform):
    spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=255, frame_step=128))
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def preprocess_audio_file(file_path):
    contents = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(
        contents,
        desired_channels=1,
        desired_samples=16000
    )

    audio = tf.squeeze(audio, axis=-1)
    spectrogram = get_spectrogram(audio)

    return spectrogram[tf.newaxis, ...]


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Spoken Digit Recognition API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None,
        "num_labels": len(labels)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio_tensor = preprocess_audio_file(tmp_path)
        probabilities = model.predict(audio_tensor, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))

        if isinstance(labels, dict):
            label_lookup = {int(k): str(v) for k, v in labels.items()}
        else:
            label_lookup = {i: str(v) for i, v in enumerate(labels)}

        predicted_digit = label_lookup.get(predicted_index, str(predicted_index))

        return JSONResponse({
            "predicted_digit": predicted_digit,
            "predicted_class_index": predicted_index,
            "probabilities": {
                label_lookup.get(i, str(i)): float(probabilities[i])
                for i in range(len(probabilities))
            }
        })
    finally:
        os.remove(tmp_path)