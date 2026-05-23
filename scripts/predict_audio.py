import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from speech_digit.config import LABELS_PATH, MODEL_DIR, PREDICTION_DIR
from speech_digit.data import load_single_audio_file
from speech_digit.env import configure_tensorflow


def load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return {int(k): v for k, v in labels.items()}
    return {i: str(i) for i in range(10)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a wav file")
    parser.add_argument("--save-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    configure_tensorflow()
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Model not found. Run: python scripts/train.py")

    model = tf.keras.models.load_model(MODEL_DIR)
    labels = load_labels()
    batch, sample_rate = load_single_audio_file(args.audio)

    probabilities = model.predict(batch)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_digit = labels[predicted_index]

    result = {
        "audio": str(Path(args.audio)),
        "sample_rate": sample_rate,
        "predicted_class_index": predicted_index,
        "predicted_spoken_number": predicted_digit,
        "probabilities_by_digit": {
            labels[i]: float(probabilities[i]) for i in range(len(probabilities))
        },
    }

    print(json.dumps(result, indent=2))

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.save_json) if args.save_json else PREDICTION_DIR / "last_prediction.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved prediction to: {output_path}")


if __name__ == "__main__":
    main()
