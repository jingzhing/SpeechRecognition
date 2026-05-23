import tensorflow as tf

from speech_digit.config import MODEL_DIR
from speech_digit.data import load_datasets
from speech_digit.env import configure_tensorflow
from speech_digit.reports import evaluate_and_save


def main():
    configure_tensorflow()
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Model not found. Run: python scripts/train.py")

    _, validation_set, test_set, holdout_set, _, _ = load_datasets()
    model = tf.keras.models.load_model(MODEL_DIR)

    evaluate_and_save(model, validation_set, "validation")
    evaluate_and_save(model, test_set, "test")
    evaluate_and_save(model, holdout_set, "holdout")


if __name__ == "__main__":
    main()
