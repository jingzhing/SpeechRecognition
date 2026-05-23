from speech_digit.config import MODEL_DIR
from speech_digit.data import download_and_prepare_dataset, load_datasets
from speech_digit.env import configure_tensorflow
from speech_digit.model import build_dnn_model, train_progressive_sgd
from speech_digit.reports import evaluate_and_save, save_history


def main():
    configure_tensorflow()
    download_and_prepare_dataset(force=False)
    train_set, validation_set, test_set, holdout_set, input_shape, label_names = load_datasets()

    print("Labels:", list(label_names))
    print("Input shape:", input_shape)

    model = build_dnn_model(input_shape, train_set)
    model.summary()

    history = train_progressive_sgd(model, train_set, validation_set)
    save_history(history)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR)
    print(f"Saved trained model to: {MODEL_DIR}")

    print("Evaluating after training...")
    evaluate_and_save(model, validation_set, "validation")
    evaluate_and_save(model, test_set, "test")
    evaluate_and_save(model, holdout_set, "holdout")


if __name__ == "__main__":
    main()
