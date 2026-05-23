import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from speech_digit.config import NUM_CLASSES, REPORT_DIR


def save_history(history):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = REPORT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"epoch": history.epoch, "history": history.history}, f, indent=2)

    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="Train Set")
    plt.plot(history.epoch, history.history["val_loss"], label="Validate Set")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(history.epoch, 100 * np.array(history.history["accuracy"]), label="Train Set")
    plt.plot(history.epoch, 100 * np.array(history.history["val_accuracy"]), label="Validate Set")
    plt.legend()
    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "accuracy_curve.png", dpi=160)
    plt.close()

    print(f"Saved training reports to: {REPORT_DIR}")


def predict_dataset(model, dataset):
    probabilities = model.predict(dataset)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = tf.concat(list(dataset.map(lambda spec, label: label)), axis=0).numpy()
    return y_true, y_pred, probabilities


def save_confusion_matrix(matrix, path, title):
    visible_matrix = matrix.copy()
    for i in range(NUM_CLASSES):
        visible_matrix[i][i] = 0

    plt.figure(figsize=(7, 7))
    plt.imshow(visible_matrix, cmap="Pastel1_r")
    plt.xticks(np.arange(NUM_CLASSES))
    plt.yticks(np.arange(NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, int(visible_matrix[i, j]), ha="center", va="center")
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate_and_save(model, dataset, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    loss, accuracy = model.evaluate(dataset, verbose=0)
    y_true, y_pred, probabilities = predict_dataset(model, dataset)
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()

    metrics = {
        "dataset": name,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "sample_count": int(len(y_true)),
        "correct_count": int(np.sum(y_true == y_pred)),
        "wrong_count": int(np.sum(y_true != y_pred)),
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
    }

    metrics_path = REPORT_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    matrix_path = REPORT_DIR / f"{name}_confusion_matrix.png"
    save_confusion_matrix(matrix, matrix_path, f"Incorrect predictions made on {name}")

    print(f"{name} accuracy: {accuracy:.4f}")
    print(f"Saved {name} metrics to: {metrics_path}")
    print(f"Saved {name} confusion matrix to: {matrix_path}")
    return metrics
