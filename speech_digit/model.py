import tensorflow as tf
from tensorflow.keras import layers as kLayers
from tensorflow.keras import models as kModels

from speech_digit.config import NUM_CLASSES


def build_dnn_model(input_shape, train_set):
    norm_layer = kLayers.Normalization()
    norm_layer.adapt(train_set.map(lambda spec, label: spec))

    model = kModels.Sequential([
        kLayers.Input(shape=input_shape),
        norm_layer,
        kLayers.Resizing(128, 64),
        kLayers.Flatten(),
        kLayers.Dense(2048, activation="relu"),
        kLayers.Dense(2048, activation="relu"),
        kLayers.Dense(88, activation="relu"),
        kLayers.Dropout(0.25),
        kLayers.Dense(256, activation="relu"),
        kLayers.Dense(128, activation="relu"),
        kLayers.Dropout(0.25),
        kLayers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


class CombinedHistory:
    def __init__(self):
        self.epoch = []
        self.history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

    def add(self, history):
        offset = len(self.epoch)
        self.epoch.extend([epoch + offset for epoch in history.epoch])
        for key in self.history:
            self.history[key].extend(history.history.get(key, []))


def train_progressive_sgd(model, train_set, validation_set):
    schedule = [
        (0.1, 5),
        (0.05, 5),
        (0.01, 5),
        (0.005, 5),
    ]

    combined = CombinedHistory()
    for learning_rate, epochs in schedule:
        print(f"Training with SGD learning_rate={learning_rate}, epochs={epochs}")
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history = model.fit(
            train_set,
            validation_data=validation_set,
            epochs=epochs,
        )
        combined.add(history)
    return combined
