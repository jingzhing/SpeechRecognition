import tensorflow as tf


def configure_tensorflow():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print("TensorFlow GPU devices detected:")
        for gpu in gpus:
            print(" -", gpu)
    else:
        print("No TensorFlow GPU detected. Training will run on CPU.")
    return gpus
