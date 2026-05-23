# Spoken Digit Recognition, Windows Version

This repo trains a spoken digit recognition model on Windows using the main DNN model from the original project.

The expected output classes are digits `0` to `9`, representing spoken numbers zero to nine.

## What this repo does

1. Downloads the Spoken Digit Dataset from the Google Drive URL used in the original project.
2. Reorganises files into Keras-compatible folders:

```text
data/filtered/0/*.wav
data/filtered/1/*.wav
...
data/filtered/9/*.wav
```

3. Loads audio with `tf.keras.utils.audio_dataset_from_directory`.
4. Pads/truncates each sample to 16,000 samples, matching 2 seconds at 8 kHz.
5. Converts audio to STFT spectrograms.
6. Trains the main DNN model with progressive SGD.
7. Saves the trained model and reports.
8. Tests a single `.wav` file and returns the predicted digit plus probabilities.
9. Allows you to build a Docker inference image after local training.

## Windows setup

Use Python 3.10.

Run:

```bat
setup_windows.bat
```

This creates `.venv` and installs:

```text
tensorflow==2.10.1
numpy<1.24
matplotlib
```

TensorFlow 2.10.1 is used because it is the last practical TensorFlow line for native Windows GPU support. If your CUDA/cuDNN setup is not configured, it will still run on CPU.

## Run everything

```bat
run_all_windows.bat
```

This runs:

```bat
python scripts\download_dataset.py
python scripts\train.py
python scripts\evaluate_model.py
python scripts\predict_audio.py --audio data\filtered\5\5_A_0.wav
```

## Run steps manually

Download dataset:

```bat
download_dataset_windows.bat
```

Train:

```bat
train_windows.bat
```

Evaluate:

```bat
evaluate_windows.bat
```

Predict sample:

```bat
predict_sample_windows.bat
```

Predict your own audio:

```bat
.venv\Scripts\activate.bat
python scripts\predict_audio.py --audio path\to\your_audio.wav
```

## Main model

The repo only includes the main DNN model.

Architecture:

```text
Input spectrogram
Normalization
Resize 128 x 64
Flatten
Dense 2048 ReLU
Dense 2048 ReLU
Dense 88 ReLU
Dropout 0.25
Dense 256 ReLU
Dense 128 ReLU
Dropout 0.25
Dense 10 Softmax
```

Training schedule:

```text
SGD lr=0.1   for 5 epochs
SGD lr=0.05  for 5 epochs
SGD lr=0.01  for 5 epochs
SGD lr=0.005 for 5 epochs
```

## Outputs

After training, check:

```text
outputs/model/
outputs/labels.json
outputs/reports/training_history.json
outputs/reports/loss_curve.png
outputs/reports/accuracy_curve.png
outputs/reports/validation_metrics.json
outputs/reports/validation_confusion_matrix.png
outputs/reports/test_metrics.json
outputs/reports/test_confusion_matrix.png
outputs/reports/holdout_metrics.json
outputs/reports/holdout_confusion_matrix.png
outputs/predictions/last_prediction.json
```

Single-audio prediction output looks like:

```json
{
  "predicted_class_index": 5,
  "predicted_spoken_number": "5",
  "probabilities_by_digit": {
    "0": 0.001,
    "1": 0.002,
    "2": 0.001,
    "3": 0.004,
    "4": 0.003,
    "5": 0.980,
    "6": 0.002,
    "7": 0.001,
    "8": 0.004,
    "9": 0.002
  }
}
```

## Docker inference after local training

Train locally first. Then build the inference image:

```bat
build_inference_image_windows.bat
```

Run prediction through Docker:

```bat
run_docker_predict_windows.bat data\filtered\5\5_A_0.wav
```

The Docker image is for inference after training. Training is intended to happen locally on Windows.

## Notes on CUDA on Windows

This repo is Windows-compatible and avoids Linux-only packages like `tensorflow[and-cuda]`.

If TensorFlow detects your GPU, the training script will print the GPU device. If not, it will train on CPU. The dataset is small, so CPU training is still workable.
