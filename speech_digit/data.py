import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from speech_digit.config import (
    BATCH_SIZE,
    DATA_DIR,
    DATASET_URL,
    FILTERED_DATA_DIR,
    LABELS_PATH,
    OUTPUT_SEQUENCE_LENGTH,
    RAW_DATA_DIR,
    SEED,
    VALIDATION_SPLIT,
)


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_and_prepare_dataset(force=False):
    ensure_dirs()

    existing_wavs = list(FILTERED_DATA_DIR.glob("*/*.wav"))
    if existing_wavs and not force:
        print(f"Dataset already exists: {FILTERED_DATA_DIR}")
        print(f"Found {len(existing_wavs)} wav files.")
        return FILTERED_DATA_DIR

    if force and FILTERED_DATA_DIR.exists():
        shutil.rmtree(FILTERED_DATA_DIR)
    FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Spoken Digit Dataset...")
    archive_path = tf.keras.utils.get_file(
        fname="audio_dataset.zip",
        origin=DATASET_URL,
        extract=True,
        cache_dir=str(RAW_DATA_DIR),
        cache_subdir=".",
    )
    print(f"Downloaded archive to: {archive_path}")

    candidate_roots = [
        RAW_DATA_DIR / "Spoken Digit Dataset" / "data",
        RAW_DATA_DIR / "data",
        RAW_DATA_DIR / "audio_dataset" / "data",
    ]

    source_dir = None
    for candidate in candidate_roots:
        if candidate.exists() and list(candidate.glob("*.wav")):
            source_dir = candidate
            break

    if source_dir is None:
        wavs = list(RAW_DATA_DIR.rglob("*.wav"))
        if not wavs:
            raise FileNotFoundError(
                "No wav files were found after downloading. If Google Drive blocks the download, "
                "manually download the dataset zip from the URL in speech_digit/config.py and extract "
                "the wav files into data/raw before rerunning this script."
            )
        source_dir = wavs[0].parent

    print(f"Preparing files from: {source_dir}")
    for digit in range(10):
        class_dir = FILTERED_DATA_DIR / str(digit)
        class_dir.mkdir(parents=True, exist_ok=True)
        files = glob.glob(str(source_dir / f"{digit}_*.wav"))
        for file_path in files:
            src = Path(file_path)
            dst = class_dir / src.name
            shutil.copy2(src, dst)

    final_wavs = list(FILTERED_DATA_DIR.glob("*/*.wav"))
    if not final_wavs:
        raise RuntimeError("Dataset preparation finished, but no wav files were copied.")

    counts = {str(digit): len(list((FILTERED_DATA_DIR / str(digit)).glob("*.wav"))) for digit in range(10)}
    print("Prepared dataset:")
    for digit, count in counts.items():
        print(f" - {digit}: {count} files")
    print(f"Total: {sum(counts.values())} wav files")
    return FILTERED_DATA_DIR


def eliminate_channel(data):
    return tf.squeeze(data, axis=-1)


def get_spectrogram(waveform):
    spectrogram = tf.abs(tf.signal.stft(waveform, frame_length=255, frame_step=128))
    return spectrogram[..., tf.newaxis]


def preprocess_audio_tensor(data):
    return get_spectrogram(eliminate_channel(data))


def preprocess_with_labels(data, labels):
    return preprocess_audio_tensor(data), labels


def load_datasets():
    if not FILTERED_DATA_DIR.exists() or not list(FILTERED_DATA_DIR.glob("*/*.wav")):
        raise FileNotFoundError("Dataset not found. Run: python scripts/download_dataset.py")

    train_ds, holdout_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=FILTERED_DATA_DIR,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        subset="both",
    )

    label_names = np.array(train_ds.class_names)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({int(i): str(label) for i, label in enumerate(label_names)}, f, indent=2)

    train_ds = train_ds.map(preprocess_with_labels, num_parallel_calls=tf.data.AUTOTUNE)
    holdout_ds = holdout_ds.map(preprocess_with_labels, num_parallel_calls=tf.data.AUTOTUNE)

    test_ds = holdout_ds.shard(num_shards=2, index=0)
    val_ds = holdout_ds.shard(num_shards=2, index=1)

    train_set = train_ds.cache().shuffle(3000, seed=SEED).prefetch(tf.data.AUTOTUNE)
    validation_set = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_set = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    holdout_set = holdout_ds.cache().prefetch(tf.data.AUTOTUNE)

    sample_spectrogram, _ = next(iter(train_set.take(1)))
    input_shape = sample_spectrogram.shape[1:]

    return train_set, validation_set, test_set, holdout_set, input_shape, label_names


def load_single_audio_file(audio_path):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    contents = tf.io.read_file(str(audio_path))
    data, sample_rate = tf.audio.decode_wav(
        contents,
        desired_channels=1,
        desired_samples=OUTPUT_SEQUENCE_LENGTH,
    )
    spectrogram = preprocess_audio_tensor(data)
    return spectrogram[tf.newaxis, ...], int(sample_rate.numpy())
