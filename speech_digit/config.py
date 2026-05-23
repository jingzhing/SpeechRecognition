from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FILTERED_DATA_DIR = DATA_DIR / "filtered"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
REPORT_DIR = OUTPUT_DIR / "reports"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
LABELS_PATH = OUTPUT_DIR / "labels.json"

DATASET_URL = "https://drive.google.com/uc?export=download&id=1cJnMi4u5AM6an3julR-T0saKUTXSwZ0j"
BATCH_SIZE = 32
SAMPLE_RATE = 8000
OUTPUT_SEQUENCE_LENGTH = 16000
VALIDATION_SPLIT = 0.1
SEED = 0
NUM_CLASSES = 10
