"""Application-wide paths, image settings, thresholds, and model names."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Storage paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
MODELS_DIR = PROJECT_ROOT / "models"

# Image processing
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# Detection and recognition thresholds
FACE_DETECTION_CONFIDENCE = 0.60
FACE_RECOGNITION_THRESHOLD = 0.60
LIVENESS_CONFIDENCE_THRESHOLD = 0.70
EMOTION_CONFIDENCE_THRESHOLD = 0.50
ATTENDANCE_DUPLICATE_WINDOW_SECONDS = 60

# Model identifiers and filenames
FACE_DETECTION_MODEL_NAME = "opencv_face_detector"
FACE_DETECTION_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_NMS_THRESHOLD = 0.30
FACE_MIN_SIZE = 40
API_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
API_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
FACE_RECOGNITION_MODEL_NAME = "face_recognition"
GENDER_MODEL_PATH = MODELS_DIR / "gender.h5"
GENDER_MODEL_NAME = "gender_classifier"
AGE_MODEL_PATH = MODELS_DIR / "age.h5"
AGE_MODEL_NAME = "age_estimator"
AGE_VALIDATION_MAE_LIMIT = 10.0
EMOTION_MODEL_NAME = "emotion_classifier"
LIVENESS_MODEL_NAME = "liveness_detector"
