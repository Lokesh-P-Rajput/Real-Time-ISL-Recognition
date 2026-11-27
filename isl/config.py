"""Central settings for the whole project.

Change values here instead of editing the other files.
The list of signs is NOT here: it lives in assets/sign_guide.csv (edit it in Excel).
"""
import csv
from pathlib import Path

# ----------------------------------------------------------------- folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
SIGN_IMAGE_DIR = ASSETS_DIR / "signs"           # one photo per sign, e.g. "Thank You.jpg"
for _folder in (DATA_DIR, MODEL_DIR, REPORT_DIR, SIGN_IMAGE_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

DATASET_CSV = DATA_DIR / "dataset.csv"          # landmark samples used for training
DB_PATH = DATA_DIR / "isl_history.db"           # SQLite database
SIGN_GUIDE_CSV = ASSETS_DIR / "sign_guide.csv"  # YOUR sign list: gesture, spoken_text, description
MODEL_PATH = MODEL_DIR / "isl_model.joblib"     # trained classifier
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"   # MediaPipe hand model
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ------------------------------------------------------------------- signs
# An extra "no sign" class (relaxed hand / random movement). It is never spoken,
# but it stops the model from forcing a wrong sign when you are not signing.
NEUTRAL_LABEL = "Neutral"

# Used only to create assets/sign_guide.csv the first time. After that the CSV is the master list.
DEFAULT_GESTURES = {
    "Hello": "Hello",
    "Thank You": "Thank you",
    "Yes": "Yes",
    "No": "No",
    "Help": "Help",
    "Sorry": "Sorry",
    "Please": "Please",
    "Stop": "Stop",
    "Welcome": "Welcome",
    "Water": "Water",
    "Food": "Food",
    "Good": "Good",
    "Bad": "Bad",
    "Come": "Come",
    "Go": "Go",
}


def load_gestures(path=None) -> dict:
    """sign name -> text that is shown and spoken, read from assets/sign_guide.csv.

    Falls back to DEFAULT_GESTURES when the file is missing or unreadable.
    """
    path = Path(path or SIGN_GUIDE_CSV)
    if not path.exists():
        return dict(DEFAULT_GESTURES)
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        print(f"Warning: could not read {path.name} ({exc}). Using the built-in sign list.")
        return dict(DEFAULT_GESTURES)
    gestures = {}
    for row in rows:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items() if k}
        name = row.get("gesture", "")
        if not name:
            continue
        if name.lower() == NEUTRAL_LABEL.lower():
            print(f"Warning: '{NEUTRAL_LABEL}' is reserved for the no-sign class and is ignored in the sign list.")
            continue
        gestures[name] = row.get("spoken_text") or name
    return gestures or dict(DEFAULT_GESTURES)


GESTURES = load_gestures()
COLLECT_LABELS = list(GESTURES) + [NEUTRAL_LABEL]


def reload_gestures() -> dict:
    """Re-read assets/sign_guide.csv into GESTURES (same dict object, so every module sees it)."""
    fresh = load_gestures()
    GESTURES.clear()
    GESTURES.update(fresh)
    return GESTURES


# ------------------------------------------------------------------ camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --------------------------------------------------------------- MediaPipe
MIN_DETECTION_CONFIDENCE = 0.6
MIN_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
# Mirror left-hand landmarks so one model works for left- and right-handed signers.
MIRROR_LEFT_HAND = True

# ----------------------------------------------------------- data collection
SAMPLES_PER_GESTURE = 300       # samples recorded per sign per run
CAPTURE_EVERY_N_FRAMES = 2      # skip frames so samples are not identical

# ---------------------------------------------------------------- training
# Model that train_model.py saves: "XGBoost", "Random Forest", "SVM", "KNN" or "auto"
# (auto = most accurate; the fastest one if several are within 1 % of each other).
# All four are always trained and compared in reports/model_comparison.csv.
DEFAULT_MODEL = "XGBoost"
TEST_SIZE = 0.20                # 80:20 train/test split
RANDOM_STATE = 42

# ------------------------------------------------------- live recognition
CONFIDENCE_THRESHOLD = 0.80     # below this the prediction is ignored
STABLE_FRAMES = 6               # same sign must be seen this many frames in a row
MIN_GAP_SECONDS = 1.0           # minimum time between two accepted signs
HAND_ABSENT_FRAMES = 5          # frames without a hand before the same sign may repeat

# ------------------------------------------------------------ text-to-speech
TTS_RATE = 160
TTS_VOLUME = 1.0
TTS_VOICE_CONTAINS = ""         # e.g. "Zira", "David" or "Heera" to pick an installed Windows voice
