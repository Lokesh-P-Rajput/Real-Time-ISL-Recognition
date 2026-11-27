"""Synthetic hand data so the tests need no webcam."""
from unittest import mock

import numpy as np
import pandas as pd

from isl import config
from isl.features import FEATURE_NAMES, normalize_landmarks

LABELS = ["Hello", "Yes", "No", "Neutral"]
TEST_GESTURES = {"Hello": "Hello", "Yes": "Yes", "No": "No"}


def use_test_gestures():
    """Run a test with a fixed sign list, whatever the student has put in assets/sign_guide.csv."""
    return mock.patch.dict(config.GESTURES, TEST_GESTURES, clear=True)


def make_dataset(path, per_class=60, seed=0):
    """Every label gets its own random 'hand shape' plus small noise, in 3 sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in LABELS:
        base = rng.random((21, 3))
        for i in range(per_class):
            features = normalize_landmarks(base + rng.normal(0, 0.01, (21, 3)))
            rows.append([label, f"s{i % 3}", *features])
    pd.DataFrame(rows, columns=["label", "session", *FEATURE_NAMES]).to_csv(path, index=False)
    return path
