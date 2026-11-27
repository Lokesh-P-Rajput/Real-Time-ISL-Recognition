"""Turn 21 MediaPipe hand landmarks into a clean 63-number feature vector."""
from __future__ import annotations

import numpy as np

from . import config

NUM_LANDMARKS = 21
FEATURE_NAMES = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
NUM_FEATURES = len(FEATURE_NAMES)   # 63


def normalize_landmarks(points, handedness: str | None = None) -> np.ndarray:
    """Make the features independent of hand position, distance and hand side.

    1. Wrist (landmark 0) becomes the origin       -> position independent
    2. Everything is divided by the hand size      -> distance independent
    3. A left hand is mirrored into a right hand   -> one model for both hands

    `points` is anything that reshapes to (21, 3). Returns a flat array of 63 floats.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[0]
    if config.MIRROR_LEFT_HAND and handedness == "Left":
        pts[:, 0] *= -1.0
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.reshape(-1)
