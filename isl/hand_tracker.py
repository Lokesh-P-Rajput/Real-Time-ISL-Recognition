"""Hand detection and landmark extraction with MediaPipe.

Newer MediaPipe versions removed the old `mp.solutions.hands` API, so this project
uses the current Tasks API (`HandLandmarker`). It needs a small model file
(hand_landmarker.task, about 8 MB) which is downloaded automatically the first
time and kept in the `models/` folder.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import config
from .features import normalize_landmarks

# The 21 standard connections of a MediaPipe hand (0 = wrist, 4/8/12/16/20 = fingertips).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
FINGERTIPS = (4, 8, 12, 16, 20)


def ensure_hand_model(path=None) -> Path:
    """Return the path of the MediaPipe hand model, downloading it if needed."""
    path = Path(path or config.HAND_MODEL_PATH)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading the MediaPipe hand model (one time, ~8 MB) to {path} ...")
    part = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(config.HAND_MODEL_URL, part)
        part.replace(path)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the hand model (check your internet connection). "
            f"Download it manually from\n  {config.HAND_MODEL_URL}\n"
            f"and save the file as\n  {path}"
        ) from exc
    return path


def draw_hand(frame_bgr: np.ndarray, points: np.ndarray) -> None:
    """Draw the hand skeleton on the frame (in place)."""
    h, w = frame_bgr.shape[:2]
    px = [(int(p[0] * w), int(p[1] * h)) for p in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, px[a], px[b], (255, 190, 40), 2, cv2.LINE_AA)
    for i, p in enumerate(px):
        cv2.circle(frame_bgr, p, 6 if i in FINGERTIPS else 4, (60, 60, 255), -1, cv2.LINE_AA)


class HandTracker:
    """Detects one hand per frame and returns its normalised 63-value feature vector."""

    def __init__(self, model_path=None, max_hands: int = 1):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_file = ensure_hand_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.VIDEO,       # tracks the hand between frames
            num_hands=max_hands,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = 0

    def process(self, frame_bgr: np.ndarray, draw: bool = True, mirror: bool = True):
        """Return (frame, features).  `features` is None when no hand is visible.

        The frame is mirrored (selfie view) so that movements feel natural and
        MediaPipe reports Left/Right correctly. Collection and live recognition
        both use this same method, so training and prediction always match.
        """
        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        if timestamp_ms <= self._last_ts:          # MediaPipe needs strictly increasing times
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return frame_bgr, None

        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        handedness = result.handedness[0][0].category_name if result.handedness else None
        if draw:
            draw_hand(frame_bgr, points)
        return frame_bgr, normalize_landmarks(points, handedness)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
