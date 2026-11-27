"""One frame in -> detected hand, predicted gesture and accepted gesture out.

Kept free of any Streamlit code so it can be tested without a browser or camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from . import config
from .stabilizer import GestureStabilizer


@dataclass
class FrameResult:
    frame: np.ndarray            # mirrored frame with the hand skeleton drawn
    hand_present: bool
    label: str | None            # raw prediction for this frame
    confidence: float
    committed: str | None        # gesture accepted on this frame (else None)
    latency_ms: float            # frame -> prediction time


class LivePipeline:
    def __init__(self, recognizer, tracker, threshold=None, stable_frames=None,
                 draw_landmarks=True):
        self.recognizer = recognizer
        self.tracker = tracker
        self.draw_landmarks = draw_landmarks
        self.stabilizer = GestureStabilizer(
            stable_frames=config.STABLE_FRAMES if stable_frames is None else stable_frames,
            threshold=config.CONFIDENCE_THRESHOLD if threshold is None else threshold,
            min_gap=config.MIN_GAP_SECONDS,
            absent_frames=config.HAND_ABSENT_FRAMES,
            ignore_labels={config.NEUTRAL_LABEL},
        )

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        start = perf_counter()
        frame, features = self.tracker.process(frame_bgr, draw=self.draw_landmarks)
        label, confidence = None, 0.0
        if features is not None:
            label, confidence = self.recognizer.predict(features)
        latency_ms = (perf_counter() - start) * 1000.0
        # A sign that was removed from the sign list (old model) is treated like "no sign".
        known = label is None or label == config.NEUTRAL_LABEL or label in config.GESTURES
        committed = self.stabilizer.update(label if known else config.NEUTRAL_LABEL,
                                           confidence, features is not None)
        return FrameResult(frame, features is not None, label, confidence, committed, latency_ms)

    def close(self) -> None:
        self.tracker.close()
