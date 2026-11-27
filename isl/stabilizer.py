"""Decides when a gesture is "real" so the text/voice is not repeated every frame."""
from __future__ import annotations

import time
from collections import deque


class GestureStabilizer:
    """Accepts a gesture only when it is confident, stable and new.

    A gesture is accepted when
      * its confidence is >= threshold,
      * the same label was predicted for `stable_frames` frames in a row,
      * it is not the gesture that was just accepted (the hand must leave the
        screen, or a different/neutral pose must appear, before it can repeat),
      * at least `min_gap` seconds passed since the last accepted gesture.
    """

    def __init__(self, stable_frames=6, threshold=0.8, min_gap=1.0,
                 absent_frames=5, ignore_labels=()):
        self.threshold = threshold
        self.min_gap = min_gap
        self.absent_frames = absent_frames
        self.ignore_labels = set(ignore_labels)
        self._buffer = deque(maxlen=max(1, int(stable_frames)))
        self.reset()

    def reset(self) -> None:
        self._buffer.clear()
        self._last_committed = None
        self._last_time = float("-inf")
        self._absent = 0

    def update(self, label, confidence, hand_present=True, now=None):
        """Feed one frame. Returns the accepted label, or None."""
        now = time.monotonic() if now is None else now

        # No hand (or the "Neutral" no-sign pose) counts as "hand absent".
        if not hand_present or label is None or label in self.ignore_labels:
            self._buffer.clear()
            self._absent += 1
            if self._absent >= self.absent_frames:
                self._last_committed = None
            return None

        self._absent = 0
        self._buffer.append(label if confidence >= self.threshold else None)
        if len(self._buffer) < self._buffer.maxlen:
            return None
        first = self._buffer[0]
        if first is None or any(item != first for item in self._buffer):
            return None
        if first == self._last_committed:
            return None
        if now - self._last_time < self.min_gap:
            return None

        self._last_committed = first
        self._last_time = now
        self._buffer.clear()
        return first
