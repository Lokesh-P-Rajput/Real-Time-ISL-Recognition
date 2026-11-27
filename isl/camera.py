"""Webcam helper (OpenCV)."""
from __future__ import annotations

import sys

import cv2

from . import config


def open_camera(index: int | None = None):
    """Open the webcam. On Windows the DirectShow backend opens much faster."""
    index = config.CAMERA_INDEX if index is None else int(index)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap
