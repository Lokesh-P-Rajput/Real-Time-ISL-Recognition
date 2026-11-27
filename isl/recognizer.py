"""Loads the trained model and predicts a gesture from a feature vector."""
from __future__ import annotations

import joblib
import numpy as np

from . import config


class GestureRecognizer:
    def __init__(self, model_path=None):
        bundle = joblib.load(model_path or config.MODEL_PATH)
        self.model = bundle["model"]              # scaler + classifier pipeline
        self.encoder = bundle["label_encoder"]    # number -> gesture name
        self.meta = bundle.get("meta", {})

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.encoder.classes_]

    def predict(self, features) -> tuple[str, float]:
        """Return (gesture label, confidence between 0 and 1)."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        best = int(np.argmax(proba))
        label = self.encoder.inverse_transform([self.model.classes_[best]])[0]
        return str(label), float(proba[best])
