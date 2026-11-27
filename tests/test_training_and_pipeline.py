import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from isl.features import FEATURE_NAMES
from isl.pipeline import LivePipeline
from isl.recognizer import GestureRecognizer
from tests.helpers import LABELS, make_dataset, use_test_gestures
from train_model import resolve_model_choice, train


class FakeTracker:
    """Pretends to be MediaPipe: returns the next prepared feature vector (or None)."""

    def __init__(self, sequence):
        self.sequence = list(sequence)

    def process(self, frame, draw=True):
        return frame, self.sequence.pop(0)

    def close(self):
        pass


class TrainingAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import xgboost  # noqa: F401
        except Exception:
            raise unittest.SkipTest("xgboost is not installed (pip install xgboost)")
        cls.gestures_patch = use_test_gestures()
        cls.gestures_patch.start()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.root = root
        cls.csv = make_dataset(root / "dataset.csv")
        cls.model_path = root / "model.joblib"
        cls.summary = train(cls.csv, cls.model_path, root / "reports", cv=3, verbose=False)   # default = XGBoost

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        cls.gestures_patch.stop()

    def test_xgboost_is_the_default_and_all_four_models_are_compared(self):
        self.assertEqual(self.summary["best_model"], "XGBoost")
        self.assertEqual({m["Model"] for m in self.summary["models"]}, {"XGBoost", "Random Forest", "SVM", "KNN"})
        self.assertGreater(self.summary["test_accuracy"], 0.95)
        for name in ("model_comparison.csv", "classification_report.txt", "confusion_matrix.png", "metrics.json"):
            self.assertTrue((self.root / "reports" / name).exists(), name)

    def test_recognizer_loads_the_xgboost_model_and_predicts_each_class(self):
        recognizer = GestureRecognizer(self.model_path)
        self.assertEqual(recognizer.meta["model_name"], "XGBoost")
        df = pd.read_csv(self.csv)
        self.assertEqual(set(recognizer.classes), set(LABELS))
        for label in LABELS:
            features = df[df["label"] == label][FEATURE_NAMES].iloc[0].to_numpy()
            predicted, confidence = recognizer.predict(features)
            self.assertEqual(predicted, label)
            self.assertTrue(0.0 <= confidence <= 1.0)

    def test_other_models_can_be_chosen(self):
        for choice, expected in (("knn", "KNN"), ("rf", "Random Forest"), ("svm", "SVM")):
            out = train(self.csv, self.root / f"{choice}.joblib", self.root / f"rep_{choice}",
                        cv=0, model_choice=choice, verbose=False)
            self.assertEqual(out["best_model"], expected)
            self.assertEqual(GestureRecognizer(self.root / f"{choice}.joblib").meta["model_name"], expected)

    def test_auto_picks_one_of_the_trained_models(self):
        out = train(self.csv, self.root / "auto.joblib", self.root / "rep_auto", cv=0, model_choice="auto", verbose=False)
        self.assertIn(out["best_model"], {"XGBoost", "Random Forest", "SVM", "KNN"})

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_choice("neural-net")

    def test_session_split_is_evaluation_only_when_save_model_false(self):
        target = self.root / "should_not_exist.joblib"
        out = train(self.csv, target, self.root / "rep_strict", split_by_session=True, cv=0,
                    save_model=False, verbose=False)
        self.assertEqual(out["split"], "session-held-out")
        self.assertFalse(target.exists())
        self.assertTrue((self.root / "rep_strict" / "model_comparison.csv").exists())

    def test_too_few_samples_is_rejected(self):
        small = self.root / "small.csv"
        pd.read_csv(self.csv).groupby("label").head(5).to_csv(small, index=False)
        with self.assertRaises(ValueError):
            train(small, self.root / "m3.joblib", self.root / "r3", verbose=False)

    def test_live_pipeline_speaks_once_per_gesture(self):
        recognizer = GestureRecognizer(self.model_path)
        df = pd.read_csv(self.csv)
        pick = lambda label: df[df["label"] == label][FEATURE_NAMES].iloc[1].to_numpy(dtype=np.float32)
        # Hello x8, hand leaves x6, Yes x8, Neutral x6, Hello x8
        sequence = [pick("Hello")] * 8 + [None] * 6 + [pick("Yes")] * 8 + [pick("Neutral")] * 6 + [pick("Hello")] * 8
        pipeline = LivePipeline(recognizer, FakeTracker(sequence), threshold=0.5, stable_frames=4)
        pipeline.stabilizer.min_gap = 0.0
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        committed = [r.committed for r in (pipeline.process(frame) for _ in sequence) if r.committed]
        self.assertEqual(committed, ["Hello", "Yes", "Hello"])


class XGBoostMissingTests(unittest.TestCase):
    """What happens when XGBoost (or its Windows C++ runtime) cannot be loaded."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = make_dataset(self.root / "d.csv")
        for patcher in (mock.patch.dict(sys.modules, {"xgboost": None}),    # makes "import xgboost" fail
                        use_test_gestures()):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_choosing_xgboost_gives_a_clear_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            train(self.csv, self.root / "m.joblib", self.root / "r", cv=0, model_choice="xgboost", verbose=False)
        self.assertIn("pip install xgboost", str(ctx.exception))
        self.assertIn("Visual C++", str(ctx.exception))

    def test_other_models_still_train_without_xgboost(self):
        out = train(self.csv, self.root / "m.joblib", self.root / "r", cv=0, model_choice="knn", verbose=False)
        self.assertEqual(out["best_model"], "KNN")
        self.assertNotIn("XGBoost", {m["Model"] for m in out["models"]})


if __name__ == "__main__":
    unittest.main()
