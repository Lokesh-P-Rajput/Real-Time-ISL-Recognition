"""Adding, changing and removing signs: dataset, database, training and the live pipeline."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import collect_data
from isl import config, guide
from isl import database as db
from isl.features import FEATURE_NAMES
from isl.pipeline import LivePipeline
from isl.recognizer import GestureRecognizer
from tests.helpers import make_dataset
from train_model import train


class FakeTracker:
    def __init__(self, sequence):
        self.sequence = list(sequence)

    def process(self, frame, draw=True):
        return frame, self.sequence.pop(0)

    def close(self):
        pass


class SignManagementTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        paths = {"ASSETS_DIR": self.root, "SIGN_IMAGE_DIR": self.root / "signs",
                 "SIGN_GUIDE_CSV": self.root / "sign_guide.csv", "DATASET_CSV": self.root / "dataset.csv",
                 "DB_PATH": self.root / "t.db"}
        for name, value in paths.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(config.GESTURES, {"Hello": "Hello", "Yes": "Yes", "No": "No"}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        guide.ensure_guide_csv(dict(config.GESTURES))
        db.init_db()
        db.seed_gestures(config.GESTURES)
        make_dataset(config.DATASET_CSV, per_class=40)

    def test_rename_label_updates_dataset_database_and_sheet(self):
        db.add_dataset_samples("Hello", [np.zeros(63)] * 3)
        collect_data.rename_label("Hello", "Namaste")
        labels = set(pd.read_csv(config.DATASET_CSV)["label"])
        self.assertIn("Namaste", labels)
        self.assertNotIn("Hello", labels)
        self.assertIn("Namaste", config.load_gestures())
        self.assertEqual(db.count_dataset_samples(), 3)
        with db.get_conn() as conn:
            names = {r["gesture_name"] for r in conn.execute("SELECT gesture_name FROM gestures")}
            self.assertIn("Namaste", names)
            self.assertNotIn("Hello", names)

    def test_rename_to_a_name_with_samples_is_refused(self):
        with self.assertRaises(SystemExit):
            collect_data.rename_label("Hello", "Yes")

    def test_delete_label_removes_samples_only(self):
        collect_data.delete_label("No")
        self.assertNotIn("No", set(pd.read_csv(config.DATASET_CSV)["label"]))
        self.assertIn("No", config.load_gestures())            # the sign itself stays in the sheet

    def test_training_ignores_samples_of_signs_removed_from_the_sheet(self):
        data = pd.read_csv(config.DATASET_CSV)
        extra = data[data["label"] == "No"].copy()
        extra["label"] = "Removed Sign"
        pd.concat([data, extra]).to_csv(config.DATASET_CSV, index=False)
        out = train(config.DATASET_CSV, self.root / "m.joblib", self.root / "rep", cv=0,
                    model_choice="knn", verbose=False)
        self.assertNotIn("Removed Sign", out["classes"])
        self.assertIn("No", out["classes"])

    def test_live_pipeline_ignores_a_removed_sign_of_an_old_model(self):
        train(config.DATASET_CSV, self.root / "m.joblib", self.root / "rep", cv=0, model_choice="knn", verbose=False)
        recognizer = GestureRecognizer(self.root / "m.joblib")
        data = pd.read_csv(config.DATASET_CSV)
        pick = lambda label: data[data["label"] == label][FEATURE_NAMES].iloc[1].to_numpy(dtype=np.float32)
        sequence = [pick("No")] * 8 + [None] * 6 + [pick("Hello")] * 8
        with mock.patch.dict(config.GESTURES, {"Hello": "Hello", "Yes": "Yes"}, clear=True):   # "No" was removed
            pipeline = LivePipeline(recognizer, FakeTracker(sequence), threshold=0.5, stable_frames=4)
            pipeline.stabilizer.min_gap = 0.0
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            committed = [r.committed for r in (pipeline.process(frame) for _ in sequence) if r.committed]
        self.assertEqual(committed, ["Hello"])


if __name__ == "__main__":
    unittest.main()
