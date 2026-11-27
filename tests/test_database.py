import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl import config
from isl import database as db


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(config, "DB_PATH", Path(self.tmp.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        db.init_db()
        db.seed_gestures({"Hello": "Hello", "Thank You": "Thank you"})

    def test_user_is_created_once(self):
        self.assertEqual(db.get_or_create_user("Ayaj"), db.get_or_create_user("Ayaj"))

    def test_translation_history_round_trip(self):
        sid = db.start_session(db.get_or_create_user("Lokesh"))
        db.add_translation(sid, "Thank You", "Thank you", 0.934)
        db.end_session(sid)
        history = db.get_history()
        self.assertEqual(len(history), 1)
        row = history.iloc[0]
        self.assertEqual((row["Gesture"], row["Translated text"], row["User"]), ("Thank You", "Thank you", "Lokesh"))
        self.assertAlmostEqual(row["Confidence %"], 93.4)

    def test_history_can_be_cleared(self):
        sid = db.start_session(db.get_or_create_user("A"))
        db.add_translation(sid, "Hello", "Hello", 0.9)
        db.clear_history()
        self.assertTrue(db.get_history().empty)

    def test_unknown_gesture_is_added_on_the_fly(self):
        sid = db.start_session(db.get_or_create_user("A"))
        db.add_translation(sid, "Brand New", "Brand new", 0.9)
        self.assertEqual(db.get_history().iloc[0]["Gesture"], "Brand New")

    def test_dataset_samples(self):
        db.add_dataset_samples("Hello", [np.zeros(63), np.ones(63)])
        self.assertEqual(db.count_dataset_samples(), 2)
        db.delete_dataset_samples("Hello")
        self.assertEqual(db.count_dataset_samples(), 0)

    def test_foreign_keys_are_enforced(self):
        with self.assertRaises(Exception):
            db.add_translation(999, "Hello", "Hello", 0.9)


if __name__ == "__main__":
    unittest.main()
