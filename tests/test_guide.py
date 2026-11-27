import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl import config, guide


class GuideBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name, value in {"ASSETS_DIR": self.root, "SIGN_IMAGE_DIR": self.root / "signs",
                            "SIGN_GUIDE_CSV": self.root / "sign_guide.csv"}.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(config.GESTURES, {"Hello": "Hello", "Yes": "Yes"}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.frame = np.full((48, 64, 3), 120, np.uint8)


class PhotoTests(GuideBase):
    def test_photo_is_saved_found_and_not_overwritten(self):
        self.assertIsNone(guide.find_sign_image("Thank You"))
        path = guide.save_reference_photo("Thank You", self.frame)
        self.assertEqual(path.name, "Thank You.jpg")
        self.assertEqual(guide.find_sign_image("Thank You"), path)
        self.assertIsNone(guide.save_reference_photo("Thank You", self.frame))          # kept
        self.assertIsNotNone(guide.save_reference_photo("Thank You", self.frame, overwrite=True))

    def test_own_png_photo_is_kept(self):
        config.SIGN_IMAGE_DIR.mkdir(parents=True)
        own = config.SIGN_IMAGE_DIR / "Yes.png"
        own.write_bytes(b"fake")
        self.assertEqual(guide.find_sign_image("Yes"), own)
        self.assertIsNone(guide.save_reference_photo("Yes", self.frame))

    def test_overwrite_replaces_a_photo_with_another_extension(self):
        config.SIGN_IMAGE_DIR.mkdir(parents=True)
        (config.SIGN_IMAGE_DIR / "Yes.png").write_bytes(b"fake")
        guide.save_reference_photo("Yes", self.frame, overwrite=True)
        self.assertEqual(guide.find_sign_image("Yes").suffix, ".jpg")
        self.assertFalse((config.SIGN_IMAGE_DIR / "Yes.png").exists())

    def test_unsafe_characters_do_not_escape_the_folder(self):
        self.assertEqual(guide.save_reference_photo("../evil", self.frame).parent, config.SIGN_IMAGE_DIR)


class SignSheetTests(GuideBase):
    def test_template_has_three_columns_and_keeps_user_edits(self):
        guide.ensure_guide_csv({"Hello": "Hello there", "Yes": "Yes"})
        self.assertEqual(config.load_gestures(), {"Hello": "Hello there", "Yes": "Yes"})
        config.SIGN_GUIDE_CSV.write_text("gesture,spoken_text,description\nHello,Namaste,Open palm\n", encoding="utf-8")
        guide.ensure_guide_csv({"Hello": "x", "Yes": "y"})                     # must not overwrite
        self.assertEqual(config.load_gestures(), {"Hello": "Namaste"})
        self.assertEqual(guide.load_descriptions(), {"Hello": "Open palm"})

    def test_old_two_column_file_is_upgraded(self):
        config.SIGN_GUIDE_CSV.write_text("gesture,description\nHello,Open palm\n", encoding="utf-8")
        guide.ensure_guide_csv()
        self.assertEqual(config.load_gestures(), {"Hello": "Hello"})
        self.assertEqual(guide.load_descriptions(), {"Hello": "Open palm"})

    def test_loader_is_forgiving(self):
        self.assertEqual(config.load_gestures(self.root / "missing.csv"), config.DEFAULT_GESTURES)
        path = self.root / "x.csv"
        path.write_text("", encoding="utf-8")
        self.assertEqual(config.load_gestures(path), config.DEFAULT_GESTURES)
        # Excel style: BOM, odd header case, blank rows, blank spoken text, reserved name
        path.write_text("\ufeffGesture , Spoken_Text,Description\nHello,,x\n,,\nNeutral,No sign,y\n Water ,I need water,z\n",
                        encoding="utf-8")
        self.assertEqual(config.load_gestures(path), {"Hello": "Hello", "Water": "I need water"})

    def test_reload_gestures_sees_edits_in_the_same_dict(self):
        same_object = config.GESTURES
        config.SIGN_GUIDE_CSV.write_text("gesture,spoken_text,description\nA,Alpha,\nB,Beta,\n", encoding="utf-8")
        config.reload_gestures()
        self.assertIs(config.GESTURES, same_object)
        self.assertEqual(dict(config.GESTURES), {"A": "Alpha", "B": "Beta"})

    def test_add_sign(self):
        guide.ensure_guide_csv({"Hello": "Hello", "Yes": "Yes"})
        self.assertTrue(guide.add_sign("Thank You", "Thank you"))
        self.assertFalse(guide.add_sign("Thank You"))
        self.assertEqual(config.load_gestures()["Thank You"], "Thank you")
        self.assertEqual(config.GESTURES["Thank You"], "Thank you")
        with self.assertRaises(ValueError):
            guide.add_sign("neutral")
        with self.assertRaises(ValueError):
            guide.add_sign("   ")

    def test_rename_sign_keeps_order_text_and_photo(self):
        guide.ensure_guide_csv({"Hello": "Hello", "Yes": "Yes"})
        guide.save_reference_photo("Hello", self.frame)
        guide.rename_sign("Hello", "Namaste")
        self.assertEqual(list(config.load_gestures()), ["Namaste", "Yes"])
        self.assertEqual(list(config.GESTURES), ["Namaste", "Yes"])
        self.assertIsNotNone(guide.find_sign_image("Namaste"))
        self.assertIsNone(guide.find_sign_image("Hello"))
        with self.assertRaises(ValueError):
            guide.rename_sign("Namaste", "Yes")
        with self.assertRaises(ValueError):
            guide.rename_sign("Unknown", "Other")


if __name__ == "__main__":
    unittest.main()
