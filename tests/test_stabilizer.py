import unittest

from isl.stabilizer import GestureStabilizer


def make(**kw):
    args = dict(stable_frames=3, threshold=0.8, min_gap=0.0, absent_frames=2, ignore_labels={"Neutral"})
    args.update(kw)
    return GestureStabilizer(**args)


class StabilizerTests(unittest.TestCase):
    def test_commits_after_stable_frames(self):
        s = make()
        self.assertEqual([s.update("Hello", 0.9) for _ in range(3)], [None, None, "Hello"])

    def test_low_confidence_never_commits(self):
        s = make()
        self.assertTrue(all(s.update("Hello", 0.5) is None for _ in range(10)))

    def test_flickering_label_never_commits(self):
        s = make()
        outs = [s.update(lab, 0.95) for lab in ["Hello", "Yes"] * 5]
        self.assertTrue(all(o is None for o in outs))

    def test_no_repeat_while_held(self):
        s = make()
        outs = [s.update("Hello", 0.9) for _ in range(12)]
        self.assertEqual(outs.count("Hello"), 1)

    def test_repeat_after_hand_removed(self):
        s = make()
        [s.update("Hello", 0.9) for _ in range(3)]
        [s.update(None, 0.0, hand_present=False) for _ in range(2)]
        self.assertEqual([s.update("Hello", 0.9) for _ in range(3)][-1], "Hello")

    def test_different_gesture_commits(self):
        s = make()
        [s.update("Hello", 0.9) for _ in range(3)]
        self.assertEqual([s.update("Yes", 0.9) for _ in range(3)][-1], "Yes")

    def test_neutral_is_never_accepted_but_resets_repeat(self):
        s = make()
        [s.update("Hello", 0.9) for _ in range(3)]
        self.assertTrue(all(s.update("Neutral", 0.99) is None for _ in range(5)))
        self.assertEqual([s.update("Hello", 0.9) for _ in range(3)][-1], "Hello")

    def test_minimum_gap_between_gestures(self):
        s = make(min_gap=1.0)
        self.assertEqual([s.update("Hello", 0.9, now=0.0) for _ in range(3)][-1], "Hello")
        self.assertIsNone(s.update("Yes", 0.9, now=0.2))
        self.assertIsNone(s.update("Yes", 0.9, now=0.3))
        self.assertIsNone(s.update("Yes", 0.9, now=0.4))
        self.assertEqual(s.update("Yes", 0.9, now=1.5), "Yes")


if __name__ == "__main__":
    unittest.main()
