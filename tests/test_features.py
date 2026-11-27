import unittest

import numpy as np

from isl.features import NUM_FEATURES, normalize_landmarks


def hand(seed=0):
    return np.random.default_rng(seed).random((21, 3)).astype(np.float32)


class FeatureTests(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(normalize_landmarks(hand()).shape, (NUM_FEATURES,))

    def test_wrist_is_origin(self):
        self.assertTrue(np.allclose(normalize_landmarks(hand()).reshape(21, 3)[0], 0))

    def test_position_independent(self):
        h = hand()
        self.assertTrue(np.allclose(normalize_landmarks(h), normalize_landmarks(h + [0.3, -0.2, 0.1]), atol=1e-5))

    def test_distance_independent(self):
        h = hand()
        self.assertTrue(np.allclose(normalize_landmarks(h), normalize_landmarks(h * 2.5), atol=1e-5))

    def test_left_hand_is_mirrored(self):
        h = hand()
        right = normalize_landmarks(h, "Right").reshape(21, 3)
        left = normalize_landmarks(h, "Left").reshape(21, 3)
        self.assertTrue(np.allclose(left[:, 0], -right[:, 0]))
        self.assertTrue(np.allclose(left[:, 1:], right[:, 1:]))

    def test_degenerate_hand_does_not_crash(self):
        self.assertTrue(np.isfinite(normalize_landmarks(np.zeros((21, 3)))).all())


if __name__ == "__main__":
    unittest.main()
