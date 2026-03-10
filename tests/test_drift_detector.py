"""
Tests for Page-Hinkley drift detector.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.core.drift_detector import PageHinkleyDetector


class TestPageHinkleyDetector(unittest.TestCase):
    """Test Page-Hinkley drift detection."""

    def setUp(self):
        self.detector = PageHinkleyDetector(
            delta=0.005, threshold=50, warmup=20
        )

    def test_no_drift_stable(self):
        """Stable stream should not trigger drift."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            drift = self.detector.update(0.3 + rng.randn() * 0.01)
            # Should not drift on stable data with small noise
        self.assertFalse(drift)

    def test_drift_on_shift(self):
        """Large distribution shift should trigger drift."""
        detector = PageHinkleyDetector(delta=0.005, threshold=10, warmup=10)
        rng = np.random.RandomState(42)

        # Stable phase
        for _ in range(30):
            detector.update(0.3 + rng.randn() * 0.01)

        # Shift phase
        detected = False
        for _ in range(100):
            d = detector.update(2.0 + rng.randn() * 0.01)
            if d:
                detected = True
                break

        self.assertTrue(detected)

    def test_reset(self):
        """Reset should clear all state."""
        for _ in range(50):
            self.detector.update(1.0)
        self.detector.reset()
        self.assertEqual(self.detector.n, 0)
        self.assertEqual(self.detector.m_t, 0.0)
        self.assertEqual(self.detector.M_t, 0.0)

    def test_warmup_suppresses_drift(self):
        """No drift should be detected during warmup period."""
        detector = PageHinkleyDetector(
            delta=0.001, threshold=1, warmup=50
        )
        # Even with extreme values, warmup should suppress detection
        for i in range(50):
            drift = detector.update(100.0)
            self.assertFalse(drift, f"Drift falsely detected at step {i}")

    def test_history_tracking(self):
        """Detector should track history of values and statistics."""
        for i in range(10):
            self.detector.update(float(i) * 0.1)
        self.assertEqual(len(self.detector.history), 10)

    def test_multiple_drifts(self):
        """Detector should be able to detect multiple drifts after reset."""
        detector = PageHinkleyDetector(
            delta=0.005, threshold=5, warmup=5
        )

        drifts = 0
        for epoch in range(3):
            # Stable
            for _ in range(20):
                detector.update(0.3)
            # Shift
            for _ in range(50):
                d = detector.update(5.0)
                if d:
                    drifts += 1
                    detector.reset()
                    break

        self.assertGreaterEqual(drifts, 2)


if __name__ == "__main__":
    unittest.main()
