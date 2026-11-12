import unittest

from detector import VehicleDetector


class DetectorConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.detector = VehicleDetector()

    def test_ambulance_stays_high_when_slider_is_low(self):
        self.detector.confidence = 0.15

        self.assertFalse(self.detector._passes_confidence("ambulance", 0.59, 0.15))
        self.assertTrue(self.detector._passes_confidence("ambulance", 0.60, 0.15))

    def test_ambulance_stays_fixed_when_slider_is_high(self):
        self.detector.confidence = 0.85

        self.assertFalse(self.detector._passes_confidence("ambulance", 0.59, 0.85))
        self.assertTrue(self.detector._passes_confidence("ambulance", 0.60, 0.85))

    def test_normal_vehicle_classes_follow_slider(self):
        self.detector.confidence = 0.40

        self.assertFalse(self.detector._passes_confidence("car", 0.39, 0.40))
        self.assertTrue(self.detector._passes_confidence("car", 0.40, 0.40))

    def test_ambulance_proximity_prefers_lower_and_larger_box(self):
        far_score = self.detector._ambulance_proximity_score(
            (100, 100, 160, 160), (720, 1280, 3)
        )
        near_score = self.detector._ambulance_proximity_score(
            (100, 400, 400, 700), (720, 1280, 3)
        )
        self.assertGreater(near_score, far_score)


if __name__ == "__main__":
    unittest.main()
