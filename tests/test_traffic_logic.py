import unittest

from traffic_logic import (
    AmbulanceDirectionTracker,
    build_lane_signal_states,
    calculate_priority_scores,
    calculate_weighted_demand,
    compute_green_duration,
    LaneWaitingTimeTracker,
    lane_for_x,
    normalise_lane_count,
    TrafficDemandSmoother,
    traffic_signal_advice,
    rank_ambulance_lanes,
    VEHICLE_WEIGHTS,
)


class TrafficLogicTests(unittest.TestCase):
    def test_lane_count_and_boundaries(self):
        self.assertEqual(normalise_lane_count("4"), 4)
        self.assertEqual(normalise_lane_count(3), 1)
        self.assertEqual(lane_for_x(0, 100, 4), 1)
        self.assertEqual(lane_for_x(99, 100, 4), 4)

    def test_empty_counts_receive_base_duration(self):
        self.assertEqual(compute_green_duration({1: 0, 2: 0}), {1: 20, 2: 20})

    def test_busiest_lane_and_tie_break(self):
        lane, _ = traffic_signal_advice({1: 7, 2: 10}, set())
        self.assertEqual(lane, 2)
        lane, _ = traffic_signal_advice({1: 10, 2: 10}, set())
        self.assertEqual(lane, 1)

    def test_emergency_overrides_normal_priority(self):
        lane, _ = traffic_signal_advice({1: 100, 2: 1}, {2})
        self.assertEqual(lane, 2)
        self.assertEqual(
            build_lane_signal_states({1: 100, 2: 1}, 2, {2}),
            {1: "YIELD", 2: "EMERGENCY"},
        )

    def test_ambulance_priority_uses_nearest_then_lane_count(self):
        ranked = rank_ambulance_lanes(
            {1, 2, 3},
            ambulance_counts={1: 5, 2: 1, 3: 3},
            ambulance_proximity={1: 0.80, 2: 0.95, 3: 0.80},
        )
        self.assertEqual(ranked, [2, 1, 3])

        lane, advice = traffic_signal_advice(
            {1: 2, 2: 1},
            {1, 2},
            ambulance_counts={1: 2, 2: 1},
            ambulance_proximity={1: 0.75, 2: 0.75},
        )
        self.assertEqual(lane, 1)
        self.assertIn("2 ambulance(s) in lane", advice)

    def test_ambulance_direction(self):
        tracker = AmbulanceDirectionTracker()
        for value in (5, 20, 45, 80):
            tracker.update(1, value)
        self.assertEqual(tracker.direction(1), "right")

    def test_weighted_demand_and_priority(self):
        demand = calculate_weighted_demand({1: ["car", "bus", "motorcycle"]})
        self.assertEqual(demand, {1: 3.5})
        scores = calculate_priority_scores(demand, {1: 10.0, 2: 100.0})
        self.assertAlmostEqual(scores[1], 5.45)

    def test_all_model_classes_have_weights_and_display_names_are_integrated(self):
        model_classes = {
            "ambulance", "auto_ricksaw", "bike", "bus", "fire_truck",
            "leguna", "pickup", "police_car", "private_car", "truck", "cng",
        }
        self.assertEqual(set(VEHICLE_WEIGHTS), model_classes)

        demand = calculate_weighted_demand({
            1: [
                "Ambulance", "Auto Ricksaw", "Motorcycle", "Bus",
                "Fire Truck", "Leguna", "Pickup Truck", "Police Car",
                "Private Car", "Truck", "CNG Ricksaw",
            ]
        })
        self.assertEqual(demand, {1: 12.5})

    def test_smoothing_and_waiting_time(self):
        smoother = TrafficDemandSmoother(alpha=0.5)
        self.assertEqual(smoother.update({1: 10.0})[1], 10.0)
        self.assertEqual(smoother.update({1: 0.0})[1], 5.0)

        tracker = LaneWaitingTimeTracker()
        self.assertEqual(tracker.update({1: 3, 2: 0}, signal_lane=1, elapsed_seconds=2), {1: 0.0, 2: 0.0})
        self.assertEqual(tracker.update({1: 3, 2: 2}, signal_lane=1, elapsed_seconds=2)[2], 2.0)


if __name__ == "__main__":
    unittest.main()
