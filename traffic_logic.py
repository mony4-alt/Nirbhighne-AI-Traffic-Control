"""Traffic logic for lane assignment, demand scoring, and signal advice.

YOLO runs once per frame. This module receives detections, assigns them to
lanes, calculates weighted traffic demand, tracks waiting time, and
recommends the next green lane. The signal advice is advisory only; it does
not control a physical traffic signal.
"""

from __future__ import annotations

import collections

import config


SUPPORTED_LANE_COUNTS = (1, 2, 4)

VEHICLE_WEIGHTS = {
    # The keys below match the 11 labels used by the YOLO model.  Values
    # represent relative road-space/traffic demand, not emergency priority.
    "ambulance": 1.0,
    "auto_ricksaw": 0.5,
    "bike": 0.5,
    "bus": 2.0,
    "fire_truck": 2.0,
    "leguna": 1.0,
    "pickup": 1.0,
    "police_car": 1.0,
    "private_car": 1.0,
    "truck": 2.0,
    "cng": 0.5,
}

# Detector results are displayed using friendly names from config.py, while
# callers/tests may still provide the older generic names.  Normalize those
# forms to the model's canonical class names before looking up a weight.
_VEHICLE_WEIGHT_ALIASES = {
    "motorcycle": "bike",
    "motorbike": "bike",
    "bicycle": "bike",
    "car": "private_car",
    "auto_rickshaw": "auto_ricksaw",
    "auto_ricksaw": "auto_ricksaw",
    "cng_ricksaw": "cng",
    "cng_rickshaw": "cng",
    "fire_truck": "fire_truck",
    "police_car": "police_car",
    "pickup_truck": "pickup",
}

TRAFFIC_WEIGHT = 0.7
WAITING_WEIGHT = 0.3
EMERGENCY_PRIORITY = 1000.0
SMOOTHING_ALPHA = 0.3
DEFAULT_TIME_STEP = 1.0


# ---------------------------------------------------------------------------
# Lane helpers
# ---------------------------------------------------------------------------

def normalise_lane_count(value: int | str) -> int:
    """Return one of the supported lane-zone counts."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    return value if value in SUPPORTED_LANE_COUNTS else 1


def lane_for_x(x: float, frame_width: int, lane_count: int) -> int:
    """Map an x-coordinate to a one-based, equal-width lane zone."""
    lane_count = normalise_lane_count(lane_count)
    if frame_width <= 0:
        return 1
    index = int(max(0, min(frame_width - 1, x)) * lane_count / frame_width)
    return min(lane_count, index + 1)


# ---------------------------------------------------------------------------
# Weighted traffic demand
# ---------------------------------------------------------------------------

def vehicle_weight(vehicle_type: str) -> float:
    """Return the configured weight for a raw or display vehicle class."""
    key = str(vehicle_type).lower().strip()
    key = key.replace("-", "_").replace(" ", "_")
    key = _VEHICLE_WEIGHT_ALIASES.get(key, key)
    return VEHICLE_WEIGHTS.get(key, 1.0)


def calculate_weighted_demand(
    vehicle_types_by_lane: dict[int, list[str]],
) -> dict[int, float]:
    """Calculate demand as the sum of the configured vehicle weights."""
    demand: dict[int, float] = {}
    for lane, vehicle_types in vehicle_types_by_lane.items():
        lane_demand = 0.0
        for vehicle_type in vehicle_types:
            lane_demand += vehicle_weight(vehicle_type)
        demand[lane] = lane_demand
    return demand


def demand_from_counts(lane_counts: dict[int, int]) -> dict[int, float]:
    """Use ordinary vehicle counts as demand when types are unavailable."""
    return {lane: float(max(0, count)) for lane, count in lane_counts.items()}


# ---------------------------------------------------------------------------
# Traffic smoothing
# ---------------------------------------------------------------------------

class TrafficDemandSmoother:
    """Smooth demand across frames using an exponential moving average."""

    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        self.alpha = max(0.0, min(1.0, alpha))
        self._smoothed: dict[int, float] = {}

    def update(self, current_demand: dict[int, float]) -> dict[int, float]:
        result: dict[int, float] = {}
        for lane, current_value in current_demand.items():
            previous_value = self._smoothed.get(lane, current_value)
            smoothed_value = (
                self.alpha * current_value
                + (1.0 - self.alpha) * previous_value
            )
            self._smoothed[lane] = smoothed_value
            result[lane] = smoothed_value
        return result

    def clear_lane(self, lane: int) -> None:
        self._smoothed.pop(lane, None)

    def clear_all(self) -> None:
        self._smoothed.clear()


# ---------------------------------------------------------------------------
# Waiting time tracker
# ---------------------------------------------------------------------------

class LaneWaitingTimeTracker:
    """Track waiting time for lanes that have vehicles but no green signal."""

    def __init__(self):
        self._waiting_time: dict[int, float] = {}

    def update(
        self,
        lane_counts: dict[int, int],
        signal_lane: int | None,
        elapsed_seconds: float = DEFAULT_TIME_STEP,
    ) -> dict[int, float]:
        elapsed_seconds = max(0.0, elapsed_seconds)
        for lane in lane_counts:
            count = lane_counts.get(lane, 0)
            if lane == signal_lane:
                self._waiting_time[lane] = 0.0
            elif count > 0:
                self._waiting_time[lane] = (
                    self._waiting_time.get(lane, 0.0) + elapsed_seconds
                )
            else:
                self._waiting_time[lane] = 0.0
        return dict(self._waiting_time)

    def get(self, lane: int) -> float:
        return self._waiting_time.get(lane, 0.0)

    def reset_lane(self, lane: int) -> None:
        self._waiting_time[lane] = 0.0

    def clear_all(self) -> None:
        self._waiting_time.clear()


# ---------------------------------------------------------------------------
# Priority and dynamic green time
# ---------------------------------------------------------------------------

def rank_ambulance_lanes(
    ambulance_lanes: set[int] | list[int] | tuple[int, ...],
    ambulance_counts: dict[int, int] | None = None,
    ambulance_proximity: dict[int, float] | None = None,
) -> list[int]:
    """Rank lanes containing ambulances for emergency signal priority.

    ``ambulance_proximity`` is a normalized score where 1.0 means the
    ambulance appears closest to the camera.  Proximity is deliberately the
    primary key; the number of ambulances in that lane is the tie-breaker.
    Lane number is the final deterministic tie-breaker.
    """
    ambulance_counts = ambulance_counts or {}
    ambulance_proximity = ambulance_proximity or {}

    lanes = {int(lane) for lane in ambulance_lanes}

    def _proximity(lane: int) -> float:
        try:
            return max(0.0, min(1.0, float(ambulance_proximity.get(lane, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _count(lane: int) -> int:
        try:
            return max(0, int(ambulance_counts.get(lane, 0)))
        except (TypeError, ValueError):
            return 0

    return sorted(
        lanes,
        key=lambda lane: (-_proximity(lane), -_count(lane), lane),
    )

def calculate_priority_scores(
    traffic_demand: dict[int, float],
    waiting_times: dict[int, float],
    ambulance_lanes: set[int] | None = None,
) -> dict[int, float]:
    """Calculate traffic-plus-waiting priority, with emergency override."""
    ambulance_lanes = ambulance_lanes or set()
    scores: dict[int, float] = {}
    for lane in traffic_demand:
        score = (
            TRAFFIC_WEIGHT * traffic_demand.get(lane, 0.0)
            + WAITING_WEIGHT * waiting_times.get(lane, 0.0)
        )
        if lane in ambulance_lanes:
            score += EMERGENCY_PRIORITY
        scores[lane] = score
    return scores


def compute_green_duration(
    lane_counts: dict[int, int],
    traffic_demand: dict[int, float] | None = None,
) -> dict[int, int]:
    """Compute proportional green time bounded by configured limits."""
    if traffic_demand is None:
        traffic_demand = demand_from_counts(lane_counts)

    total_demand = sum(traffic_demand.values())
    if total_demand <= 0:
        return {
            lane: config.SIGNAL_BASE_GREEN_SEC
            for lane in lane_counts
        }

    span = config.MAX_SIGNAL_GREEN_SEC - config.MIN_SIGNAL_GREEN_SEC
    result: dict[int, int] = {}
    for lane in lane_counts:
        demand = max(0.0, traffic_demand.get(lane, 0.0))
        seconds = config.MIN_SIGNAL_GREEN_SEC + round(
            (demand / total_demand) * span
        )
        result[lane] = max(
            config.MIN_SIGNAL_GREEN_SEC,
            min(config.MAX_SIGNAL_GREEN_SEC, seconds),
        )
    return result


# ---------------------------------------------------------------------------
# Ambulance direction tracker
# ---------------------------------------------------------------------------

class AmbulanceDirectionTracker:
    """Track ambulance centroid-x history and infer left/right movement."""

    def __init__(self):
        self._history: dict[int, collections.deque] = {}

    def update(self, lane: int, cx: float) -> None:
        if lane not in self._history:
            self._history[lane] = collections.deque(
                maxlen=config.AMBULANCE_DIRECTION_HISTORY
            )
        self._history[lane].append(cx)

    def direction(self, lane: int) -> str:
        hist = self._history.get(lane)
        if not hist or len(hist) < config.AMBULANCE_DIRECTION_MIN_FRAMES:
            return "unknown"

        half = len(hist) // 2
        values = list(hist)
        early = sum(values[:half]) / half
        late = sum(values[half:]) / (len(values) - half)
        delta = late - early
        if delta > 10:
            return "right"
        if delta < -10:
            return "left"
        return "unknown"

    def clear_lane(self, lane: int) -> None:
        self._history.pop(lane, None)

    def clear_all(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Signal advice and state
# ---------------------------------------------------------------------------

def traffic_signal_advice(
    lane_counts: dict[int, int],
    ambulance_lanes: set[int],
    green_durations: dict[int, int] | None = None,
    traffic_demand: dict[int, float] | None = None,
    waiting_times: dict[int, float] | None = None,
    ambulance_counts: dict[int, int] | None = None,
    ambulance_proximity: dict[int, float] | None = None,
) -> tuple[int | None, str]:
    """Choose an advisory green lane using emergency, demand, and wait time.

    For an emergency, the nearest ambulance wins first.  If the nearest
    proximity scores are equal, the lane with more ambulances wins.
    """
    green_durations = green_durations or {}
    waiting_times = waiting_times or {}

    if ambulance_lanes:
        ranked_lanes = rank_ambulance_lanes(
            ambulance_lanes,
            ambulance_counts=ambulance_counts,
            ambulance_proximity=ambulance_proximity,
        )
        lane = ranked_lanes[0]
        duration = green_durations.get(lane, config.MIN_SIGNAL_GREEN_SEC)
        count = max(1, int((ambulance_counts or {}).get(lane, 1)))
        proximity = (ambulance_proximity or {}).get(lane)
        proximity_text = (
            f"proximity={float(proximity) * 100:.0f}%"
            if proximity is not None else "nearest ambulance"
        )
        return lane, (
            f"EMERGENCY PRIORITY: GREEN Lane {lane} - "
            f"{proximity_text}, {count} ambulance(s) in lane; "
            f"all other lanes YIELD [{duration}s]"
        )

    if not lane_counts or not any(lane_counts.values()):
        return None, "No vehicles detected - keep current signal plan"

    if traffic_demand is None:
        traffic_demand = demand_from_counts(lane_counts)

    priority_scores = calculate_priority_scores(
        traffic_demand,
        waiting_times,
        ambulance_lanes,
    )
    lane = max(
        priority_scores,
        key=lambda number: (priority_scores[number], -number),
    )
    duration = green_durations.get(lane, config.SIGNAL_BASE_GREEN_SEC)
    return lane, (
        f"Suggested GREEN: Lane {lane} ({lane_counts[lane]} vehicles, "
        f"demand={traffic_demand[lane]:.1f}, "
        f"priority={priority_scores[lane]:.1f}) [{duration}s green]"
    )


def build_lane_signal_states(
    lane_counts: dict[int, int],
    signal_lane: int | None,
    ambulance_lanes: set[int],
) -> dict[int, str]:
    """Return each lane's GREEN, RED, YIELD, or EMERGENCY state."""
    states: dict[int, str] = {}
    for lane in lane_counts:
        if lane in ambulance_lanes:
            states[lane] = "EMERGENCY"
        elif lane == signal_lane and not ambulance_lanes:
            states[lane] = "GREEN"
        else:
            states[lane] = "YIELD" if ambulance_lanes else "RED"
    return states
