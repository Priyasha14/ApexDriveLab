from dataclasses import dataclass


@dataclass
class TireState:
    lateral_grip_usage: float = 0.0
    longitudinal_grip_usage: float = 0.0
    combined_grip_usage: float = 0.0

