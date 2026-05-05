from dataclasses import dataclass


@dataclass
class TireState:
    temperature: float = 70.0
    wear: float = 0.0
    condition_grip_multiplier: float = 1.0
    front_slip_angle: float = 0.0
    rear_slip_angle: float = 0.0
    front_lateral_force: float = 0.0
    rear_lateral_force: float = 0.0
    front_longitudinal_force: float = 0.0
    rear_longitudinal_force: float = 0.0
    front_load: float = 0.0
    rear_load: float = 0.0
    lateral_grip_usage: float = 0.0
    longitudinal_grip_usage: float = 0.0
    combined_grip_usage: float = 0.0
    front_grip_usage: float = 0.0
    rear_grip_usage: float = 0.0
