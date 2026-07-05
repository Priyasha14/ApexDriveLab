"""Advanced tire model: Pacejka Magic Formula, load sensitivity, per-axle thermals.

The lateral force curve follows the Magic Formula:

    F(alpha) = -D * sin(C * atan(B*alpha - E*(B*alpha - atan(B*alpha))))

where D is the peak force (grip limit for the axle), and B is derived from the
setup's cornering stiffness so the linear region matches the previous linear
tire model exactly. Past the peak slip angle the force falls away, which gives
the car a realistic breakaway: push beyond the limit and grip *drops*, instead
of plateauing forever.
"""

import math
from dataclasses import dataclass

from config import (
    TIRE_LOAD_SENSITIVITY,
    TIRE_MF_SHAPE_C,
    TIRE_MF_CURVATURE_E,
    TIRE_OPTIMAL_TEMP_LOW,
    TIRE_OPTIMAL_TEMP_HIGH,
)


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
    # --- new channels (advanced model) ---
    front_temperature: float = 70.0
    rear_temperature: float = 70.0
    front_wear: float = 0.0
    rear_wear: float = 0.0
    front_peak_slip: float = 0.0
    rear_peak_slip: float = 0.0
    front_saturation: float = 0.0  # how far past the peak of the MF curve (0..1+)
    rear_saturation: float = 0.0


def magic_formula(slip_angle: float, cornering_stiffness: float, peak_force: float,
                  shape_c: float = TIRE_MF_SHAPE_C, curvature_e: float = TIRE_MF_CURVATURE_E) -> float:
    """Pacejka lateral force. Linear region matches -cornering_stiffness * slip."""
    if peak_force <= 1e-6:
        return 0.0
    stiffness_b = cornering_stiffness / max(shape_c * peak_force, 1e-6)
    x = stiffness_b * slip_angle
    inner = x - curvature_e * (x - math.atan(x))
    return -peak_force * math.sin(shape_c * math.atan(inner))


def peak_slip_angle(cornering_stiffness: float, peak_force: float,
                    shape_c: float = TIRE_MF_SHAPE_C) -> float:
    """Slip angle at which the Magic Formula reaches its peak (E=0 approximation)."""
    if cornering_stiffness <= 1e-6 or peak_force <= 1e-6:
        return 0.0
    stiffness_b = cornering_stiffness / (shape_c * peak_force)
    return math.tan(math.pi / (2.0 * shape_c)) / stiffness_b


def load_sensitivity_multiplier(load_fraction: float) -> float:
    """Real tires lose friction coefficient as vertical load rises.

    Normalised so a 50/50 axle carries multiplier 1.0; the heavier axle gains
    less grip than a linear model would predict, the lighter one loses less.
    """
    return max(0.6, 1.0 - TIRE_LOAD_SENSITIVITY * (load_fraction - 0.5))


def temperature_grip_multiplier(temperature: float) -> float:
    """Grip vs temperature: cold rubber is greasy, overheated rubber melts."""
    if temperature < TIRE_OPTIMAL_TEMP_LOW:
        return 0.45 + max(0.0, temperature - 20.0) / (TIRE_OPTIMAL_TEMP_LOW - 20.0) * 0.55
    if temperature <= TIRE_OPTIMAL_TEMP_HIGH:
        return 1.0
    return max(0.70, 1.0 - (temperature - TIRE_OPTIMAL_TEMP_HIGH) / 55.0 * 0.30)


def wear_grip_multiplier(wear: float) -> float:
    return max(0.38, 1.0 - wear * 0.62)
