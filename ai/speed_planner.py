import math
from dataclasses import dataclass

from physics.vector_utils import clamp


@dataclass
class SpeedPlan:
    target_speed_kmh: float
    upcoming_curvature: float
    braking_distance: float


def ellipse_curvature(angle: float, radius: tuple[float, float]) -> float:
    a, b = radius
    numerator = a * b
    denominator = ((b * math.cos(angle)) ** 2 + (a * math.sin(angle)) ** 2) ** 1.5
    return numerator / max(denominator, 1e-6)


def plan_speed(angle: float, speed_kmh: float, outer_radius: tuple[float, float], corner_speed_multiplier: float, braking_safety_margin: float) -> SpeedPlan:
    lookahead_angle = (angle - 0.42) % math.tau
    curvature = ellipse_curvature(lookahead_angle, outer_radius)
    curvature_norm = clamp((curvature - 0.0018) / 0.0035, 0.0, 1.0)
    target = (150.0 - 72.0 * curvature_norm) * corner_speed_multiplier
    speed_mps = speed_kmh / 3.6
    target_mps = target / 3.6
    braking_distance = max(0.0, (speed_mps * speed_mps - target_mps * target_mps) / (2.0 * 13.0)) * braking_safety_margin
    return SpeedPlan(target_speed_kmh=target, upcoming_curvature=curvature, braking_distance=braking_distance)

