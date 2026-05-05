import math

import numpy as np

from physics.vector_utils import clamp


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    return angle


def steering_to_target(car, target: np.ndarray) -> tuple[float, float]:
    delta = target - car.position
    target_heading = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = wrap_angle(target_heading - car.heading)
    steer = clamp(heading_error / 0.75, -1.0, 1.0)
    return steer, heading_error

