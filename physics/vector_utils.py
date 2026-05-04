import math

import numpy as np


def vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=float)


def from_angle(angle: float) -> np.ndarray:
    return vec2(math.cos(angle), math.sin(angle))


def length(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = length(vector)
    if magnitude <= 1e-9:
        return vec2(0.0, 0.0)
    return vector / magnitude


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rotate_points(points: list[tuple[float, float]], angle: float, origin: np.ndarray) -> list[tuple[float, float]]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for x, y in points:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        rotated.append((float(origin[0] + rx), float(origin[1] + ry)))
    return rotated

