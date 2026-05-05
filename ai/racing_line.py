import math
from dataclasses import dataclass

import numpy as np

from physics.vector_utils import vec2


@dataclass
class RacingLine:
    center: tuple[float, float]
    inner_radius: tuple[float, float]
    outer_radius: tuple[float, float]

    def point_at(self, angle: float) -> np.ndarray:
        blend = 0.56 + 0.10 * math.sin(angle * 2.0 - 0.6)
        radius_x = self.inner_radius[0] + (self.outer_radius[0] - self.inner_radius[0]) * blend
        radius_y = self.inner_radius[1] + (self.outer_radius[1] - self.inner_radius[1]) * blend
        return vec2(self.center[0] + math.cos(angle) * radius_x, self.center[1] + math.sin(angle) * radius_y)

    def target_ahead(self, current_angle: float, lookahead_distance: float) -> np.ndarray:
        average_radius = (self.inner_radius[0] + self.outer_radius[0] + self.inner_radius[1] + self.outer_radius[1]) / 4.0
        angle_step = lookahead_distance / max(average_radius, 1.0)
        return self.point_at((current_angle - angle_step) % math.tau)

    def tangent_heading(self, angle: float) -> float:
        sample_a = self.point_at(angle)
        sample_b = self.point_at((angle - 0.02) % math.tau)
        delta = sample_b - sample_a
        return math.atan2(float(delta[1]), float(delta[0]))

    def path_error(self, position: np.ndarray, angle: float) -> float:
        target = self.point_at(angle)
        return float(np.linalg.norm(position - target))

