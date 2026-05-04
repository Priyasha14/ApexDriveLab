import math

import pygame

from config import (
    BOUNDARY_COLOR,
    CHECKPOINT_COUNT,
    GRASS_COLOR,
    HEIGHT,
    START_FINISH_COLOR,
    TRACK_CENTER,
    TRACK_COLOR,
    TRACK_INNER_RADIUS,
    TRACK_OUTER_RADIUS,
    WIDTH,
)


class Track:
    def __init__(self) -> None:
        self.center = TRACK_CENTER
        self.outer_radius = TRACK_OUTER_RADIUS
        self.inner_radius = TRACK_INNER_RADIUS

    def normalized_radius(self, position) -> float:
        dx = (float(position[0]) - self.center[0]) / self.outer_radius[0]
        dy = (float(position[1]) - self.center[1]) / self.outer_radius[1]
        return math.sqrt(dx * dx + dy * dy)

    def inner_normalized_radius(self, position) -> float:
        dx = (float(position[0]) - self.center[0]) / self.inner_radius[0]
        dy = (float(position[1]) - self.center[1]) / self.inner_radius[1]
        return math.sqrt(dx * dx + dy * dy)

    def is_on_track(self, position) -> bool:
        return self.normalized_radius(position) <= 1.0 and self.inner_normalized_radius(position) >= 1.0

    def angle_for_position(self, position) -> float:
        dx = (float(position[0]) - self.center[0]) / self.outer_radius[0]
        dy = (float(position[1]) - self.center[1]) / self.outer_radius[1]
        return math.atan2(dy, dx) % (math.tau)

    def checkpoint_index_for_position(self, position) -> int:
        angle = self.angle_for_position(position)
        return int((angle / math.tau) * CHECKPOINT_COUNT) % CHECKPOINT_COUNT

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(GRASS_COLOR)
        outer_rect = pygame.Rect(0, 0, self.outer_radius[0] * 2, self.outer_radius[1] * 2)
        outer_rect.center = self.center
        inner_rect = pygame.Rect(0, 0, self.inner_radius[0] * 2, self.inner_radius[1] * 2)
        inner_rect.center = self.center

        pygame.draw.ellipse(screen, TRACK_COLOR, outer_rect)
        pygame.draw.ellipse(screen, BOUNDARY_COLOR, outer_rect, 4)
        pygame.draw.ellipse(screen, GRASS_COLOR, inner_rect)
        pygame.draw.ellipse(screen, BOUNDARY_COLOR, inner_rect, 4)

        for index in range(CHECKPOINT_COUNT):
            angle = math.tau * index / CHECKPOINT_COUNT
            start = self._ellipse_point(angle, self.inner_radius)
            end = self._ellipse_point(angle, self.outer_radius)
            color = START_FINISH_COLOR if index == 0 else (66, 170, 255)
            width = 5 if index == 0 else 2
            pygame.draw.line(screen, color, start, end, width)

        pygame.draw.rect(screen, (20, 24, 28), pygame.Rect(0, 0, WIDTH, 1))
        pygame.draw.rect(screen, (20, 24, 28), pygame.Rect(0, HEIGHT - 1, WIDTH, 1))

    def _ellipse_point(self, angle: float, radius: tuple[float, float]) -> tuple[int, int]:
        return (
            int(self.center[0] + math.cos(angle) * radius[0]),
            int(self.center[1] + math.sin(angle) * radius[1]),
        )

