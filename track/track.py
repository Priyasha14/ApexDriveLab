import math

import pygame

from config import (
    BOUNDARY_COLOR,
    CHECKPOINT_COUNT,
    GRASS_COLOR,
    GRASS_DARK_COLOR,
    HEIGHT,
    KERB_RED,
    KERB_WHITE,
    START_FINISH_COLOR,
    TRACK_CENTER,
    TRACK_COLOR,
    TRACK_GROOVE_COLOR,
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
        self._draw_grass(screen)
        outer_rect = pygame.Rect(0, 0, self.outer_radius[0] * 2, self.outer_radius[1] * 2)
        outer_rect.center = self.center
        inner_rect = pygame.Rect(0, 0, self.inner_radius[0] * 2, self.inner_radius[1] * 2)
        inner_rect.center = self.center

        pygame.draw.ellipse(screen, TRACK_COLOR, outer_rect)
        self._draw_racing_groove(screen)
        pygame.draw.ellipse(screen, BOUNDARY_COLOR, outer_rect, 4)
        pygame.draw.ellipse(screen, GRASS_COLOR, inner_rect)
        pygame.draw.ellipse(screen, BOUNDARY_COLOR, inner_rect, 4)
        self._draw_kerbs(screen)

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

    def _draw_grass(self, screen: pygame.Surface) -> None:
        screen.fill(GRASS_COLOR)
        stripe_width = 46
        for x in range(-HEIGHT, WIDTH, stripe_width * 2):
            points = [(x, 0), (x + stripe_width, 0), (x + stripe_width + HEIGHT, HEIGHT), (x + HEIGHT, HEIGHT)]
            pygame.draw.polygon(screen, GRASS_DARK_COLOR, points)

    def _draw_racing_groove(self, screen: pygame.Surface) -> None:
        groove_radius = (
            (self.outer_radius[0] + self.inner_radius[0]) / 2.0,
            (self.outer_radius[1] + self.inner_radius[1]) / 2.0,
        )
        groove_rect = pygame.Rect(0, 0, groove_radius[0] * 2, groove_radius[1] * 2)
        groove_rect.center = self.center
        pygame.draw.ellipse(screen, TRACK_GROOVE_COLOR, groove_rect, 34)

    def _draw_kerbs(self, screen: pygame.Surface) -> None:
        segments = 72
        for index in range(segments):
            color = KERB_RED if index % 2 == 0 else KERB_WHITE
            start_angle = math.tau * index / segments
            mid_angle = start_angle + math.tau / segments * 0.5
            self._draw_kerb_segment(screen, mid_angle, self.outer_radius, 13, color)
            self._draw_kerb_segment(screen, mid_angle, self.inner_radius, 11, color)

    def _draw_kerb_segment(self, screen: pygame.Surface, angle: float, radius: tuple[float, float], size: int, color: tuple[int, int, int]) -> None:
        point = self._ellipse_point(angle, radius)
        tangent_angle = angle + math.pi / 2.0
        normal = (math.cos(angle), math.sin(angle))
        tangent = (math.cos(tangent_angle), math.sin(tangent_angle))
        half_len = size
        half_width = 5
        corners = []
        for t, n in [(-half_len, -half_width), (half_len, -half_width), (half_len, half_width), (-half_len, half_width)]:
            corners.append((int(point[0] + tangent[0] * t + normal[0] * n), int(point[1] + tangent[1] * t + normal[1] * n)))
        pygame.draw.polygon(screen, color, corners)
