import math

import pygame

from config import (
    BOUNDARY_COLOR,
    BRAKE_ZONE_COLOR,
    CHECKPOINT_COLOR,
    CHECKPOINT_COUNT,
    GRASS_COLOR,
    HEIGHT,
    KERB_RED,
    KERB_WHITE,
    RACING_LINE_COLOR,
    START_FINISH_COLOR,
    TRACK_CENTER,
    TRACK_COLOR,
    TRACK_GROOVE_COLOR,
    TRACK_INNER_RADIUS,
    TRACK_OUTER_RADIUS,
    TURN_IN_COLOR,
    WIDTH,
)


def monaco_buildings() -> list[tuple[pygame.Rect, float, tuple[int, int, int]]]:
    return [
        (pygame.Rect(900, 454, 112, 70), 72.0, (126, 111, 95)),
        (pygame.Rect(690, 242, 132, 70), 108.0, (132, 123, 108)),
        (pygame.Rect(570, 176, 122, 70), 96.0, (150, 137, 116)),
        (pygame.Rect(432, 208, 92, 70), 82.0, (102, 91, 82)),
        (pygame.Rect(276, 330, 86, 76), 78.0, (116, 103, 96)),
        (pygame.Rect(720, 334, 188, 52), 62.0, (96, 103, 112)),
        (pygame.Rect(1030, 514, 96, 80), 68.0, (136, 124, 107)),
        (pygame.Rect(700, 610, 110, 66), 74.0, (115, 122, 130)),
    ]


class Track:
    def __init__(self) -> None:
        self.center = TRACK_CENTER
        self.outer_radius = TRACK_OUTER_RADIUS
        self.inner_radius = TRACK_INNER_RADIUS
        self.road_width = 104.0
        self.centerline = [
            (1010.0, 570.0),  # Start/finish straight
            (875.0, 566.0),
            (760.0, 536.0),  # Sainte Devote
            (700.0, 466.0),  # Beau Rivage climb
            (650.0, 392.0),
            (600.0, 322.0),  # Massenet
            (524.0, 282.0),  # Casino Square
            (438.0, 306.0),
            (382.0, 350.0),  # Mirabeau Haute
            (338.0, 384.0),
            (308.0, 420.0),  # Fairmont hairpin entry
            (326.0, 462.0),
            (390.0, 470.0),  # Mirabeau Bas
            (468.0, 444.0),
            (540.0, 410.0),  # Portier
            (648.0, 374.0),  # Tunnel
            (788.0, 364.0),
            (924.0, 386.0),
            (982.0, 426.0),  # Nouvelle Chicane
            (936.0, 470.0),
            (1018.0, 506.0),
            (1098.0, 520.0),  # Tabac
            (1138.0, 560.0),  # Swimming Pool
            (1082.0, 600.0),
            (990.0, 578.0),
            (930.0, 616.0),
            (844.0, 642.0),  # La Rascasse
            (772.0, 626.0),  # Antony Noghes
            (760.0, 584.0),
        ]
        self._segments = self._build_segments()
        self.total_length = sum(segment["length"] for segment in self._segments)

    def _build_segments(self) -> list[dict[str, float | tuple[float, float]]]:
        segments = []
        distance = 0.0
        points = self.centerline
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            segments.append({"start": start, "end": end, "dx": dx, "dy": dy, "length": length, "distance": distance})
            distance += length
        return segments

    def project_to_centerline(self, position) -> tuple[float, float, tuple[float, float], int]:
        px = float(position[0])
        py = float(position[1])
        best_distance = float("inf")
        best_progress = 0.0
        best_point = self.centerline[0]
        best_index = 0
        for index, segment in enumerate(self._segments):
            start = segment["start"]
            dx = float(segment["dx"])
            dy = float(segment["dy"])
            length = max(float(segment["length"]), 1e-6)
            t = ((px - start[0]) * dx + (py - start[1]) * dy) / (length * length)
            t = max(0.0, min(1.0, t))
            point = (start[0] + dx * t, start[1] + dy * t)
            distance = math.hypot(px - point[0], py - point[1])
            if distance < best_distance:
                best_distance = distance
                best_progress = float(segment["distance"]) + length * t
                best_point = point
                best_index = index
        return best_distance, best_progress % self.total_length, best_point, best_index

    def point_at_progress(self, progress: float) -> tuple[float, float]:
        progress %= self.total_length
        for segment in self._segments:
            start = segment["start"]
            length = float(segment["length"])
            distance = float(segment["distance"])
            if progress <= distance + length:
                t = (progress - distance) / max(length, 1e-6)
                return (start[0] + float(segment["dx"]) * t, start[1] + float(segment["dy"]) * t)
        return self.centerline[0]

    def progress_for_angle(self, angle: float) -> float:
        return ((-angle) % math.tau) / math.tau * self.total_length

    def angle_for_progress(self, progress: float) -> float:
        return (-(progress / self.total_length) * math.tau) % math.tau

    def normalized_radius(self, position) -> float:
        distance, _, _, _ = self.project_to_centerline(position)
        return distance / (self.road_width * 0.5)

    def inner_normalized_radius(self, position) -> float:
        return 1.0

    def is_on_track(self, position) -> bool:
        distance, _, _, _ = self.project_to_centerline(position)
        return distance <= self.road_width * 0.5

    def angle_for_position(self, position) -> float:
        _, progress, _, _ = self.project_to_centerline(position)
        return self.angle_for_progress(progress)

    def checkpoint_index_for_position(self, position) -> int:
        _, progress, _, _ = self.project_to_centerline(position)
        normalized = progress / self.total_length
        return int(((-normalized) % 1.0) * CHECKPOINT_COUNT) % CHECKPOINT_COUNT

    def draw(self, screen: pygame.Surface) -> None:
        self._draw_city(screen)
        self._draw_harbor(screen)
        self._draw_road(screen)
        self._draw_reference_line(screen)
        self._draw_kerbs(screen)
        self._draw_checkpoints(screen)
        self._draw_landmarks(screen)
        pygame.draw.rect(screen, (20, 24, 28), pygame.Rect(0, 0, WIDTH, 1))
        pygame.draw.rect(screen, (20, 24, 28), pygame.Rect(0, HEIGHT - 1, WIDTH, 1))

    def _draw_city(self, screen: pygame.Surface) -> None:
        screen.fill((32, 52, 58))
        pygame.draw.rect(screen, (28, 79, 49), pygame.Rect(0, 520, WIDTH, 200))
        pygame.draw.rect(screen, (31, 94, 56), pygame.Rect(0, 0, 270, 720))
        pygame.draw.rect(screen, (35, 111, 68), pygame.Rect(0, 0, 1280, 96))
        for rect, _, color in monaco_buildings():
            pygame.draw.rect(screen, color, rect, border_radius=3)
            pygame.draw.rect(screen, (48, 52, 56), rect, 2, border_radius=3)
        for x in range(12, WIDTH, 58):
            pygame.draw.line(screen, (40, 76, 54), (x, 0), (x - 110, HEIGHT), 20)

    def _draw_harbor(self, screen: pygame.Surface) -> None:
        pygame.draw.polygon(screen, (23, 75, 101), [(760, 470), (1280, 480), (1280, 720), (940, 720), (870, 620), (930, 548)])
        pygame.draw.polygon(screen, (41, 102, 132), [(925, 540), (1280, 545), (1280, 610), (1000, 595)])
        for y in range(520, 700, 34):
            pygame.draw.line(screen, (48, 113, 144), (940, y), (1260, y + 14), 2)
        for x, y in [(990, 622), (1085, 640), (1180, 626), (1135, 690)]:
            pygame.draw.polygon(screen, (240, 238, 218), [(x, y), (x + 28, y + 24), (x - 18, y + 28)])
            pygame.draw.line(screen, (220, 220, 210), (x, y), (x, y - 36), 2)

    def _draw_road(self, screen: pygame.Surface) -> None:
        points = [(int(x), int(y)) for x, y in self.centerline]
        closed = points + [points[0]]
        pygame.draw.lines(screen, BOUNDARY_COLOR, False, closed, int(self.road_width + 16))
        pygame.draw.lines(screen, (12, 15, 18), False, closed, int(self.road_width + 8))
        pygame.draw.lines(screen, TRACK_COLOR, False, closed, int(self.road_width))
        pygame.draw.lines(screen, TRACK_GROOVE_COLOR, False, closed, 24)
        pygame.draw.lines(screen, (74, 78, 82), False, closed, 2)
        for point in points:
            pygame.draw.circle(screen, TRACK_COLOR, point, int(self.road_width * 0.5))

    def _draw_reference_line(self, screen: pygame.Surface) -> None:
        racing_points = []
        for index in range(220):
            progress = self.total_length * index / 220
            point = self.point_at_progress(progress)
            racing_points.append((int(point[0]), int(point[1])))
        pygame.draw.lines(screen, RACING_LINE_COLOR, True, racing_points, 3)
        for start, end, color in [(0.05, 0.10, BRAKE_ZONE_COLOR), (0.34, 0.40, BRAKE_ZONE_COLOR), (0.58, 0.64, BRAKE_ZONE_COLOR), (0.81, 0.87, TURN_IN_COLOR)]:
            segment = []
            for index in range(int(start * 220), int(end * 220)):
                point = self.point_at_progress(self.total_length * index / 220)
                segment.append((int(point[0]), int(point[1])))
            if len(segment) > 1:
                pygame.draw.lines(screen, color, False, segment, 7)

    def _draw_kerbs(self, screen: pygame.Surface) -> None:
        for index, segment in enumerate(self._segments):
            if index % 2:
                continue
            start = segment["start"]
            end = segment["end"]
            dx = float(segment["dx"])
            dy = float(segment["dy"])
            length = max(float(segment["length"]), 1e-6)
            nx = -dy / length
            ny = dx / length
            for side in (-1, 1):
                color = KERB_RED if (index + side) % 3 else KERB_WHITE
                offset = self.road_width * 0.5 * side
                p0 = (int(start[0] + nx * offset), int(start[1] + ny * offset))
                p1 = (int(end[0] + nx * offset), int(end[1] + ny * offset))
                pygame.draw.line(screen, color, p0, p1, 8)

    def _draw_checkpoints(self, screen: pygame.Surface) -> None:
        for index in range(CHECKPOINT_COUNT):
            progress = self.total_length * index / CHECKPOINT_COUNT
            point = self.point_at_progress(progress)
            next_point = self.point_at_progress(progress + 8.0)
            tangent = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            normal = tangent + math.pi / 2.0
            half = self.road_width * 0.5
            start = (int(point[0] + math.cos(normal) * half), int(point[1] + math.sin(normal) * half))
            end = (int(point[0] - math.cos(normal) * half), int(point[1] - math.sin(normal) * half))
            color = START_FINISH_COLOR if index == 0 else CHECKPOINT_COLOR
            pygame.draw.line(screen, color, start, end, 5 if index == 0 else 2)

    def _draw_landmarks(self, screen: pygame.Surface) -> None:
        font = pygame.font.Font(None, 22)
        labels = [
            ("Sainte Devote", (738, 500)),
            ("Massenet", (628, 350)),
            ("Casino", (500, 250)),
            ("Mirabeau", (370, 322)),
            ("Fairmont Hairpin", (232, 408)),
            ("Portier", (512, 390)),
            ("Tunnel", (720, 332)),
            ("Nouvelle Chicane", (938, 430)),
            ("Tabac", (1084, 494)),
            ("Swimming Pool", (1048, 594)),
            ("Rascasse", (822, 650)),
            ("Anthony Noghes", (712, 610)),
        ]
        for text, pos in labels:
            surface = font.render(text, True, (225, 230, 225))
            screen.blit(surface, pos)
