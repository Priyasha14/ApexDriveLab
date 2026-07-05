import math
from datetime import datetime
from pathlib import Path

import pygame

from ai.racing_line import RacingLine
from ai.rule_driver import RuleBasedDriver
from config import (
    BACKGROUND_COLOR,
    CAR_COLOR,
    CAR_NOSE_COLOR,
    FPS,
    HEIGHT,
    KERB_RED,
    KERB_WHITE,
    OFF_TRACK_GRIP_SCALE,
    WIDTH,
    WINDOW_TITLE,
)
from physics.car import Car, CarInputs
from physics.setup import SETUPS
from physics.vector_utils import from_angle, length
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track
from ui.hud import HUD


def read_inputs(aero_override: str | None) -> CarInputs:
    keys = pygame.key.get_pressed()
    throttle = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
    steer = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer += 1.0
    deploy_hybrid = keys[pygame.K_SPACE]
    drs = True if keys[pygame.K_e] else None  # E requests DRS; otherwise automatic
    return CarInputs(throttle=throttle, brake=brake, steer=steer, deploy_hybrid=deploy_hybrid, aero_mode=aero_override, drs=drs)


def draw_car(screen: pygame.Surface, car: Car) -> None:
    draw_car_shadow(screen, car)
    draw_wheels(screen, car)
    draw_f1_body(screen, car)


def car_point(car: Car, forward, right, x: float, y: float) -> tuple[int, int]:
    point = car.position + forward * x + right * y
    return int(point[0]), int(point[1])


def draw_oriented_polygon(screen: pygame.Surface, car: Car, points: list[tuple[float, float]], color: tuple[int, int, int], outline: tuple[int, int, int] | None = None, width: int = 1) -> None:
    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    polygon = [car_point(car, forward, right, x, y) for x, y in points]
    pygame.draw.polygon(screen, color, polygon)
    if outline is not None:
        pygame.draw.polygon(screen, outline, polygon, width)


def draw_rotated_rect(screen: pygame.Surface, center: tuple[int, int], size: tuple[int, int], angle: float, color: tuple[int, int, int], border_radius: int = 2, outline: tuple[int, int, int] | None = None) -> None:
    surface = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(surface, color, surface.get_rect(), border_radius=border_radius)
    if outline is not None:
        pygame.draw.rect(surface, outline, surface.get_rect(), 1, border_radius=border_radius)
    rotated = pygame.transform.rotate(surface, -angle * 57.2958)
    screen.blit(rotated, rotated.get_rect(center=center))


def draw_car_shadow(screen: pygame.Surface, car: Car) -> None:
    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    shadow_points = [
        car_point(car, forward, right, 28, 0),
        car_point(car, forward, right, 16, 18),
        car_point(car, forward, right, -27, 18),
        car_point(car, forward, right, -32, 13),
        car_point(car, forward, right, -32, -13),
        car_point(car, forward, right, -27, -18),
        car_point(car, forward, right, 16, -18),
    ]
    shadow = [(x + 5, y + 7) for x, y in shadow_points]
    pygame.draw.polygon(screen, (6, 8, 10), shadow)


def draw_f1_body(screen: pygame.Surface, car: Car) -> None:
    dark = (14, 16, 20)
    carbon = (24, 27, 31)
    highlight = (255, 232, 132)
    sidepod = (178, 21, 38)
    halo = (33, 36, 40)

    draw_oriented_polygon(screen, car, [(-9, -9), (12, -7), (24, -2), (27, 0), (24, 2), (12, 7), (-9, 9), (-17, 5), (-17, -5)], CAR_COLOR, dark, 2)
    draw_oriented_polygon(screen, car, [(3, -4), (29, -2), (34, 0), (29, 2), (3, 4), (-2, 0)], CAR_NOSE_COLOR, dark, 1)
    draw_oriented_polygon(screen, car, [(-5, -16), (9, -15), (13, -7), (-10, -8), (-18, -13)], sidepod, dark, 1)
    draw_oriented_polygon(screen, car, [(-5, 16), (9, 15), (13, 7), (-10, 8), (-18, 13)], sidepod, dark, 1)
    draw_oriented_polygon(screen, car, [(-31, -17), (-22, -15), (-20, 15), (-31, 17), (-35, 12), (-35, -12)], carbon, dark, 1)
    draw_oriented_polygon(screen, car, [(20, -20), (31, -19), (33, 19), (20, 20), (18, 14), (18, -14)], carbon, dark, 1)
    draw_oriented_polygon(screen, car, [(-7, -4), (5, -4), (9, 0), (5, 4), (-7, 4), (-10, 0)], (22, 25, 29), (5, 7, 9), 1)

    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    cockpit = car_point(car, forward, right, -3, 0)
    pygame.draw.circle(screen, (8, 10, 12), cockpit, 6)
    pygame.draw.circle(screen, (74, 92, 104), cockpit, 3)
    for side in (-1, 1):
        start = car_point(car, forward, right, -8, side * 7)
        end = car_point(car, forward, right, 6, side * 3)
        pygame.draw.line(screen, halo, start, end, 3)

    centerline_a = car_point(car, forward, right, -26, 0)
    centerline_b = car_point(car, forward, right, 30, 0)
    pygame.draw.line(screen, highlight, centerline_a, centerline_b, 2)
    pygame.draw.circle(screen, highlight, car_point(car, forward, right, 34, 0), 3)


def draw_wheels(screen: pygame.Surface, car: Car) -> None:
    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    wheel_offsets = [
        (forward * 15 + right * 14, car.heading + car.steering_angle),
        (forward * 15 - right * 14, car.heading + car.steering_angle),
        (-forward * 17 + right * 14, car.heading),
        (-forward * 17 - right * 14, car.heading),
    ]
    for offset, angle in wheel_offsets:
        center = car.position + offset
        wheel_center = (int(center[0]), int(center[1]))
        draw_rotated_rect(screen, wheel_center, (8, 16), angle, (7, 8, 10), 3, (35, 38, 42))
        rim = center - forward * 1
        pygame.draw.circle(screen, (58, 62, 66), (int(rim[0]), int(rim[1])), 2)


def draw_debug_vectors(screen: pygame.Surface, car: Car) -> None:
    origin = (int(car.position[0]), int(car.position[1]))
    heading = car.position + from_angle(car.heading) * 70.0
    pygame.draw.line(screen, (80, 180, 255), origin, (int(heading[0]), int(heading[1])), 3)

    if length(car.velocity) > 1.0:
        velocity = car.position + car.velocity * 0.22
        pygame.draw.line(screen, (120, 255, 150), origin, (int(velocity[0]), int(velocity[1])), 3)


def spawn_particles(particles: list[dict], car: Car, on_track: bool) -> None:
    grip = car.tire_state.combined_grip_usage
    if grip < 0.82 and on_track:
        return

    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    color = (156, 160, 164) if on_track else (117, 86, 49)
    life = 0.45 if on_track else 0.65
    for side in (-1, 1):
        origin = car.position - forward * 16 + right * side * 10
        particles.append(
            {
                "position": origin.copy(),
                "velocity": -forward * (30 + grip * 40) + right * side * 12,
                "life": life,
                "max_life": life,
                "radius": 4 + grip * 4,
                "color": color,
            }
        )


def update_particles(particles: list[dict], dt: float) -> None:
    for particle in particles[:]:
        particle["life"] -= dt
        particle["position"] += particle["velocity"] * dt
        particle["velocity"] *= max(0.0, 1.0 - 2.8 * dt)
        if particle["life"] <= 0:
            particles.remove(particle)


def draw_particles(screen: pygame.Surface, particles: list[dict]) -> None:
    for particle in particles:
        alpha = max(0.0, particle["life"] / particle["max_life"])
        radius = max(1, int(particle["radius"] * alpha))
        color = tuple(int(channel * alpha + 18 * (1.0 - alpha)) for channel in particle["color"])
        pygame.draw.circle(screen, color, particle["position"].astype(int), radius)


def ellipse_world_point(track: Track, angle: float, radius: tuple[float, float]) -> tuple[float, float]:
    return (
        track.center[0] + math.cos(angle) * radius[0],
        track.center[1] + math.sin(angle) * radius[1],
    )


def track_edge_point(track: Track, progress: float, side: float) -> tuple[float, float]:
    center = track.point_at_progress(progress)
    ahead = track.point_at_progress(progress + 8.0)
    tangent = math.atan2(ahead[1] - center[1], ahead[0] - center[0])
    normal = tangent + math.pi / 2.0
    offset = track.road_width * 0.5 * side
    return center[0] + math.cos(normal) * offset, center[1] + math.sin(normal) * offset


def wrap_angle_delta(angle: float) -> float:
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    return angle


def update_cockpit_camera(camera_state: dict, car: Car, dt: float, snap: bool = False) -> None:
    if snap:
        camera_state["position"] = car.position.copy()
        camera_state["heading"] = car.heading
        return

    position_response = min(1.0, dt * 4.5)
    heading_response = min(1.0, dt * 3.2)
    camera_state["position"] += (car.position - camera_state["position"]) * position_response
    camera_state["heading"] += wrap_angle_delta(car.heading - camera_state["heading"]) * heading_response


def project_cockpit_point(camera_position, camera_heading: float, point: tuple[float, float], horizon_y: int, focal: float, camera_height: float) -> tuple[int, int, float] | None:
    forward = from_angle(camera_heading)
    right = from_angle(camera_heading + 1.5707963267948966)
    dx = point[0] - float(camera_position[0])
    dy = point[1] - float(camera_position[1])
    ahead = dx * float(forward[0]) + dy * float(forward[1])
    lateral = dx * float(right[0]) + dy * float(right[1])
    if ahead < 10.0:
        return None

    depth = ahead + 28.0
    x = WIDTH / 2 + lateral * focal / depth
    y = horizon_y + camera_height * focal / depth
    return int(x), int(y), ahead


def camera_axes(camera_heading: float) -> tuple[tuple[float, float], tuple[float, float]]:
    forward = from_angle(camera_heading)
    right = from_angle(camera_heading + 1.5707963267948966)
    return (float(forward[0]), float(forward[1])), (float(right[0]), float(right[1]))


def project_world_3d(camera_position, camera_heading: float, point: tuple[float, float, float], focal: float = 720.0, camera_height: float = 34.0) -> tuple[int, int, float] | None:
    forward, right = camera_axes(camera_heading)
    dx = point[0] - float(camera_position[0])
    dz = point[2] - float(camera_position[1])
    lateral = dx * right[0] + dz * right[1]
    depth = dx * forward[0] + dz * forward[1]
    vertical = point[1] - camera_height
    if depth < 8.0:
        return None
    x = WIDTH / 2 + lateral * focal / depth
    y = HEIGHT * 0.50 - vertical * focal / depth
    return int(x), int(y), depth


def add_3d_polygon(polygons: list[tuple[float, list[tuple[int, int]], tuple[int, int, int]]], camera_position, camera_heading: float, points: list[tuple[float, float, float]], color: tuple[int, int, int]) -> None:
    projected = [project_world_3d(camera_position, camera_heading, point) for point in points]
    if any(point is None for point in projected):
        return
    screen_points = [(point[0], point[1]) for point in projected if point is not None]
    depth = sum(point[2] for point in projected if point is not None) / len(projected)
    polygons.append((depth, screen_points, color))


def add_3d_wall(polygons: list[tuple[float, list[tuple[int, int]], tuple[int, int, int]]], camera_position, camera_heading: float, a: tuple[float, float], b: tuple[float, float], height: float, color: tuple[int, int, int]) -> None:
    add_3d_polygon(
        polygons,
        camera_position,
        camera_heading,
        [(a[0], 0.0, a[1]), (b[0], 0.0, b[1]), (b[0], height, b[1]), (a[0], height, a[1])],
        color,
    )


def add_building(polygons: list[tuple[float, list[tuple[int, int]], tuple[int, int, int]]], camera_position, camera_heading: float, rect: pygame.Rect, height: float, color: tuple[int, int, int]) -> None:
    x0, z0, x1, z1 = rect.left, rect.top, rect.right, rect.bottom
    top_color = tuple(min(255, channel + 24) for channel in color)
    side_color = tuple(max(0, channel - 28) for channel in color)
    add_3d_polygon(polygons, camera_position, camera_heading, [(x0, height, z0), (x1, height, z0), (x1, height, z1), (x0, height, z1)], top_color)
    add_3d_wall(polygons, camera_position, camera_heading, (x0, z0), (x1, z0), height, color)
    add_3d_wall(polygons, camera_position, camera_heading, (x1, z0), (x1, z1), height, side_color)
    add_3d_wall(polygons, camera_position, camera_heading, (x1, z1), (x0, z1), height, color)
    add_3d_wall(polygons, camera_position, camera_heading, (x0, z1), (x0, z0), height, side_color)


def monaco_buildings() -> list[tuple[pygame.Rect, float, tuple[int, int, int]]]:
    return [
        (pygame.Rect(350, 154, 118, 96), 82.0, (135, 123, 106)),
        (pygame.Rect(495, 118, 126, 92), 108.0, (164, 150, 126)),
        (pygame.Rect(642, 112, 108, 74), 74.0, (120, 112, 106)),
        (pygame.Rect(780, 128, 150, 94), 96.0, (150, 136, 114)),
        (pygame.Rect(964, 150, 130, 90), 70.0, (112, 121, 130)),
        (pygame.Rect(342, 430, 136, 88), 76.0, (144, 128, 105)),
        (pygame.Rect(520, 456, 108, 74), 58.0, (112, 116, 118)),
        (pygame.Rect(664, 440, 150, 88), 64.0, (96, 105, 116)),
        (pygame.Rect(858, 422, 132, 92), 72.0, (146, 129, 103)),
        (pygame.Rect(1010, 430, 118, 92), 80.0, (120, 118, 112)),
    ]


def draw_cockpit_road(screen: pygame.Surface, car: Car, track: Track, camera_state: dict) -> None:
    camera_position = camera_state["position"]
    camera_heading = camera_state["heading"]
    horizon_y = 235
    screen.fill((104, 153, 190))
    pygame.draw.rect(screen, (19, 82, 54), pygame.Rect(0, horizon_y, WIDTH, HEIGHT - horizon_y))
    pygame.draw.rect(screen, (22, 92, 60), pygame.Rect(0, horizon_y + 92, WIDTH, HEIGHT - horizon_y - 92))

    for x in range(-140, WIDTH + 180, 150):
        pygame.draw.polygon(screen, (24, 105, 66), [(x, horizon_y), (x + 56, horizon_y), (x + 250, HEIGHT), (x + 130, HEIGHT)])

    for rect, height, color in monaco_buildings():
        building_bottom = project_world_3d(camera_position, camera_heading, (rect.centerx, 0.0, rect.centery), focal=520.0, camera_height=34.0)
        if not building_bottom:
            continue
        scale = max(0.16, min(1.3, 130.0 / max(building_bottom[2], 1.0)))
        width = int(rect.width * scale)
        height_px = int(height * scale * 1.45)
        if width < 10 or height_px < 14:
            continue
        base_x, base_y = building_bottom[0], building_bottom[1]
        block = pygame.Rect(base_x - width // 2, base_y - height_px, width, height_px)
        if block.bottom < horizon_y - 20 or block.left > WIDTH or block.right < 0:
            continue
        pygame.draw.rect(screen, tuple(max(0, channel - 22) for channel in color), block.move(7, 8))
        pygame.draw.rect(screen, color, block, border_radius=2)
        pygame.draw.rect(screen, (42, 46, 52), block, 2, border_radius=2)

    pygame.draw.polygon(screen, (24, 88, 124), [(780, horizon_y + 36), (WIDTH, horizon_y + 8), (WIDTH, HEIGHT), (890, HEIGHT)])
    for y in range(horizon_y + 64, HEIGHT, 42):
        pygame.draw.line(screen, (46, 116, 148), (830, y), (WIDTH, y - 28), 2)

    forward, right = camera_axes(camera_heading)
    _, current_progress, _, _ = track.project_to_centerline(camera_position)

    samples = []
    distances = [24, 34, 48, 66, 88, 116, 150, 190, 238, 296, 366, 450, 550, 668, 806, 968]
    for distance in distances:
        point = track.point_at_progress(current_progress + distance)
        dx = point[0] - float(camera_position[0])
        dz = point[1] - float(camera_position[1])
        ahead = dx * forward[0] + dz * forward[1]
        lateral = dx * right[0] + dz * right[1]
        if ahead < 10.0:
            continue
        depth = ahead + 70.0
        perspective = 1.0 / depth
        screen_y = horizon_y + 6200.0 * perspective
        road_width = max(30.0, track.road_width * 720.0 * perspective)
        center_x = WIDTH / 2 + lateral * 760.0 * perspective
        samples.append((center_x, screen_y, road_width, distance))

    samples = [sample for sample in samples if horizon_y - 12 <= sample[1] <= HEIGHT + 120]
    if len(samples) < 2:
        return

    for index in range(len(samples) - 1, 0, -1):
        far = samples[index]
        near = samples[index - 1]
        far_left = (far[0] - far[2] * 0.5, far[1])
        far_right = (far[0] + far[2] * 0.5, far[1])
        near_left = (near[0] - near[2] * 0.5, near[1])
        near_right = (near[0] + near[2] * 0.5, near[1])
        road_color = (47, 50, 53) if index % 2 else (41, 44, 47)
        pygame.draw.polygon(screen, road_color, [near_left, near_right, far_right, far_left])

        edge_width = max(3, int(near[2] * 0.045))
        kerb_color = KERB_RED if index % 2 else KERB_WHITE
        pygame.draw.line(screen, kerb_color, near_left, far_left, edge_width)
        pygame.draw.line(screen, kerb_color, near_right, far_right, edge_width)
        pygame.draw.line(screen, (230, 234, 236), (near_left[0] - edge_width * 1.2, near_left[1]), (far_left[0] - edge_width * 1.2, far_left[1]), max(2, edge_width // 2))
        pygame.draw.line(screen, (230, 234, 236), (near_right[0] + edge_width * 1.2, near_right[1]), (far_right[0] + edge_width * 1.2, far_right[1]), max(2, edge_width // 2))

        wall_height = max(9, int(near[2] * 0.13))
        pygame.draw.polygon(
            screen,
            (70, 78, 84),
            [(near_left[0] - edge_width * 3.0, near_left[1]), (far_left[0] - edge_width * 3.0, far_left[1]), (far_left[0] - edge_width * 3.0, far_left[1] - wall_height), (near_left[0] - edge_width * 3.0, near_left[1] - wall_height)],
        )
        pygame.draw.polygon(
            screen,
            (70, 78, 84),
            [(near_right[0] + edge_width * 3.0, near_right[1]), (far_right[0] + edge_width * 3.0, far_right[1]), (far_right[0] + edge_width * 3.0, far_right[1] - wall_height), (near_right[0] + edge_width * 3.0, near_right[1] - wall_height)],
        )

    centerline = [(sample[0], sample[1]) for sample in samples if sample[3] > 45]
    if len(centerline) > 1:
        pygame.draw.lines(screen, (67, 207, 126), False, centerline, 4)

    speed_haze = min(car.speed_kmh / 240.0, 1.0)
    haze = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(haze, (210, 230, 240, int(18 * speed_haze)), pygame.Rect(0, horizon_y - 50, WIDTH, 120))
    screen.blit(haze, (0, 0))


def draw_cockpit_overlay(screen: pygame.Surface, car: Car) -> None:
    width = screen.get_width()
    height = screen.get_height()
    base_y = height - 126
    center_x = width // 2

    pygame.draw.polygon(screen, (8, 10, 12), [(0, height), (width, height), (width, height - 58), (center_x + 250, base_y), (center_x - 250, base_y), (0, height - 58)])
    pygame.draw.polygon(screen, (22, 25, 29), [(center_x - 310, height), (center_x + 310, height), (center_x + 180, base_y), (center_x - 180, base_y)])
    pygame.draw.polygon(screen, (120, 14, 28), [(center_x - 235, height), (center_x + 235, height), (center_x + 118, base_y + 18), (center_x - 118, base_y + 18)])
    pygame.draw.polygon(screen, (228, 36, 54), [(center_x - 84, height), (center_x + 84, height), (center_x + 44, base_y + 6), (center_x - 44, base_y + 6)])
    pygame.draw.line(screen, CAR_NOSE_COLOR, (center_x, base_y + 8), (center_x, height), 4)

    wheel_radius = 58
    wheel_center = (center_x, height - 58)
    pygame.draw.circle(screen, (6, 7, 9), wheel_center, wheel_radius)
    pygame.draw.circle(screen, (32, 36, 40), wheel_center, wheel_radius, 12)
    marker_angle = car.inputs.steer * 1.15 - 1.5707963267948966
    marker_end = (
        int(wheel_center[0] + 44 * pygame.math.Vector2(1, 0).rotate_rad(marker_angle).x),
        int(wheel_center[1] + 44 * pygame.math.Vector2(1, 0).rotate_rad(marker_angle).y),
    )
    pygame.draw.line(screen, (72, 182, 255), wheel_center, marker_end, 5)
    pygame.draw.circle(screen, (12, 15, 18), wheel_center, 18)

    for side in (-1, 1):
        anchor = (center_x + side * 205, height)
        top = (center_x + side * 72, base_y - 86)
        pygame.draw.line(screen, (18, 21, 25), anchor, top, 13)
        pygame.draw.line(screen, (57, 62, 68), anchor, top, 3)
    pygame.draw.arc(screen, (18, 21, 25), pygame.Rect(center_x - 118, base_y - 120, 236, 116), 3.45, 5.98, 12)
    pygame.draw.arc(screen, (73, 78, 84), pygame.Rect(center_x - 118, base_y - 120, 236, 116), 3.45, 5.98, 3)


def telemetry_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"telemetry_{timestamp}.csv"


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    track = Track()
    racing_line = RacingLine.from_track(track)
    ai_driver = RuleBasedDriver(racing_line)
    car = Car()
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))
    hud = HUD()
    telemetry = TelemetryLogger()
    sim_time = 0.0
    particles: list[dict] = []

    running = True
    debug_enabled = False
    aero_override: str | None = None
    ai_enabled = False
    cockpit_camera = False
    cockpit_camera_state = {"position": car.position.copy(), "heading": car.heading}
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    telemetry.save_csv(telemetry_path())
                    telemetry.clear()
                    sim_time = 0.0
                    car.reset()
                    update_cockpit_camera(cockpit_camera_state, car, dt, snap=True)
                    checkpoints.reset(track.checkpoint_index_for_position(car.position))
                elif event.key == pygame.K_F1:
                    debug_enabled = not debug_enabled
                elif event.key == pygame.K_v:
                    cockpit_camera = not cockpit_camera
                elif event.key == pygame.K_p:
                    ai_enabled = not ai_enabled
                elif event.key == pygame.K_t:
                    telemetry.save_csv(telemetry_path())
                elif event.key == pygame.K_1:
                    car.apply_setup(SETUPS["balanced"])
                elif event.key == pygame.K_2:
                    car.apply_setup(SETUPS["stable"])
                elif event.key == pygame.K_3:
                    car.apply_setup(SETUPS["rotation"])
                elif event.key == pygame.K_4:
                    car.apply_setup(SETUPS["high_downforce"])
                elif event.key == pygame.K_5:
                    car.apply_setup(SETUPS["low_drag"])
                elif event.key == pygame.K_6:
                    car.apply_setup(SETUPS["front_aero"])
                elif event.key == pygame.K_7:
                    car.apply_setup(SETUPS["rear_aero"])
                elif event.key == pygame.K_z:
                    aero_override = None
                elif event.key == pygame.K_x:
                    aero_override = "corner"
                elif event.key == pygame.K_c:
                    aero_override = "straight"

        inputs = ai_driver.control(car, track) if ai_enabled else read_inputs(aero_override)
        grip_scale = 1.0 if track.is_on_track(car.position) else OFF_TRACK_GRIP_SCALE
        car.update(dt, inputs, grip_scale)
        on_track = track.is_on_track(car.position)
        checkpoint_index = track.checkpoint_index_for_position(car.position)
        checkpoints.update(dt, checkpoint_index)
        sim_time += dt
        telemetry.log(sim_time, car, on_track, ai_driver.state if ai_enabled else None, track)
        spawn_particles(particles, car, on_track)
        update_particles(particles, dt)
        update_cockpit_camera(cockpit_camera_state, car, dt)

        render_target = pygame.Surface((WIDTH, HEIGHT))
        render_target.fill(BACKGROUND_COLOR)
        track.draw(render_target)
        draw_particles(render_target, particles)
        if not cockpit_camera:
            draw_car(render_target, car)
        if debug_enabled:
            draw_debug_vectors(render_target, car)

        if cockpit_camera:
            draw_cockpit_road(screen, car, track, cockpit_camera_state)
            draw_cockpit_overlay(screen, car)
        else:
            screen.blit(render_target, (0, 0))
        hud.draw(screen, car, checkpoints.state, on_track, debug_enabled, ai_enabled, ai_driver.state)
        pygame.display.flip()

    telemetry.save_csv(telemetry_path())
    pygame.quit()


if __name__ == "__main__":
    main()
