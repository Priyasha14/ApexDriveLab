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
    return CarInputs(throttle=throttle, brake=brake, steer=steer, deploy_hybrid=deploy_hybrid, aero_mode=aero_override)


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


def render_cockpit_camera(world: pygame.Surface, screen: pygame.Surface, car: Car) -> None:
    camera_size = 1900
    camera = pygame.Surface((camera_size, camera_size), pygame.SRCALPHA)
    center = camera_size // 2
    camera.fill(BACKGROUND_COLOR)
    camera.blit(world, (center - int(car.position[0]), center - int(car.position[1])))

    rotation_degrees = car.heading * 57.2958 + 90.0
    rotated = pygame.transform.rotate(camera, rotation_degrees)
    crop = pygame.Rect(0, 0, WIDTH, HEIGHT)
    crop.center = rotated.get_rect().center
    cockpit_view = rotated.subsurface(crop).copy()
    screen.blit(cockpit_view, (0, 0))


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
    racing_line = RacingLine(track.center, track.inner_radius, track.outer_radius)
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

        render_target = pygame.Surface((WIDTH, HEIGHT))
        render_target.fill(BACKGROUND_COLOR)
        track.draw(render_target)
        draw_particles(render_target, particles)
        if not cockpit_camera:
            draw_car(render_target, car)
        if debug_enabled:
            draw_debug_vectors(render_target, car)

        if cockpit_camera:
            render_cockpit_camera(render_target, screen, car)
            draw_cockpit_overlay(screen, car)
        else:
            screen.blit(render_target, (0, 0))
        hud.draw(screen, car, checkpoints.state, on_track, debug_enabled, ai_enabled, ai_driver.state)
        pygame.display.flip()

    telemetry.save_csv(telemetry_path())
    pygame.quit()


if __name__ == "__main__":
    main()
