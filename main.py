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
    polygon = car.body_polygon()
    shadow = [(x + 5, y + 6) for x, y in polygon]
    pygame.draw.polygon(screen, (7, 9, 12), shadow)
    pygame.draw.polygon(screen, CAR_COLOR, polygon)
    pygame.draw.polygon(screen, (18, 20, 24), polygon, 2)
    pygame.draw.circle(screen, CAR_NOSE_COLOR, polygon[0], 4)
    draw_wheels(screen, car)


def draw_wheels(screen: pygame.Surface, car: Car) -> None:
    forward = from_angle(car.heading)
    right = from_angle(car.heading + 1.5707963267948966)
    wheel_offsets = [
        forward * 13 + right * 11,
        forward * 13 - right * 11,
        -forward * 14 + right * 11,
        -forward * 14 - right * 11,
    ]
    for offset in wheel_offsets:
        center = car.position + offset
        rect = pygame.Rect(0, 0, 6, 13)
        rect.center = (int(center[0]), int(center[1]))
        wheel_surface = pygame.Surface((6, 13), pygame.SRCALPHA)
        pygame.draw.rect(wheel_surface, (8, 10, 12), wheel_surface.get_rect(), border_radius=2)
        rotated = pygame.transform.rotate(wheel_surface, -car.heading * 57.2958)
        screen.blit(rotated, rotated.get_rect(center=rect.center))


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
        telemetry.log(sim_time, car, on_track, ai_driver.state if ai_enabled else None)
        spawn_particles(particles, car, on_track)
        update_particles(particles, dt)

        screen.fill(BACKGROUND_COLOR)
        track.draw(screen)
        draw_particles(screen, particles)
        draw_car(screen, car)
        if debug_enabled:
            draw_debug_vectors(screen, car)
        hud.draw(screen, car, checkpoints.state, on_track, debug_enabled, ai_enabled, ai_driver.state)
        pygame.display.flip()

    telemetry.save_csv(telemetry_path())
    pygame.quit()


if __name__ == "__main__":
    main()
