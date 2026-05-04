import pygame

from config import (
    BACKGROUND_COLOR,
    CAR_COLOR,
    CAR_NOSE_COLOR,
    FPS,
    HEIGHT,
    WIDTH,
    WINDOW_TITLE,
)
from physics.car import Car, CarInputs
from track.checkpoints import CheckpointManager
from track.track import Track
from ui.hud import HUD


def read_inputs() -> CarInputs:
    keys = pygame.key.get_pressed()
    throttle = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
    steer = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer += 1.0
    return CarInputs(throttle=throttle, brake=brake, steer=steer)


def draw_car(screen: pygame.Surface, car: Car) -> None:
    polygon = car.body_polygon()
    pygame.draw.polygon(screen, CAR_COLOR, polygon)
    pygame.draw.polygon(screen, (18, 20, 24), polygon, 2)
    pygame.draw.circle(screen, CAR_NOSE_COLOR, polygon[0], 4)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    track = Track()
    car = Car()
    checkpoints = CheckpointManager()
    hud = HUD()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset()
                    checkpoints.reset()

        inputs = read_inputs()
        car.update(dt, inputs)
        on_track = track.is_on_track(car.position)
        checkpoint_index = track.checkpoint_index_for_position(car.position)
        checkpoints.update(dt, checkpoint_index)

        screen.fill(BACKGROUND_COLOR)
        track.draw(screen)
        draw_car(screen, car)
        hud.draw(screen, car, checkpoints.state, on_track)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
