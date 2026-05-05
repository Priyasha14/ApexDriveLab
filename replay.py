import argparse
import csv
import sys
from pathlib import Path

import pygame

from config import BACKGROUND_COLOR, FPS, HEIGHT, WIDTH, WINDOW_TITLE
from main import draw_car
from physics.car import Car
from physics.vector_utils import vec2
from track.track import Track


def load_replay(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def apply_frame(car: Car, row: dict[str, str]) -> None:
    car.position = vec2(as_float(row, "x"), as_float(row, "y"))
    car.heading = as_float(row, "heading")
    car.tire_state.combined_grip_usage = as_float(row, "tire_grip_usage")
    car.tire_state.temperature = as_float(row, "tire_temperature")
    car.tire_state.wear = as_float(row, "tire_wear")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved ApexDriveLab telemetry CSV.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    rows = load_replay(args.path)
    if not rows:
        print(f"No replay rows in {args.path}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"{WINDOW_TITLE} Replay")
    clock = pygame.time.Clock()
    track = Track()
    car = Car()
    frame = 0
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    frame = 0

        if not paused:
            frame = min(frame + 1, len(rows) - 1)
        apply_frame(car, rows[frame])

        screen.fill(BACKGROUND_COLOR)
        track.draw(screen)
        draw_car(screen, car)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
