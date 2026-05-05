import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.racing_line import RacingLine
from ai.rule_driver import DriverParameters, RuleBasedDriver
from config import FPS, OFF_TRACK_GRIP_SCALE
from physics.car import Car
from physics.setup import SETUPS
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track


@dataclass
class LapResult:
    completed_laps: int
    elapsed_time: float
    off_track_pct: float
    max_grip_usage: float
    best_lap_time: float | None
    average_lap_time: float | None


def run_ai_laps(
    laps: int = 1,
    max_time: float = 120.0,
    setup_name: str = "balanced",
    params: DriverParameters | None = None,
    save_telemetry: bool = False,
) -> LapResult:
    track = Track()
    racing_line = RacingLine(track.center, track.inner_radius, track.outer_radius)
    driver = RuleBasedDriver(racing_line, params or DriverParameters())
    car = Car()
    car.apply_setup(SETUPS.get(setup_name, SETUPS["balanced"]))
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))
    telemetry = TelemetryLogger()

    dt = 1.0 / FPS
    elapsed = 0.0
    off_track_samples = 0
    samples = 0
    max_grip = 0.0

    while elapsed < max_time and checkpoints.state.lap_count < laps:
        on_track = track.is_on_track(car.position)
        grip_scale = 1.0 if on_track else OFF_TRACK_GRIP_SCALE
        inputs = driver.control(car, track)
        car.update(dt, inputs, grip_scale)
        on_track = track.is_on_track(car.position)
        checkpoints.update(dt, track.checkpoint_index_for_position(car.position))

        elapsed += dt
        samples += 1
        off_track_samples += 0 if on_track else 1
        max_grip = max(max_grip, car.tire_state.combined_grip_usage)
        telemetry.log(elapsed, car, on_track)

    if save_telemetry:
        telemetry.save_csv(Path("runs") / "ai_lap.csv")

    return LapResult(
        completed_laps=checkpoints.state.lap_count,
        elapsed_time=elapsed,
        off_track_pct=off_track_samples / max(samples, 1) * 100.0,
        max_grip_usage=max_grip,
        best_lap_time=checkpoints.state.best_lap_time,
        average_lap_time=elapsed / checkpoints.state.lap_count if checkpoints.state.lap_count else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based driver in headless simulation.")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--setup", default="balanced")
    parser.add_argument("--save-telemetry", action="store_true")
    args = parser.parse_args()

    result = run_ai_laps(laps=args.laps, max_time=args.max_time, setup_name=args.setup, save_telemetry=args.save_telemetry)
    print(result)


if __name__ == "__main__":
    main()
