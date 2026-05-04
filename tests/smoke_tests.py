from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.car import Car, CarInputs
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track


def test_straight_line_stability() -> None:
    car = Car()
    for _ in range(180):
        car.update(1 / 60, CarInputs(throttle=1.0), 1.0)

    assert car.speed_kmh > 80.0
    assert abs(car.yaw_rate) < 0.01
    assert car.handling_balance == "neutral"


def test_steering_produces_tire_state() -> None:
    car = Car()
    for _ in range(90):
        car.update(1 / 60, CarInputs(throttle=1.0, steer=-0.25), 1.0)

    assert abs(car.steering_angle) > 0.05
    assert abs(car.tire_state.front_slip_angle) > 0.01
    assert car.tire_state.combined_grip_usage > 0.0


def test_clockwise_checkpoint_lap() -> None:
    checkpoints = CheckpointManager()
    checkpoints.reset(0)

    for checkpoint_index in [7, 6, 5, 4, 3, 2, 1, 0]:
        checkpoints.update(0.1, checkpoint_index)

    assert checkpoints.state.lap_count == 1
    assert checkpoints.state.current_checkpoint == 0
    assert checkpoints.state.last_lap_time is not None


def test_telemetry_export(tmp_path: Path) -> None:
    car = Car()
    logger = TelemetryLogger()
    car.update(1 / 60, CarInputs(throttle=1.0, steer=-0.2), 1.0)
    logger.log(1 / 60, car, True)

    path = logger.save_csv(tmp_path / "telemetry.csv")

    assert path is not None
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "lateral_acceleration" in header
    assert "front_slip_angle" in header
    assert "tire_grip_usage" in header


def run_all() -> None:
    temp_dir = Path("runs") / "_smoke"
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_straight_line_stability()
    test_steering_produces_tire_state()
    test_clockwise_checkpoint_lap()
    test_telemetry_export(temp_dir)
    print("smoke tests passed")


if __name__ == "__main__":
    run_all()
