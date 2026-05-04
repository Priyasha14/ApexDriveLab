import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TelemetryLogger:
    samples: list[dict] = field(default_factory=list)

    def log(self, time_s: float, car, on_track: bool) -> None:
        self.samples.append(
            {
                "time_s": time_s,
                "speed": car.speed,
                "speed_kmh": car.speed_kmh,
                "throttle": car.inputs.throttle,
                "brake": car.inputs.brake,
                "steer": car.inputs.steer,
                "steering_angle": car.steering_angle,
                "yaw_rate": car.yaw_rate,
                "ax": float(car.acceleration[0]),
                "ay": float(car.acceleration[1]),
                "longitudinal_acceleration": car.longitudinal_acceleration,
                "lateral_acceleration": car.lateral_acceleration,
                "front_slip_angle": car.tire_state.front_slip_angle,
                "rear_slip_angle": car.tire_state.rear_slip_angle,
                "front_grip_usage": car.tire_state.front_grip_usage,
                "rear_grip_usage": car.tire_state.rear_grip_usage,
                "tire_grip_usage": car.tire_state.combined_grip_usage,
                "front_load": car.tire_state.front_load,
                "rear_load": car.tire_state.rear_load,
                "lateral_load_transfer": car.lateral_load_transfer,
                "handling_balance": car.handling_balance,
                "on_track": on_track,
            }
        )

    def clear(self) -> None:
        self.samples.clear()

    def save_csv(self, path: Path) -> Path | None:
        if not self.samples:
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.samples[0].keys())
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)
        return path
