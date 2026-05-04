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
                "ax": float(car.acceleration[0]),
                "ay": float(car.acceleration[1]),
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
