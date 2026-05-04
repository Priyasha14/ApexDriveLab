from dataclasses import dataclass, field


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

