import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TelemetryLogger:
    samples: list[dict] = field(default_factory=list)

    def log(self, time_s: float, car, on_track: bool, ai_state=None) -> None:
        self.samples.append(
            {
                "time_s": time_s,
                "speed": car.speed,
                "speed_kmh": car.speed_kmh,
                "throttle": car.inputs.throttle,
                "brake": car.inputs.brake,
                "steer": car.inputs.steer,
                "setup": car.setup.name,
                "aero_mode": car.aero_state.mode,
                "drag_force": car.aero_state.drag_force,
                "downforce": car.aero_state.downforce,
                "front_downforce": car.aero_state.front_downforce,
                "rear_downforce": car.aero_state.rear_downforce,
                "battery_energy": car.hybrid_state.energy,
                "battery_charge": car.hybrid_state.charge_fraction,
                "deployment_power": car.hybrid_state.deploy_power,
                "recovery_power": car.hybrid_state.recovery_power,
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
                "ai_target_speed": getattr(ai_state, "target_speed", 0.0) if ai_state else 0.0,
                "ai_path_error": getattr(ai_state, "path_error", 0.0) if ai_state else 0.0,
                "ai_heading_error": getattr(ai_state, "heading_error", 0.0) if ai_state else 0.0,
                "ai_decision": getattr(ai_state, "decision", "manual") if ai_state else "manual",
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
