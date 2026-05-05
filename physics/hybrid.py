from dataclasses import dataclass


@dataclass
class HybridState:
    capacity: float = 4_000_000.0
    energy: float = 2_800_000.0
    max_deploy_power: float = 120_000.0
    max_recovery_power: float = 90_000.0
    deploy_power: float = 0.0
    recovery_power: float = 0.0

    @property
    def charge_fraction(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return max(0.0, min(self.energy / self.capacity, 1.0))

    def reset(self) -> None:
        self.energy = min(self.capacity, self.capacity * 0.70)
        self.deploy_power = 0.0
        self.recovery_power = 0.0

    def update(self, dt: float, speed_mps: float, deploy_requested: bool, brake: float, throttle: float) -> float:
        self.deploy_power = 0.0
        self.recovery_power = 0.0
        deploy_accel_mps2 = 0.0

        if deploy_requested and throttle > 0.2 and self.energy > 1.0 and speed_mps > 5.0:
            usable_power = min(self.max_deploy_power, self.energy / max(dt, 1e-6))
            self.deploy_power = usable_power
            self.energy -= usable_power * dt
            deploy_accel_mps2 = usable_power / max(speed_mps, 5.0) / 798.0

        recovery_request = max(brake, 0.35 if throttle < 0.05 and speed_mps > 8.0 else 0.0)
        if recovery_request > 0.0 and self.energy < self.capacity:
            self.recovery_power = min(self.max_recovery_power * recovery_request, (self.capacity - self.energy) / max(dt, 1e-6))
            self.energy += self.recovery_power * dt

        self.energy = max(0.0, min(self.energy, self.capacity))
        return deploy_accel_mps2
