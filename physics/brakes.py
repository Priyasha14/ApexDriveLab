"""Brake thermodynamics: carbon discs need heat to bite, and fade when cooked."""

from dataclasses import dataclass

from config import (
    BRAKE_AMBIENT_TEMP,
    BRAKE_COLD_TEMP,
    BRAKE_FADE_TEMP,
    BRAKE_MAX_TEMP,
    BRAKE_HEATING_RATE,
    BRAKE_COOLING_RATE,
)


@dataclass
class BrakeState:
    temperature: float = BRAKE_AMBIENT_TEMP
    performance: float = 1.0

    def reset(self) -> None:
        self.temperature = BRAKE_AMBIENT_TEMP
        self.performance = 1.0

    def update(self, dt: float, brake_input: float, speed: float) -> float:
        heating = brake_input * speed * BRAKE_HEATING_RATE
        cooling = (self.temperature - BRAKE_AMBIENT_TEMP) * (BRAKE_COOLING_RATE + speed * 0.00035)
        self.temperature = min(BRAKE_MAX_TEMP, max(BRAKE_AMBIENT_TEMP, self.temperature + (heating - cooling) * dt))

        if self.temperature < BRAKE_COLD_TEMP:
            self.performance = 0.88 + 0.12 * (self.temperature - BRAKE_AMBIENT_TEMP) / max(BRAKE_COLD_TEMP - BRAKE_AMBIENT_TEMP, 1.0)
        elif self.temperature <= BRAKE_FADE_TEMP:
            self.performance = 1.0
        else:
            overheat = (self.temperature - BRAKE_FADE_TEMP) / max(BRAKE_MAX_TEMP - BRAKE_FADE_TEMP, 1.0)
            self.performance = max(0.70, 1.0 - overheat * 0.30)
        return self.performance
