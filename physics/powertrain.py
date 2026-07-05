"""Powertrain: torque curve, sequential gearbox with shift cuts, power limiting.

Replaces the old constant-acceleration engine. Below the power corner speed
the car is torque/traction limited (flat acceleration, shaped by the torque
curve within each gear); above it the engine is power limited and available
acceleration falls off as P/v - which is why real cars pull hard out of
corners and crawl toward top speed.
"""

import math
from dataclasses import dataclass, field

from config import (
    GEAR_COUNT,
    GEAR_TOP_SPEEDS,
    IDLE_RPM,
    REDLINE_RPM,
    SHIFT_UP_RPM,
    SHIFT_DOWN_RPM,
    SHIFT_CUT_TIME,
    SHIFT_CUT_TORQUE,
    POWER_CORNER_SPEED,
)


def torque_curve(rpm_fraction: float) -> float:
    """Normalised torque vs normalised rpm - broad plateau, soft ends."""
    x = max(0.0, min(rpm_fraction, 1.0))
    return 0.85 + 0.15 * math.sin(math.pi * min(1.0, 0.25 + x * 0.85))


@dataclass
class PowertrainState:
    gear: int = 1
    rpm: float = IDLE_RPM
    rpm_fraction: float = 0.0
    shift_cut_timer: float = 0.0
    engine_torque_fraction: float = 0.0
    power_limited: bool = False

    def reset(self) -> None:
        self.gear = 1
        self.rpm = IDLE_RPM
        self.rpm_fraction = 0.0
        self.shift_cut_timer = 0.0
        self.engine_torque_fraction = 0.0
        self.power_limited = False

    def _rpm_for(self, speed: float, gear: int) -> float:
        top = GEAR_TOP_SPEEDS[gear - 1]
        fraction = max(0.0, min(speed / top, 1.15))
        return IDLE_RPM + (REDLINE_RPM - IDLE_RPM) * fraction

    def update(self, dt: float, speed: float, throttle: float, max_engine_accel: float) -> float:
        """Advance gearbox state and return the available drive acceleration."""
        self.shift_cut_timer = max(0.0, self.shift_cut_timer - dt)

        self.rpm = self._rpm_for(speed, self.gear)
        if self.rpm > SHIFT_UP_RPM and self.gear < GEAR_COUNT:
            self.gear += 1
            self.shift_cut_timer = SHIFT_CUT_TIME
            self.rpm = self._rpm_for(speed, self.gear)
        elif self.rpm < SHIFT_DOWN_RPM and self.gear > 1:
            self.gear -= 1
            self.shift_cut_timer = SHIFT_CUT_TIME * 0.6
            self.rpm = self._rpm_for(speed, self.gear)

        self.rpm_fraction = max(0.0, min((self.rpm - IDLE_RPM) / (REDLINE_RPM - IDLE_RPM), 1.0))

        torque_fraction = torque_curve(self.rpm_fraction)
        if self.shift_cut_timer > 0.0:
            torque_fraction *= SHIFT_CUT_TORQUE

        torque_limited = max_engine_accel * torque_fraction
        peak_power = max_engine_accel * POWER_CORNER_SPEED
        power_limited = peak_power / max(speed, POWER_CORNER_SPEED)
        available = min(torque_limited, power_limited)
        self.power_limited = power_limited < torque_limited
        self.engine_torque_fraction = torque_fraction

        return available * throttle
