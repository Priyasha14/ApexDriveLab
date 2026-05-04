import math
from dataclasses import dataclass, field

import numpy as np

from config import (
    CAR_LENGTH,
    CAR_START_HEADING,
    CAR_START_POSITION,
    CAR_WIDTH,
    LINEAR_DRAG,
    MAX_BRAKE_ACCEL,
    MAX_ENGINE_ACCEL,
    MAX_REVERSE_ACCEL,
    MAX_SPEED_FOR_STEERING,
    MAX_STEER_RATE,
    ROLLING_RESISTANCE,
)
from physics.tires import TireState
from physics.vector_utils import clamp, from_angle, length, rotate_points, vec2


@dataclass
class CarInputs:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


@dataclass
class Car:
    position: np.ndarray = field(default_factory=lambda: vec2(*CAR_START_POSITION))
    velocity: np.ndarray = field(default_factory=lambda: vec2(0.0, 0.0))
    heading: float = CAR_START_HEADING
    acceleration: np.ndarray = field(default_factory=lambda: vec2(0.0, 0.0))
    inputs: CarInputs = field(default_factory=CarInputs)
    tire_state: TireState = field(default_factory=TireState)

    def reset(self) -> None:
        self.position = vec2(*CAR_START_POSITION)
        self.velocity = vec2(0.0, 0.0)
        self.heading = CAR_START_HEADING
        self.acceleration = vec2(0.0, 0.0)
        self.inputs = CarInputs()
        self.tire_state = TireState()

    @property
    def speed(self) -> float:
        return length(self.velocity)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6 / 10.0

    def update(self, dt: float, inputs: CarInputs) -> None:
        self.inputs = inputs
        forward = from_angle(self.heading)
        signed_forward_speed = float(np.dot(self.velocity, forward))

        throttle_accel = inputs.throttle * MAX_ENGINE_ACCEL
        brake_accel = inputs.brake * MAX_BRAKE_ACCEL

        if signed_forward_speed < -15.0:
            drive_accel = throttle_accel * 0.35
        else:
            drive_accel = throttle_accel

        longitudinal_accel = drive_accel
        if self.speed > 2.0:
            longitudinal_accel -= math.copysign(brake_accel, signed_forward_speed or 1.0)
        elif inputs.brake > 0.0 and inputs.throttle <= 0.0:
            longitudinal_accel -= MAX_REVERSE_ACCEL

        drag = -self.velocity * LINEAR_DRAG
        rolling = -forward * math.copysign(ROLLING_RESISTANCE, signed_forward_speed) if self.speed > 1.0 else vec2(0.0, 0.0)
        right = from_angle(self.heading + math.pi / 2.0)
        lateral_speed = float(np.dot(self.velocity, right))
        lateral_grip = -right * lateral_speed * 6.0

        self.acceleration = forward * longitudinal_accel + drag + rolling + lateral_grip
        self.velocity += self.acceleration * dt

        if self.speed < 1.0 and inputs.throttle == 0.0 and inputs.brake == 0.0:
            self.velocity *= 0.0

        speed_factor = clamp(self.speed / MAX_SPEED_FOR_STEERING, 0.0, 1.0)
        steering_strength = 0.22 + 0.78 * speed_factor
        reverse_factor = -1.0 if signed_forward_speed < -5.0 else 1.0
        self.heading += inputs.steer * MAX_STEER_RATE * steering_strength * reverse_factor * dt
        self.position += self.velocity * dt

    def body_polygon(self) -> list[tuple[float, float]]:
        half_l = CAR_LENGTH / 2.0
        half_w = CAR_WIDTH / 2.0
        local_points = [
            (half_l, 0.0),
            (half_l * 0.45, half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
            (half_l * 0.45, -half_w),
        ]
        return rotate_points(local_points, self.heading, self.position)
