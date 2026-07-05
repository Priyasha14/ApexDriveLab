"""Vehicle dynamics core: nonlinear bicycle model.

Pipeline each step:
  1. Suspension: previous-frame aero load sets ride height (ground effect).
  2. Aerodynamics: wing + floor downforce, DRS, yaw sensitivity, drag.
  3. Powertrain: torque curve + gearbox -> drive acceleration, plus ERS deploy.
  4. Brakes: thermal model scales braking performance.
  5. Tires: Pacejka Magic Formula lateral forces per axle with load
     sensitivity, per-axle temperature and wear; combined-slip friction circle.
  6. Rigid body: integrate accelerations, yaw dynamics, position.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from config import (
    AIR_DENSITY,
    CAR_LENGTH,
    CAR_MASS,
    CAR_START_HEADING,
    CAR_START_POSITION,
    CAR_WIDTH,
    FRONTAL_AREA,
    FRONT_AXLE_DISTANCE,
    LINEAR_DRAG,
    LOAD_TRANSFER_LATERAL,
    LOAD_TRANSFER_LONGITUDINAL,
    MAX_REVERSE_ACCEL,
    MAX_STEER_ANGLE,
    MIN_SLIP_SPEED,
    OFF_TRACK_DRAG_MULTIPLIER,
    REAR_AXLE_DISTANCE,
    ROLLING_RESISTANCE,
    SIM_ACCEL_SCALE,
    STATIC_FRONT_LOAD,
    STATIC_REAR_LOAD,
    STEER_RESPONSE,
    RIDE_HEIGHT_STATIC,
)
from physics.aero import (
    AeroState,
    choose_aero_mode,
    compute_aero_state,
    drs_available,
    ride_height_from_load,
)
from physics.brakes import BrakeState
from physics.hybrid import HybridState
from physics.powertrain import PowertrainState
from physics.setup import SETUPS, CarSetup
from physics.tires import (
    TireState,
    load_sensitivity_multiplier,
    magic_formula,
    peak_slip_angle,
    temperature_grip_multiplier,
    wear_grip_multiplier,
)
from physics.vector_utils import clamp, from_angle, length, rotate_points, vec2


@dataclass
class CarInputs:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    deploy_hybrid: bool = False
    aero_mode: str | None = None
    drs: bool | None = None  # None = automatic, True/False = manual override


@dataclass
class Car:
    position: np.ndarray = field(default_factory=lambda: vec2(*CAR_START_POSITION))
    velocity: np.ndarray = field(default_factory=lambda: vec2(0.0, 0.0))
    heading: float = CAR_START_HEADING
    yaw_rate: float = 0.0
    steering_angle: float = 0.0
    acceleration: np.ndarray = field(default_factory=lambda: vec2(0.0, 0.0))
    longitudinal_acceleration: float = 0.0
    lateral_acceleration: float = 0.0
    lateral_load_transfer: float = 0.0
    handling_balance: str = "neutral"
    setup: CarSetup = field(default_factory=lambda: SETUPS["balanced"])
    aero_state: AeroState = field(default_factory=AeroState)
    hybrid_state: HybridState = field(default_factory=HybridState)
    inputs: CarInputs = field(default_factory=CarInputs)
    tire_state: TireState = field(default_factory=TireState)
    powertrain_state: PowertrainState = field(default_factory=PowertrainState)
    brake_state: BrakeState = field(default_factory=BrakeState)
    ride_height: float = RIDE_HEIGHT_STATIC

    def reset(self) -> None:
        self.position = vec2(*CAR_START_POSITION)
        self.velocity = vec2(0.0, 0.0)
        self.heading = CAR_START_HEADING
        self.yaw_rate = 0.0
        self.steering_angle = 0.0
        self.acceleration = vec2(0.0, 0.0)
        self.longitudinal_acceleration = 0.0
        self.lateral_acceleration = 0.0
        self.lateral_load_transfer = 0.0
        self.handling_balance = "neutral"
        self.inputs = CarInputs()
        self.tire_state = TireState()
        self.aero_state = AeroState()
        self.hybrid_state.reset()
        self.powertrain_state.reset()
        self.brake_state.reset()
        self.ride_height = RIDE_HEIGHT_STATIC

    @property
    def speed(self) -> float:
        return length(self.velocity)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 3.6 / 10.0

    def update(self, dt: float, inputs: CarInputs, grip_scale: float = 1.0) -> None:
        self.inputs = inputs
        forward = from_angle(self.heading)
        right = from_angle(self.heading + math.pi / 2.0)
        vx = float(np.dot(self.velocity, forward))
        vy = float(np.dot(self.velocity, right))
        speed_mps = self.speed / SIM_ACCEL_SCALE

        # ------ 1. Suspension / ride height from last frame's aero load ------
        self.ride_height = ride_height_from_load(self.aero_state.downforce_acceleration)

        # ------ 2. Aerodynamics ------
        aero_mode = choose_aero_mode(self.speed_kmh, inputs.steer, inputs.aero_mode)
        if inputs.drs is None:
            drs_open = drs_available(self.speed_kmh, inputs.steer, inputs.throttle)
        else:
            drs_open = inputs.drs and drs_available(self.speed_kmh, inputs.steer, max(inputs.throttle, 0.71))
        self.aero_state = compute_aero_state(
            speed_mps=speed_mps,
            mode_name=aero_mode,
            air_density=AIR_DENSITY,
            drag_coefficient=self.setup.drag_coefficient,
            lift_coefficient=self.setup.downforce_coefficient,
            frontal_area=FRONTAL_AREA,
            aero_balance_front=self.setup.aero_balance_front,
            mass=CAR_MASS,
            sim_accel_scale=SIM_ACCEL_SCALE,
            steer_fraction=inputs.steer,
            ride_height=self.ride_height,
            drs_open=drs_open,
        )

        # ------ Steering with speed-sensitive authority ------
        high_speed_tightening = 1.0 / (1.0 + (self.speed_kmh / 320.0) ** 2)
        target_steering = inputs.steer * MAX_STEER_ANGLE * (0.55 + 0.45 * high_speed_tightening)
        response = clamp(STEER_RESPONSE * dt, 0.0, 1.0)
        self.steering_angle += (target_steering - self.steering_angle) * response

        # ------ 3. Powertrain + hybrid ------
        hybrid_accel = self.hybrid_state.update(dt, speed_mps, inputs.deploy_hybrid, inputs.brake, inputs.throttle) * SIM_ACCEL_SCALE
        drive_accel = self.powertrain_state.update(dt, self.speed, inputs.throttle, self.setup.engine_accel) + hybrid_accel

        # ------ 4. Brakes with thermal fade ------
        brake_performance = self.brake_state.update(dt, inputs.brake, self.speed)
        brake_accel = inputs.brake * self.setup.brake_accel * brake_performance

        rear_longitudinal = drive_accel if vx > -15.0 else drive_accel * 0.35
        front_longitudinal = 0.0
        if self.speed > 2.0:
            brake_direction = -math.copysign(1.0, vx or 1.0)
            front_longitudinal += brake_direction * brake_accel * self.setup.brake_bias_front
            rear_longitudinal += brake_direction * brake_accel * (1.0 - self.setup.brake_bias_front)
        elif inputs.brake > 0.0 and inputs.throttle <= 0.0:
            rear_longitudinal -= MAX_REVERSE_ACCEL

        # ------ 5. Tires: loads, Pacejka lateral forces, friction circle ------
        front_slip, rear_slip = self._slip_angles(vx, vy)
        front_load, rear_load, lateral_transfer = self._load_distribution(
            rear_longitudinal + front_longitudinal,
            self.lateral_acceleration,
        )

        aero_front_load = self.aero_state.front_downforce / CAR_MASS * SIM_ACCEL_SCALE / max(self.setup.tire_grip_accel, 1.0)
        aero_rear_load = self.aero_state.rear_downforce / CAR_MASS * SIM_ACCEL_SCALE / max(self.setup.tire_grip_accel, 1.0)

        front_condition = temperature_grip_multiplier(self.tire_state.front_temperature) * wear_grip_multiplier(self.tire_state.front_wear)
        rear_condition = temperature_grip_multiplier(self.tire_state.rear_temperature) * wear_grip_multiplier(self.tire_state.rear_wear)

        front_limit = (
            self.setup.tire_grip_accel
            * (front_load + aero_front_load)
            * load_sensitivity_multiplier(front_load)
            * grip_scale
            * front_condition
        )
        rear_limit = (
            self.setup.tire_grip_accel
            * (rear_load + aero_rear_load)
            * load_sensitivity_multiplier(rear_load)
            * grip_scale
            * rear_condition
        )

        front_lateral = magic_formula(front_slip, self.setup.front_cornering_stiffness, front_limit)
        rear_lateral = magic_formula(rear_slip, self.setup.rear_cornering_stiffness, rear_limit)

        front_longitudinal, front_lateral, front_usage = self._apply_friction_circle(front_longitudinal, front_lateral, front_limit)
        rear_longitudinal, rear_lateral, rear_usage = self._apply_friction_circle(rear_longitudinal, rear_lateral, rear_limit)

        # ------ Resistive forces ------
        drag_multiplier = 1.0 if grip_scale >= 0.99 else OFF_TRACK_DRAG_MULTIPLIER
        drag = -vx * abs(vx) * LINEAR_DRAG * 0.004 * drag_multiplier
        rolling = -math.copysign(ROLLING_RESISTANCE, vx) if abs(vx) > 1.0 else 0.0
        aero_drag = -math.copysign(self.aero_state.drag_acceleration, vx) if abs(vx) > 1.0 else 0.0

        # ------ 6. Rigid body integration ------
        local_ax = front_longitudinal + rear_longitudinal + drag + rolling + aero_drag
        local_ay = front_lateral + rear_lateral
        self.longitudinal_acceleration = local_ax
        self.lateral_acceleration = local_ay
        self.lateral_load_transfer = lateral_transfer
        self.acceleration = forward * local_ax + right * local_ay
        self.velocity += self.acceleration * dt

        yaw_accel = (FRONT_AXLE_DISTANCE * front_lateral - REAR_AXLE_DISTANCE * rear_lateral) * self.setup.yaw_response
        self.yaw_rate += yaw_accel * dt
        self.yaw_rate *= max(0.0, 1.0 - self.setup.yaw_damping * dt)
        self.heading += self.yaw_rate * dt
        self.position += self.velocity * dt
        self._settle_when_stopped(inputs)
        self._update_tire_state(
            front_slip,
            rear_slip,
            front_lateral,
            rear_lateral,
            front_longitudinal,
            rear_longitudinal,
            front_load,
            rear_load,
            front_usage,
            rear_usage,
            front_limit,
            rear_limit,
            dt,
        )

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

    def apply_setup(self, setup: CarSetup) -> None:
        self.setup = setup

    def set_tire_condition(self, temperature: float | None = None, wear: float | None = None) -> None:
        if temperature is not None:
            self.tire_state.temperature = temperature
            self.tire_state.front_temperature = temperature
            self.tire_state.rear_temperature = temperature
        if wear is not None:
            wear = clamp(wear, 0.0, 1.0)
            self.tire_state.wear = wear
            self.tire_state.front_wear = wear
            self.tire_state.rear_wear = wear

    def _slip_angles(self, vx: float, vy: float) -> tuple[float, float]:
        speed_for_slip = math.copysign(max(abs(vx), MIN_SLIP_SPEED), vx if abs(vx) > 1.0 else 1.0)
        front_slip = math.atan2(vy + FRONT_AXLE_DISTANCE * self.yaw_rate, abs(speed_for_slip)) - self.steering_angle
        rear_slip = math.atan2(vy - REAR_AXLE_DISTANCE * self.yaw_rate, abs(speed_for_slip))
        return front_slip, rear_slip

    def _load_distribution(self, longitudinal_accel: float, lateral_accel: float) -> tuple[float, float, float]:
        transfer = clamp(longitudinal_accel * LOAD_TRANSFER_LONGITUDINAL, -0.18, 0.18)
        front_load = STATIC_FRONT_LOAD - transfer
        rear_load = STATIC_REAR_LOAD + transfer
        lateral_transfer = clamp(abs(lateral_accel) * LOAD_TRANSFER_LATERAL, 0.0, 0.22)
        front_load = clamp(front_load, 0.30, 0.70)
        rear_load = clamp(rear_load, 0.30, 0.70)
        total = front_load + rear_load
        return front_load / total, rear_load / total, lateral_transfer

    def _apply_friction_circle(self, longitudinal: float, lateral: float, limit: float) -> tuple[float, float, float]:
        demand = math.hypot(longitudinal, lateral)
        if limit <= 1e-6:
            return 0.0, 0.0, 0.0
        if demand <= limit:
            return longitudinal, lateral, demand / limit
        scale = limit / demand
        return longitudinal * scale, lateral * scale, 1.0

    def _settle_when_stopped(self, inputs: CarInputs) -> None:
        if self.speed < 1.0 and inputs.throttle == 0.0 and inputs.brake == 0.0:
            self.velocity *= 0.0
            self.yaw_rate = 0.0

    def _update_tire_state(
        self,
        front_slip: float,
        rear_slip: float,
        front_lateral: float,
        rear_lateral: float,
        front_longitudinal: float,
        rear_longitudinal: float,
        front_load: float,
        rear_load: float,
        front_usage: float,
        rear_usage: float,
        front_limit: float,
        rear_limit: float,
        dt: float,
    ) -> None:
        previous = self.tire_state
        front_lateral_usage = abs(front_lateral) / max(self.setup.tire_grip_accel * front_load, 1.0)
        rear_lateral_usage = abs(rear_lateral) / max(self.setup.tire_grip_accel * rear_load, 1.0)
        lateral_usage = max(front_lateral_usage, rear_lateral_usage)
        longitudinal_usage = max(abs(front_longitudinal), abs(rear_longitudinal)) / max(self.setup.tire_grip_accel, 1.0)
        combined_usage = max(front_usage, rear_usage)
        if front_lateral_usage > rear_lateral_usage + 0.08:
            self.handling_balance = "understeer"
        elif rear_lateral_usage > front_lateral_usage + 0.08:
            self.handling_balance = "oversteer"
        else:
            self.handling_balance = "neutral"

        # Peak slip angles and saturation (how far past the MF peak each axle is).
        front_peak = peak_slip_angle(self.setup.front_cornering_stiffness, max(front_limit, 1.0))
        rear_peak = peak_slip_angle(self.setup.rear_cornering_stiffness, max(rear_limit, 1.0))
        front_saturation = abs(front_slip) / max(front_peak, 1e-6)
        rear_saturation = abs(rear_slip) / max(rear_peak, 1e-6)

        # Per-axle thermals: sliding (post-peak) heats tires far faster than grip usage alone.
        front_slide_heat = max(0.0, front_saturation - 1.0) * 22.0
        rear_slide_heat = max(0.0, rear_saturation - 1.0) * 22.0
        front_target = 62.0 + front_usage * 45.0 + abs(self.lateral_acceleration) * 0.006 + front_slide_heat
        rear_target = 62.0 + rear_usage * 45.0 + abs(self.lateral_acceleration) * 0.006 + rear_slide_heat
        front_temperature = previous.front_temperature + (front_target - previous.front_temperature) * min(1.0, dt * 0.18)
        rear_temperature = previous.rear_temperature + (rear_target - previous.rear_temperature) * min(1.0, dt * 0.18)

        # Per-axle wear: quadratic in usage, with a sliding penalty.
        front_wear = previous.front_wear + (front_usage * front_usage + max(0.0, front_saturation - 1.0) * 0.6) * dt * 0.00075
        rear_wear = previous.rear_wear + (rear_usage * rear_usage + max(0.0, rear_saturation - 1.0) * 0.6) * dt * 0.00075
        front_wear = clamp(front_wear, 0.0, 1.0)
        rear_wear = clamp(rear_wear, 0.0, 1.0)

        temperature = (front_temperature + rear_temperature) / 2.0
        wear = max(front_wear, rear_wear)
        condition = (
            temperature_grip_multiplier(front_temperature) * wear_grip_multiplier(front_wear)
            + temperature_grip_multiplier(rear_temperature) * wear_grip_multiplier(rear_wear)
        ) / 2.0

        self.tire_state = TireState(
            temperature=temperature,
            wear=wear,
            condition_grip_multiplier=clamp(condition, 0.30, 1.05),
            front_slip_angle=front_slip,
            rear_slip_angle=rear_slip,
            front_lateral_force=front_lateral,
            rear_lateral_force=rear_lateral,
            front_longitudinal_force=front_longitudinal,
            rear_longitudinal_force=rear_longitudinal,
            front_load=front_load,
            rear_load=rear_load,
            lateral_grip_usage=clamp(lateral_usage, 0.0, 1.0),
            longitudinal_grip_usage=clamp(longitudinal_usage, 0.0, 1.0),
            combined_grip_usage=combined_usage,
            front_grip_usage=front_usage,
            rear_grip_usage=rear_usage,
            front_temperature=front_temperature,
            rear_temperature=rear_temperature,
            front_wear=front_wear,
            rear_wear=rear_wear,
            front_peak_slip=front_peak,
            rear_peak_slip=rear_peak,
            front_saturation=front_saturation,
            rear_saturation=rear_saturation,
        )
