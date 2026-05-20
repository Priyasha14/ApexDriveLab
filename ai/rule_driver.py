from dataclasses import dataclass, field

from ai.pure_pursuit import steering_to_target, wrap_angle
from ai.racing_line import RacingLine
from ai.speed_planner import plan_speed
from physics.car import CarInputs
from physics.vector_utils import clamp


@dataclass
class DriverParameters:
    lookahead_distance: float = 92.0
    speed_lookahead_gain: float = 0.38
    corner_speed_multiplier: float = 0.70
    braking_safety_margin: float = 1.45
    throttle_aggressiveness: float = 0.018
    brake_aggressiveness: float = 0.030
    aero_switch_speed: float = 118.0
    hybrid_deploy_speed: float = 80.0
    enable_active_aero: bool = True
    enable_hybrid: bool = True


@dataclass
class DriverState:
    target_speed: float = 0.0
    path_error: float = 0.0
    heading_error: float = 0.0
    decision: str = "idle"
    target_aero_mode: str = "corner"


@dataclass
class RuleBasedDriver:
    racing_line: RacingLine
    params: DriverParameters = field(default_factory=DriverParameters)
    state: DriverState = field(default_factory=DriverState)

    def control(self, car, track) -> CarInputs:
        angle = track.angle_for_position(car.position)
        lookahead = self.params.lookahead_distance + car.speed_kmh * self.params.speed_lookahead_gain
        target = self.racing_line.target_ahead(angle, lookahead)
        steer, heading_error = steering_to_target(car, target)
        speed_plan = plan_speed(angle, car.speed_kmh, track.outer_radius, self.params.corner_speed_multiplier, self.params.braking_safety_margin)

        speed_error = speed_plan.target_speed_kmh - car.speed_kmh
        throttle = clamp(speed_error * self.params.throttle_aggressiveness, 0.0, 1.0)
        brake = clamp(-speed_error * self.params.brake_aggressiveness, 0.0, 1.0)

        if speed_plan.braking_distance > lookahead * 0.72:
            brake = max(brake, 0.35)
            throttle = 0.0

        aero_mode = "straight" if self.params.enable_active_aero and speed_plan.target_speed_kmh > self.params.aero_switch_speed and abs(steer) < 0.22 else "corner"
        deploy = self.params.enable_hybrid and speed_error > 12.0 and car.speed_kmh > self.params.hybrid_deploy_speed and car.hybrid_state.charge_fraction > 0.18

        self.state = DriverState(
            target_speed=speed_plan.target_speed_kmh,
            path_error=self.racing_line.path_error(car.position, angle),
            heading_error=wrap_angle(heading_error),
            decision="brake" if brake > 0.05 else "deploy" if deploy else "throttle" if throttle > 0.05 else "coast",
            target_aero_mode=aero_mode,
        )
        return CarInputs(throttle=throttle, brake=brake, steer=steer, deploy_hybrid=deploy, aero_mode=aero_mode)
