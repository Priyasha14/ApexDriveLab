from dataclasses import dataclass


@dataclass(frozen=True)
class AeroMode:
    name: str
    drag_multiplier: float
    downforce_multiplier: float


@dataclass
class AeroState:
    mode: str = "corner"
    drag_force: float = 0.0
    downforce: float = 0.0
    front_downforce: float = 0.0
    rear_downforce: float = 0.0
    drag_acceleration: float = 0.0
    downforce_acceleration: float = 0.0


AERO_MODES = {
    "corner": AeroMode(name="corner", drag_multiplier=1.0, downforce_multiplier=1.0),
    "straight": AeroMode(name="straight", drag_multiplier=0.62, downforce_multiplier=0.58),
}


def choose_aero_mode(speed_kmh: float, steer: float, requested_mode: str | None) -> str:
    if requested_mode in AERO_MODES:
        return requested_mode
    if speed_kmh > 120.0 and abs(steer) < 0.18:
        return "straight"
    return "corner"


def compute_aero_state(
    speed_mps: float,
    mode_name: str,
    air_density: float,
    drag_coefficient: float,
    lift_coefficient: float,
    frontal_area: float,
    aero_balance_front: float,
    mass: float,
    sim_accel_scale: float,
) -> AeroState:
    mode = AERO_MODES[mode_name]
    dynamic_pressure = 0.5 * air_density * speed_mps * speed_mps
    drag_force = dynamic_pressure * drag_coefficient * frontal_area * mode.drag_multiplier
    downforce = dynamic_pressure * lift_coefficient * frontal_area * mode.downforce_multiplier
    front_downforce = downforce * aero_balance_front
    rear_downforce = downforce - front_downforce

    return AeroState(
        mode=mode.name,
        drag_force=drag_force,
        downforce=downforce,
        front_downforce=front_downforce,
        rear_downforce=rear_downforce,
        drag_acceleration=drag_force / mass * sim_accel_scale,
        downforce_acceleration=downforce / mass * sim_accel_scale,
    )
