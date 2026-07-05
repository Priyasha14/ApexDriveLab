"""Advanced aerodynamics: wing + ground-effect floor, DRS, yaw sensitivity, slipstream.

Downforce is split into two physical sources:

* **Wings** - classic q*Cl*A downforce. The rear wing can open (DRS), dumping
  most of its downforce and a large chunk of its drag.
* **Ground-effect floor** - Venturi tunnels whose downforce grows as the car
  runs closer to the ground. Aero load compresses the (quasi-static)
  suspension, lowering ride height, which increases floor downforce - a real
  nonlinearity of modern F1 cars. Below a stall threshold the floor chokes and
  sheds load abruptly (the mechanism behind porpoising).

Yaw/steer angle bleeds downforce, because wings and floors are designed for
clean, straight onset flow. A slipstream helper models running in another
car's wake: less drag (tow) but also less downforce (dirty air).
"""

from dataclasses import dataclass

from config import (
    DRS_DRAG_MULTIPLIER,
    DRS_MIN_SPEED_KMH,
    DRS_REAR_DOWNFORCE_MULTIPLIER,
    DRS_MAX_STEER,
    GROUND_EFFECT_FRACTION,
    GROUND_EFFECT_GAIN,
    RIDE_HEIGHT_STATIC,
    RIDE_HEIGHT_MIN,
    RIDE_HEIGHT_STALL,
    FLOOR_STALL_MULTIPLIER,
    YAW_DOWNFORCE_LOSS,
)


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
    # --- new channels (advanced model) ---
    wing_downforce: float = 0.0
    ground_effect_downforce: float = 0.0
    drs_active: bool = False
    floor_stalled: bool = False
    ride_height: float = RIDE_HEIGHT_STATIC
    aero_efficiency: float = 0.0  # downforce / drag (L/D)


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


def drs_available(speed_kmh: float, steer: float, throttle: float) -> bool:
    """DRS opens automatically on straights: fast, near-zero steer, full send."""
    return speed_kmh > DRS_MIN_SPEED_KMH and abs(steer) < DRS_MAX_STEER and throttle > 0.7


def ride_height_from_load(downforce_acceleration: float) -> float:
    """Quasi-static suspension: aero load compresses the car toward the floor."""
    compression = downforce_acceleration * 0.00085
    return max(RIDE_HEIGHT_MIN, RIDE_HEIGHT_STATIC - compression)


def slipstream_multipliers(gap_lengths: float) -> tuple[float, float]:
    """(drag_multiplier, downforce_multiplier) for following another car.

    gap_lengths: distance to the car ahead, in car lengths. Close behind you
    get a strong tow (less drag) but lose downforce in the dirty air.
    """
    if gap_lengths >= 8.0 or gap_lengths < 0.0:
        return 1.0, 1.0
    proximity = 1.0 - gap_lengths / 8.0
    return 1.0 - 0.35 * proximity, 1.0 - 0.30 * proximity


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
    steer_fraction: float = 0.0,
    ride_height: float = RIDE_HEIGHT_STATIC,
    drs_open: bool = False,
    slipstream_drag_multiplier: float = 1.0,
    slipstream_downforce_multiplier: float = 1.0,
) -> AeroState:
    mode = AERO_MODES[mode_name]
    dynamic_pressure = 0.5 * air_density * speed_mps * speed_mps

    # Yaw sensitivity: downforce bleeds away as the car takes steering/yaw.
    yaw_loss = 1.0 - YAW_DOWNFORCE_LOSS * min(1.0, abs(steer_fraction))

    # --- Wing downforce (subject to mode + DRS on the rear element) ---
    wing_cl = lift_coefficient * (1.0 - GROUND_EFFECT_FRACTION)
    wing_downforce = dynamic_pressure * wing_cl * frontal_area * mode.downforce_multiplier
    front_wing = wing_downforce * aero_balance_front
    rear_wing = wing_downforce - front_wing
    if drs_open:
        rear_wing *= DRS_REAR_DOWNFORCE_MULTIPLIER

    # --- Ground-effect floor (subject to ride height, immune to DRS) ---
    floor_cl = lift_coefficient * GROUND_EFFECT_FRACTION
    height_gain = 1.0 + GROUND_EFFECT_GAIN * max(0.0, (RIDE_HEIGHT_STATIC - ride_height) / RIDE_HEIGHT_STATIC)
    floor_stalled = ride_height <= RIDE_HEIGHT_STALL
    if floor_stalled:
        height_gain *= FLOOR_STALL_MULTIPLIER
    ground_effect = dynamic_pressure * floor_cl * frontal_area * height_gain
    front_floor = ground_effect * 0.5
    rear_floor = ground_effect * 0.5

    front_downforce = (front_wing + front_floor) * yaw_loss * slipstream_downforce_multiplier
    rear_downforce = (rear_wing + rear_floor) * yaw_loss * slipstream_downforce_multiplier
    downforce = front_downforce + rear_downforce

    # --- Drag ---
    drag_multiplier = mode.drag_multiplier
    if drs_open:
        drag_multiplier *= DRS_DRAG_MULTIPLIER
    drag_force = dynamic_pressure * drag_coefficient * frontal_area * drag_multiplier * slipstream_drag_multiplier

    return AeroState(
        mode=mode.name,
        drag_force=drag_force,
        downforce=downforce,
        front_downforce=front_downforce,
        rear_downforce=rear_downforce,
        drag_acceleration=drag_force / mass * sim_accel_scale,
        downforce_acceleration=downforce / mass * sim_accel_scale,
        wing_downforce=(front_wing + rear_wing) * yaw_loss,
        ground_effect_downforce=ground_effect * yaw_loss,
        drs_active=drs_open,
        floor_stalled=floor_stalled,
        ride_height=ride_height,
        aero_efficiency=downforce / drag_force if drag_force > 1e-6 else 0.0,
    )
