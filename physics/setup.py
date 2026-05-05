from dataclasses import dataclass

from config import (
    BRAKE_BIAS_FRONT,
    AERO_BALANCE_FRONT,
    DOWNFORCE_COEFFICIENT,
    DRAG_COEFFICIENT,
    FRONT_CORNERING_STIFFNESS,
    MAX_BRAKE_ACCEL,
    MAX_ENGINE_ACCEL,
    REAR_CORNERING_STIFFNESS,
    TIRE_GRIP_ACCEL,
    YAW_DAMPING,
    YAW_RESPONSE,
)


@dataclass(frozen=True)
class CarSetup:
    name: str
    engine_accel: float = MAX_ENGINE_ACCEL
    brake_accel: float = MAX_BRAKE_ACCEL
    brake_bias_front: float = BRAKE_BIAS_FRONT
    front_cornering_stiffness: float = FRONT_CORNERING_STIFFNESS
    rear_cornering_stiffness: float = REAR_CORNERING_STIFFNESS
    tire_grip_accel: float = TIRE_GRIP_ACCEL
    yaw_response: float = YAW_RESPONSE
    yaw_damping: float = YAW_DAMPING
    drag_coefficient: float = DRAG_COEFFICIENT
    downforce_coefficient: float = DOWNFORCE_COEFFICIENT
    aero_balance_front: float = AERO_BALANCE_FRONT


SETUPS = {
    "balanced": CarSetup(name="balanced"),
    "stable": CarSetup(
        name="stable",
        brake_bias_front=BRAKE_BIAS_FRONT + 0.04,
        front_cornering_stiffness=FRONT_CORNERING_STIFFNESS * 1.04,
        rear_cornering_stiffness=REAR_CORNERING_STIFFNESS * 0.94,
        yaw_damping=YAW_DAMPING * 1.18,
    ),
    "rotation": CarSetup(
        name="rotation",
        brake_bias_front=BRAKE_BIAS_FRONT - 0.03,
        front_cornering_stiffness=FRONT_CORNERING_STIFFNESS * 0.98,
        rear_cornering_stiffness=REAR_CORNERING_STIFFNESS * 1.08,
        yaw_response=YAW_RESPONSE * 1.16,
        yaw_damping=YAW_DAMPING * 0.92,
    ),
    "high_downforce": CarSetup(
        name="high_downforce",
        drag_coefficient=DRAG_COEFFICIENT * 1.10,
        downforce_coefficient=DOWNFORCE_COEFFICIENT * 1.18,
    ),
    "low_drag": CarSetup(
        name="low_drag",
        drag_coefficient=DRAG_COEFFICIENT * 0.86,
        downforce_coefficient=DOWNFORCE_COEFFICIENT * 0.82,
    ),
    "front_aero": CarSetup(
        name="front_aero",
        aero_balance_front=0.50,
        front_cornering_stiffness=FRONT_CORNERING_STIFFNESS * 1.03,
    ),
    "rear_aero": CarSetup(
        name="rear_aero",
        aero_balance_front=0.40,
        rear_cornering_stiffness=REAR_CORNERING_STIFFNESS * 1.03,
    ),
}
