from dataclasses import dataclass

from config import (
    BRAKE_BIAS_FRONT,
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
}

