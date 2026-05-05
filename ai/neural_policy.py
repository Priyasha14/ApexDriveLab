import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai.pure_pursuit import wrap_angle
from ai.racing_line import RacingLine
from ai.speed_planner import ellipse_curvature
from config import MAX_STEER_ANGLE
from physics.car import CarInputs
from physics.vector_utils import clamp


ACTION_SIZE = 5
OBSERVATION_SIZE = 14


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _signed_path_error(car, track, racing_line: RacingLine, angle: float) -> float:
    target = racing_line.point_at(angle)
    radial = target - np.asarray(track.center, dtype=float)
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-6:
        return 0.0
    return float(np.dot(car.position - target, radial / radial_norm))


def build_observation(car, track, racing_line: RacingLine) -> np.ndarray:
    angle = track.angle_for_position(car.position)
    tangent_heading = racing_line.tangent_heading(angle)
    heading_error = wrap_angle(tangent_heading - car.heading)
    curvature = ellipse_curvature(angle, track.outer_radius)
    tire_temp = (car.tire_state.temperature - 85.0) / 45.0
    tire_wear = car.tire_state.wear * 2.0 - 1.0

    return np.array(
        [
            car.speed_kmh / 220.0,
            car.yaw_rate / 2.5,
            car.steering_angle / MAX_STEER_ANGLE,
            _signed_path_error(car, track, racing_line, angle) / 180.0,
            heading_error / math.pi,
            curvature * 180.0,
            car.tire_state.combined_grip_usage,
            car.hybrid_state.charge_fraction * 2.0 - 1.0,
            1.0 if car.aero_state.mode == "straight" else -1.0,
            1.0 if track.is_on_track(car.position) else -1.0,
            tire_temp,
            tire_wear,
            math.sin(angle),
            math.cos(angle),
        ],
        dtype=np.float32,
    )


def inputs_to_action(inputs: CarInputs) -> np.ndarray:
    return np.array(
        [
            clamp(inputs.steer, -1.0, 1.0),
            clamp(inputs.throttle, 0.0, 1.0),
            clamp(inputs.brake, 0.0, 1.0),
            1.0 if inputs.aero_mode == "straight" else 0.0,
            1.0 if inputs.deploy_hybrid else 0.0,
        ],
        dtype=np.float32,
    )


def action_to_inputs(action: np.ndarray) -> CarInputs:
    steer = clamp(float(action[0]), -1.0, 1.0)
    throttle = clamp(float(action[1]), 0.0, 1.0)
    brake = clamp(float(action[2]), 0.0, 1.0)

    if brake > 0.12 and brake > throttle:
        throttle = 0.0
    elif throttle > brake:
        brake = 0.0

    return CarInputs(
        throttle=throttle,
        brake=brake,
        steer=steer,
        aero_mode="straight" if float(action[3]) > 0.5 else "corner",
        deploy_hybrid=bool(float(action[4]) > 0.5),
    )


@dataclass
class TrainingHistory:
    losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)


@dataclass
class NeuralPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    observation_mean: np.ndarray = field(default_factory=lambda: np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    observation_std: np.ndarray = field(default_factory=lambda: np.ones(OBSERVATION_SIZE, dtype=np.float32))

    @classmethod
    def create(cls, hidden_1: int = 64, hidden_2: int = 64, seed: int = 7) -> "NeuralPolicy":
        rng = np.random.default_rng(seed)
        return cls(
            w1=(rng.normal(0.0, 0.18, size=(OBSERVATION_SIZE, hidden_1))).astype(np.float32),
            b1=np.zeros(hidden_1, dtype=np.float32),
            w2=(rng.normal(0.0, 0.14, size=(hidden_1, hidden_2))).astype(np.float32),
            b2=np.zeros(hidden_2, dtype=np.float32),
            w3=(rng.normal(0.0, 0.10, size=(hidden_2, ACTION_SIZE))).astype(np.float32),
            b3=np.zeros(ACTION_SIZE, dtype=np.float32),
        )

    def set_normalization(self, observations: np.ndarray) -> None:
        self.observation_mean = observations.mean(axis=0).astype(np.float32)
        self.observation_std = np.maximum(observations.std(axis=0), 1e-3).astype(np.float32)

    def _normalize(self, observations: np.ndarray) -> np.ndarray:
        return (observations - self.observation_mean) / self.observation_std

    def _forward_raw(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self._normalize(observations)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        z = h2 @ self.w3 + self.b3
        actions = np.concatenate([np.tanh(z[:, :1]), _sigmoid(z[:, 1:])], axis=1)
        return x, h1, h2, z, actions

    def predict(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return self._forward_raw(obs)[-1][0]

    def control(self, car, track, racing_line: RacingLine) -> CarInputs:
        return action_to_inputs(self.predict(build_observation(car, track, racing_line)))

    def train(
        self,
        observations: np.ndarray,
        targets: np.ndarray,
        epochs: int = 250,
        learning_rate: float = 0.004,
        batch_size: int = 128,
        validation_fraction: float = 0.15,
        loss_weights: np.ndarray | None = None,
        seed: int = 11,
    ) -> TrainingHistory:
        rng = np.random.default_rng(seed)
        observations = observations.astype(np.float32)
        targets = targets.astype(np.float32)
        self.set_normalization(observations)

        order = rng.permutation(len(observations))
        split = max(1, int(len(order) * (1.0 - validation_fraction)))
        train_idx = order[:split]
        val_idx = order[split:] if split < len(order) else order[:1]
        history = TrainingHistory()
        weights = np.asarray(loss_weights if loss_weights is not None else np.ones(ACTION_SIZE), dtype=np.float32)

        for _ in range(epochs):
            rng.shuffle(train_idx)
            batch_losses = []
            for start in range(0, len(train_idx), batch_size):
                idx = train_idx[start : start + batch_size]
                x, h1, h2, z, pred = self._forward_raw(observations[idx])
                target = targets[idx]
                error = pred - target
                batch_losses.append(float(np.mean((error * error) * weights)))

                dz = 2.0 * error * weights / len(idx)
                dz[:, 0] *= 1.0 - np.tanh(z[:, 0]) ** 2
                sigmoid_outputs = pred[:, 1:]
                dz[:, 1:] *= sigmoid_outputs * (1.0 - sigmoid_outputs)

                dw3 = h2.T @ dz
                db3 = dz.sum(axis=0)
                dh2 = dz @ self.w3.T * (1.0 - h2 * h2)
                dw2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)
                dh1 = dh2 @ self.w2.T * (1.0 - h1 * h1)
                dw1 = x.T @ dh1
                db1 = dh1.sum(axis=0)

                self.w3 -= learning_rate * dw3
                self.b3 -= learning_rate * db3
                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1

            history.losses.append(float(np.mean(batch_losses)))
            validation_pred = self._forward_raw(observations[val_idx])[-1]
            validation_error = validation_pred - targets[val_idx]
            history.validation_losses.append(float(np.mean((validation_error * validation_error) * weights)))

        return history

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            observation_mean=self.observation_mean,
            observation_std=self.observation_std,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "NeuralPolicy":
        with np.load(path) as data:
            return cls(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                w3=data["w3"],
                b3=data["b3"],
                observation_mean=data["observation_mean"],
                observation_std=data["observation_std"],
            )
