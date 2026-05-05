from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai.neural_policy import ACTION_SIZE, action_to_inputs
from physics.car import CarInputs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class LatentPolicyHistory:
    losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)


@dataclass
class LatentPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    latent_mean: np.ndarray
    latent_std: np.ndarray

    @classmethod
    def create(cls, latent_size: int, hidden_1: int = 48, hidden_2: int = 48, seed: int = 41) -> "LatentPolicy":
        rng = np.random.default_rng(seed)
        return cls(
            w1=rng.normal(0.0, 0.18, size=(latent_size, hidden_1)).astype(np.float32),
            b1=np.zeros(hidden_1, dtype=np.float32),
            w2=rng.normal(0.0, 0.14, size=(hidden_1, hidden_2)).astype(np.float32),
            b2=np.zeros(hidden_2, dtype=np.float32),
            w3=rng.normal(0.0, 0.10, size=(hidden_2, ACTION_SIZE)).astype(np.float32),
            b3=np.zeros(ACTION_SIZE, dtype=np.float32),
            latent_mean=np.zeros(latent_size, dtype=np.float32),
            latent_std=np.ones(latent_size, dtype=np.float32),
        )

    @property
    def latent_size(self) -> int:
        return int(self.w1.shape[0])

    def set_normalization(self, latent: np.ndarray) -> None:
        self.latent_mean = latent.mean(axis=0).astype(np.float32)
        self.latent_std = np.maximum(latent.std(axis=0), 1e-3).astype(np.float32)

    def _normalize(self, latent: np.ndarray) -> np.ndarray:
        return (latent - self.latent_mean) / self.latent_std

    def _forward_raw(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self._normalize(latent)
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        z = h2 @ self.w3 + self.b3
        actions = np.concatenate([np.tanh(z[:, :1]), _sigmoid(z[:, 1:])], axis=1)
        return x, h1, h2, z, actions

    def predict(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype=np.float32).reshape(1, -1)
        return self._forward_raw(z)[-1][0]

    def control(self, latent: np.ndarray) -> CarInputs:
        return action_to_inputs(self.predict(latent))

    def train(
        self,
        latent: np.ndarray,
        targets: np.ndarray,
        epochs: int = 400,
        learning_rate: float = 0.003,
        batch_size: int = 128,
        validation_fraction: float = 0.15,
        loss_weights: np.ndarray | None = None,
        seed: int = 43,
    ) -> LatentPolicyHistory:
        rng = np.random.default_rng(seed)
        latent = latent.astype(np.float32)
        targets = targets.astype(np.float32)
        self.set_normalization(latent)

        order = rng.permutation(len(latent))
        split = max(1, int(len(order) * (1.0 - validation_fraction)))
        train_idx = order[:split]
        val_idx = order[split:] if split < len(order) else order[:1]
        weights = np.asarray(loss_weights if loss_weights is not None else np.ones(ACTION_SIZE), dtype=np.float32)
        history = LatentPolicyHistory()

        for _ in range(epochs):
            rng.shuffle(train_idx)
            batch_losses = []
            for start in range(0, len(train_idx), batch_size):
                idx = train_idx[start : start + batch_size]
                x, h1, h2, z, pred = self._forward_raw(latent[idx])
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
            validation_pred = self._forward_raw(latent[val_idx])[-1]
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
            latent_mean=self.latent_mean,
            latent_std=self.latent_std,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "LatentPolicy":
        with np.load(path) as data:
            return cls(
                w1=data["w1"],
                b1=data["b1"],
                w2=data["w2"],
                b2=data["b2"],
                w3=data["w3"],
                b3=data["b3"],
                latent_mean=data["latent_mean"],
                latent_std=data["latent_std"],
            )
