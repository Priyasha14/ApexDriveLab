from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai.neural_policy import OBSERVATION_SIZE


@dataclass
class VAEHistory:
    losses: list[float] = field(default_factory=list)
    reconstruction_losses: list[float] = field(default_factory=list)
    kl_losses: list[float] = field(default_factory=list)


@dataclass
class DrivingStateVAE:
    encoder_w1: np.ndarray
    encoder_b1: np.ndarray
    mean_w: np.ndarray
    mean_b: np.ndarray
    logvar_w: np.ndarray
    logvar_b: np.ndarray
    decoder_w1: np.ndarray
    decoder_b1: np.ndarray
    decoder_w2: np.ndarray
    decoder_b2: np.ndarray
    observation_mean: np.ndarray = field(default_factory=lambda: np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    observation_std: np.ndarray = field(default_factory=lambda: np.ones(OBSERVATION_SIZE, dtype=np.float32))

    @classmethod
    def create(cls, latent_size: int = 2, hidden_size: int = 48, seed: int = 23) -> "DrivingStateVAE":
        rng = np.random.default_rng(seed)
        return cls(
            encoder_w1=rng.normal(0.0, 0.16, size=(OBSERVATION_SIZE, hidden_size)).astype(np.float32),
            encoder_b1=np.zeros(hidden_size, dtype=np.float32),
            mean_w=rng.normal(0.0, 0.12, size=(hidden_size, latent_size)).astype(np.float32),
            mean_b=np.zeros(latent_size, dtype=np.float32),
            logvar_w=rng.normal(0.0, 0.08, size=(hidden_size, latent_size)).astype(np.float32),
            logvar_b=np.zeros(latent_size, dtype=np.float32),
            decoder_w1=rng.normal(0.0, 0.14, size=(latent_size, hidden_size)).astype(np.float32),
            decoder_b1=np.zeros(hidden_size, dtype=np.float32),
            decoder_w2=rng.normal(0.0, 0.16, size=(hidden_size, OBSERVATION_SIZE)).astype(np.float32),
            decoder_b2=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        )

    def set_normalization(self, observations: np.ndarray) -> None:
        self.observation_mean = observations.mean(axis=0).astype(np.float32)
        self.observation_std = np.maximum(observations.std(axis=0), 1e-3).astype(np.float32)

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        return (observations - self.observation_mean) / self.observation_std

    def denormalize(self, observations: np.ndarray) -> np.ndarray:
        return observations * self.observation_std + self.observation_mean

    def encode(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.normalize(np.asarray(observations, dtype=np.float32))
        h = np.tanh(x @ self.encoder_w1 + self.encoder_b1)
        mean = h @ self.mean_w + self.mean_b
        logvar = np.clip(h @ self.logvar_w + self.logvar_b, -8.0, 4.0)
        return mean, logvar

    def decode_normalized(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype=np.float32)
        h = np.tanh(z @ self.decoder_w1 + self.decoder_b1)
        return h @ self.decoder_w2 + self.decoder_b2

    def reconstruct(self, observations: np.ndarray) -> np.ndarray:
        mean, _ = self.encode(observations)
        return self.denormalize(self.decode_normalized(mean))

    def train(
        self,
        observations: np.ndarray,
        epochs: int = 350,
        learning_rate: float = 0.003,
        batch_size: int = 128,
        beta: float = 0.0015,
        seed: int = 29,
    ) -> VAEHistory:
        rng = np.random.default_rng(seed)
        observations = observations.astype(np.float32)
        self.set_normalization(observations)
        normalized = self.normalize(observations)
        history = VAEHistory()

        for _ in range(epochs):
            order = rng.permutation(len(normalized))
            epoch_loss = []
            epoch_reconstruction = []
            epoch_kl = []

            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                x = normalized[idx]
                batch_count = len(idx)

                h_enc = np.tanh(x @ self.encoder_w1 + self.encoder_b1)
                mean = h_enc @ self.mean_w + self.mean_b
                logvar = np.clip(h_enc @ self.logvar_w + self.logvar_b, -8.0, 4.0)
                std = np.exp(0.5 * logvar)
                epsilon = rng.normal(0.0, 1.0, size=mean.shape).astype(np.float32)
                z = mean + std * epsilon

                h_dec = np.tanh(z @ self.decoder_w1 + self.decoder_b1)
                reconstruction = h_dec @ self.decoder_w2 + self.decoder_b2

                reconstruction_error = reconstruction - x
                reconstruction_loss = float(np.mean(reconstruction_error * reconstruction_error))
                kl_per_sample = -0.5 * np.sum(1.0 + logvar - mean * mean - np.exp(logvar), axis=1)
                kl_loss = float(np.mean(kl_per_sample))
                epoch_reconstruction.append(reconstruction_loss)
                epoch_kl.append(kl_loss)
                epoch_loss.append(reconstruction_loss + beta * kl_loss)

                d_reconstruction = 2.0 * reconstruction_error / batch_count / OBSERVATION_SIZE
                d_decoder_w2 = h_dec.T @ d_reconstruction
                d_decoder_b2 = d_reconstruction.sum(axis=0)
                d_h_dec = d_reconstruction @ self.decoder_w2.T * (1.0 - h_dec * h_dec)
                d_decoder_w1 = z.T @ d_h_dec
                d_decoder_b1 = d_h_dec.sum(axis=0)
                d_z = d_h_dec @ self.decoder_w1.T

                d_mean = d_z + beta * mean / batch_count
                d_logvar = d_z * epsilon * 0.5 * std
                d_logvar += beta * 0.5 * (np.exp(logvar) - 1.0) / batch_count

                d_mean_w = h_enc.T @ d_mean
                d_mean_b = d_mean.sum(axis=0)
                d_logvar_w = h_enc.T @ d_logvar
                d_logvar_b = d_logvar.sum(axis=0)
                d_h_enc = (d_mean @ self.mean_w.T + d_logvar @ self.logvar_w.T) * (1.0 - h_enc * h_enc)
                d_encoder_w1 = x.T @ d_h_enc
                d_encoder_b1 = d_h_enc.sum(axis=0)

                self.decoder_w2 -= learning_rate * d_decoder_w2
                self.decoder_b2 -= learning_rate * d_decoder_b2
                self.decoder_w1 -= learning_rate * d_decoder_w1
                self.decoder_b1 -= learning_rate * d_decoder_b1
                self.mean_w -= learning_rate * d_mean_w
                self.mean_b -= learning_rate * d_mean_b
                self.logvar_w -= learning_rate * d_logvar_w
                self.logvar_b -= learning_rate * d_logvar_b
                self.encoder_w1 -= learning_rate * d_encoder_w1
                self.encoder_b1 -= learning_rate * d_encoder_b1

            history.losses.append(float(np.mean(epoch_loss)))
            history.reconstruction_losses.append(float(np.mean(epoch_reconstruction)))
            history.kl_losses.append(float(np.mean(epoch_kl)))

        return history

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            encoder_w1=self.encoder_w1,
            encoder_b1=self.encoder_b1,
            mean_w=self.mean_w,
            mean_b=self.mean_b,
            logvar_w=self.logvar_w,
            logvar_b=self.logvar_b,
            decoder_w1=self.decoder_w1,
            decoder_b1=self.decoder_b1,
            decoder_w2=self.decoder_w2,
            decoder_b2=self.decoder_b2,
            observation_mean=self.observation_mean,
            observation_std=self.observation_std,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "DrivingStateVAE":
        with np.load(path) as data:
            return cls(
                encoder_w1=data["encoder_w1"],
                encoder_b1=data["encoder_b1"],
                mean_w=data["mean_w"],
                mean_b=data["mean_b"],
                logvar_w=data["logvar_w"],
                logvar_b=data["logvar_b"],
                decoder_w1=data["decoder_w1"],
                decoder_b1=data["decoder_b1"],
                decoder_w2=data["decoder_w2"],
                decoder_b2=data["decoder_b2"],
                observation_mean=data["observation_mean"],
                observation_std=data["observation_std"],
            )
