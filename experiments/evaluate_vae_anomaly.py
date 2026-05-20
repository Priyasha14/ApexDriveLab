import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.neural_policy import build_observation
from ai.racing_line import RacingLine
from ai.rule_driver import DriverParameters, RuleBasedDriver
from ai.vae import DrivingStateVAE
from config import FPS, OFF_TRACK_GRIP_SCALE
from experiments.train_neural_policy import collect_demonstrations
from physics.car import Car, CarInputs
from track.checkpoints import CheckpointManager
from track.track import Track


def reconstruction_error(vae: DrivingStateVAE, observations: np.ndarray) -> np.ndarray:
    reconstruction = vae.reconstruct(observations)
    normalized_input = vae.normalize(observations)
    normalized_reconstruction = vae.normalize(reconstruction)
    return np.mean((normalized_input - normalized_reconstruction) ** 2, axis=1)


def collect_stressed_observations(samples: int, seed: int = 71) -> np.ndarray:
    rng = np.random.default_rng(seed)
    track = Track()
    racing_line = RacingLine.from_track(track)
    rule_driver = RuleBasedDriver(racing_line, DriverParameters())
    car = Car()
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))

    dt = 1.0 / FPS
    observations = []
    elapsed = 0.0
    previous_random_steer = 0.0

    while len(observations) < samples and elapsed < 140.0:
        expert = rule_driver.control(car, track)
        previous_random_steer = 0.82 * previous_random_steer + 0.18 * float(rng.uniform(-1.0, 1.0))
        inputs = CarInputs(
            throttle=float(np.clip(expert.throttle + rng.normal(0.18, 0.24), 0.0, 1.0)),
            brake=float(np.clip(expert.brake + rng.normal(0.08, 0.28), 0.0, 1.0)),
            steer=float(np.clip(expert.steer + previous_random_steer * 0.75, -1.0, 1.0)),
            aero_mode=expert.aero_mode,
            deploy_hybrid=expert.deploy_hybrid or bool(rng.random() > 0.6),
        )
        on_track = track.is_on_track(car.position)
        grip_scale = (1.0 if on_track else OFF_TRACK_GRIP_SCALE) * 0.72
        car.update(dt, inputs, grip_scale)
        checkpoints.update(dt, track.checkpoint_index_for_position(car.position))
        elapsed += dt

        if elapsed > 1.0:
            observations.append(build_observation(car, track, racing_line))

        if not math.isfinite(car.position[0]) or not math.isfinite(car.position[1]):
            break

    return np.asarray(observations, dtype=np.float32)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.max() + 1e-6], thresholds[::-1], [scores.min() - 1e-6]))
    positives = max(int(labels.sum()), 1)
    negatives = max(int(len(labels) - labels.sum()), 1)
    tpr = []
    fpr = []
    accuracies = []

    for threshold in thresholds:
        predictions = scores >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        tn = int(np.sum((~predictions) & (labels == 0)))
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        accuracies.append((tp + tn) / len(labels))

    order = np.argsort(fpr)
    sorted_fpr = np.asarray(fpr)[order]
    sorted_tpr = np.asarray(tpr)[order]
    auc = float(np.trapezoid(sorted_tpr, sorted_fpr))
    best_index = int(np.argmax(accuracies))
    return np.asarray(fpr), np.asarray(tpr), thresholds, auc, float(accuracies[best_index]), float(thresholds[best_index])


def save_roc_csv(path: Path, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["threshold", "fpr", "tpr"])
        for threshold, false_positive_rate, true_positive_rate in zip(thresholds, fpr, tpr):
            writer.writerow([f"{threshold:.8f}", f"{false_positive_rate:.8f}", f"{true_positive_rate:.8f}"])
    return path


def plot_roc(path: Path, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, label=f"VAE reconstruction error AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("VAE anomaly ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_report(path: Path, auc: float, accuracy: float, threshold: float, normal_count: int, anomaly_count: int, roc_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# VAE Anomaly ROC",
                "",
                "This treats clean rule-driver states as normal and low-grip/noisy-control driving states as anomalies.",
                "",
                f"- Normal samples: {normal_count}",
                f"- Anomaly samples: {anomaly_count}",
                f"- ROC AUC: {auc:.4f}",
                f"- Best threshold accuracy: {accuracy:.4f}",
                f"- Best reconstruction-error threshold: {threshold:.6f}",
                f"- Plot: `{roc_path}`",
                "",
                "ROC/accuracy are meaningful here because anomaly detection is a binary classification task. They are not the right metric for raw VAE reconstruction by itself.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the VAE as an anomaly detector and plot ROC.")
    parser.add_argument("--model", type=Path, default=Path("models") / "driving_state_vae_8d.npz")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--roc-csv", type=Path, default=Path("experiments") / "results" / "vae_anomaly_roc.csv")
    parser.add_argument("--plot", type=Path, default=Path("experiments") / "results" / "vae_anomaly_roc.png")
    parser.add_argument("--report", type=Path, default=Path("experiments") / "results" / "vae_anomaly_roc.md")
    args = parser.parse_args()

    vae = DrivingStateVAE.load(args.model)
    normal, _ = collect_demonstrations(laps=3, rollouts=1, control_noise=0.0)
    normal = normal[: args.samples]
    anomalies = collect_stressed_observations(samples=len(normal))
    count = min(len(normal), len(anomalies))
    normal = normal[:count]
    anomalies = anomalies[:count]

    scores = np.concatenate([reconstruction_error(vae, normal), reconstruction_error(vae, anomalies)])
    labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomalies), dtype=int)])
    fpr, tpr, thresholds, auc, accuracy, threshold = roc_curve(labels, scores)
    save_roc_csv(args.roc_csv, fpr, tpr, thresholds)
    plot_roc(args.plot, fpr, tpr, auc)
    save_report(args.report, auc, accuracy, threshold, len(normal), len(anomalies), args.plot)

    print(f"auc={auc:.4f}")
    print(f"accuracy={accuracy:.4f}")
    print(f"threshold={threshold:.6f}")
    print(f"plot={args.plot}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
