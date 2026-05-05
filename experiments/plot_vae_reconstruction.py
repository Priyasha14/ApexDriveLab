import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.neural_policy import OBSERVATION_SIZE
from ai.vae import DrivingStateVAE
from experiments.train_neural_policy import collect_demonstrations


FEATURE_NAMES = [
    "speed",
    "yaw rate",
    "steering angle",
    "path error",
    "heading error",
    "curvature",
    "grip usage",
    "battery",
    "aero mode",
    "on track",
    "tire temp",
    "tire wear",
    "sin progress",
    "cos progress",
]


def plot_reconstruction(model_path: Path, output_path: Path, samples: int = 360) -> tuple[Path, float]:
    import matplotlib.pyplot as plt

    observations, _ = collect_demonstrations(laps=2, rollouts=1, control_noise=0.0)
    observations = observations[: min(samples, len(observations))]
    vae = DrivingStateVAE.load(model_path)
    reconstruction = vae.reconstruct(observations)
    mae = float(np.mean(np.abs(reconstruction - observations)))

    x = np.arange(len(observations))
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()

    for index in range(OBSERVATION_SIZE):
        ax = axes[index]
        ax.plot(x, observations[:, index], label="input", linewidth=1.4)
        ax.plot(x, reconstruction[:, index], label="reconstruction", linewidth=1.2, linestyle="--")
        ax.set_ylabel(FEATURE_NAMES[index])
        ax.grid(alpha=0.2)
        if index == 0:
            ax.legend(loc="upper right")

    axes[-2].set_xlabel("sample")
    axes[-1].set_xlabel("sample")
    fig.suptitle(f"VAE input vs reconstruction | MAE {mae:.4f}", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path, mae


def save_report(path: Path, plot_path: Path, mae: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# VAE Reconstruction Check",
                "",
                "This plot compares original driving-state signals against the VAE reconstruction.",
                "",
                f"- Mean absolute reconstruction error: {mae:.6f}",
                f"- Plot: `{plot_path}`",
                "",
                "Smooth reconstruction is expected because the VAE compresses many simulator signals into a two-dimensional latent space.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VAE input signals against reconstructed signals.")
    parser.add_argument("--model", type=Path, default=Path("models") / "driving_state_vae.npz")
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--output", type=Path, default=Path("experiments") / "results" / "vae_reconstruction_signals.png")
    parser.add_argument("--report", type=Path, default=Path("experiments") / "results" / "vae_reconstruction.md")
    args = parser.parse_args()

    plot_path, mae = plot_reconstruction(args.model, args.output, args.samples)
    save_report(args.report, plot_path, mae)
    print(f"plot={plot_path}")
    print(f"mae={mae:.6f}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
