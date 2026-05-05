import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.vae import DrivingStateVAE
from experiments.train_neural_policy import collect_demonstrations


def save_history(path: Path, losses: list[float], reconstruction_losses: list[float], kl_losses: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "reconstruction_loss", "kl_loss"])
        for epoch, values in enumerate(zip(losses, reconstruction_losses, kl_losses), start=1):
            loss, reconstruction_loss, kl_loss = values
            writer.writerow([epoch, f"{loss:.8f}", f"{reconstruction_loss:.8f}", f"{kl_loss:.8f}"])
    return path


def save_report(path: Path, sample_count: int, final_reconstruction: float, final_kl: float, model_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# VAE Driving-State Model",
                "",
                "The VAE learns a compressed latent representation of simulator driving states.",
                "",
                f"- Demonstration samples: {sample_count}",
                f"- Final reconstruction loss: {final_reconstruction:.6f}",
                f"- Final KL loss: {final_kl:.6f}",
                f"- Saved model: `{model_path}`",
                "",
                "Use the encoder output as a compact state vector for later policy learning or anomaly detection.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE on ApexDriveLab driving-state observations.")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=0.0015)
    parser.add_argument("--latent-size", type=int, default=2)
    parser.add_argument("--model-path", type=Path, default=Path("models") / "driving_state_vae.npz")
    parser.add_argument("--history-path", type=Path, default=Path("experiments") / "results" / "vae_training.csv")
    parser.add_argument("--report-path", type=Path, default=Path("experiments") / "results" / "vae_training.md")
    args = parser.parse_args()

    observations, _ = collect_demonstrations(laps=args.laps, rollouts=args.rollouts, control_noise=0.08)
    vae = DrivingStateVAE.create(latent_size=args.latent_size)
    history = vae.train(observations, epochs=args.epochs, learning_rate=args.learning_rate, beta=args.beta)
    vae.save(args.model_path)
    save_history(args.history_path, history.losses, history.reconstruction_losses, history.kl_losses)
    save_report(args.report_path, len(observations), history.reconstruction_losses[-1], history.kl_losses[-1], args.model_path)

    reconstruction = vae.reconstruct(observations[: min(512, len(observations))])
    reconstruction_mae = float(np.mean(np.abs(reconstruction - observations[: len(reconstruction)])))

    print(f"collected_samples={len(observations)}")
    print(f"final_reconstruction_loss={history.reconstruction_losses[-1]:.6f}")
    print(f"final_kl_loss={history.kl_losses[-1]:.6f}")
    print(f"sample_reconstruction_mae={reconstruction_mae:.6f}")
    print(f"saved_model={args.model_path}")


if __name__ == "__main__":
    main()
