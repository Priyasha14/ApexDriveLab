import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.latent_policy import LatentPolicy
from ai.vae import DrivingStateVAE
from experiments.train_neural_policy import collect_demonstrations


def save_history(path: Path, losses: list[float], validation_losses: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "validation_loss"])
        for epoch, (loss, validation_loss) in enumerate(zip(losses, validation_losses), start=1):
            writer.writerow([epoch, f"{loss:.8f}", f"{validation_loss:.8f}"])
    return path


def save_report(path: Path, sample_count: int, latent_size: int, final_loss: float, final_validation_loss: float, vae_path: Path, policy_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# VAE-Latent Policy Training",
                "",
                "This policy uses the VAE encoder as a learned feature extractor, then learns controls from the latent vector.",
                "",
                f"- Demonstration samples: {sample_count}",
                f"- Latent size: {latent_size}",
                f"- Final training loss: {final_loss:.6f}",
                f"- Final validation loss: {final_validation_loss:.6f}",
                f"- VAE model: `{vae_path}`",
                f"- Policy model: `{policy_path}`",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a driver policy on VAE latent driving states.")
    parser.add_argument("--vae-model", type=Path, default=Path("models") / "driving_state_vae_8d.npz")
    parser.add_argument("--policy-path", type=Path, default=Path("models") / "vae_latent_policy.npz")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--history-path", type=Path, default=Path("experiments") / "results" / "vae_policy_training.csv")
    parser.add_argument("--report-path", type=Path, default=Path("experiments") / "results" / "vae_policy_training.md")
    args = parser.parse_args()

    observations, actions = collect_demonstrations(laps=args.laps, rollouts=args.rollouts, control_noise=0.06)
    vae = DrivingStateVAE.load(args.vae_model)
    latent, _ = vae.encode(observations)
    policy = LatentPolicy.create(latent_size=latent.shape[1])
    weights = np.array([3.0, 2.0, 2.0, 0.35, 0.35], dtype=np.float32)
    history = policy.train(latent, actions, epochs=args.epochs, learning_rate=args.learning_rate, loss_weights=weights)
    policy.save(args.policy_path)
    save_history(args.history_path, history.losses, history.validation_losses)
    save_report(args.report_path, len(observations), latent.shape[1], history.losses[-1], history.validation_losses[-1], args.vae_model, args.policy_path)

    print(f"collected_samples={len(observations)}")
    print(f"latent_size={latent.shape[1]}")
    print(f"final_loss={history.losses[-1]:.6f}")
    print(f"final_validation_loss={history.validation_losses[-1]:.6f}")
    print(f"saved_policy={args.policy_path}")


if __name__ == "__main__":
    main()
