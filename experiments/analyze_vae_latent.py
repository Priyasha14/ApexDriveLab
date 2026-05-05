import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.vae import DrivingStateVAE
from experiments.train_neural_policy import collect_demonstrations


def classify_phase(action: np.ndarray) -> str:
    steer, throttle, brake, _, deploy = action
    if brake > 0.15:
        return "braking"
    if abs(steer) > 0.35:
        return "cornering"
    if deploy > 0.5:
        return "deploy"
    if throttle > 0.4:
        return "acceleration"
    return "coast"


def save_latent_csv(path: Path, latent: np.ndarray, actions: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["z0", "z1", "phase"])
        for point, action in zip(latent, actions):
            z0 = float(point[0])
            z1 = float(point[1]) if point.shape[0] > 1 else 0.0
            writer.writerow([f"{z0:.6f}", f"{z1:.6f}", classify_phase(action)])
    return path


def plot_latent(path: Path, latent: np.ndarray, actions: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    phase_colors = {
        "acceleration": "tab:green",
        "braking": "tab:red",
        "cornering": "tab:blue",
        "deploy": "tab:cyan",
        "coast": "tab:gray",
    }
    phases = [classify_phase(action) for action in actions]

    fig, ax = plt.subplots(figsize=(9, 7))
    for phase, color in phase_colors.items():
        idx = [i for i, value in enumerate(phases) if value == phase]
        if idx:
            ax.scatter(latent[idx, 0], latent[idx, 1] if latent.shape[1] > 1 else np.zeros(len(idx)), s=10, alpha=0.55, label=phase, color=color)
    ax.set_xlabel("latent z0")
    ax.set_ylabel("latent z1")
    ax.set_title("VAE latent driving states")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_report(path: Path, latent: np.ndarray, actions: np.ndarray, plot_path: Path) -> Path:
    phases = [classify_phase(action) for action in actions]
    counts = {phase: phases.count(phase) for phase in sorted(set(phases))}
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VAE Latent Analysis",
        "",
        "The VAE encoder maps each driving state into a compact latent point. Similar driving situations should appear near each other.",
        "",
        f"- Samples analyzed: {len(latent)}",
        f"- Latent mean: {np.mean(latent, axis=0).round(4).tolist()}",
        f"- Latent std: {np.std(latent, axis=0).round(4).tolist()}",
        f"- Plot: `{plot_path}`",
        "",
        "Phase counts:",
    ]
    lines.extend(f"- {phase}: {count}" for phase, count in counts.items())
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and plot the VAE latent driving-state space.")
    parser.add_argument("--model", type=Path, default=Path("models") / "driving_state_vae.npz")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--csv-path", type=Path, default=Path("experiments") / "results" / "vae_latent_points.csv")
    parser.add_argument("--plot-path", type=Path, default=Path("experiments") / "results" / "vae_latent_space.png")
    parser.add_argument("--report-path", type=Path, default=Path("experiments") / "results" / "vae_latent_analysis.md")
    args = parser.parse_args()

    observations, actions = collect_demonstrations(laps=args.laps, rollouts=args.rollouts, control_noise=0.06)
    vae = DrivingStateVAE.load(args.model)
    latent, _ = vae.encode(observations)
    save_latent_csv(args.csv_path, latent, actions)
    plot_latent(args.plot_path, latent, actions)
    save_report(args.report_path, latent, actions, args.plot_path)

    print(f"samples={len(latent)}")
    print(f"latent_csv={args.csv_path}")
    print(f"latent_plot={args.plot_path}")
    print(f"report={args.report_path}")


if __name__ == "__main__":
    main()
