import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_float_column(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key, "")]


def plot_learning_curves(output: Path) -> Path:
    import matplotlib.pyplot as plt

    vae_2d = read_rows(Path("experiments") / "results" / "vae_training.csv")
    vae_8d = read_rows(Path("experiments") / "results" / "vae_8d_training.csv")
    raw_policy = read_rows(Path("experiments") / "results" / "neural_policy_training.csv")
    vae_policy = read_rows(Path("experiments") / "results" / "vae_policy_training.csv")
    optimizer = read_rows(Path("experiments") / "results" / "vae_policy_optimization.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(read_float_column(vae_2d, "epoch"), read_float_column(vae_2d, "reconstruction_loss"), label="2D reconstruction")
    axes[0, 0].plot(read_float_column(vae_8d, "epoch"), read_float_column(vae_8d, "reconstruction_loss"), label="8D reconstruction")
    axes[0, 0].set_title("VAE reconstruction loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(read_float_column(vae_2d, "epoch"), read_float_column(vae_2d, "kl_loss"), label="2D KL")
    axes[0, 1].plot(read_float_column(vae_8d, "epoch"), read_float_column(vae_8d, "kl_loss"), label="8D KL")
    axes[0, 1].set_title("VAE KL loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("KL")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(read_float_column(raw_policy, "epoch"), read_float_column(raw_policy, "loss"), label="raw train")
    axes[1, 0].plot(read_float_column(raw_policy, "epoch"), read_float_column(raw_policy, "validation_loss"), label="raw validation")
    axes[1, 0].plot(read_float_column(vae_policy, "epoch"), read_float_column(vae_policy, "loss"), label="VAE policy train")
    axes[1, 0].plot(read_float_column(vae_policy, "epoch"), read_float_column(vae_policy, "validation_loss"), label="VAE policy validation")
    axes[1, 0].set_title("Policy imitation loss")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("weighted MSE")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    iterations = read_float_column(optimizer, "iteration")
    scores = read_float_column(optimizer, "score")
    best_scores = []
    best = float("inf")
    for score in scores:
        best = min(best, score)
        best_scores.append(best)
    axes[1, 1].plot(iterations, scores, label="candidate score", alpha=0.45)
    axes[1, 1].plot(iterations, best_scores, label="best score", linewidth=2.0)
    axes[1, 1].set_title("Evolutionary policy optimization")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("score")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def save_report(path: Path, plot_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Learning Curves",
                "",
                "These curves track VAE reconstruction/KL loss, policy train/validation loss, and evolutionary optimization score.",
                "",
                f"- Plot: `{plot_path}`",
                "",
                "The train/validation policy curves are imitation-learning losses. The optimizer curve is a lap-score objective, where lower is better.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training, validation, and optimization curves.")
    parser.add_argument("--output", type=Path, default=Path("experiments") / "results" / "learning_curves.png")
    parser.add_argument("--report", type=Path, default=Path("experiments") / "results" / "learning_curves.md")
    args = parser.parse_args()

    plot_path = plot_learning_curves(args.output)
    save_report(args.report, plot_path)
    print(f"plot={plot_path}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
