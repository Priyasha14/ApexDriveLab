import argparse
import csv
from pathlib import Path


def load_column(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "0") or 0.0))
        except ValueError:
            values.append(0.0)
    return values


def plot_run(path: Path, output: Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No telemetry rows in {path}")

    time = load_column(rows, "time_s")
    speed = load_column(rows, "speed_kmh")
    grip = load_column(rows, "tire_grip_usage")
    battery = load_column(rows, "battery_charge")
    target_speed = load_column(rows, "ai_target_speed")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, speed, label="speed")
    if any(target_speed):
        axes[0].plot(time, target_speed, label="target speed", linestyle="--")
    axes[0].set_ylabel("km/h")
    axes[0].legend()

    axes[1].plot(time, grip, color="tab:orange")
    axes[1].set_ylabel("grip usage")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(time, battery, color="tab:green")
    axes[2].set_ylabel("battery")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylim(0.0, 1.05)

    fig.tight_layout()
    output_path = output or path.with_suffix(".png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an ApexDriveLab telemetry CSV.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(plot_run(args.path, args.output))


if __name__ == "__main__":
    main()
