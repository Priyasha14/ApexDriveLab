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
    distance = load_column(rows, "lap_distance")
    x_axis = distance if any(distance) else time
    x_label = "lap distance" if any(distance) else "time (s)"
    speed = load_column(rows, "speed_kmh")
    grip = load_column(rows, "tire_grip_usage")
    battery = load_column(rows, "battery_charge")
    target_speed = load_column(rows, "ai_target_speed")
    throttle = load_column(rows, "throttle")
    brake = load_column(rows, "brake")
    steering = load_column(rows, "steering_angle")
    tire_temp = load_column(rows, "tire_temperature")
    tire_wear = load_column(rows, "tire_wear")
    deployment = load_column(rows, "deployment_power")
    recovery = load_column(rows, "recovery_power")
    lateral_g = [value / 98.1 for value in load_column(rows, "lateral_acceleration")]
    longitudinal_g = [value / 98.1 for value in load_column(rows, "longitudinal_acceleration")]
    aero_mode = [1.0 if row.get("aero_mode") == "straight" else 0.0 for row in rows]

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(x_axis, speed, label="speed")
    if any(target_speed):
        axes[0].plot(x_axis, target_speed, label="target speed", linestyle="--")
    axes[0].set_ylabel("km/h")
    axes[0].legend()

    axes[1].plot(x_axis, throttle, label="throttle", color="tab:green")
    axes[1].plot(x_axis, brake, label="brake", color="tab:red")
    axes[1].plot(x_axis, steering, label="steering angle", color="tab:blue")
    axes[1].set_ylabel("inputs")
    axes[1].legend()

    axes[2].plot(x_axis, tire_temp, label="tire temp", color="tab:orange")
    axes[2].plot(x_axis, tire_wear, label="tire wear", color="tab:brown")
    axes[2].plot(x_axis, grip, label="grip usage", color="tab:purple")
    axes[2].set_ylabel("tires")
    axes[2].legend()

    axes[3].plot(x_axis, battery, label="battery charge", color="tab:green")
    axes[3].plot(x_axis, deployment, label="deploy power", color="tab:cyan")
    axes[3].plot(x_axis, recovery, label="recovery power", color="tab:olive")
    axes[3].step(x_axis, aero_mode, label="straight aero", color="tab:gray")
    axes[3].set_ylabel("energy/aero")
    axes[3].legend()

    axes[4].plot(x_axis, lateral_g, label="lateral G", color="tab:red")
    axes[4].plot(x_axis, longitudinal_g, label="longitudinal G", color="tab:blue")
    axes[4].set_ylabel("G")
    axes[4].set_xlabel(x_label)
    axes[4].legend()

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
