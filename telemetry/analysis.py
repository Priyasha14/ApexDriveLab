import argparse
import csv
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def summarize(path: Path) -> dict[str, object]:
    rows = load_rows(path)
    if not rows:
        return {"path": str(path), "samples": 0}

    speeds = [as_float(row, "speed_kmh") for row in rows]
    grip = [as_float(row, "tire_grip_usage") for row in rows]
    lateral_accel = [abs(as_float(row, "lateral_acceleration")) for row in rows]
    longitudinal_accel = [as_float(row, "longitudinal_acceleration") for row in rows]
    off_track_samples = sum(1 for row in rows if row.get("on_track") == "False")
    balance_counts = Counter(row.get("handling_balance", "unknown") for row in rows)
    setup_counts = Counter(row.get("setup", "unknown") for row in rows)

    duration = as_float(rows[-1], "time_s") - as_float(rows[0], "time_s")
    saturated_samples = sum(1 for value in grip if value >= 0.98)

    return {
        "path": str(path),
        "samples": len(rows),
        "duration_s": duration,
        "max_speed_kmh": max(speeds),
        "avg_speed_kmh": sum(speeds) / len(speeds),
        "max_grip_usage": max(grip),
        "grip_saturation_pct": saturated_samples / len(rows) * 100.0,
        "off_track_pct": off_track_samples / len(rows) * 100.0,
        "max_lateral_accel": max(lateral_accel),
        "max_braking_accel": min(longitudinal_accel),
        "max_drive_accel": max(longitudinal_accel),
        "handling_balance": dict(balance_counts),
        "setups": dict(setup_counts),
    }


def print_summary(summary: dict[str, object]) -> None:
    if summary.get("samples") == 0:
        print(f"{summary['path']}: no samples")
        return

    print(f"Run: {summary['path']}")
    print(f"Samples: {summary['samples']}")
    print(f"Duration: {summary['duration_s']:.2f} s")
    print(f"Speed: avg {summary['avg_speed_kmh']:.1f} km/h | max {summary['max_speed_kmh']:.1f} km/h")
    print(f"Grip: max {summary['max_grip_usage'] * 100:.1f}% | saturated {summary['grip_saturation_pct']:.1f}%")
    print(f"Off track: {summary['off_track_pct']:.1f}%")
    print(f"Acceleration: lateral max {summary['max_lateral_accel']:.1f} | brake max {summary['max_braking_accel']:.1f} | drive max {summary['max_drive_accel']:.1f}")
    print(f"Handling balance: {summary['handling_balance']}")
    print(f"Setups: {summary['setups']}")


def latest_run(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("telemetry_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No telemetry CSV files found in {runs_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ApexDriveLab telemetry CSV files.")
    parser.add_argument("path", nargs="?", type=Path, help="Telemetry CSV path. Defaults to latest file in runs/.")
    args = parser.parse_args()

    path = args.path or latest_run(Path("runs"))
    print_summary(summarize(path))


if __name__ == "__main__":
    main()
