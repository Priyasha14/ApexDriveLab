import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.optimizer import format_result, random_search, score_lap
from ai.rule_driver import DriverParameters
from experiments.run_lap import run_ai_laps
from telemetry.plots import plot_run


RESULTS_DIR = Path("experiments") / "results"


def lap_row(label: str, setup: str, result, score: float, notes: str = "") -> dict[str, object]:
    return {
        "label": label,
        "setup": setup,
        "completed_laps": result.completed_laps,
        "elapsed_time": f"{result.elapsed_time:.3f}",
        "best_lap_time": "" if result.best_lap_time is None else f"{result.best_lap_time:.3f}",
        "average_lap_time": "" if result.average_lap_time is None else f"{result.average_lap_time:.3f}",
        "off_track_pct": f"{result.off_track_pct:.3f}",
        "max_grip_usage": f"{result.max_grip_usage:.3f}",
        "score": f"{score:.3f}",
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], optimized_lines: list[str], plot_path: Path | None) -> None:
    lines = [
        "# ApexDriveLab Months 3-5 Validation",
        "",
        "This report records the current validation pass for aero, hybrid, rule-based AI, and random-search optimization.",
        "",
        "## Summary Table",
        "",
        "| Label | Setup | Laps | Best Lap | Avg Lap | Off Track | Max Grip | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['setup']} | {row['completed_laps']} | {row['best_lap_time']} | "
            f"{row['average_lap_time']} | {row['off_track_pct']}% | {row['max_grip_usage']} | {row['score']} |"
        )

    lines.extend(["", "## Optimized Candidates", ""])
    lines.extend(f"- {line}" for line in optimized_lines)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Aero and hybrid are active because the AI lap telemetry includes aero mode, downforce, drag, battery charge, deployment, and recovery columns.",
            "- Ablation rows compare the same rule driver with active aero and/or hybrid disabled.",
            "- The rule-based driver is considered stable when it completes repeated laps with low off-track percentage.",
            "- Random search is the first optimization layer; it is intentionally simple so parameter effects remain understandable.",
            "- Grip saturation near 1.0 means the car is driving at the tire limit. That is useful, but future tuning should reduce excessive saturation if it makes behavior unrealistic.",
        ]
    )
    if plot_path:
        lines.extend(["", "## Plot", "", f"![Telemetry plot]({plot_path.name})"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    optimized_lines: list[str] = []

    default_params = DriverParameters()
    for setup in ["balanced", "high_downforce", "low_drag", "front_aero", "rear_aero"]:
        result = run_ai_laps(laps=3, max_time=90.0, setup_name=setup, params=default_params, save_telemetry=(setup == "balanced"))
        rows.append(lap_row("default_ai", setup, result, score_lap(result)))

    for label, params in [
        ("no_active_aero", DriverParameters(enable_active_aero=False)),
        ("no_hybrid", DriverParameters(enable_hybrid=False)),
        ("no_aero_or_hybrid", DriverParameters(enable_active_aero=False, enable_hybrid=False)),
    ]:
        result = run_ai_laps(laps=3, max_time=90.0, setup_name="balanced", params=params)
        rows.append(lap_row(label, "balanced", result, score_lap(result)))

    for setup in ["balanced", "low_drag", "high_downforce"]:
        optimized = random_search(iterations=10, seed=23, setup_name=setup)
        validation_result = run_ai_laps(laps=3, max_time=90.0, setup_name=setup, params=optimized.params)
        rows.append(lap_row("optimized_ai", setup, validation_result, score_lap(validation_result), "random_search_10_retested_3_laps"))
        optimized_lines.append(f"{setup}: {format_result(optimized)} | retested_3_lap_avg={validation_result.average_lap_time:.3f}")

    csv_path = RESULTS_DIR / "months_3_5_validation.csv"
    report_path = RESULTS_DIR / "months_3_5_validation.md"
    write_csv(csv_path, rows)

    plot_path = None
    telemetry_path = Path("runs") / "ai_lap.csv"
    if telemetry_path.exists():
        plot_path = RESULTS_DIR / "ai_lap_plot.png"
        plot_run(telemetry_path, plot_path)

    write_report(report_path, rows, optimized_lines, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    if plot_path:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
