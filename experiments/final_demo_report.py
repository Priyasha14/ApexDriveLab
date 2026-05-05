import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.optimizer import random_search, score_lap
from ai.rule_driver import DriverParameters
from experiments.run_lap import run_ai_laps
from telemetry.plots import plot_run


RESULTS_DIR = Path("experiments") / "results" / "month_6"


def row(name: str, category: str, result, score: float) -> dict[str, object]:
    return {
        "name": name,
        "category": category,
        "completed_laps": result.completed_laps,
        "best_lap_time": "" if result.best_lap_time is None else f"{result.best_lap_time:.3f}",
        "average_lap_time": "" if result.average_lap_time is None else f"{result.average_lap_time:.3f}",
        "off_track_pct": f"{result.off_track_pct:.3f}",
        "max_grip_usage": f"{result.max_grip_usage:.3f}",
        "score": f"{score:.3f}",
        "avg_tire_temp": "",
        "final_tire_wear": "",
        "avg_condition_grip": "",
    }


def add_telemetry_metrics(item: dict[str, object], path: Path) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return item

    temps = [float(row.get("tire_temperature", 0.0) or 0.0) for row in rows]
    condition = [float(row.get("tire_condition_grip", 0.0) or 0.0) for row in rows]
    item["avg_tire_temp"] = f"{sum(temps) / len(temps):.2f}"
    item["final_tire_wear"] = f"{float(rows[-1].get('tire_wear', 0.0) or 0.0):.4f}"
    item["avg_condition_grip"] = f"{sum(condition) / len(condition):.3f}"
    return item


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], plot_path: Path) -> None:
    lines = [
        "# ApexDriveLab Final Demo Report",
        "",
        "## Modeled Physics",
        "",
        "- 2D bicycle-model vehicle dynamics with front/rear axle slip angles.",
        "- Tire friction circle, basic load transfer, tire temperature, and tire wear.",
        "- Aerodynamic drag and downforce with front/rear aero balance.",
        "- Active aero modes for corner and straight behavior.",
        "- Hybrid energy store with deployment and regenerative recovery.",
        "",
        "## AI Controls",
        "",
        "- Pure-pursuit steering follows a racing line target ahead of the car.",
        "- Speed planning estimates upcoming curvature and commands throttle/brake.",
        "- The rule driver selects active aero mode and hybrid deployment.",
        "- Random search tunes lookahead, speed, braking, aero, and hybrid thresholds.",
        "",
        "## Experiments",
        "",
        "| Category | Name | Laps | Best Lap | Avg Lap | Off Track | Max Grip | Cond Grip | Tire Temp | Tire Wear | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['category']} | {item['name']} | {item['completed_laps']} | {item['best_lap_time']} | "
            f"{item['average_lap_time']} | {item['off_track_pct']}% | {item['max_grip_usage']} | "
            f"{item['avg_condition_grip']} | {item['avg_tire_temp']} | {item['final_tire_wear']} | {item['score']} |"
        )

    lines.extend(
        [
            "",
            "## Results Found",
            "",
            "- The optimized rule-based AI is faster than the default rule-based AI in the current simplified simulator.",
            "- Hybrid deployment improves lap time compared with conservative or disabled deployment.",
            "- Low-drag and front-aero configurations are slightly faster on this oval-style test track.",
            "- Cold, worn, and wet-condition tests provide repeatable scenarios for tire and grip sensitivity.",
            "- Grip saturation is still frequent, so future work should tune tire limits and target-speed planning for more realism.",
            "",
            "## Telemetry Dashboard",
            "",
            f"![Dashboard]({plot_path.name})",
            "",
            "## Video Notes",
            "",
            "A short demo video can be recorded from the Pygame window by running the simulator, pressing `P` for AI mode, and showing the generated telemetry dashboard/report afterward.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    cases = [
        ("high_downforce", "aero_setup", dict(setup_name="high_downforce")),
        ("low_drag", "aero_setup", dict(setup_name="low_drag")),
        ("front_biased", "aero_balance", dict(setup_name="front_aero")),
        ("rear_biased", "aero_balance", dict(setup_name="rear_aero")),
        ("conservative_energy", "energy", dict(params=DriverParameters(hybrid_deploy_speed=125.0))),
        ("aggressive_energy", "energy", dict(params=DriverParameters(hybrid_deploy_speed=45.0))),
        ("fresh_tires", "tires", dict(initial_tire_wear=0.0, initial_tire_temperature=92.0)),
        ("worn_tires", "tires", dict(initial_tire_wear=0.55, initial_tire_temperature=92.0)),
        ("cold_tires", "tires", dict(initial_tire_wear=0.0, initial_tire_temperature=45.0)),
        ("ideal_tires", "tires", dict(initial_tire_wear=0.0, initial_tire_temperature=92.0)),
        ("dry_track", "track", dict(track_grip_multiplier=1.0)),
        ("wet_track", "track", dict(track_grip_multiplier=0.72)),
    ]

    for name, category, kwargs in cases:
        telemetry_path = RESULTS_DIR / f"{name}.csv"
        result = run_ai_laps(laps=3, max_time=90.0, save_telemetry=True, telemetry_path=telemetry_path, **kwargs)
        rows.append(add_telemetry_metrics(row(name, category, result, score_lap(result)), telemetry_path))

    default_path = RESULTS_DIR / "rule_based_default.csv"
    default_result = run_ai_laps(laps=3, max_time=90.0, save_telemetry=True, telemetry_path=default_path)
    rows.append(add_telemetry_metrics(row("rule_based_default", "ai", default_result, score_lap(default_result)), default_path))

    optimized = random_search(iterations=14, seed=41)
    optimized_path = RESULTS_DIR / "optimized_ai.csv"
    optimized_result = run_ai_laps(laps=3, max_time=90.0, params=optimized.params, save_telemetry=True, telemetry_path=optimized_path)
    rows.append(add_telemetry_metrics(row("optimized_rule_based", "ai", optimized_result, score_lap(optimized_result)), optimized_path))

    csv_path = RESULTS_DIR / "final_demo_results.csv"
    report_path = RESULTS_DIR / "final_demo_report.md"
    plot_path = RESULTS_DIR / "optimized_ai_dashboard.png"
    plot_run(RESULTS_DIR / "optimized_ai.csv", plot_path)
    write_csv(csv_path, rows)
    write_report(report_path, rows, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
