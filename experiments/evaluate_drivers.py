import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.latent_policy import LatentPolicy
from ai.neural_policy import NeuralPolicy, build_observation
from ai.racing_line import RacingLine
from ai.rule_driver import DriverParameters, RuleBasedDriver
from ai.vae import DrivingStateVAE
from config import FPS, OFF_TRACK_GRIP_SCALE
from physics.car import Car
from physics.setup import SETUPS
from track.checkpoints import CheckpointManager
from track.track import Track


@dataclass(frozen=True)
class EvaluationCondition:
    name: str
    setup_name: str = "balanced"
    grip_multiplier: float = 1.0
    tire_temperature: float | None = None
    tire_wear: float | None = None
    max_time: float = 90.0


@dataclass
class DriverModels:
    raw_policy: NeuralPolicy
    vae: DrivingStateVAE
    vae_policy: LatentPolicy
    optimized_vae_policy: LatentPolicy


@dataclass
class EvaluationResult:
    driver: str
    condition: str
    repeat: int
    completed_laps: int
    lap_time: float | None
    elapsed_time: float
    off_track_pct: float
    max_grip_usage: float
    avg_speed_kmh: float
    score: float


CONDITIONS = [
    EvaluationCondition("balanced"),
    EvaluationCondition("high_downforce", setup_name="high_downforce"),
    EvaluationCondition("low_drag", setup_name="low_drag"),
    EvaluationCondition("cold_tires", tire_temperature=45.0),
    EvaluationCondition("worn_tires", tire_wear=0.55),
    EvaluationCondition("wet_track", grip_multiplier=0.72, max_time=110.0),
]


DRIVERS = ["rule_based", "raw_neural", "vae_latent", "optimized_vae_latent"]


def score_result(completed_laps: int, lap_time: float | None, elapsed_time: float, off_track_pct: float, max_grip_usage: float) -> float:
    base_time = lap_time or elapsed_time + 50.0
    incomplete_penalty = 180.0 if completed_laps == 0 else 0.0
    off_track_penalty = off_track_pct * 2.0
    grip_penalty = max(0.0, max_grip_usage - 0.98) * 25.0
    return base_time + incomplete_penalty + off_track_penalty + grip_penalty


def load_models(
    raw_policy_path: Path,
    vae_path: Path,
    vae_policy_path: Path,
    optimized_vae_policy_path: Path,
) -> DriverModels:
    return DriverModels(
        raw_policy=NeuralPolicy.load(raw_policy_path),
        vae=DrivingStateVAE.load(vae_path),
        vae_policy=LatentPolicy.load(vae_policy_path),
        optimized_vae_policy=LatentPolicy.load(optimized_vae_policy_path),
    )


def control_for_driver(driver: str, models: DriverModels, rule_driver: RuleBasedDriver, car: Car, track: Track, racing_line: RacingLine):
    if driver == "rule_based":
        return rule_driver.control(car, track)
    if driver == "raw_neural":
        return models.raw_policy.control(car, track, racing_line)

    observation = build_observation(car, track, racing_line).reshape(1, -1)
    latent, _ = models.vae.encode(observation)
    if driver == "vae_latent":
        return models.vae_policy.control(latent[0])
    if driver == "optimized_vae_latent":
        return models.optimized_vae_policy.control(latent[0])
    raise ValueError(f"Unknown driver: {driver}")


def run_driver_condition(driver: str, condition: EvaluationCondition, models: DriverModels, repeat: int) -> EvaluationResult:
    track = Track()
    racing_line = RacingLine(track.center, track.inner_radius, track.outer_radius)
    rule_driver = RuleBasedDriver(racing_line, DriverParameters())
    car = Car()
    car.apply_setup(SETUPS.get(condition.setup_name, SETUPS["balanced"]))
    car.set_tire_condition(condition.tire_temperature, condition.tire_wear)
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))

    dt = 1.0 / FPS
    elapsed = 0.0
    samples = 0
    off_track_samples = 0
    max_grip = 0.0
    speed_sum = 0.0

    while elapsed < condition.max_time and checkpoints.state.lap_count < 1:
        inputs = control_for_driver(driver, models, rule_driver, car, track, racing_line)
        on_track = track.is_on_track(car.position)
        grip_scale = (1.0 if on_track else OFF_TRACK_GRIP_SCALE) * condition.grip_multiplier
        car.update(dt, inputs, grip_scale)
        on_track = track.is_on_track(car.position)
        checkpoints.update(dt, track.checkpoint_index_for_position(car.position))

        elapsed += dt
        samples += 1
        off_track_samples += 0 if on_track else 1
        max_grip = max(max_grip, car.tire_state.combined_grip_usage)
        speed_sum += car.speed_kmh

    off_track_pct = off_track_samples / max(samples, 1) * 100.0
    lap_time = checkpoints.state.best_lap_time
    return EvaluationResult(
        driver=driver,
        condition=condition.name,
        repeat=repeat,
        completed_laps=checkpoints.state.lap_count,
        lap_time=lap_time,
        elapsed_time=elapsed,
        off_track_pct=off_track_pct,
        max_grip_usage=max_grip,
        avg_speed_kmh=speed_sum / max(samples, 1),
        score=score_result(checkpoints.state.lap_count, lap_time, elapsed, off_track_pct, max_grip),
    )


def write_csv(path: Path, results: list[EvaluationResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "driver",
                "condition",
                "repeat",
                "completed_laps",
                "lap_time",
                "elapsed_time",
                "off_track_pct",
                "max_grip_usage",
                "avg_speed_kmh",
                "score",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "driver": result.driver,
                    "condition": result.condition,
                    "repeat": result.repeat,
                    "completed_laps": result.completed_laps,
                    "lap_time": "" if result.lap_time is None else f"{result.lap_time:.6f}",
                    "elapsed_time": f"{result.elapsed_time:.6f}",
                    "off_track_pct": f"{result.off_track_pct:.6f}",
                    "max_grip_usage": f"{result.max_grip_usage:.6f}",
                    "avg_speed_kmh": f"{result.avg_speed_kmh:.6f}",
                    "score": f"{result.score:.6f}",
                }
            )
    return path


def summarize_driver(results: list[EvaluationResult], driver: str) -> list[str]:
    driver_results = [result for result in results if result.driver == driver]
    completed = [result for result in driver_results if result.completed_laps > 0 and result.lap_time is not None]
    completion_rate = len(completed) / max(len(driver_results), 1) * 100.0
    avg_score = statistics.mean(result.score for result in driver_results)
    avg_off_track = statistics.mean(result.off_track_pct for result in driver_results)
    if completed:
        avg_lap = statistics.mean(result.lap_time for result in completed if result.lap_time is not None)
        best_lap = min(result.lap_time for result in completed if result.lap_time is not None)
        return [driver, f"{completion_rate:.1f}%", f"{avg_lap:.3f}s", f"{best_lap:.3f}s", f"{avg_off_track:.2f}%", f"{avg_score:.3f}"]
    return [driver, f"{completion_rate:.1f}%", "n/a", "n/a", f"{avg_off_track:.2f}%", f"{avg_score:.3f}"]


def write_report(path: Path, results: list[EvaluationResult], csv_path: Path, repeats: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Driver Evaluation Suite",
        "",
        "This experiment compares four controllers across setup, tire, and grip changes.",
        "",
        f"- Repeats per condition: {repeats}",
        f"- CSV: `{csv_path}`",
        "",
        "| Driver | Completion | Avg completed lap | Best lap | Avg off-track | Avg score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for driver in DRIVERS:
        lines.append("| " + " | ".join(summarize_driver(results, driver)) + " |")

    lines.extend(["", "## Condition Winners", ""])
    for condition in CONDITIONS:
        condition_results = [result for result in results if result.condition == condition.name]
        best = min(condition_results, key=lambda result: result.score)
        lap_text = "incomplete" if best.lap_time is None else f"{best.lap_time:.3f}s"
        lines.append(f"- {condition.name}: {best.driver} ({lap_text}, score {best.score:.3f}, off-track {best.off_track_pct:.2f}%)")

    lines.extend(
        [
            "",
            "Interpretation: a driver that is fast only in the balanced condition may be overfit. A stronger controller should keep completing laps as grip, tires, and aero setup change.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate_all(
    repeats: int,
    output_csv: Path,
    output_report: Path,
    raw_policy_path: Path,
    vae_path: Path,
    vae_policy_path: Path,
    optimized_vae_policy_path: Path,
) -> list[EvaluationResult]:
    models = load_models(raw_policy_path, vae_path, vae_policy_path, optimized_vae_policy_path)
    results = []
    for condition in CONDITIONS:
        for driver in DRIVERS:
            for repeat in range(1, repeats + 1):
                results.append(run_driver_condition(driver, condition, models, repeat))
    write_csv(output_csv, results)
    write_report(output_report, results, output_csv, repeats)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all drivers across multiple simulator conditions.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-csv", type=Path, default=Path("experiments") / "results" / "driver_evaluation.csv")
    parser.add_argument("--output-report", type=Path, default=Path("experiments") / "results" / "driver_evaluation.md")
    parser.add_argument("--raw-policy", type=Path, default=Path("models") / "neural_policy.npz")
    parser.add_argument("--vae", type=Path, default=Path("models") / "driving_state_vae_8d.npz")
    parser.add_argument("--vae-policy", type=Path, default=Path("models") / "vae_latent_policy.npz")
    parser.add_argument("--optimized-vae-policy", type=Path, default=Path("models") / "optimized_vae_latent_policy.npz")
    args = parser.parse_args()

    results = evaluate_all(
        repeats=args.repeats,
        output_csv=args.output_csv,
        output_report=args.output_report,
        raw_policy_path=args.raw_policy,
        vae_path=args.vae,
        vae_policy_path=args.vae_policy,
        optimized_vae_policy_path=args.optimized_vae_policy,
    )
    print(f"evaluations={len(results)}")
    print(f"csv={args.output_csv}")
    print(f"report={args.output_report}")


if __name__ == "__main__":
    main()
