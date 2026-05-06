import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.latent_policy import LatentPolicy
from ai.neural_policy import build_observation
from ai.racing_line import RacingLine
from ai.rule_driver import DriverState
from ai.vae import DrivingStateVAE
from config import FPS, OFF_TRACK_GRIP_SCALE
from experiments.run_lap import run_ai_laps
from experiments.run_neural_lap import run_neural_laps
from experiments.run_vae_policy_lap import run_vae_policy_laps
from physics.car import Car
from physics.setup import SETUPS
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track


@dataclass
class PolicyEvaluation:
    score: float
    completed_laps: int
    elapsed_time: float
    off_track_pct: float
    max_grip_usage: float
    best_lap_time: float | None


def evaluate_policy(
    vae: DrivingStateVAE,
    policy: LatentPolicy,
    laps: int = 1,
    max_time: float = 60.0,
    setup_name: str = "balanced",
    save_telemetry: bool = False,
    telemetry_path: Path | None = None,
) -> PolicyEvaluation:
    track = Track()
    racing_line = RacingLine(track.center, track.inner_radius, track.outer_radius)
    car = Car()
    car.apply_setup(SETUPS.get(setup_name, SETUPS["balanced"]))
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))
    telemetry = TelemetryLogger()
    driver_state = DriverState(decision="optimized_vae_policy")

    dt = 1.0 / FPS
    elapsed = 0.0
    samples = 0
    off_track_samples = 0
    max_grip = 0.0

    while elapsed < max_time and checkpoints.state.lap_count < laps:
        observation = build_observation(car, track, racing_line).reshape(1, -1)
        latent, _ = vae.encode(observation)
        inputs = policy.control(latent[0])
        on_track = track.is_on_track(car.position)
        grip_scale = 1.0 if on_track else OFF_TRACK_GRIP_SCALE
        car.update(dt, inputs, grip_scale)
        on_track = track.is_on_track(car.position)
        checkpoints.update(dt, track.checkpoint_index_for_position(car.position))

        elapsed += dt
        samples += 1
        off_track_samples += 0 if on_track else 1
        max_grip = max(max_grip, car.tire_state.combined_grip_usage)
        if save_telemetry:
            telemetry.log(elapsed, car, on_track, driver_state, track)

    if save_telemetry:
        telemetry.save_csv(telemetry_path or Path("runs") / "optimized_vae_policy_lap.csv")

    off_track_pct = off_track_samples / max(samples, 1) * 100.0
    return PolicyEvaluation(
        score=score_result(checkpoints.state.lap_count, elapsed, checkpoints.state.best_lap_time, off_track_pct, max_grip),
        completed_laps=checkpoints.state.lap_count,
        elapsed_time=elapsed,
        off_track_pct=off_track_pct,
        max_grip_usage=max_grip,
        best_lap_time=checkpoints.state.best_lap_time,
    )


def score_result(completed_laps: int, elapsed_time: float, best_lap_time: float | None, off_track_pct: float, max_grip_usage: float) -> float:
    lap_time = best_lap_time or elapsed_time + 40.0
    incomplete_penalty = 160.0 if completed_laps == 0 else 0.0
    off_track_penalty = off_track_pct * 1.8
    grip_penalty = max(0.0, max_grip_usage - 0.98) * 20.0
    return lap_time + incomplete_penalty + off_track_penalty + grip_penalty


def mutate_policy(base: LatentPolicy, rng: np.random.Generator, mutation_scale: float, structured_strength: float) -> LatentPolicy:
    candidate = base.copy()
    for array in [candidate.w1, candidate.b1, candidate.w2, candidate.b2, candidate.w3, candidate.b3]:
        array += rng.normal(0.0, mutation_scale, size=array.shape).astype(np.float32)

    throttle_bias = rng.normal(0.02, structured_strength)
    brake_bias = rng.normal(-0.01, structured_strength)
    steer_gain = float(np.clip(rng.normal(1.0, structured_strength * 1.6), 0.82, 1.18))
    candidate.b3[1] += throttle_bias
    candidate.b3[2] += brake_bias
    candidate.w3[:, 0] *= steer_gain
    candidate.b3[0] *= steer_gain
    return candidate


def save_history(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["iteration", "score", "lap_time", "completed_laps", "off_track_pct", "max_grip_usage", "accepted"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_report(
    path: Path,
    baseline: PolicyEvaluation,
    optimized: PolicyEvaluation,
    rule_lap: float | None,
    raw_neural_lap: float | None,
    optimized_policy_path: Path,
    history_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    improvement = (baseline.best_lap_time or baseline.elapsed_time) - (optimized.best_lap_time or optimized.elapsed_time)
    path.write_text(
        "\n".join(
            [
                "# Optimized VAE-Latent Policy",
                "",
                "The optimizer mutates the trained VAE-latent policy and keeps candidates that improve the lap score.",
                "",
                f"- Baseline VAE policy lap: {(baseline.best_lap_time or baseline.elapsed_time):.3f} s",
                f"- Optimized VAE policy lap: {(optimized.best_lap_time or optimized.elapsed_time):.3f} s",
                f"- Lap-time improvement: {improvement:.3f} s",
                f"- Optimized off-track: {optimized.off_track_pct:.2f}%",
                f"- Optimized score: {optimized.score:.3f}",
                f"- Optimized model: `{optimized_policy_path}`",
                f"- History: `{history_path}`",
                "",
                "Reference baselines:",
                f"- Rule-based driver: {rule_lap:.3f} s" if rule_lap is not None else "- Rule-based driver: incomplete",
                f"- Raw neural imitation driver: {raw_neural_lap:.3f} s" if raw_neural_lap is not None else "- Raw neural imitation driver: incomplete",
                "",
                "This is evolutionary policy optimization, not PPO. It is intentionally small and easy to inspect.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def optimize_policy(
    vae_model: Path,
    policy_path: Path,
    output_policy_path: Path,
    iterations: int,
    seed: int,
    mutation_scale: float,
    structured_strength: float,
    history_path: Path,
    report_path: Path,
) -> PolicyEvaluation:
    rng = np.random.default_rng(seed)
    vae = DrivingStateVAE.load(vae_model)
    best_policy = LatentPolicy.load(policy_path)
    best_eval = evaluate_policy(vae, best_policy)
    baseline_eval = best_eval
    rows = [
        {
            "iteration": 0,
            "score": f"{best_eval.score:.6f}",
            "lap_time": f"{(best_eval.best_lap_time or best_eval.elapsed_time):.6f}",
            "completed_laps": best_eval.completed_laps,
            "off_track_pct": f"{best_eval.off_track_pct:.6f}",
            "max_grip_usage": f"{best_eval.max_grip_usage:.6f}",
            "accepted": True,
        }
    ]

    for iteration in range(1, iterations + 1):
        candidate = mutate_policy(best_policy, rng, mutation_scale, structured_strength)
        evaluation = evaluate_policy(vae, candidate)
        accepted = evaluation.score < best_eval.score
        if accepted:
            best_policy = candidate
            best_eval = evaluation
        rows.append(
            {
                "iteration": iteration,
                "score": f"{evaluation.score:.6f}",
                "lap_time": f"{(evaluation.best_lap_time or evaluation.elapsed_time):.6f}",
                "completed_laps": evaluation.completed_laps,
                "off_track_pct": f"{evaluation.off_track_pct:.6f}",
                "max_grip_usage": f"{evaluation.max_grip_usage:.6f}",
                "accepted": accepted,
            }
        )

    best_policy.save(output_policy_path)
    final_eval = evaluate_policy(vae, best_policy, save_telemetry=True, telemetry_path=Path("runs") / "optimized_vae_policy_lap.csv")
    save_history(history_path, rows)
    rule = run_ai_laps(laps=1, max_time=45.0)
    raw_neural = run_neural_laps(laps=1, max_time=90.0)
    save_report(report_path, baseline_eval, final_eval, rule.best_lap_time, raw_neural.best_lap_time, output_policy_path, history_path)
    return final_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the VAE-latent driver with simple evolutionary search.")
    parser.add_argument("--vae-model", type=Path, default=Path("models") / "driving_state_vae_8d.npz")
    parser.add_argument("--policy", type=Path, default=Path("models") / "vae_latent_policy.npz")
    parser.add_argument("--output-policy", type=Path, default=Path("models") / "optimized_vae_latent_policy.npz")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--mutation-scale", type=float, default=0.004)
    parser.add_argument("--structured-strength", type=float, default=0.035)
    parser.add_argument("--history-path", type=Path, default=Path("experiments") / "results" / "vae_policy_optimization.csv")
    parser.add_argument("--report-path", type=Path, default=Path("experiments") / "results" / "vae_policy_optimization.md")
    args = parser.parse_args()

    result = optimize_policy(
        vae_model=args.vae_model,
        policy_path=args.policy,
        output_policy_path=args.output_policy,
        iterations=args.iterations,
        seed=args.seed,
        mutation_scale=args.mutation_scale,
        structured_strength=args.structured_strength,
        history_path=args.history_path,
        report_path=args.report_path,
    )
    print(result)


if __name__ == "__main__":
    main()
