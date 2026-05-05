import random
from dataclasses import dataclass

from ai.rule_driver import DriverParameters
from experiments.run_lap import LapResult, run_ai_laps


@dataclass
class OptimizationResult:
    params: DriverParameters
    score: float
    lap_result: LapResult


def score_lap(result: LapResult) -> float:
    lap_time = result.best_lap_time or result.elapsed_time + 30.0
    incomplete_penalty = 120.0 if result.completed_laps == 0 else 0.0
    off_track_penalty = result.off_track_pct * 0.65
    grip_penalty = max(0.0, result.max_grip_usage - 0.96) * 25.0
    return lap_time + incomplete_penalty + off_track_penalty + grip_penalty


def sample_params(rng: random.Random) -> DriverParameters:
    return DriverParameters(
        lookahead_distance=rng.uniform(82.0, 160.0),
        speed_lookahead_gain=rng.uniform(0.25, 0.90),
        corner_speed_multiplier=rng.uniform(0.68, 0.98),
        braking_safety_margin=rng.uniform(0.95, 1.55),
        throttle_aggressiveness=rng.uniform(0.012, 0.030),
        brake_aggressiveness=rng.uniform(0.020, 0.050),
        aero_switch_speed=rng.uniform(95.0, 145.0),
        hybrid_deploy_speed=rng.uniform(55.0, 115.0),
    )


def random_search(iterations: int = 12, seed: int = 7, setup_name: str = "balanced") -> OptimizationResult:
    rng = random.Random(seed)
    best: OptimizationResult | None = None

    default_params = DriverParameters()
    candidates = [default_params] + [sample_params(rng) for _ in range(max(0, iterations - 1))]
    for params in candidates:
        lap_result = run_ai_laps(laps=1, max_time=60.0, setup_name=setup_name, params=params)
        score = score_lap(lap_result)
        result = OptimizationResult(params=params, score=score, lap_result=lap_result)
        if best is None or result.score < best.score:
            best = result

    assert best is not None
    return best


def format_result(result: OptimizationResult) -> str:
    return (
        f"score={result.score:.3f}, "
        f"lap={result.lap_result.best_lap_time}, "
        f"off_track={result.lap_result.off_track_pct:.1f}%, "
        f"grip={result.lap_result.max_grip_usage:.2f}, "
        f"params={result.params}"
    )
