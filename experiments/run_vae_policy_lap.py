import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.latent_policy import LatentPolicy
from ai.neural_policy import build_observation
from ai.racing_line import RacingLine
from ai.rule_driver import DriverState
from ai.vae import DrivingStateVAE
from config import FPS, OFF_TRACK_GRIP_SCALE
from physics.car import Car
from physics.setup import SETUPS
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track


@dataclass
class VAEPolicyLapResult:
    completed_laps: int
    elapsed_time: float
    off_track_pct: float
    max_grip_usage: float
    best_lap_time: float | None


def run_vae_policy_laps(
    vae_model: Path = Path("models") / "driving_state_vae_8d.npz",
    policy_path: Path = Path("models") / "vae_latent_policy.npz",
    laps: int = 1,
    max_time: float = 120.0,
    setup_name: str = "balanced",
    save_telemetry: bool = False,
    telemetry_path: Path | None = None,
) -> VAEPolicyLapResult:
    track = Track()
    racing_line = RacingLine(track.center, track.inner_radius, track.outer_radius)
    vae = DrivingStateVAE.load(vae_model)
    policy = LatentPolicy.load(policy_path)
    car = Car()
    car.apply_setup(SETUPS.get(setup_name, SETUPS["balanced"]))
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))
    telemetry = TelemetryLogger()

    dt = 1.0 / FPS
    elapsed = 0.0
    off_track_samples = 0
    samples = 0
    max_grip = 0.0
    driver_state = DriverState(decision="vae_policy")

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
        telemetry.log(elapsed, car, on_track, driver_state, track)

    if save_telemetry:
        telemetry.save_csv(telemetry_path or Path("runs") / "vae_policy_lap.csv")

    return VAEPolicyLapResult(
        completed_laps=checkpoints.state.lap_count,
        elapsed_time=elapsed,
        off_track_pct=off_track_samples / max(samples, 1) * 100.0,
        max_grip_usage=max_grip,
        best_lap_time=checkpoints.state.best_lap_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VAE-latent neural policy in headless simulation.")
    parser.add_argument("--vae-model", type=Path, default=Path("models") / "driving_state_vae_8d.npz")
    parser.add_argument("--policy", type=Path, default=Path("models") / "vae_latent_policy.npz")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--setup", default="balanced")
    parser.add_argument("--save-telemetry", action="store_true")
    args = parser.parse_args()

    result = run_vae_policy_laps(
        vae_model=args.vae_model,
        policy_path=args.policy,
        laps=args.laps,
        max_time=args.max_time,
        setup_name=args.setup,
        save_telemetry=args.save_telemetry,
    )
    print(result)


if __name__ == "__main__":
    main()
