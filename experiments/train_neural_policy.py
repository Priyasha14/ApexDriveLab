import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.neural_policy import NeuralPolicy, build_observation, inputs_to_action
from ai.racing_line import RacingLine
from ai.rule_driver import DriverParameters, RuleBasedDriver
from config import FPS, OFF_TRACK_GRIP_SCALE
from physics.car import Car, CarInputs
from physics.setup import SETUPS
from physics.vector_utils import clamp
from track.checkpoints import CheckpointManager
from track.track import Track


def noisy_inputs(inputs: CarInputs, rng: np.random.Generator, noise: float) -> CarInputs:
    if noise <= 0.0:
        return inputs
    steer = clamp(inputs.steer + float(rng.normal(0.0, 0.55 * noise)), -1.0, 1.0)
    throttle = clamp(inputs.throttle + float(rng.normal(0.0, 0.25 * noise)), 0.0, 1.0)
    brake = clamp(inputs.brake + float(rng.normal(0.0, 0.20 * noise)), 0.0, 1.0)
    if brake > 0.15 and brake > throttle:
        throttle = 0.0
    elif throttle > brake:
        brake = 0.0
    return CarInputs(
        throttle=throttle,
        brake=brake,
        steer=steer,
        aero_mode=inputs.aero_mode,
        deploy_hybrid=inputs.deploy_hybrid,
    )


def collect_demonstrations(
    laps: int = 5,
    max_time: float = 180.0,
    setup_name: str = "balanced",
    rollouts: int = 4,
    control_noise: float = 0.22,
    seed: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    track = Track()
    racing_line = RacingLine.from_track(track)
    rng = np.random.default_rng(seed)
    dt = 1.0 / FPS
    observations = []
    actions = []

    for rollout in range(max(1, rollouts)):
        driver = RuleBasedDriver(racing_line, DriverParameters())
        car = Car()
        car.apply_setup(SETUPS.get(setup_name, SETUPS["balanced"]))
        car.heading += float(rng.normal(0.0, 0.03 * rollout))
        checkpoints = CheckpointManager()
        checkpoints.reset(track.checkpoint_index_for_position(car.position))
        elapsed = 0.0
        rollout_noise = 0.0 if rollout == 0 else control_noise

        while elapsed < max_time and checkpoints.state.lap_count < laps:
            expert_inputs = driver.control(car, track)
            observations.append(build_observation(car, track, racing_line))
            actions.append(inputs_to_action(expert_inputs))

            executed_inputs = noisy_inputs(expert_inputs, rng, rollout_noise)
            on_track = track.is_on_track(car.position)
            grip_scale = 1.0 if on_track else OFF_TRACK_GRIP_SCALE
            car.update(dt, executed_inputs, grip_scale)
            checkpoints.update(dt, track.checkpoint_index_for_position(car.position))
            elapsed += dt

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def save_history(path: Path, losses: list[float], validation_losses: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "validation_loss"])
        for epoch, (loss, validation_loss) in enumerate(zip(losses, validation_losses), start=1):
            writer.writerow([epoch, f"{loss:.8f}", f"{validation_loss:.8f}"])
    return path


def save_report(path: Path, sample_count: int, final_loss: float, final_validation_loss: float, model_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Neural Policy Training",
                "",
                "The neural driver is trained with imitation learning. It observes the simulator state and learns to copy the rule-based driver's control outputs.",
                "",
                f"- Demonstration samples: {sample_count}",
                f"- Final training loss: {final_loss:.6f}",
                f"- Final validation loss: {final_validation_loss:.6f}",
                f"- Saved model: `{model_path}`",
                "",
                "This is a baseline neural network, not reinforcement learning. The next step would be to fine-tune it with repeated lap rewards.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy neural driver from rule-driver demonstrations.")
    parser.add_argument("--laps", type=int, default=5)
    parser.add_argument("--max-time", type=float, default=180.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--control-noise", type=float, default=0.22)
    parser.add_argument("--model-path", type=Path, default=Path("models") / "neural_policy.npz")
    parser.add_argument("--history-path", type=Path, default=Path("experiments") / "results" / "neural_policy_training.csv")
    parser.add_argument("--report-path", type=Path, default=Path("experiments") / "results" / "neural_policy_training.md")
    args = parser.parse_args()

    observations, actions = collect_demonstrations(laps=args.laps, max_time=args.max_time, rollouts=args.rollouts, control_noise=args.control_noise)
    policy = NeuralPolicy.create()
    loss_weights = np.array([3.0, 2.0, 2.0, 0.35, 0.35], dtype=np.float32)
    history = policy.train(observations, actions, epochs=args.epochs, learning_rate=args.learning_rate, loss_weights=loss_weights)
    policy.save(args.model_path)
    save_history(args.history_path, history.losses, history.validation_losses)
    save_report(args.report_path, len(observations), history.losses[-1], history.validation_losses[-1], args.model_path)

    print(f"collected_samples={len(observations)}")
    print(f"final_loss={history.losses[-1]:.6f}")
    print(f"final_validation_loss={history.validation_losses[-1]:.6f}")
    print(f"saved_model={args.model_path}")


if __name__ == "__main__":
    main()
