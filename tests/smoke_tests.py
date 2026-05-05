from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.car import Car, CarInputs
from physics.setup import SETUPS
from experiments.run_lap import run_ai_laps
from ai.optimizer import random_search
from ai.neural_policy import NeuralPolicy, action_to_inputs
from ai.latent_policy import LatentPolicy
from ai.vae import DrivingStateVAE
from telemetry.analysis import summarize
from telemetry.logger import TelemetryLogger
from track.checkpoints import CheckpointManager
from track.track import Track


def test_straight_line_stability() -> None:
    car = Car()
    for _ in range(180):
        car.update(1 / 60, CarInputs(throttle=1.0), 1.0)

    assert car.speed_kmh > 80.0
    assert abs(car.yaw_rate) < 0.01
    assert car.handling_balance == "neutral"


def test_steering_produces_tire_state() -> None:
    car = Car()
    for _ in range(90):
        car.update(1 / 60, CarInputs(throttle=1.0, steer=-0.25), 1.0)

    assert abs(car.steering_angle) > 0.05
    assert abs(car.tire_state.front_slip_angle) > 0.01
    assert car.tire_state.combined_grip_usage > 0.0


def test_setup_switch_changes_vehicle_response() -> None:
    balanced = Car()
    rotation = Car()
    rotation.apply_setup(SETUPS["rotation"])

    for _ in range(120):
        balanced.update(1 / 60, CarInputs(throttle=1.0, steer=-0.25), 1.0)
        rotation.update(1 / 60, CarInputs(throttle=1.0, steer=-0.25), 1.0)

    assert rotation.setup.name == "rotation"
    assert abs(rotation.yaw_rate - balanced.yaw_rate) > 0.01


def test_aero_and_hybrid_states_update() -> None:
    car = Car()
    for _ in range(180):
        car.update(1 / 60, CarInputs(throttle=1.0, deploy_hybrid=True, aero_mode="straight"), 1.0)

    assert car.aero_state.mode == "straight"
    assert car.aero_state.drag_force > 0.0
    assert car.aero_state.downforce > 0.0
    assert car.hybrid_state.energy < car.hybrid_state.capacity * 0.70

    energy_after_deploy = car.hybrid_state.energy
    for _ in range(60):
        car.update(1 / 60, CarInputs(brake=1.0, aero_mode="corner"), 1.0)

    assert car.aero_state.mode == "corner"
    assert car.hybrid_state.energy >= energy_after_deploy


def test_clockwise_checkpoint_lap() -> None:
    checkpoints = CheckpointManager()
    checkpoints.reset(0)

    for checkpoint_index in [7, 6, 5, 4, 3, 2, 1, 0]:
        checkpoints.update(0.1, checkpoint_index)

    assert checkpoints.state.lap_count == 1
    assert checkpoints.state.current_checkpoint == 0
    assert checkpoints.state.last_lap_time is not None


def test_telemetry_export(tmp_path: Path) -> None:
    car = Car()
    logger = TelemetryLogger()
    car.update(1 / 60, CarInputs(throttle=1.0, steer=-0.2), 1.0)
    logger.log(1 / 60, car, True)

    path = logger.save_csv(tmp_path / "telemetry.csv")

    assert path is not None
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "lateral_acceleration" in header
    assert "front_slip_angle" in header
    assert "tire_grip_usage" in header

    summary = summarize(path)
    assert summary["samples"] == 1
    assert "max_speed_kmh" in summary
    assert "handling_balance" in summary


def test_rule_driver_completes_clean_lap() -> None:
    result = run_ai_laps(laps=1, max_time=45.0)
    assert result.completed_laps == 1
    assert result.off_track_pct < 5.0


def test_optimizer_returns_valid_candidate() -> None:
    result = random_search(iterations=2, seed=3)
    assert result.lap_result.completed_laps == 1
    assert result.score > 0.0


def test_neural_policy_predicts_valid_inputs(tmp_path: Path) -> None:
    policy = NeuralPolicy.create(seed=5)
    action = policy.predict(policy.observation_mean)
    inputs = action_to_inputs(action)
    model_path = tmp_path / "policy.npz"
    policy.save(model_path)
    loaded = NeuralPolicy.load(model_path)
    loaded_action = loaded.predict(loaded.observation_mean)

    assert action.shape == (5,)
    assert -1.0 <= inputs.steer <= 1.0
    assert 0.0 <= inputs.throttle <= 1.0
    assert 0.0 <= inputs.brake <= 1.0
    assert loaded_action.shape == (5,)


def test_vae_encode_decode_roundtrip(tmp_path: Path) -> None:
    vae = DrivingStateVAE.create(seed=8)
    observations = np.zeros((6, 14), dtype=np.float32)
    vae.set_normalization(observations)
    mean, logvar = vae.encode(observations)
    reconstruction = vae.reconstruct(observations)
    model_path = tmp_path / "vae.npz"
    vae.save(model_path)
    loaded = DrivingStateVAE.load(model_path)
    loaded_mean, _ = loaded.encode(observations)

    assert mean.shape == (6, 2)
    assert logvar.shape == (6, 2)
    assert reconstruction.shape == observations.shape
    assert loaded_mean.shape == mean.shape


def test_latent_policy_predicts_valid_inputs(tmp_path: Path) -> None:
    policy = LatentPolicy.create(latent_size=8, seed=10)
    action = policy.predict(np.zeros(8, dtype=np.float32))
    inputs = policy.control(np.zeros(8, dtype=np.float32))
    model_path = tmp_path / "latent_policy.npz"
    policy.save(model_path)
    loaded = LatentPolicy.load(model_path)

    assert action.shape == (5,)
    assert -1.0 <= inputs.steer <= 1.0
    assert 0.0 <= inputs.throttle <= 1.0
    assert 0.0 <= inputs.brake <= 1.0
    assert loaded.latent_size == 8


def run_all() -> None:
    temp_dir = Path("runs") / "_smoke"
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_straight_line_stability()
    test_steering_produces_tire_state()
    test_setup_switch_changes_vehicle_response()
    test_aero_and_hybrid_states_update()
    test_clockwise_checkpoint_lap()
    test_telemetry_export(temp_dir)
    test_rule_driver_completes_clean_lap()
    test_optimizer_returns_valid_candidate()
    test_neural_policy_predicts_valid_inputs(temp_dir)
    test_vae_encode_decode_roundtrip(temp_dir)
    test_latent_policy_predicts_valid_inputs(temp_dir)
    print("smoke tests passed")


if __name__ == "__main__":
    run_all()
