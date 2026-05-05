import math

import numpy as np

from ai.racing_line import RacingLine
from ai.speed_planner import ellipse_curvature
from config import FPS, OFF_TRACK_GRIP_SCALE
from physics.car import Car, CarInputs
from track.checkpoints import CheckpointManager
from track.track import Track

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


class ApexDriveEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self):
        if gym is None:
            raise ImportError("Install gymnasium to use ApexDriveEnv.")
        self.track = Track()
        self.racing_line = RacingLine(self.track.center, self.track.inner_radius, self.track.outer_radius)
        self.car = Car()
        self.checkpoints = CheckpointManager()
        self.dt = 1.0 / FPS
        self.elapsed = 0.0
        self.previous_angle = self.track.angle_for_position(self.car.position)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset()
        self.checkpoints.reset(self.track.checkpoint_index_for_position(self.car.position))
        self.elapsed = 0.0
        self.previous_angle = self.track.angle_for_position(self.car.position)
        return self._observation(), {}

    def step(self, action):
        steer, throttle, brake, aero_value, deploy_value = action
        aero_mode = "straight" if aero_value > 0.5 else "corner"
        inputs = CarInputs(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
            aero_mode=aero_mode,
            deploy_hybrid=bool(deploy_value > 0.5),
        )

        on_track = self.track.is_on_track(self.car.position)
        self.car.update(self.dt, inputs, 1.0 if on_track else OFF_TRACK_GRIP_SCALE)
        on_track = self.track.is_on_track(self.car.position)
        self.checkpoints.update(self.dt, self.track.checkpoint_index_for_position(self.car.position))
        self.elapsed += self.dt

        angle = self.track.angle_for_position(self.car.position)
        progress = (self.previous_angle - angle) % math.tau
        if progress > math.pi:
            progress = 0.0
        self.previous_angle = angle

        reward = progress * 25.0
        reward += 20.0 if self.checkpoints.state.last_lap_time is not None else 0.0
        reward -= 2.0 if not on_track else 0.0
        reward -= max(0.0, self.car.tire_state.combined_grip_usage - 0.98) * 0.4
        reward -= 0.1 if throttle > 0.2 and brake > 0.2 else 0.0

        terminated = self.checkpoints.state.lap_count >= 1
        truncated = self.elapsed >= 90.0
        return self._observation(), reward, terminated, truncated, {}

    def _observation(self):
        angle = self.track.angle_for_position(self.car.position)
        path_error = self.racing_line.path_error(self.car.position, angle) / 240.0
        curvature = ellipse_curvature(angle, self.track.outer_radius) * 180.0
        checkpoint_delta = self.checkpoints.state.current_checkpoint / 8.0
        return np.array(
            [
                self.car.speed_kmh / 220.0,
                self.car.yaw_rate / 2.5,
                path_error,
                curvature,
                self.car.tire_state.combined_grip_usage,
                self.car.hybrid_state.charge_fraction,
                1.0 if self.car.aero_state.mode == "straight" else 0.0,
                checkpoint_delta,
                1.0 if self.track.is_on_track(self.car.position) else -1.0,
            ],
            dtype=np.float32,
        )
