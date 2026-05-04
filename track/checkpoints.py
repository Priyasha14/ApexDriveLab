from dataclasses import dataclass

from config import CHECKPOINT_COUNT


@dataclass
class LapState:
    current_checkpoint: int = 0
    lap_count: int = 0
    lap_time: float = 0.0
    best_lap_time: float | None = None
    last_lap_time: float | None = None


class CheckpointManager:
    def __init__(self, direction: int = -1) -> None:
        self.direction = -1 if direction < 0 else 1
        self.state = LapState()

    def reset(self, initial_checkpoint: int = 0) -> None:
        self.state = LapState(current_checkpoint=initial_checkpoint)

    def update(self, dt: float, checkpoint_index: int) -> None:
        self.state.lap_time += dt
        expected = (self.state.current_checkpoint + self.direction) % CHECKPOINT_COUNT

        if checkpoint_index != expected:
            return

        self.state.current_checkpoint = expected
        if expected == 0:
            self.state.lap_count += 1
            self.state.last_lap_time = self.state.lap_time
            if self.state.best_lap_time is None or self.state.lap_time < self.state.best_lap_time:
                self.state.best_lap_time = self.state.lap_time
            self.state.lap_time = 0.0
