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
    def __init__(self) -> None:
        self.state = LapState()

    def reset(self) -> None:
        self.state = LapState()

    def update(self, dt: float, checkpoint_index: int) -> None:
        self.state.lap_time += dt
        expected = (self.state.current_checkpoint + 1) % CHECKPOINT_COUNT

        if checkpoint_index != expected:
            return

        self.state.current_checkpoint = expected
        if expected == 0:
            self.state.lap_count += 1
            self.state.last_lap_time = self.state.lap_time
            if self.state.best_lap_time is None or self.state.lap_time < self.state.best_lap_time:
                self.state.best_lap_time = self.state.lap_time
            self.state.lap_time = 0.0

