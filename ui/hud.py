import pygame

from config import CHECKPOINT_COUNT, HUD_COLOR, HUD_WARNING_COLOR


class HUD:
    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

    def draw(self, screen: pygame.Surface, car, lap_state, on_track: bool, debug_enabled: bool) -> None:
        lines = [
            f"Speed: {car.speed_kmh:6.1f} km/h",
            f"Lap: {lap_state.lap_count}",
            f"Lap time: {self._format_time(lap_state.lap_time)}",
            f"Best: {self._format_time(lap_state.best_lap_time)}",
            f"Checkpoint: {lap_state.current_checkpoint + 1}/{CHECKPOINT_COUNT}",
            f"Off track: {'NO' if on_track else 'YES'}",
            f"Debug: {'ON' if debug_enabled else 'OFF'}",
        ]
        x, y = 20, 18
        for line in lines:
            color = HUD_WARNING_COLOR if line.endswith("YES") else HUD_COLOR
            surface = self.font.render(line, True, color)
            screen.blit(surface, (x, y))
            y += 28

        help_text = "Controls: W/Up throttle | S/Down brake/reverse | A/D steer | T save data | F1 debug | R reset | Esc quit"
        surface = self.small_font.render(help_text, True, HUD_COLOR)
        screen.blit(surface, (20, screen.get_height() - 32))

    def _format_time(self, value: float | None) -> str:
        if value is None:
            return "--:--.---"
        minutes = int(value // 60)
        seconds = value % 60
        return f"{minutes:02d}:{seconds:06.3f}"
