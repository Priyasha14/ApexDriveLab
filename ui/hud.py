import pygame

from config import CHECKPOINT_COUNT, HUD_COLOR, HUD_PANEL_BORDER, HUD_PANEL_COLOR, HUD_WARNING_COLOR


class HUD:
    def __init__(self) -> None:
        self.title_font = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

    def draw(self, screen: pygame.Surface, car, lap_state, on_track: bool, debug_enabled: bool, ai_enabled: bool = False, ai_state=None) -> None:
        panel = pygame.Rect(16, 14, 310, 422)
        pygame.draw.rect(screen, HUD_PANEL_COLOR, panel, border_radius=8)
        pygame.draw.rect(screen, HUD_PANEL_BORDER, panel, 1, border_radius=8)

        title = self.title_font.render("ApexDriveLab", True, HUD_COLOR)
        screen.blit(title, (30, 26))
        self._draw_speed_bar(screen, car, panel)

        lines = [
            f"Speed: {car.speed_kmh:6.1f} km/h",
            f"Lap: {lap_state.lap_count}",
            f"Lap time: {self._format_time(lap_state.lap_time)}",
            f"Best: {self._format_time(lap_state.best_lap_time)}",
            f"Checkpoint: {lap_state.current_checkpoint + 1}/{CHECKPOINT_COUNT}",
            f"Off track: {'NO' if on_track else 'YES'}",
            f"Grip use: {car.tire_state.combined_grip_usage * 100:5.1f}%",
            f"Balance: {car.handling_balance}",
            f"Setup: {car.setup.name}",
            f"Aero: {car.aero_state.mode}",
            f"Battery: {car.hybrid_state.charge_fraction * 100:5.1f}%",
            f"Driver: {'AI' if ai_enabled else 'manual'}",
            f"Debug: {'ON' if debug_enabled else 'OFF'}",
        ]
        x, y = 30, 92
        for line in lines:
            color = HUD_WARNING_COLOR if line.endswith("YES") else HUD_COLOR
            surface = self.font.render(line, True, color)
            screen.blit(surface, (x, y))
            y += 28

        self._draw_grip_meter(screen, car, pygame.Rect(950, 18, 300, 88))
        self._draw_input_meter(screen, car, pygame.Rect(950, 118, 300, 116))
        if ai_enabled and ai_state:
            self._draw_ai_meter(screen, ai_state, pygame.Rect(950, 246, 300, 96))
        help_text = "Controls: W/S/A/D drive | P AI | 1-7 setup | Space deploy | Z/X/C aero | T data | F1 | R | Esc"
        surface = self.small_font.render(help_text, True, HUD_COLOR)
        help_rect = surface.get_rect(center=(screen.get_width() // 2, screen.get_height() - 24))
        pygame.draw.rect(screen, HUD_PANEL_COLOR, help_rect.inflate(24, 12), border_radius=7)
        screen.blit(surface, help_rect)

    def _format_time(self, value: float | None) -> str:
        if value is None:
            return "--:--.---"
        minutes = int(value // 60)
        seconds = value % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    def _draw_speed_bar(self, screen: pygame.Surface, car, panel: pygame.Rect) -> None:
        bar = pygame.Rect(panel.x + 14, panel.y + 58, panel.width - 28, 12)
        pygame.draw.rect(screen, (31, 38, 45), bar, border_radius=6)
        fill = bar.copy()
        fill.width = int(bar.width * min(car.speed_kmh / 240.0, 1.0))
        pygame.draw.rect(screen, (66, 170, 255), fill, border_radius=6)

    def _draw_grip_meter(self, screen: pygame.Surface, car, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, HUD_PANEL_COLOR, rect, border_radius=8)
        pygame.draw.rect(screen, HUD_PANEL_BORDER, rect, 1, border_radius=8)
        title = self.small_font.render("Tire Load", True, HUD_COLOR)
        screen.blit(title, (rect.x + 14, rect.y + 10))

        for index, (label, usage) in enumerate(
            [
                ("Front", car.tire_state.front_grip_usage),
                ("Rear", car.tire_state.rear_grip_usage),
            ]
        ):
            y = rect.y + 38 + index * 22
            text = self.small_font.render(label, True, HUD_COLOR)
            screen.blit(text, (rect.x + 14, y - 4))
            bar = pygame.Rect(rect.x + 78, y, 190, 10)
            pygame.draw.rect(screen, (31, 38, 45), bar, border_radius=5)
            fill = bar.copy()
            fill.width = int(bar.width * min(usage, 1.0))
            color = (73, 220, 132) if usage < 0.82 else HUD_WARNING_COLOR
            pygame.draw.rect(screen, color, fill, border_radius=5)

    def _draw_input_meter(self, screen: pygame.Surface, car, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, HUD_PANEL_COLOR, rect, border_radius=8)
        pygame.draw.rect(screen, HUD_PANEL_BORDER, rect, 1, border_radius=8)
        title = self.small_font.render("Driver Input", True, HUD_COLOR)
        screen.blit(title, (rect.x + 14, rect.y + 10))

        self._draw_horizontal_meter(screen, "Throttle", car.inputs.throttle, rect.x + 14, rect.y + 42, (73, 220, 132))
        self._draw_horizontal_meter(screen, "Brake", car.inputs.brake, rect.x + 14, rect.y + 68, HUD_WARNING_COLOR)
        self._draw_steering_meter(screen, car.inputs.steer, rect.x + 14, rect.y + 94)

    def _draw_horizontal_meter(self, screen: pygame.Surface, label: str, value: float, x: int, y: int, color: tuple[int, int, int]) -> None:
        text = self.small_font.render(label, True, HUD_COLOR)
        screen.blit(text, (x, y - 5))
        bar = pygame.Rect(x + 82, y, 180, 10)
        pygame.draw.rect(screen, (31, 38, 45), bar, border_radius=5)
        fill = bar.copy()
        fill.width = int(bar.width * max(0.0, min(value, 1.0)))
        pygame.draw.rect(screen, color, fill, border_radius=5)

    def _draw_steering_meter(self, screen: pygame.Surface, value: float, x: int, y: int) -> None:
        text = self.small_font.render("Steer", True, HUD_COLOR)
        screen.blit(text, (x, y - 5))
        bar = pygame.Rect(x + 82, y, 180, 10)
        pygame.draw.rect(screen, (31, 38, 45), bar, border_radius=5)
        center_x = bar.x + bar.width // 2
        marker_x = int(center_x + max(-1.0, min(value, 1.0)) * bar.width * 0.48)
        pygame.draw.line(screen, (66, 170, 255), (marker_x, y - 4), (marker_x, y + 14), 4)
        pygame.draw.line(screen, HUD_PANEL_BORDER, (center_x, y - 3), (center_x, y + 13), 1)

    def _draw_ai_meter(self, screen: pygame.Surface, ai_state, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, HUD_PANEL_COLOR, rect, border_radius=8)
        pygame.draw.rect(screen, HUD_PANEL_BORDER, rect, 1, border_radius=8)
        lines = [
            f"AI target: {ai_state.target_speed:5.1f} km/h",
            f"Path error: {ai_state.path_error:5.1f}",
            f"Decision: {ai_state.decision}",
        ]
        for index, line in enumerate(lines):
            surface = self.small_font.render(line, True, HUD_COLOR)
            screen.blit(surface, (rect.x + 14, rect.y + 14 + index * 24))
