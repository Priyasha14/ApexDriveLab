# ApexDriveLab

ApexDriveLab is a lightweight 2D top-down racing simulator focused on clean vehicle dynamics, track logic, lap timing, and telemetry-friendly structure.

## Run

```powershell
cd E:\ApexDriveLab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main.py
```

Controls:

- `W` / `Up`: throttle
- `S` / `Down`: brake, then reverse at low speed
- `A` / `Left`: steer left
- `D` / `Right`: steer right
- `R`: reset car and lap state
- `Esc`: quit

## Architecture

- `main.py`: Pygame setup, event loop, update order, drawing order.
- `config.py`: constants that are safe to tune first.
- `physics/car.py`: car state, inputs, acceleration, braking, drag, steering.
- `physics/tires.py`: tire state structure for later dynamics work.
- `physics/vector_utils.py`: small math helpers around NumPy vectors.
- `track/track.py`: oval track, boundaries, off-track detection, checkpoint sectors.
- `track/checkpoints.py`: checkpoint order, lap count, lap timing, best lap.
- `ui/hud.py`: speed, lap time, checkpoint, and off-track display.
- `telemetry/logger.py`: lightweight telemetry logger.

## Physics

The car uses position, velocity, heading, and acceleration.

Acceleration is applied in the car's forward direction. Drag acts against velocity.
Braking removes forward speed. Steering rotates the heading more strongly as speed
increases. A small lateral damping term keeps the basic model from sliding forever.

Good first tuning constants:

- `MAX_ENGINE_ACCEL`: how quickly the car gains speed.
- `MAX_BRAKE_ACCEL`: braking strength.
- `LINEAR_DRAG`: high-speed speed loss.
- `MAX_STEER_RATE`: how quickly the car rotates.
- `TRACK_OUTER_RADIUS` / `TRACK_INNER_RADIUS`: track width and shape.

## Debugging Behavior

If the car is hard to keep on track, reduce engine acceleration or increase steering.
If it feels like it never slows down, increase drag or rolling resistance. If it feels
too floaty, increase lateral damping in `physics/car.py`.
