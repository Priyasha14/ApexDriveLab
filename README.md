# ApexDriveLab

ApexDriveLab is a lightweight 2D top-down racing simulator focused on clean vehicle dynamics, track logic, lap timing, and telemetry-friendly structure.

## Run

```powershell
cd E:\ApexDriveLab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main.py
```

## Test

```powershell
cd E:\ApexDriveLab
.\.venv\Scripts\python.exe .\tests\smoke_tests.py
```

## Headless Experiments

Run one AI lap:

```powershell
.\.venv\Scripts\python.exe .\experiments\run_lap.py --laps 1
```

Compare setups with random-search tuning:

```powershell
.\.venv\Scripts\python.exe .\experiments\compare_setups.py
```

Compare two saved telemetry runs:

```powershell
.\.venv\Scripts\python.exe .\experiments\compare_runs.py .\runs\manual.csv .\runs\ai_lap.csv
```

Generate the Months 3-5 validation report:

```powershell
.\.venv\Scripts\python.exe .\experiments\validate_months_3_5.py
```

Plot telemetry after installing Matplotlib:

```powershell
.\.venv\Scripts\python.exe .\telemetry\plots.py .\runs\ai_lap.csv
```

Controls:

- `W` / `Up`: throttle
- `S` / `Down`: brake, then reverse at low speed
- `A` / `Left`: steer left
- `D` / `Right`: steer right
- `1`: balanced setup
- `2`: stable setup
- `3`: rotation setup
- `4`: high-downforce aero setup
- `5`: low-drag aero setup
- `6`: front-biased aero setup
- `7`: rear-biased aero setup
- `Space`: deploy hybrid energy
- `Z`: automatic active aero
- `X`: force corner aero mode
- `C`: force straight aero mode
- `T`: save telemetry
- `F1`: toggle debug vectors
- `R`: reset car and lap state
- `Esc`: quit

## Architecture

- `main.py`: Pygame setup, event loop, update order, drawing order.
- `config.py`: constants that are safe to tune first.
- `physics/car.py`: car state, inputs, acceleration, braking, drag, steering.
- `physics/aero.py`: drag, downforce, aero balance, and active aero modes.
- `physics/hybrid.py`: battery energy, deployment, and recovery.
- `physics/setup.py`: setup presets for handling comparison.
- `physics/tires.py`: tire slip, force, load, and grip usage state.
- `physics/vector_utils.py`: small math helpers around NumPy vectors.
- `track/track.py`: oval track, boundaries, off-track detection, checkpoint sectors.
- `track/checkpoints.py`: checkpoint order, lap count, lap timing, best lap.
- `ai/rule_driver.py`: pure-pursuit steering, speed planning, braking, aero, and hybrid decisions.
- `ai/optimizer.py`: random-search tuning for rule-driver parameters.
- `ai/gym_env.py`: optional Gymnasium wrapper for later reinforcement learning.
- `experiments/run_lap.py`: headless AI lap runner.
- `experiments/compare_setups.py`: setup comparison experiment.
- `experiments/compare_runs.py`: telemetry comparison helper for manual vs AI or default vs optimized.
- `experiments/validate_months_3_5.py`: repeatable validation report generator.
- `telemetry/plots.py`: Matplotlib telemetry plotting helper.
- `ui/hud.py`: speed, lap time, checkpoint, and off-track display.
- `telemetry/logger.py`: lightweight telemetry logger.

## Physics

The car uses position, velocity, heading, and acceleration.

The vehicle model is a simplified bicycle model. The front and rear axles are each
represented by one tire pair. Steering changes the front wheel angle, slip angle
creates lateral tire force, and the difference between front and rear lateral force
rotates the car.

Longitudinal force comes from throttle and braking. Lateral and longitudinal tire
forces share a friction circle, so heavy braking or acceleration reduces the grip
left for cornering. Basic weight transfer shifts load forward under braking, rearward
under acceleration, and laterally while cornering.

Aerodynamic drag and downforce scale with speed squared. Corner aero mode produces
more downforce and drag; straight aero mode reduces both. Downforce increases tire
normal load, so high speed can create more grip while drag limits top speed.

The hybrid system stores energy, deploys extra power when requested, and recovers
energy during braking or lift-off. This makes energy management part of the driving
problem instead of a free speed boost.

Good first tuning constants:

- `MAX_ENGINE_ACCEL`: how quickly the car gains speed.
- `MAX_BRAKE_ACCEL`: braking strength.
- `LINEAR_DRAG`: high-speed speed loss.
- `MAX_STEER_ANGLE`: steering limit.
- `FRONT_CORNERING_STIFFNESS` / `REAR_CORNERING_STIFFNESS`: handling balance.
- `TIRE_GRIP_ACCEL`: available tire grip.
- `DRAG_COEFFICIENT` / `DOWNFORCE_COEFFICIENT`: aero tradeoff.
- `AERO_BALANCE_FRONT`: front/rear downforce split.
- `TRACK_OUTER_RADIUS` / `TRACK_INNER_RADIUS`: track width and shape.

## Debugging Behavior

If the car pushes wide, reduce `FRONT_CORNERING_STIFFNESS` less or increase it
relative to the rear. If the rear rotates too quickly, reduce `REAR_CORNERING_STIFFNESS`
or increase `YAW_DAMPING`. If braking makes the car unstable, lower `MAX_BRAKE_ACCEL`
or move `BRAKE_BIAS_FRONT` slightly forward.

## Telemetry

Telemetry is saved as CSV files in `runs/`. Press `T` to save during a session.
Closing the simulator also saves the current session automatically. The CSV includes
speed, steering, throttle, brake, longitudinal and lateral acceleration, slip angles,
load transfer, tire grip usage, handling balance, aero mode, downforce, drag, battery
energy, deployment power, and recovery power.

Analyze the latest saved run:

```powershell
.\.venv\Scripts\python.exe .\telemetry\analysis.py
```

Analyze a specific run:

```powershell
.\.venv\Scripts\python.exe .\telemetry\analysis.py .\runs\telemetry_YYYYMMDD_HHMMSS.csv
```
