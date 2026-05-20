import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    TRACK_CENTER,
    TRACK_INNER_RADIUS,
    TRACK_OUTER_RADIUS,
)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def load_frames(path: Path, step: int) -> list[dict[str, float | str | bool]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    frames = []
    for row in rows[:: max(1, step)]:
        frames.append(
            {
                "time": as_float(row, "time_s"),
                "x": as_float(row, "x") - TRACK_CENTER[0],
                "z": as_float(row, "y") - TRACK_CENTER[1],
                "heading": as_float(row, "heading"),
                "speed": as_float(row, "speed_kmh"),
                "throttle": as_float(row, "throttle"),
                "brake": as_float(row, "brake"),
                "steer": as_float(row, "steer"),
                "grip": as_float(row, "tire_grip_usage"),
                "battery": as_float(row, "battery_charge"),
                "aero": row.get("aero_mode", "corner"),
                "on_track": row.get("on_track", "True") == "True",
            }
        )
    return frames


def html_template(frames: list[dict[str, float | str | bool]], source: Path) -> str:
    payload = {
        "source": str(source),
        "track": {
            "outerRadius": TRACK_OUTER_RADIUS,
            "innerRadius": TRACK_INNER_RADIUS,
        },
        "frames": frames,
    }
    data_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ApexDriveLab 3D Replay</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #081018;
      color: #e8edf2;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
      background: linear-gradient(#132333, #0a1219 42%, #183f26 43%);
    }}
    .hud {{
      position: fixed;
      left: 18px;
      top: 18px;
      display: grid;
      gap: 8px;
      min-width: 280px;
      padding: 14px 16px;
      background: rgba(6, 11, 16, 0.78);
      border: 1px solid rgba(185, 210, 230, 0.22);
      border-radius: 8px;
      backdrop-filter: blur(10px);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
    }}
    .title {{
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #7dd3fc;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      font-size: 13px;
    }}
    .label {{
      color: #9fb0bf;
    }}
    .value {{
      font-variant-numeric: tabular-nums;
      color: #f7fafc;
    }}
    .help {{
      position: fixed;
      right: 18px;
      bottom: 18px;
      padding: 10px 12px;
      color: #b8c6d2;
      background: rgba(6, 11, 16, 0.68);
      border: 1px solid rgba(185, 210, 230, 0.18);
      border-radius: 8px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <canvas id="view"></canvas>
  <div class="hud">
    <div class="title">ApexDriveLab 3D Replay</div>
    <div class="row"><span class="label">Source</span><span class="value" id="source"></span></div>
    <div class="row"><span class="label">Time</span><span class="value" id="time"></span></div>
    <div class="row"><span class="label">Speed</span><span class="value" id="speed"></span></div>
    <div class="row"><span class="label">Aero</span><span class="value" id="aero"></span></div>
    <div class="row"><span class="label">Grip</span><span class="value" id="grip"></span></div>
    <div class="row"><span class="label">Battery</span><span class="value" id="battery"></span></div>
    <div class="row"><span class="label">Input</span><span class="value" id="input"></span></div>
  </div>
  <div class="help">Space: play/pause · R: restart · ←/→: scrub · C: camera</div>
  <script>
const replay = {data_json};
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const ui = {{
  source: document.getElementById("source"),
  time: document.getElementById("time"),
  speed: document.getElementById("speed"),
  aero: document.getElementById("aero"),
  grip: document.getElementById("grip"),
  battery: document.getElementById("battery"),
  input: document.getElementById("input"),
}};

let frameIndex = 0;
let playing = true;
let cameraMode = 0;
const focal = 760;

function resize() {{
  canvas.width = Math.floor(window.innerWidth * devicePixelRatio);
  canvas.height = Math.floor(window.innerHeight * devicePixelRatio);
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}
window.addEventListener("resize", resize);
resize();

document.addEventListener("keydown", (event) => {{
  if (event.code === "Space") playing = !playing;
  if (event.code === "KeyR") frameIndex = 0;
  if (event.code === "KeyC") cameraMode = (cameraMode + 1) % 2;
  if (event.code === "ArrowRight") frameIndex = Math.min(replay.frames.length - 1, frameIndex + 12);
  if (event.code === "ArrowLeft") frameIndex = Math.max(0, frameIndex - 12);
}});

function sub(a, b) {{ return {{x: a.x - b.x, y: a.y - b.y, z: a.z - b.z}}; }}
function add(a, b) {{ return {{x: a.x + b.x, y: a.y + b.y, z: a.z + b.z}}; }}
function mul(a, s) {{ return {{x: a.x * s, y: a.y * s, z: a.z * s}}; }}
function dot(a, b) {{ return a.x * b.x + a.y * b.y + a.z * b.z; }}
function cross(a, b) {{ return {{x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x}}; }}
function norm(a) {{
  const len = Math.hypot(a.x, a.y, a.z) || 1;
  return {{x: a.x / len, y: a.y / len, z: a.z / len}};
}}

function frameCamera(frame) {{
  const forward = {{x: Math.cos(frame.heading), y: 0, z: Math.sin(frame.heading)}};
  const car = {{x: frame.x, y: 0, z: frame.z}};
  if (cameraMode === 1) {{
    const pos = {{x: 0, y: 820, z: 760}};
    const target = {{x: frame.x * 0.25, y: 0, z: frame.z * 0.25}};
    return makeCamera(pos, target);
  }}
  const pos = add(add(car, mul(forward, -220)), {{x: 0, y: 112, z: 0}});
  const target = add(add(car, mul(forward, 135)), {{x: 0, y: 12, z: 0}});
  return makeCamera(pos, target);
}}

function makeCamera(pos, target) {{
  const forward = norm(sub(target, pos));
  const worldUp = {{x: 0, y: 1, z: 0}};
  const right = norm(cross(forward, worldUp));
  const up = norm(cross(right, forward));
  return {{pos, forward, right, up}};
}}

function project(point, camera) {{
  const rel = sub(point, camera.pos);
  const x = dot(rel, camera.right);
  const y = dot(rel, camera.up);
  const z = dot(rel, camera.forward);
  if (z < 8) return null;
  const width = window.innerWidth;
  const height = window.innerHeight;
  return {{
    x: width / 2 + x / z * focal,
    y: height / 2 - y / z * focal,
    z,
  }};
}}

function drawPoly(points, camera, fill, stroke = null, width = 1) {{
  const projected = points.map((point) => project(point, camera));
  if (projected.some((point) => point === null)) return null;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i += 1) ctx.lineTo(projected[i].x, projected[i].y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {{
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }}
  return projected.reduce((sum, point) => sum + point.z, 0) / projected.length;
}}

function ellipsePoint(angle, radius) {{
  return {{x: Math.cos(angle) * radius[0], y: 0, z: Math.sin(angle) * radius[1]}};
}}

function drawTrack(camera) {{
  const segments = 96;
  const outer = replay.track.outerRadius;
  const inner = replay.track.innerRadius;
  const strips = [];
  for (let i = 0; i < segments; i += 1) {{
    const a0 = Math.PI * 2 * i / segments;
    const a1 = Math.PI * 2 * (i + 1) / segments;
    const groove = i % 2 === 0 ? "#34383c" : "#303438";
    strips.push({{
      points: [ellipsePoint(a0, inner), ellipsePoint(a1, inner), ellipsePoint(a1, outer), ellipsePoint(a0, outer)],
      color: groove,
    }});
  }}
  strips.sort((a, b) => {{
    const az = a.points.reduce((sum, point) => sum + project(point, camera)?.z || 0, 0);
    const bz = b.points.reduce((sum, point) => sum + project(point, camera)?.z || 0, 0);
    return bz - az;
  }});
  for (const strip of strips) drawPoly(strip.points, camera, strip.color, "#59626a", 0.7);

  for (let i = 0; i < 64; i += 1) {{
    if (i % 2 !== 0) continue;
    const a0 = Math.PI * 2 * i / 64;
    const a1 = Math.PI * 2 * (i + 0.55) / 64;
    const color = i % 4 === 0 ? "#e43f4b" : "#f7f2e8";
    const outerKerb = [ellipsePoint(a0, outer), ellipsePoint(a1, outer), ellipsePoint(a1, [outer[0] + 16, outer[1] + 10]), ellipsePoint(a0, [outer[0] + 16, outer[1] + 10])];
    drawPoly(outerKerb, camera, color);
  }}

  for (let i = 0; i < 8; i += 1) {{
    const a = Math.PI * 2 * i / 8;
    const p0 = ellipsePoint(a, inner);
    const p1 = ellipsePoint(a, outer);
    const p0h = {{x: p0.x, y: 2, z: p0.z}};
    const p1h = {{x: p1.x, y: 2, z: p1.z}};
    drawLine3D(p0h, p1h, camera, i === 0 ? "#ffffff" : "#42aaff", i === 0 ? 4 : 2);
  }}
}}

function drawLine3D(a, b, camera, color, width) {{
  const pa = project(a, camera);
  const pb = project(b, camera);
  if (!pa || !pb) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
}}

function carCorners(frame) {{
  const c = Math.cos(frame.heading);
  const s = Math.sin(frame.heading);
  const length = 44;
  const width = 18;
  const h = 11;
  const local = [
    {{x: length / 2, z: 0}},
    {{x: length * 0.22, z: width / 2}},
    {{x: -length / 2, z: width / 2}},
    {{x: -length / 2, z: -width / 2}},
    {{x: length * 0.22, z: -width / 2}},
  ];
  return local.map((p) => ({{
    x: frame.x + p.x * c - p.z * s,
    y: h,
    z: frame.z + p.x * s + p.z * c,
  }}));
}}

function drawCar(frame, camera) {{
  const shadow = carCorners(frame).map((p) => {{ return {{x: p.x, y: 1, z: p.z}}; }});
  drawPoly(shadow, camera, "rgba(0, 0, 0, 0.35)");
  const body = carCorners(frame);
  const color = frame.on_track ? "#ef3346" : "#ffb84a";
  drawPoly(body, camera, color, "#ffef8a", 1.5);
  const nose = body[0];
  const tail = {{x: (body[2].x + body[3].x) / 2, y: 13, z: (body[2].z + body[3].z) / 2}};
  drawLine3D(tail, nose, camera, "#ffe16a", 3);
}}

function drawTrails(frame, camera) {{
  const start = Math.max(0, frameIndex - 90);
  for (let i = start; i < frameIndex; i += 2) {{
    const a = replay.frames[i];
    const b = replay.frames[i + 1] || a;
    const alpha = (i - start) / Math.max(1, frameIndex - start);
    drawLine3D({{x: a.x, y: 3, z: a.z}}, {{x: b.x, y: 3, z: b.z}}, camera, `rgba(125, 211, 252, ${{alpha * 0.6}})`, 2);
  }}
}}

function draw(frame) {{
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  const gradient = ctx.createLinearGradient(0, 0, 0, window.innerHeight);
  gradient.addColorStop(0, "#142437");
  gradient.addColorStop(0.42, "#0a1118");
  gradient.addColorStop(0.43, "#183f26");
  gradient.addColorStop(1, "#0d281a");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  const camera = frameCamera(frame);
  drawTrack(camera);
  drawTrails(frame, camera);
  drawCar(frame, camera);
  updateHud(frame);
}}

function updateHud(frame) {{
  ui.source.textContent = replay.source.split(/[\\\\/]/).pop();
  ui.time.textContent = `${{frame.time.toFixed(2)}} s`;
  ui.speed.textContent = `${{frame.speed.toFixed(1)}} km/h`;
  ui.aero.textContent = frame.aero;
  ui.grip.textContent = `${{(frame.grip * 100).toFixed(0)}}%`;
  ui.battery.textContent = `${{(frame.battery * 100).toFixed(0)}}%`;
  ui.input.textContent = `T ${{frame.throttle.toFixed(2)}} · B ${{frame.brake.toFixed(2)}} · S ${{frame.steer.toFixed(2)}}`;
}}

function tick() {{
  if (playing) frameIndex = (frameIndex + 1) % replay.frames.length;
  draw(replay.frames[frameIndex]);
  requestAnimationFrame(tick);
}}

tick();
  </script>
</body>
</html>
"""


def export_html(telemetry_path: Path, output_path: Path, step: int) -> Path:
    frames = load_frames(telemetry_path, step)
    if not frames:
        raise ValueError(f"No telemetry frames found in {telemetry_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_template(frames, telemetry_path), encoding="utf-8")
    return output_path


def latest_telemetry(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No telemetry CSV files found in {runs_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a self-contained browser 3D replay from telemetry CSV.")
    parser.add_argument("telemetry", nargs="?", type=Path, help="Telemetry CSV. Defaults to latest runs/*.csv.")
    parser.add_argument("--output", type=Path, default=Path("experiments") / "results" / "apex_3d_replay.html")
    parser.add_argument("--step", type=int, default=2, help="Use every Nth telemetry frame.")
    args = parser.parse_args()

    telemetry_path = args.telemetry or latest_telemetry(Path("runs"))
    output = export_html(telemetry_path, args.output, args.step)
    print(output)


if __name__ == "__main__":
    main()
