import math

import numpy as np
from ursina import AmbientLight, DirectionalLight, Entity, Text, Ursina, Vec3, camera, color, held_keys, time, window

from ai.racing_line import RacingLine
from ai.rule_driver import RuleBasedDriver
from config import HEIGHT, OFF_TRACK_GRIP_SCALE, WIDTH
from physics.car import Car, CarInputs
from physics.setup import SETUPS
from track.checkpoints import CheckpointManager
from track.track import Track, monaco_buildings


WORLD_SCALE = 31.0


def to_world(point) -> Vec3:
    return Vec3((float(point[0]) - WIDTH / 2) / WORLD_SCALE, 0.0, (float(point[1]) - HEIGHT / 2) / WORLD_SCALE)


def heading_to_yaw(heading: float) -> float:
    return math.degrees(heading) + 90.0


def create_segment(start, end, width: float, height: float, y: float, material_color, name: str) -> Entity:
    start_world = to_world(start)
    end_world = to_world(end)
    delta = end_world - start_world
    length = max(0.1, math.hypot(delta.x, delta.z))
    midpoint = (start_world + end_world) * 0.5
    yaw = math.degrees(math.atan2(delta.x, delta.z))
    return Entity(model="cube", name=name, color=material_color, position=(midpoint.x, y, midpoint.z), scale=(width / WORLD_SCALE, height, length), rotation_y=yaw)


def segment_normal(start, end) -> np.ndarray:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = max(math.hypot(dx, dy), 1e-6)
    return np.array([-dy / length, dx / length], dtype=float)


def create_track_scene(track: Track) -> None:
    Entity(model="plane", color=color.rgb(23, 90, 55), scale=(80, 1, 50), position=(0, -0.08, 0))
    Entity(model="cube", color=color.rgb(24, 88, 125), scale=(18, 0.08, 10), position=(14, -0.04, 9))

    for rect, building_height, building_color in monaco_buildings():
        center = to_world(rect.center)
        Entity(
            model="cube",
            color=color.rgb(*building_color),
            position=(center.x, building_height / WORLD_SCALE * 0.5, center.z),
            scale=(rect.width / WORLD_SCALE, building_height / WORLD_SCALE, rect.height / WORLD_SCALE),
        )

    for index, segment in enumerate(track._segments):
        start = segment["start"]
        end = segment["end"]
        create_segment(start, end, track.road_width, 0.06, 0.0, color.rgb(42, 44, 46) if index % 2 else color.rgb(50, 52, 54), "road")
        normal = segment_normal(start, end)
        for side in (-1.0, 1.0):
            edge_start = np.array(start) + normal * track.road_width * 0.5 * side
            edge_end = np.array(end) + normal * track.road_width * 0.5 * side
            kerb_inner_start = np.array(start) + normal * track.road_width * 0.40 * side
            kerb_inner_end = np.array(end) + normal * track.road_width * 0.40 * side
            barrier_start = np.array(start) + normal * track.road_width * 0.66 * side
            barrier_end = np.array(end) + normal * track.road_width * 0.66 * side
            kerb_color = color.rgb(214, 42, 52) if index % 2 == 0 else color.rgb(248, 248, 242)
            create_segment(kerb_inner_start, edge_start, 8.0, 0.08, 0.05, kerb_color, "kerb")
            create_segment(kerb_inner_end, edge_end, 8.0, 0.08, 0.05, kerb_color, "kerb")
            create_segment(barrier_start, barrier_end, 10.0, 0.75, 0.38, color.rgb(222, 226, 229), "barrier")

    for progress in np.linspace(0, track.total_length, 9)[:-1]:
        point = track.point_at_progress(progress)
        marker = to_world(point)
        Entity(model="cube", color=color.azure, position=(marker.x, 0.08, marker.z), scale=(0.35, 0.04, 2.2), rotation_y=heading_to_yaw(track.angle_for_progress(progress)))


class F1Car3D:
    def __init__(self) -> None:
        self.root = Entity()
        Entity(parent=self.root, model="cube", color=color.rgb(220, 30, 48), scale=(0.72, 0.22, 1.35), position=(0, 0.26, 0))
        Entity(parent=self.root, model="cube", color=color.rgb(255, 225, 80), scale=(0.22, 0.16, 1.15), position=(0, 0.32, 0.63))
        Entity(parent=self.root, model="cube", color=color.rgb(25, 28, 32), scale=(1.15, 0.08, 0.16), position=(0, 0.24, 0.78))
        Entity(parent=self.root, model="cube", color=color.rgb(25, 28, 32), scale=(1.22, 0.10, 0.20), position=(0, 0.36, -0.72))
        Entity(parent=self.root, model="cube", color=color.rgb(15, 18, 22), scale=(0.28, 0.24, 0.28), position=(0, 0.43, -0.08))
        for x in (-0.48, 0.48):
            for z in (-0.47, 0.48):
                Entity(parent=self.root, model="cube", color=color.black, scale=(0.22, 0.30, 0.40), position=(x, 0.17, z))

    def update(self, car: Car) -> None:
        world = to_world(car.position)
        self.root.position = Vec3(world.x, 0.08, world.z)
        self.root.rotation_y = heading_to_yaw(car.heading)


def read_ursina_inputs(aero_override: str | None) -> CarInputs:
    throttle = 1.0 if held_keys["w"] or held_keys["up arrow"] else 0.0
    brake = 1.0 if held_keys["s"] or held_keys["down arrow"] else 0.0
    steer = float(held_keys["d"] or held_keys["right arrow"]) - float(held_keys["a"] or held_keys["left arrow"])
    deploy = bool(held_keys["space"])
    return CarInputs(throttle=throttle, brake=brake, steer=steer, deploy_hybrid=deploy, aero_mode=aero_override)


def main() -> None:
    app = Ursina(borderless=False)
    window.title = "ApexDriveLab 3D"
    window.color = color.rgb(96, 150, 190)
    track = Track()
    racing_line = RacingLine.from_track(track)
    ai_driver = RuleBasedDriver(racing_line)
    car = Car()
    car_visual = F1Car3D()
    checkpoints = CheckpointManager()
    checkpoints.reset(track.checkpoint_index_for_position(car.position))
    create_track_scene(track)
    DirectionalLight(rotation=(45, -35, 35), shadows=True)
    AmbientLight(color=color.rgba(120, 130, 140, 0.45))
    status = Text(text="", origin=(-0.5, 0.5), position=(-0.86, 0.46), scale=0.9, background=True)
    state = {"ai": False, "cockpit": False, "aero": None}

    def input(key):
        if key == "p":
            state["ai"] = not state["ai"]
        elif key == "v":
            state["cockpit"] = not state["cockpit"]
        elif key == "r":
            car.reset()
            checkpoints.reset(track.checkpoint_index_for_position(car.position))
        elif key in {"1", "2", "3", "4", "5", "6", "7"}:
            setup_names = ["balanced", "stable", "rotation", "high_downforce", "low_drag", "front_aero", "rear_aero"]
            car.apply_setup(SETUPS[setup_names[int(key) - 1]])
        elif key == "z":
            state["aero"] = None
        elif key == "x":
            state["aero"] = "corner"
        elif key == "c":
            state["aero"] = "straight"

    def update():
        dt = min(time.dt, 1 / 30)
        controls = ai_driver.control(car, track) if state["ai"] else read_ursina_inputs(state["aero"])
        on_track = track.is_on_track(car.position)
        car.update(dt, controls, 1.0 if on_track else OFF_TRACK_GRIP_SCALE)
        checkpoints.update(dt, track.checkpoint_index_for_position(car.position))
        car_visual.update(car)
        forward = Vec3(math.cos(car.heading), 0, math.sin(car.heading))
        car_world = to_world(car.position)
        if state["cockpit"]:
            camera.position = car_world + Vec3(0, 0.95, 0) + forward * 0.32
            camera.rotation = (6, heading_to_yaw(car.heading), 0)
            car_visual.root.enabled = False
        else:
            camera.position = car_world + Vec3(0, 8.5, 0) - forward * 8.5
            camera.look_at(car_world + Vec3(0, 0.3, 0) + forward * 2.5)
            car_visual.root.enabled = True
        status.text = (
            f"Speed {car.speed_kmh:5.1f} km/h\n"
            f"Lap {checkpoints.state.lap_count}  CP {checkpoints.state.current_checkpoint + 1}/8\n"
            f"{'AI' if state['ai'] else 'Manual'} | {'Cockpit' if state['cockpit'] else 'Chase'} | {'ON' if on_track else 'OFF TRACK'}\n"
            "WASD drive | V cockpit | P AI | R reset"
        )

    app.run()


if __name__ == "__main__":
    main()
