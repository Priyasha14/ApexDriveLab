import math
from pathlib import Path

import numpy as np
import pygame
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    KeyboardButton,
    Filename,
    LPoint3,
    LVector3,
    TextNode,
)

from ai.racing_line import RacingLine
from ai.rule_driver import RuleBasedDriver
from config import HEIGHT, OFF_TRACK_GRIP_SCALE, WIDTH
from physics.car import Car, CarInputs
from physics.setup import SETUPS
from track.checkpoints import CheckpointManager
from track.track import Track, monaco_buildings


WORLD_SCALE = 32.0
ROAD_Z = 0.04
CAR_TEXTURE_PATH = Path(__file__).with_name(".cache") / "f1_car_top.png"


def to_world(point) -> LPoint3:
    return LPoint3((float(point[0]) - WIDTH / 2) / WORLD_SCALE, (HEIGHT / 2 - float(point[1])) / WORLD_SCALE, 0.0)


def car_forward(car: Car) -> LVector3:
    return LVector3(math.cos(car.heading), -math.sin(car.heading), 0.0)


def normalized(vector: np.ndarray) -> np.ndarray:
    length = max(float(np.linalg.norm(vector)), 1e-6)
    return vector / length


def segment_normal(start, end) -> np.ndarray:
    delta = np.array(end, dtype=float) - np.array(start, dtype=float)
    direction = normalized(delta)
    return np.array([-direction[1], direction[0]], dtype=float)


def vertex_normals(points: list[tuple[float, float]]) -> list[np.ndarray]:
    normals = []
    for index, point in enumerate(points):
        previous_point = np.array(points[index - 1], dtype=float)
        current_point = np.array(point, dtype=float)
        next_point = np.array(points[(index + 1) % len(points)], dtype=float)
        incoming = normalized(current_point - previous_point)
        outgoing = normalized(next_point - current_point)
        normal = np.array([-(incoming[1] + outgoing[1]), incoming[0] + outgoing[0]], dtype=float)
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([-outgoing[1], outgoing[0]], dtype=float)
        normals.append(normalized(normal))
    return normals


def make_mesh(name: str, vertices: list[LPoint3], triangles: list[tuple[int, int, int]], color: tuple[float, float, float, float]) -> GeomNode:
    data = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vertex_writer = GeomVertexWriter(data, "vertex")
    color_writer = GeomVertexWriter(data, "color")
    for vertex in vertices:
        vertex_writer.addData3(vertex)
        color_writer.addData4(*color)

    primitive = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        primitive.addVertices(*triangle)
    primitive.closePrimitive()

    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_strip(name: str, points: list[tuple[float, float]], width: float, z: float, color: tuple[float, float, float, float], offset: float = 0.0) -> GeomNode:
    vertices = []
    triangles = []
    normals = vertex_normals(points)
    for point, normal in zip(points, normals):
        center = np.array(point, dtype=float) + normal * offset
        left = to_world(center + normal * width * 0.5)
        right = to_world(center - normal * width * 0.5)
        vertices.append(LPoint3(left.x, left.y, z))
        vertices.append(LPoint3(right.x, right.y, z))

    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        left = index * 2
        right = left + 1
        next_left = next_index * 2
        next_right = next_left + 1
        triangles.append((left, next_left, right))
        triangles.append((right, next_left, next_right))
    return make_mesh(name, vertices, triangles, color)


def make_tunnel_shell(name: str, centerline: list[tuple[float, float]], width: float, height: float, color: tuple[float, float, float, float]) -> GeomNode:
    vertices = []
    triangles = []
    arch_steps = 8
    for point_index, point in enumerate(centerline):
        if point_index == 0:
            tangent = np.array(centerline[1], dtype=float) - np.array(point, dtype=float)
        elif point_index == len(centerline) - 1:
            tangent = np.array(point, dtype=float) - np.array(centerline[point_index - 1], dtype=float)
        else:
            tangent = np.array(centerline[point_index + 1], dtype=float) - np.array(centerline[point_index - 1], dtype=float)
        normal = np.array([-normalized(tangent)[1], normalized(tangent)[0]], dtype=float)
        center = np.array(point, dtype=float)
        for step in range(arch_steps + 1):
            angle = math.pi * step / arch_steps
            lateral = math.cos(angle) * width * 0.5
            z = 0.10 + math.sin(angle) * height
            world = to_world(center + normal * lateral)
            vertices.append(LPoint3(world.x, world.y, z))

    row = arch_steps + 1
    for point_index in range(len(centerline) - 1):
        for step in range(arch_steps):
            a = point_index * row + step
            b = a + 1
            c = (point_index + 1) * row + step
            d = c + 1
            triangles.append((a, c, b))
            triangles.append((b, c, d))
    return make_mesh(name, vertices, triangles, color)


def create_f1_car_texture(path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    surface = pygame.Surface((320, 640), pygame.SRCALPHA)
    red = (230, 16, 36, 255)
    dark_red = (150, 8, 22, 255)
    yellow = (255, 215, 20, 255)
    black = (8, 10, 13, 255)
    charcoal = (28, 32, 36, 255)
    tire = (4, 5, 6, 255)
    rim = (210, 210, 190, 255)

    def poly(points, fill):
        pygame.draw.polygon(surface, fill, points)

    # Image coordinates: top is the front of the car.
    poly([(160, 30), (262, 70), (258, 96), (62, 96), (58, 70)], black)
    poly([(88, 104), (232, 104), (214, 124), (106, 124)], charcoal)
    poly([(160, 76), (183, 250), (177, 410), (160, 470), (143, 410), (137, 250)], yellow)
    poly([(160, 210), (205, 284), (214, 430), (184, 540), (160, 585), (136, 540), (106, 430), (115, 284)], red)
    poly([(112, 320), (60, 350), (70, 470), (122, 444)], dark_red)
    poly([(208, 320), (260, 350), (250, 470), (198, 444)], dark_red)
    poly([(160, 360), (190, 410), (182, 470), (160, 492), (138, 470), (130, 410)], black)
    poly([(82, 528), (238, 528), (256, 568), (64, 568)], black)
    poly([(112, 492), (208, 492), (196, 530), (124, 530)], charcoal)
    poly([(150, 286), (170, 286), (176, 338), (160, 368), (144, 338)], charcoal)
    pygame.draw.line(surface, black, (160, 250), (160, 112), 8)
    pygame.draw.line(surface, black, (136, 300), (112, 372), 7)
    pygame.draw.line(surface, black, (184, 300), (208, 372), 7)

    for x in (52, 228):
        for y in (176, 440):
            pygame.draw.rect(surface, tire, pygame.Rect(x, y, 42, 92), border_radius=10)
            pygame.draw.rect(surface, rim, pygame.Rect(x + 12, y + 26, 18, 40), border_radius=5)

    pygame.image.save(surface, path)


class ApexDrive3D(ShowBase):
    def __init__(self) -> None:
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.55, 0.72, 0.86, 1.0)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.track = Track()
        self.racing_line = RacingLine.from_track(self.track)
        self.ai_driver = RuleBasedDriver(self.racing_line)
        self.car = Car()
        self.checkpoints = CheckpointManager()
        self.checkpoints.reset(self.track.checkpoint_index_for_position(self.car.position))
        self.ai_enabled = False
        self.cockpit = False
        self.aero_override = None
        self.manual_steer = 0.0
        self.car.apply_setup(SETUPS["stable"])
        self.setup_controls()
        self.build_scene()
        self.build_car()
        self.build_hud()
        self.taskMgr.add(self.update, "update-simulator")

    def setup_controls(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("v", self.toggle_camera)
        self.accept("p", self.toggle_ai)
        self.accept("r", self.reset_car)
        self.accept("z", self.set_aero, [None])
        self.accept("x", self.set_aero, ["corner"])
        self.accept("c", self.set_aero, ["straight"])
        names = ["balanced", "stable", "rotation", "high_downforce", "low_drag", "front_aero", "rear_aero"]
        for index, name in enumerate(names, start=1):
            self.accept(str(index), self.apply_setup, [name])

    def build_scene(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.56, 0.58, 0.60, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))
        sun = DirectionalLight("sun")
        sun.setColor((0.95, 0.90, 0.80, 1.0))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(-40, -55, 0)
        self.render.setLight(sun_node)

        self.add_ground()
        self.render.attachNewNode(make_strip("road-shadow", self.track.centerline, self.track.road_width + 36, 0.018, (0.04, 0.045, 0.05, 1)))
        self.render.attachNewNode(make_strip("monaco-road", self.track.centerline, self.track.road_width, ROAD_Z, (0.17, 0.18, 0.19, 1)))
        self.render.attachNewNode(make_strip("racing-groove", self.track.centerline, 24.0, 0.055, (0.11, 0.12, 0.13, 1)))
        self.add_buildings()
        self.add_monaco_landmarks()

    def add_ground(self) -> None:
        card = CardMaker("ground")
        size = 48
        card.setFrame(-size, size, -size, size)
        ground = self.render.attachNewNode(card.generate())
        ground.setP(-90)
        ground.setZ(-0.02)
        ground.setColor(0.12, 0.36, 0.20, 1)
        ground.setTextureOff(1)

        water_card = CardMaker("harbor")
        water_card.setFrame(5, 24, -5, 8)
        water = self.render.attachNewNode(water_card.generate())
        water.setP(-90)
        water.setZ(-0.01)
        water.setColor(0.04, 0.30, 0.48, 1)
        water.setTextureOff(1)

    def add_box(self, name: str, position: LPoint3, scale: tuple[float, float, float], color: tuple[float, float, float, float]):
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setName(name)
        box.setPos(position)
        box.setScale(*scale)
        box.setColor(*color)
        box.setColorScale(*color)
        box.setTextureOff(1)
        box.setMaterialOff(1)
        return box

    def add_buildings(self) -> None:
        for rect, height, building_color in monaco_buildings():
            center = to_world(rect.center)
            rgb = tuple(channel / 255 for channel in building_color)
            side = 1.0 if center.x >= 0 else -1.0
            self.add_box(
                "building",
                LPoint3(center.x + side * 10.0, center.y, height / WORLD_SCALE * 0.10),
                (rect.width / WORLD_SCALE * 0.24, rect.height / WORLD_SCALE * 0.24, height / WORLD_SCALE * 0.20),
                (*rgb, 1.0),
            )

    def add_sign(self, label: str, point: tuple[float, float], offset: tuple[float, float] = (0.0, 0.0), color=(0.95, 0.95, 0.88, 1.0)) -> None:
        world = to_world((point[0] + offset[0], point[1] + offset[1]))
        post = self.add_box(f"{label}-post", LPoint3(world.x, world.y, 0.35), (0.035, 0.035, 0.70), (0.08, 0.08, 0.08, 1.0))
        board = self.add_box(f"{label}-board", LPoint3(world.x, world.y, 0.84), (0.92, 0.045, 0.22), (0.04, 0.05, 0.06, 1.0))
        board.setBillboardPointEye()
        text = TextNode(f"{label}-text")
        text.setText(label)
        text.setAlign(TextNode.ACenter)
        text.setTextColor(*color)
        text.setCardColor(0, 0, 0, 0)
        node = self.render.attachNewNode(text)
        node.setScale(0.18)
        node.setPos(world.x, world.y - 0.04, 0.77)
        node.setBillboardPointEye()

    def add_start_finish(self) -> None:
        start = self.track.centerline[0]
        next_point = self.track.centerline[1]
        world = to_world(start)
        next_world = to_world(next_point)
        yaw = math.degrees(math.atan2((next_world - world).x, (next_world - world).y)) + 90
        for index in range(8):
            for side in (-1, 1):
                stripe = self.add_box(
                    "checkered-start",
                    LPoint3(world.x + side * (index - 3.5) * 0.22, world.y, 0.09),
                    (0.10, 0.20, 0.025),
                    (0.96, 0.96, 0.90, 1.0) if index % 2 == 0 else (0.02, 0.02, 0.025, 1.0),
                )
                stripe.setH(yaw)
        self.add_sign("START / FINISH", start, (0, -92))

    def add_tunnel(self) -> None:
        tunnel_points = [
            (620.0, 382.0),
            (680.0, 374.0),
            (748.0, 368.0),
            (820.0, 374.0),
            (890.0, 386.0),
        ]
        shell = self.render.attachNewNode(make_tunnel_shell("tunnel-shell", tunnel_points, 126.0, 1.50, (0.10, 0.105, 0.11, 1.0)))
        shell.setTwoSided(True)
        self.render.attachNewNode(make_tunnel_shell("tunnel-inner", tunnel_points, 104.0, 1.30, (0.035, 0.038, 0.042, 1.0))).setTwoSided(True)
        for point in tunnel_points:
            world = to_world(point)
            lamp = self.add_box("tunnel-lamp", LPoint3(world.x, world.y, 1.23), (0.18, 0.05, 0.035), (1.0, 0.86, 0.38, 1.0))
            lamp.setBillboardPointEye()
        self.add_sign("TUNNEL", (788.0, 364.0), (0, -78), (1.0, 0.90, 0.36, 1.0))

    def add_paddock(self) -> None:
        for index in range(7):
            world = to_world((1040.0 - index * 42.0, 652.0))
            garage = self.add_box("paddock-garage", LPoint3(world.x, world.y, 0.32), (0.52, 0.30, 0.50), (0.78, 0.78, 0.72, 1.0))
            garage.setH(8)
            door = self.add_box("paddock-door", LPoint3(world.x, world.y - 0.17, 0.18), (0.38, 0.025, 0.28), (0.18, 0.20, 0.22, 1.0))
            door.setH(8)
        self.add_sign("PADDOCK / PITS", (900.0, 652.0), (0, 46), (0.78, 0.95, 1.0, 1.0))

    def add_monaco_landmarks(self) -> None:
        self.add_start_finish()
        self.add_tunnel()
        self.add_paddock()
        self.add_sign("SAINTE DEVOTE", (760.0, 536.0), (78, 22))
        self.add_sign("CASINO", (524.0, 282.0), (-30, -70))
        self.add_sign("MIRABEAU", (382.0, 350.0), (-76, -20))
        self.add_sign("FAIRMONT HAIRPIN", (308.0, 420.0), (-100, 48), (1.0, 0.85, 0.38, 1.0))
        self.add_sign("PORTIER", (540.0, 410.0), (36, 54))
        self.add_sign("NOUVELLE CHICANE", (982.0, 426.0), (70, -42), (1.0, 0.85, 0.38, 1.0))
        self.add_sign("TABAC", (1098.0, 520.0), (54, -50))
        self.add_sign("SWIMMING POOL", (1082.0, 600.0), (58, 42), (0.78, 0.95, 1.0, 1.0))
        self.add_sign("RASCASSE", (844.0, 642.0), (-24, 54), (1.0, 0.85, 0.38, 1.0))
        self.add_sign("ANTHONY NOGHES", (772.0, 626.0), (-92, 18))

    def build_car(self) -> None:
        self.car_node = self.render.attachNewNode("car")
        create_f1_car_texture(CAR_TEXTURE_PATH)
        card = CardMaker("f1-car-card")
        card.setFrame(-0.64, 0.64, -1.70, 1.70)
        self.car_sprite = self.car_node.attachNewNode(card.generate())
        self.car_sprite.setP(-90)
        self.car_sprite.setZ(0.18)
        self.car_sprite.setTransparency(True)
        texture_path = Filename.fromOsSpecific(str(CAR_TEXTURE_PATH)).getFullpath()
        self.car_sprite.setTexture(self.loader.loadTexture(texture_path), 1)
        self.car_sprite.setColor(1, 1, 1, 1)
        self.car_sprite.setTwoSided(True)

    def build_hud(self) -> None:
        self.hud = OnscreenText(
            text="",
            pos=(-1.28, 0.90),
            align=TextNode.ALeft,
            scale=0.045,
            fg=(0.95, 0.97, 1.0, 1.0),
            bg=(0.02, 0.025, 0.03, 0.72),
            mayChange=True,
        )

    def toggle_camera(self) -> None:
        self.cockpit = not self.cockpit

    def toggle_ai(self) -> None:
        self.ai_enabled = not self.ai_enabled

    def set_aero(self, mode: str | None) -> None:
        self.aero_override = mode

    def apply_setup(self, name: str) -> None:
        self.car.apply_setup(SETUPS[name])

    def reset_car(self) -> None:
        self.car.reset()
        self.car.apply_setup(SETUPS["stable"])
        self.manual_steer = 0.0
        self.checkpoints.reset(self.track.checkpoint_index_for_position(self.car.position))

    def key_down(self, key: str) -> bool:
        return self.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key(key))

    def special_key_down(self, button: KeyboardButton) -> bool:
        return self.mouseWatcherNode.is_button_down(button)

    def controls(self) -> CarInputs:
        throttle = 1.0 if self.key_down("w") or self.mouseWatcherNode.is_button_down(KeyboardButton.up()) else 0.0
        brake = 1.0 if self.key_down("s") or self.mouseWatcherNode.is_button_down(KeyboardButton.down()) else 0.0
        target_steer = float(self.key_down("d") or self.mouseWatcherNode.is_button_down(KeyboardButton.right())) - float(
            self.key_down("a") or self.mouseWatcherNode.is_button_down(KeyboardButton.left())
        )
        dt = min(globalClock.getDt(), 1 / 30)
        response = min(1.0, 4.5 * dt)
        self.manual_steer += (target_steer - self.manual_steer) * response
        if target_steer == 0.0:
            self.manual_steer *= max(0.0, 1.0 - 8.0 * dt)
        speed_factor = max(0.20, 1.0 - self.car.speed_kmh / 180.0)
        steer = self.manual_steer * 0.24 * speed_factor
        throttle *= max(0.38, 1.0 - abs(self.manual_steer) * 0.55)
        return CarInputs(throttle=throttle, brake=brake, steer=steer, deploy_hybrid=self.special_key_down(KeyboardButton.space()), aero_mode=self.aero_override)

    def update_car_visual(self) -> None:
        world = to_world(self.car.position)
        car_pos = LPoint3(world.x, world.y, 0.12)
        self.car_node.setPos(car_pos)
        self.car_node.lookAt(car_pos + car_forward(self.car))

    def update_camera(self) -> None:
        world = to_world(self.car.position)
        forward = car_forward(self.car)
        if self.cockpit:
            camera_pos = LPoint3(world.x, world.y, 0.68) + forward * 0.50
            target = LPoint3(world.x, world.y, 0.36) + forward * 16.0
            self.car_node.hide()
        else:
            camera_pos = LPoint3(world.x, world.y, 2.85) - forward * 5.7
            target = LPoint3(world.x, world.y, 0.14) + forward * 8.0
            self.car_node.show()
        self.camera.setPos(camera_pos)
        self.camera.lookAt(target)
        self.camLens.setFov(62)
        self.camLens.setNearFar(0.05, 500)

    def update_hud(self, on_track: bool) -> None:
        self.hud.setText(
            f"ApexDriveLab 3D\n"
            f"Speed: {self.car.speed_kmh:5.1f} km/h\n"
            f"Lap: {self.checkpoints.state.lap_count}   CP: {self.checkpoints.state.current_checkpoint + 1}/8\n"
            f"Mode: {'AI' if self.ai_enabled else 'Manual'}   Camera: {'Cockpit' if self.cockpit else 'Chase'}\n"
            f"Track: {'ON' if on_track else 'OFF'}\n"
            "WASD drive | V camera | P AI | R reset | Esc quit"
        )

    def update(self, task):
        dt = min(globalClock.getDt(), 1 / 30)
        controls = self.ai_driver.control(self.car, self.track) if self.ai_enabled else self.controls()
        on_track = self.track.is_on_track(self.car.position)
        self.car.update(dt, controls, 1.0 if on_track else OFF_TRACK_GRIP_SCALE)
        self.checkpoints.update(dt, self.track.checkpoint_index_for_position(self.car.position))
        self.update_car_visual()
        self.update_camera()
        self.update_hud(on_track)
        return task.cont


def main() -> None:
    app = ApexDrive3D()
    app.run()


if __name__ == "__main__":
    main()
