"""ApexDriveLab 3D - game-style Panda3D viewer.

Procedurally built (no asset files): full 3D car with spinning/steering wheels
and an animated DRS flap, sun with real-time shadows, gradient sky dome, kerbs,
barriers, harbor with yachts, tire smoke, exhaust shift flames, brake glow, and
a speed-sensitive chase camera. Drives the v2 physics engine.

Run:  python visual_3d.py
Keys: WASD drive | E DRS | Space ERS | V camera | P AI | 1-7 setup | R reset | Esc quit
"""

import math
import random

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    KeyboardButton,
    LPoint3,
    LVector3,
    PNMImage,
    TextNode,
    Texture,
    TransparencyAttrib,
)

from ai.racing_line import RacingLine
from ai.rule_driver import RuleBasedDriver
from config import HEIGHT, OFF_TRACK_GRIP_SCALE, WIDTH
from physics.car import Car, CarInputs
from physics.setup import SETUPS
from track.checkpoints import CheckpointManager
from track.track import Track, monaco_buildings

WORLD_SCALE = 22.0   # smaller divisor = bigger world relative to the car
CAR_SCALE = 0.78
SCENE_K = 32.0 / WORLD_SCALE  # scales hand-tuned scenery positions from the old world size
ROAD_Z = 0.04


def to_world(point) -> LPoint3:
    return LPoint3((float(point[0]) - WIDTH / 2) / WORLD_SCALE, (HEIGHT / 2 - float(point[1])) / WORLD_SCALE, 0.0)


def car_forward(car: Car) -> LVector3:
    return LVector3(math.cos(car.heading), -math.sin(car.heading), 0.0)


def normalized(vector: np.ndarray) -> np.ndarray:
    length = max(float(np.linalg.norm(vector)), 1e-6)
    return vector / length


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


# ---------------------------------------------------------------------------
# Procedural geometry
# ---------------------------------------------------------------------------

def make_mesh(name, vertices, normals, colors, triangles) -> GeomNode:
    data = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
    vw = GeomVertexWriter(data, "vertex")
    nw = GeomVertexWriter(data, "normal")
    cw = GeomVertexWriter(data, "color")
    for vertex, normal, color in zip(vertices, normals, colors):
        vw.addData3(vertex)
        nw.addData3(normal)
        cw.addData4(*color)
    primitive = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        primitive.addVertices(*triangle)
    primitive.closePrimitive()
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def make_strip(name, points, width, z, color, offset=0.0) -> GeomNode:
    vertices, normals, colors, triangles = [], [], [], []
    up = LVector3(0, 0, 1)
    for point, normal in zip(points, vertex_normals(points)):
        center = np.array(point, dtype=float) + normal * offset
        left = to_world(center + normal * width * 0.5)
        right = to_world(center - normal * width * 0.5)
        vertices += [LPoint3(left.x, left.y, z), LPoint3(right.x, right.y, z)]
        normals_2 = [up, up]
        normals += normals_2
        colors += [color, color]
    count = len(points)
    for index in range(count):
        nxt = (index + 1) % count
        a, b, c, d = index * 2, index * 2 + 1, nxt * 2, nxt * 2 + 1
        triangles += [(a, c, b), (b, c, d)]
    return make_mesh(name, vertices, normals, colors, triangles)


def make_striped_strip(name, points, width, z, color_a, color_b, offset=0.0) -> GeomNode:
    """Kerb-style strip: alternating crisp color blocks (separate quads)."""
    vertices, normals, colors, triangles = [], [], [], []
    up = LVector3(0, 0, 1)
    norms = vertex_normals(points)
    count = len(points)
    for index in range(count):
        nxt = (index + 1) % count
        color = color_a if index % 2 == 0 else color_b
        for p_index, n in ((index, norms[index]), (nxt, norms[nxt])):
            center = np.array(points[p_index], dtype=float) + n * offset
            left = to_world(center + n * width * 0.5)
            right = to_world(center - n * width * 0.5)
            vertices += [LPoint3(left.x, left.y, z), LPoint3(right.x, right.y, z)]
            normals += [up, up]
            colors += [color, color, color, color][:2] if False else [color, color]
        base = index * 4
        triangles += [(base, base + 2, base + 1), (base + 1, base + 2, base + 3)]
    return make_mesh(name, vertices, normals, colors, triangles)


def make_wall(name, points, offset, height, color, base_z=0.0) -> GeomNode:
    """Vertical barrier following the centerline at a lateral offset."""
    vertices, normals, colors, triangles = [], [], [], []
    norms = vertex_normals(points)
    count = len(points)
    for index in range(count):
        nxt = (index + 1) % count
        for p_index in (index, nxt):
            n = norms[p_index]
            center = np.array(points[p_index], dtype=float) + n * offset
            world = to_world(center)
            face = LVector3(n[0], -n[1], 0.0)
            vertices += [LPoint3(world.x, world.y, base_z), LPoint3(world.x, world.y, base_z + height)]
            normals += [face, face]
            colors += [color, color]
        base = index * 4
        triangles += [(base, base + 2, base + 1), (base + 1, base + 2, base + 3),
                      (base, base + 1, base + 2), (base + 1, base + 3, base + 2)]
    return make_mesh(name, vertices, normals, colors, triangles)


def make_cylinder(name, radius, length, color, segments=14, cap_color=None, taper=1.0) -> GeomNode:
    """Cylinder along +Y (wheel axle orientation), centered at origin."""
    vertices, normals, colors, triangles = [], [], [], []
    cap_color = cap_color or color
    half = length / 2.0
    for seg in range(segments + 1):
        angle = 2.0 * math.pi * seg / segments
        nx, nz = math.cos(angle), math.sin(angle)
        vertices += [LPoint3(nx * radius, -half, nz * radius), LPoint3(nx * radius * taper, half, nz * radius * taper)]
        normals += [LVector3(nx, 0, nz), LVector3(nx, 0, nz)]
        colors += [color, color]
    for seg in range(segments):
        a = seg * 2
        triangles += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)]
    # End caps
    for side, y, nrm in ((0, -half, LVector3(0, -1, 0)), (1, half, LVector3(0, 1, 0))):
        center_index = len(vertices)
        vertices.append(LPoint3(0, y, 0))
        normals.append(nrm)
        colors.append(cap_color)
        ring = []
        r = radius * (taper if side else 1.0)
        for seg in range(segments):
            angle = 2.0 * math.pi * seg / segments
            ring.append(len(vertices))
            vertices.append(LPoint3(math.cos(angle) * r, y, math.sin(angle) * r))
            normals.append(nrm)
            colors.append(cap_color)
        for seg in range(segments):
            a, b = ring[seg], ring[(seg + 1) % segments]
            triangles.append((center_index, b, a) if side == 0 else (center_index, a, b))
    return make_mesh(name, vertices, normals, colors, triangles)


def make_sky_dome(name, radius=175.0, segments=24, rings=8) -> GeomNode:
    """Gradient sky: warm horizon fading to deep blue zenith."""
    horizon = (0.78, 0.86, 0.94, 1.0)
    zenith = (0.22, 0.44, 0.78, 1.0)
    vertices, normals, colors, triangles = [], [], [], []
    for ring in range(rings + 1):
        pitch = (math.pi / 2) * ring / rings
        t = math.sin(pitch)
        color = tuple(h + (z - h) * (t ** 1.2) for h, z in zip(horizon, zenith))
        for seg in range(segments + 1):
            yaw = 2 * math.pi * seg / segments
            x = math.cos(yaw) * math.cos(pitch) * radius
            y = math.sin(yaw) * math.cos(pitch) * radius
            z = math.sin(pitch) * radius - 2.0
            vertices.append(LPoint3(x, y, z))
            normals.append(LVector3(0, 0, -1))
            colors.append(color)
    row = segments + 1
    for ring in range(rings):
        for seg in range(segments):
            a = ring * row + seg
            b = a + 1
            c = a + row
            d = c + 1
            triangles += [(a, b, c), (b, d, c)]
    return make_mesh(name, vertices, normals, colors, triangles)


def make_soft_circle_texture(size=64) -> Texture:
    image = PNMImage(size, size, 4)
    image.addAlpha()
    for y in range(size):
        for x in range(size):
            dx = (x - size / 2) / (size / 2)
            dy = (y - size / 2) / (size / 2)
            r = math.hypot(dx, dy)
            alpha = max(0.0, 1.0 - r) ** 2
            image.setXelA(x, y, 0.92, 0.92, 0.94, alpha)
    texture = Texture("soft-circle")
    texture.load(image)
    return texture


def make_window_texture() -> Texture:
    size = 64
    image = PNMImage(size, size, 4)
    for y in range(size):
        for x in range(size):
            wx, wy = x % 16, y % 16
            lit = 3 <= wx <= 11 and 3 <= wy <= 10
            if lit:
                shade = 0.55 + 0.25 * (((x // 16) * 7 + (y // 16) * 13) % 5) / 4.0
                image.setXel(x, y, shade * 0.72, shade * 0.80, shade * 0.92)
            else:
                image.setXel(x, y, 1.0, 1.0, 1.0)
    texture = Texture("windows")
    texture.load(image)
    texture.setWrapU(Texture.WMRepeat)
    texture.setWrapV(Texture.WMRepeat)
    return texture


class ApexDrive3D(ShowBase):
    def __init__(self) -> None:
        super().__init__()
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.track = Track()
        self.racing_line = RacingLine.from_track(self.track)
        self.ai_driver = RuleBasedDriver(self.racing_line)
        self.car = Car()
        self.checkpoints = CheckpointManager()
        self.checkpoints.reset(self.track.checkpoint_index_for_position(self.car.position))
        self.ai_enabled = False
        self.camera_mode = 0  # 0 chase, 1 cockpit, 2 TV
        self.aero_override = None
        self.manual_steer = 0.0
        self.time_elapsed = 0.0
        self.last_gear = 1
        self.shift_flash = 0.0
        self.cam_pos = LPoint3(0, -8, 3)
        self.car.apply_setup(SETUPS["stable"])

        self.smoke_texture = make_soft_circle_texture()
        self.setup_lights()
        self.setup_controls()
        self.build_scene()
        self.build_car()
        self.build_effects()
        self.build_hud()
        self.setup_post_processing()
        self.taskMgr.add(self.update, "update-simulator")

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.42, 0.46, 0.52, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        self.sun = DirectionalLight("sun")
        self.sun.setColor((1.00, 0.94, 0.82, 1.0))
        self.sun_node = self.render.attachNewNode(self.sun)
        self.sun_node.setHpr(-42, -52, 0)
        self.render.setLight(self.sun_node)

        fill = DirectionalLight("sky-fill")
        fill.setColor((0.18, 0.22, 0.30, 1.0))
        fill_node = self.render.attachNewNode(fill)
        fill_node.setHpr(140, -30, 0)
        self.render.setLight(fill_node)

        try:
            if self.win and self.win.getGsg().getSupportsBasicShaders():
                self.sun.setShadowCaster(True, 2048, 2048)
                lens = self.sun.getLens()
                lens.setFilmSize(52, 52)
                lens.setNearFar(-60, 90)
                self.render.setShaderAuto()
        except Exception:
            pass

        fog = Fog("haze")
        fog.setColor(0.74, 0.82, 0.90)
        fog.setExpDensity(0.0065 / SCENE_K)
        self.render.setFog(fog)
        self.setBackgroundColor(0.74, 0.82, 0.90, 1.0)

    def setup_post_processing(self) -> None:
        try:
            from direct.filter.CommonFilters import CommonFilters
            filters = CommonFilters(self.win, self.cam)
            filters.setBloom(blend=(0.35, 0.45, 0.2, 0.0), desat=-0.3, intensity=0.9, size="small")
        except Exception:
            pass

    def setup_controls(self) -> None:
        self.accept("escape", self.userExit)
        self.accept("v", self.cycle_camera)
        self.accept("p", self.toggle_ai)
        self.accept("r", self.reset_car)
        self.accept("z", self.set_aero, [None])
        self.accept("x", self.set_aero, ["corner"])
        self.accept("c", self.set_aero, ["straight"])
        names = ["balanced", "stable", "rotation", "high_downforce", "low_drag", "front_aero", "rear_aero"]
        for index, name in enumerate(names, start=1):
            self.accept(str(index), self.apply_setup, [name])

    def build_scene(self) -> None:
        sky = self.render.attachNewNode(make_sky_dome("sky"))
        sky.setLightOff(1)
        sky.setFogOff(1)
        sky.setTwoSided(True)
        sky.setBin("background", 0)
        sky.setDepthWrite(False)

        self.add_ground()
        cl = self.track.centerline
        rw = self.track.road_width
        self.render.attachNewNode(make_strip("road-base", cl, rw + 30, 0.018, (0.05, 0.055, 0.06, 1)))
        self.render.attachNewNode(make_strip("road", cl, rw, ROAD_Z, (0.16, 0.165, 0.175, 1)))
        self.render.attachNewNode(make_strip("groove", cl, 26.0, 0.052, (0.10, 0.105, 0.115, 1)))
        # Kerbs
        kerb_w = 9.0
        red = (0.82, 0.10, 0.12, 1.0)
        white = (0.93, 0.93, 0.90, 1.0)
        self.render.attachNewNode(make_striped_strip("kerb-out", cl, kerb_w, 0.058, red, white, offset=rw * 0.5 - kerb_w * 0.4))
        self.render.attachNewNode(make_striped_strip("kerb-in", cl, kerb_w, 0.058, white, red, offset=-(rw * 0.5 - kerb_w * 0.4)))
        # Barriers
        wall_off = rw * 0.5 + 16.0
        self.render.attachNewNode(make_wall("barrier-out", cl, wall_off, 0.30, (0.88, 0.89, 0.90, 1.0)))
        self.render.attachNewNode(make_wall("barrier-out-top", cl, wall_off, 0.065, (0.78, 0.10, 0.12, 1.0), base_z=0.30))
        self.render.attachNewNode(make_wall("barrier-in", cl, -wall_off, 0.26, (0.45, 0.47, 0.50, 1.0)))
        self.add_buildings()
        self.add_harbor_life()
        self.add_infield_trees()
        self.add_start_gantry()

    def add_ground(self) -> None:
        card = CardMaker("ground")
        card.setFrame(-60 * SCENE_K, 60 * SCENE_K, -60 * SCENE_K, 60 * SCENE_K)
        ground = self.render.attachNewNode(card.generate())
        ground.setP(-90)
        ground.setZ(-0.02)
        ground.setColor(0.14, 0.34, 0.19, 1)

        water_card = CardMaker("harbor")
        water_card.setFrame(5 * SCENE_K, 40 * SCENE_K, -20 * SCENE_K, 8 * SCENE_K)
        self.water = self.render.attachNewNode(water_card.generate())
        self.water.setP(-90)
        self.water.setZ(-0.012)
        self.water.setColor(0.05, 0.30, 0.46, 1)

        glint = CardMaker("harbor-glint")
        glint.setFrame(6 * SCENE_K, 39 * SCENE_K, -19 * SCENE_K, 7 * SCENE_K)
        self.water_glint = self.render.attachNewNode(glint.generate())
        self.water_glint.setP(-90)
        self.water_glint.setZ(-0.008)
        self.water_glint.setTransparency(TransparencyAttrib.MAlpha)
        self.water_glint.setColor(0.55, 0.75, 0.88, 0.10)

    def add_box(self, name, position, scale, color):
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setName(name)
        box.setPos(position - LPoint3(scale[0] / 2, scale[1] / 2, 0))
        box.setScale(*scale)
        box.setColor(*color)
        return box

    def add_buildings(self) -> None:
        windows = make_window_texture()
        for rect, height, building_color in monaco_buildings():
            center = to_world(rect.center)
            rgb = tuple(channel / 255 for channel in building_color)
            side = 1.0 if center.x >= 0 else -1.0
            h = height / WORLD_SCALE * 0.30
            box = self.add_box(
                "building",
                LPoint3(center.x + side * 10.0, center.y, 0.0),
                (rect.width / WORLD_SCALE * 0.30, rect.height / WORLD_SCALE * 0.30, h),
                (*rgb, 1.0),
            )
            box.setTexture(windows, 1)
            box.setTexScale(self.render.findAllTextureStages()[0] if False else box.findTextureStage('*'), 2.0, max(1.0, h * 2.5))

    def add_harbor_life(self) -> None:
        random.seed(7)
        for bx, by, yaw in ((15.5, -2.0, 20), (18.5, 2.5, -35), (21.5, -4.5, 60), (24.0, 0.5, 10)):
            x, y = bx * SCENE_K, by * SCENE_K
            hull = self.add_box("yacht-hull", LPoint3(x, y, 0.0), (0.85, 0.34, 0.14), (0.96, 0.96, 0.94, 1.0))
            hull.setH(yaw)
            cabin = self.add_box("yacht-cabin", LPoint3(x, y, 0.14), (0.42, 0.22, 0.10), (0.80, 0.83, 0.86, 1.0))
            cabin.setH(yaw)

    def add_infield_trees(self) -> None:
        random.seed(21)
        center = to_world((WIDTH / 2, HEIGHT / 2))
        for _ in range(16):
            angle = random.uniform(0, 2 * math.pi)
            rx = random.uniform(0.15, 0.78)
            x = center.x + math.cos(angle) * (300.0 / WORLD_SCALE) * rx
            y = center.y + math.sin(angle) * (125.0 / WORLD_SCALE) * rx
            height = random.uniform(0.55, 0.95)
            trunk = self.render.attachNewNode(make_cylinder("trunk", 0.045, height, (0.36, 0.26, 0.16, 1.0)))
            trunk.setPos(x, y, height / 2)
            trunk.setP(90)
            for layer, (radius, dz) in enumerate(((0.34, 0.0), (0.26, 0.18), (0.17, 0.34))):
                cone = self.render.attachNewNode(make_cylinder("canopy", radius, 0.30, (0.10 + layer * 0.02, 0.38 + layer * 0.04, 0.14, 1.0), taper=0.05))
                cone.setPos(x, y, height + dz)
                cone.setP(90)

    def add_start_gantry(self) -> None:
        start = to_world(self.track.centerline[0])
        nxt = to_world(self.track.centerline[1])
        yaw = math.degrees(math.atan2((nxt - start).x, (nxt - start).y)) + 90
        beam_len = self.track.road_width / WORLD_SCALE + 0.8
        beam = self.add_box("gantry-beam", LPoint3(start.x, start.y, 1.05), (beam_len, 0.10, 0.16), (0.10, 0.10, 0.11, 1.0))
        beam.setH(yaw)
        for post_side in (-beam_len / 2, beam_len / 2):
            post = self.add_box("gantry-post", LPoint3(start.x + post_side * math.cos(math.radians(yaw)), start.y - post_side * math.sin(math.radians(yaw)), 0.0), (0.08, 0.08, 1.05), (0.10, 0.10, 0.11, 1.0))
        for index in range(5):
            offset = (index - 2) * 0.34
            light = self.add_box("gantry-light", LPoint3(start.x + offset * math.cos(math.radians(yaw)), start.y - offset * math.sin(math.radians(yaw)), 0.90), (0.14, 0.06, 0.14), (0.30, 0.02, 0.02, 1.0))
            light.setH(yaw)

    # ------------------------------------------------------------------
    # Car model
    # ------------------------------------------------------------------
    def build_car(self) -> None:
        self.car_node = self.render.attachNewNode("car")
        self.car_node.setScale(CAR_SCALE)
        body = self.car_node.attachNewNode("body")
        self.car_body = body

        red = (0.82, 0.07, 0.13, 1.0)
        dark_red = (0.55, 0.04, 0.09, 1.0)
        carbon = (0.07, 0.075, 0.085, 1.0)
        yellow = (1.0, 0.84, 0.10, 1.0)

        def part(parent, pos, scale, color, name="part"):
            box = self.loader.loadModel("models/box")
            box.reparentTo(parent)
            box.setName(name)
            box.setPos(pos[0] - scale[0] / 2, pos[1] - scale[1] / 2, pos[2])
            box.setScale(*scale)
            box.setColor(*color)
            return box

        # Floor + tub (car points +Y)
        part(body, (0, 0.0, 0.055), (0.66, 2.30, 0.045), carbon, "floor")
        part(body, (0, -0.10, 0.10), (0.46, 1.30, 0.22), red, "tub")
        part(body, (0, 0.72, 0.10), (0.30, 0.85, 0.15), red, "nose")
        part(body, (0, 1.14, 0.085), (0.16, 0.34, 0.10), yellow, "nose-tip")
        # Front wing
        part(body, (0, 1.28, 0.055), (1.00, 0.26, 0.035), carbon, "front-wing")
        part(body, (-0.50, 1.28, 0.055), (0.035, 0.30, 0.14), red, "fw-endplate-l")
        part(body, (0.50, 1.28, 0.055), (0.035, 0.30, 0.14), red, "fw-endplate-r")
        # Sidepods
        part(body, (-0.34, -0.18, 0.10), (0.26, 0.92, 0.17), dark_red, "sidepod-l")
        part(body, (0.34, -0.18, 0.10), (0.26, 0.92, 0.17), dark_red, "sidepod-r")
        # Engine cover + shark fin
        part(body, (0, -0.62, 0.135), (0.30, 0.85, 0.19), red, "engine-cover")
        part(body, (0, -0.78, 0.32), (0.03, 0.62, 0.16), dark_red, "shark-fin")
        # Halo + helmet
        part(body, (0, 0.16, 0.26), (0.30, 0.05, 0.045), carbon, "halo-front")
        part(body, (-0.15, 0.02, 0.24), (0.045, 0.32, 0.05), carbon, "halo-l")
        part(body, (0.15, 0.02, 0.24), (0.045, 0.32, 0.05), carbon, "halo-r")
        part(body, (0, -0.06, 0.20), (0.16, 0.16, 0.13), yellow, "helmet")
        # Airbox
        part(body, (0, -0.28, 0.29), (0.14, 0.22, 0.10), carbon, "airbox")
        # Rear wing: fixed main plane + DRS flap that rotates open
        part(body, (0, -1.10, 0.30), (0.86, 0.045, 0.055), carbon, "rw-main")
        part(body, (-0.43, -1.10, 0.10), (0.04, 0.24, 0.30), carbon, "rw-endplate-l")
        part(body, (0.43, -1.10, 0.10), (0.04, 0.24, 0.30), carbon, "rw-endplate-r")
        self.drs_flap = body.attachNewNode("drs-flap")
        self.drs_flap.setPos(0, -1.075, 0.40)
        part(self.drs_flap, (0, -0.03, -0.045), (0.82, 0.035, 0.09), red, "rw-flap")
        # Rear light (ERS/brake)
        self.rear_light = part(body, (0, -1.24, 0.14), (0.07, 0.05, 0.09), (0.25, 0.02, 0.02, 1.0), "rear-light")
        # Exhaust flame quad
        flame_card = CardMaker("flame")
        flame_card.setFrame(-0.10, 0.10, -0.16, 0.16)
        self.flame = body.attachNewNode(flame_card.generate())
        self.flame.setPos(0, -1.30, 0.16)
        self.flame.setP(-90)
        self.flame.setTexture(self.smoke_texture, 1)
        self.flame.setTransparency(TransparencyAttrib.MAlpha)
        self.flame.setColor(1.0, 0.55, 0.10, 0.0)
        self.flame.setLightOff(1)
        self.flame.setBillboardPointEye()

        # Wheels: steering nodes for fronts, spin nodes for all
        tire = (0.035, 0.035, 0.04, 1.0)
        rim = (0.72, 0.70, 0.62, 1.0)
        self.wheel_spin_nodes = []
        self.front_steer_nodes = []
        self.brake_discs = []
        wheel_radius = 0.155
        for x, y, is_front in ((-0.42, 0.78, True), (0.42, 0.78, True), (-0.42, -0.72, False), (0.42, -0.72, False)):
            anchor = body.attachNewNode("wheel-anchor")
            anchor.setPos(x, y, wheel_radius)
            if is_front:
                self.front_steer_nodes.append(anchor)
            spin = anchor.attachNewNode("wheel-spin")
            wheel = spin.attachNewNode(make_cylinder("wheel", wheel_radius, 0.16, tire, cap_color=rim))
            wheel.setH(90)  # axle across the car
            disc = spin.attachNewNode(make_cylinder("brake-disc", wheel_radius * 0.55, 0.05, (0.16, 0.16, 0.17, 1.0)))
            disc.setH(90)
            self.brake_discs.append(disc)
            self.wheel_spin_nodes.append(spin)
        self.wheel_radius = wheel_radius
        self.wheel_angle = 0.0

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------
    def build_effects(self) -> None:
        self.smoke_pool = []
        for _ in range(48):
            card = CardMaker("smoke")
            card.setFrame(-0.5, 0.5, -0.5, 0.5)
            node = self.render.attachNewNode(card.generate())
            node.setTexture(self.smoke_texture, 1)
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setBillboardPointEye()
            node.setLightOff(1)
            node.hide()
            self.smoke_pool.append({"node": node, "life": 0.0, "max_life": 0.0, "vel": LVector3(0, 0, 0)})
        self.smoke_cursor = 0
        self.smoke_accumulator = 0.0

    def spawn_smoke(self, position: LPoint3, intensity: float) -> None:
        puff = self.smoke_pool[self.smoke_cursor]
        self.smoke_cursor = (self.smoke_cursor + 1) % len(self.smoke_pool)
        puff["life"] = puff["max_life"] = 0.55 + random.random() * 0.35
        jitter = LVector3(random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08), 0)
        puff["node"].setPos(position + jitter)
        puff["node"].setScale(0.14 + 0.10 * intensity)
        puff["vel"] = LVector3(random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25), 0.9 + random.random() * 0.5)
        puff["node"].show()

    def update_effects(self, dt: float) -> None:
        # Tire smoke when an axle slides past the Magic Formula peak
        rear_sat = self.car.tire_state.rear_saturation
        front_sat = self.car.tire_state.front_saturation
        sliding = max(rear_sat, front_sat)
        if sliding > 1.05 and self.car.speed_kmh > 25.0:
            self.smoke_accumulator += dt * min(3.0, sliding) * 22.0
            while self.smoke_accumulator >= 1.0:
                self.smoke_accumulator -= 1.0
                use_rear = rear_sat >= front_sat
                indices = (2, 3) if use_rear else (0, 1)
                for wheel_index in indices:
                    spin = self.wheel_spin_nodes[wheel_index]
                    pos = spin.getPos(self.render)
                    self.spawn_smoke(LPoint3(pos.x, pos.y, 0.10), min(sliding - 1.0, 1.5))
        for puff in self.smoke_pool:
            if puff["life"] > 0.0:
                puff["life"] -= dt
                if puff["life"] <= 0.0:
                    puff["node"].hide()
                    continue
                t = 1.0 - puff["life"] / puff["max_life"]
                node = puff["node"]
                node.setPos(node.getPos() + puff["vel"] * dt)
                node.setScale(0.14 + t * 0.85)
                node.setColor(0.92, 0.92, 0.94, 0.55 * (1.0 - t))

        # Shift flame
        gear = self.car.powertrain_state.gear
        if gear != self.last_gear:
            self.shift_flash = 0.09 if gear > self.last_gear else 0.05
            self.last_gear = gear
        self.shift_flash = max(0.0, self.shift_flash - dt)
        self.flame.setColor(1.0, 0.55, 0.10, min(1.0, self.shift_flash * 11.0))

        # Brake disc glow + rear light
        brake_temp = self.car.brake_state.temperature
        glow = max(0.0, min((brake_temp - 350.0) / 600.0, 1.0))
        disc_color = (0.16 + glow * 0.9, 0.16 + glow * 0.25, 0.17, 1.0)
        for disc in self.brake_discs:
            disc.setColor(*disc_color)
        harvesting = self.car.hybrid_state.recovery_power > 1000.0
        braking = self.car.inputs.brake > 0.3
        blink = harvesting and (int(self.time_elapsed * 8) % 2 == 0)
        if braking or blink:
            self.rear_light.setColor(1.0, 0.08, 0.08, 1.0)
        else:
            self.rear_light.setColor(0.25, 0.02, 0.02, 1.0)

        # DRS flap animation
        target_pitch = -62.0 if self.car.aero_state.drs_active else 0.0
        current = self.drs_flap.getP()
        self.drs_flap.setP(current + (target_pitch - current) * min(1.0, 10.0 * dt))

        # Water shimmer
        self.water_glint.setColor(0.55, 0.75, 0.88, 0.06 + 0.05 * math.sin(self.time_elapsed * 1.7))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def build_hud(self) -> None:
        self.hud = OnscreenText(
            text="", pos=(-1.28, 0.90), align=TextNode.ALeft, scale=0.045,
            fg=(0.95, 0.97, 1.0, 1.0), bg=(0.02, 0.025, 0.03, 0.66), mayChange=True,
        )
        self.gear_hud = OnscreenText(
            text="", pos=(1.02, -0.78), align=TextNode.ACenter, scale=0.16,
            fg=(1.0, 0.95, 0.85, 1.0), bg=(0.02, 0.025, 0.03, 0.55), mayChange=True,
        )
        self.speed_hud = OnscreenText(
            text="", pos=(1.02, -0.92), align=TextNode.ACenter, scale=0.06,
            fg=(0.85, 0.90, 1.0, 1.0), mayChange=True,
        )
        self.drs_hud = OnscreenText(
            text="DRS", pos=(1.02, -0.62), align=TextNode.ACenter, scale=0.055,
            fg=(0.3, 0.32, 0.35, 1.0), mayChange=True,
        )

    def update_hud(self, on_track: bool) -> None:
        pt = self.car.powertrain_state
        state = self.checkpoints.state
        self.hud.setText(
            f"ApexDriveLab 3D\n"
            f"Lap {state.lap_count}   CP {state.current_checkpoint + 1}/8   Best {self.format_time(state.best_lap_time)}\n"
            f"Mode {'AI' if self.ai_enabled else 'Manual'}   Cam {('Chase', 'Cockpit', 'TV')[self.camera_mode]}   Track {'ON' if on_track else 'OFF'}\n"
            f"Battery {self.car.hybrid_state.charge_fraction * 100:4.0f}%   Brakes {self.car.brake_state.temperature:4.0f}C   "
            f"Tires F{self.car.tire_state.front_temperature:3.0f}/R{self.car.tire_state.rear_temperature:3.0f}C\n"
            "WASD drive | E DRS | Space ERS | V camera | P AI | R reset | Esc"
        )
        self.gear_hud.setText(f"{pt.gear}")
        self.speed_hud.setText(f"{self.car.speed_kmh:5.0f} km/h   {pt.rpm:5.0f} rpm")
        if self.car.aero_state.drs_active:
            self.drs_hud.setFg((0.2, 1.0, 0.4, 1.0))
        else:
            self.drs_hud.setFg((0.3, 0.32, 0.35, 1.0))

    @staticmethod
    def format_time(value) -> str:
        if value is None:
            return "--:--.---"
        minutes = int(value // 60)
        return f"{minutes:02d}:{value % 60:06.3f}"

    # ------------------------------------------------------------------
    # Controls & camera
    # ------------------------------------------------------------------
    def cycle_camera(self) -> None:
        self.camera_mode = (self.camera_mode + 1) % 3

    def toggle_ai(self) -> None:
        self.ai_enabled = not self.ai_enabled

    def set_aero(self, mode) -> None:
        self.aero_override = mode

    def apply_setup(self, name) -> None:
        self.car.apply_setup(SETUPS[name])

    def reset_car(self) -> None:
        self.car.reset()
        self.car.apply_setup(SETUPS["stable"])
        self.manual_steer = 0.0
        self.checkpoints.reset(self.track.checkpoint_index_for_position(self.car.position))

    def key_down(self, key: str) -> bool:
        return self.mouseWatcherNode.is_button_down(KeyboardButton.ascii_key(key))

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
        drs_request = True if self.key_down("e") else None
        return CarInputs(
            throttle=throttle, brake=brake, steer=steer,
            deploy_hybrid=self.mouseWatcherNode.is_button_down(KeyboardButton.space()),
            aero_mode=self.aero_override, drs=drs_request,
        )

    def update_car_visual(self, dt: float) -> None:
        world = to_world(self.car.position)
        squat = (1.0 - self.car.ride_height) * 0.05
        self.car_node.setPos(LPoint3(world.x, world.y, 0.0))
        self.car_node.lookAt(LPoint3(world.x, world.y, 0.0) + car_forward(self.car))
        self.car_body.setZ(0.045 - squat)
        # Body roll & pitch from accelerations
        roll = max(-6.0, min(self.car.lateral_acceleration * 0.010, 6.0))
        pitch = max(-4.0, min(-self.car.longitudinal_acceleration * 0.007, 4.0))
        self.car_body.setHpr(0, pitch, roll)
        # Wheels
        self.wheel_angle += math.degrees(self.car.speed / WORLD_SCALE / (self.wheel_radius * CAR_SCALE)) * dt
        steer_deg = -math.degrees(self.car.steering_angle) * 0.9
        for node in self.front_steer_nodes:
            node.setH(steer_deg)
        for spin in self.wheel_spin_nodes:
            spin.setP(-self.wheel_angle % 360.0)

    def update_camera(self, dt: float) -> None:
        world = to_world(self.car.position)
        forward = car_forward(self.car)
        speed_frac = min(self.car.speed_kmh / 260.0, 1.0)
        if self.camera_mode == 1:  # cockpit
            camera_pos = LPoint3(world.x, world.y, 0.50) + forward * 0.28
            target = LPoint3(world.x, world.y, 0.27) + forward * 16.0
            self.car_body.hide()
            self.camera.setPos(camera_pos)
        else:
            self.car_body.show()
            if self.camera_mode == 2:  # TV orbit
                angle = self.time_elapsed * 0.10
                camera_pos = LPoint3(world.x + math.cos(angle) * 11.0, world.y + math.sin(angle) * 11.0, 5.2)
                target = LPoint3(world.x, world.y, 0.2)
                self.camera.setPos(camera_pos)
            else:  # chase with lag + speed shake
                desired = LPoint3(world.x, world.y, 1.75 + speed_frac * 0.6) - forward * (3.9 + speed_frac * 1.8)
                blend = min(1.0, 5.5 * dt)
                self.cam_pos = self.cam_pos + (desired - self.cam_pos) * blend
                shake = speed_frac * 0.015
                jitter = LVector3(
                    math.sin(self.time_elapsed * 31.0) * shake,
                    math.sin(self.time_elapsed * 27.0 + 1.7) * shake,
                    math.sin(self.time_elapsed * 23.0 + 0.6) * shake,
                )
                self.camera.setPos(self.cam_pos + jitter)
                target = LPoint3(world.x, world.y, 0.12) + forward * 5.2
        self.camera.lookAt(target)
        self.camLens.setFov(58 + speed_frac * 20)
        self.camLens.setNearFar(0.05, 500)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def update(self, task):
        dt = min(globalClock.getDt(), 1 / 30)
        self.time_elapsed += dt
        controls = self.ai_driver.control(self.car, self.track) if self.ai_enabled else self.controls()
        on_track = self.track.is_on_track(self.car.position)
        self.car.update(dt, controls, 1.0 if on_track else OFF_TRACK_GRIP_SCALE)
        self.checkpoints.update(dt, self.track.checkpoint_index_for_position(self.car.position))
        self.update_car_visual(dt)
        self.update_camera(dt)
        self.update_effects(dt)
        self.update_hud(on_track)
        return task.cont


def main() -> None:
    app = ApexDrive3D()
    app.run()


if __name__ == "__main__":
    main()
