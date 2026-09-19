import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
from PIL import Image, ImageDraw, ImageFont

sys.path.append(".")
from envs.utils import create_actor


CAN_MODEL_NAME = "071_can"
CAN_MODEL_IDS = (0, 1, 2, 3, 5, 6)


def _load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fit_font(text, max_width, max_height, start_size, min_size=10):
    for size in range(int(start_size), int(min_size) - 1, -1):
        font = _load_font(size)
        bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
    return _load_font(min_size)


def _draw_centered(draw, xyxy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = xyxy
    x = x0 + (x1 - x0 - text_w) * 0.5
    y = y0 + (y1 - y0 - text_h) * 0.5 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def _draw_fit_centered(draw, xyxy, text, start_size, fill):
    x0, y0, x1, y1 = xyxy
    font = _fit_font(text, x1 - x0, y1 - y0, start_size)
    _draw_centered(draw, xyxy, text, font, fill)


def make_date_label_texture(
    path,
    production_date,
    vertical=True,
    label_text="PROD DATE",
    label_font_size=76,
    date_font_size=150,
):
    image = Image.new("RGBA", (768, 512), (246, 247, 239, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 750, 494), radius=34, outline=(32, 32, 32, 255), width=10)
    _draw_fit_centered(draw, (44, 58, 724, 166), label_text, label_font_size, (32, 32, 32, 255))
    _draw_fit_centered(draw, (44, 174, 724, 440), str(production_date), date_font_size, (10, 10, 10, 255))

    if vertical:
        image = image.rotate(90, expand=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def setup_scene():
    sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.3, 0.5, -1.0], [1.0, 1.0, 1.0], shadow=True)
    scene.add_point_light([0.7, -0.6, 0.8], [1.0, 1.0, 1.0], shadow=True)
    scene.add_point_light([-0.7, 0.5, 0.8], [0.6, 0.6, 0.6], shadow=False)
    return scene


def make_table(scene):
    material = sapien.render.RenderMaterial(base_color=[0.78, 0.78, 0.72, 1.0])
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_box_collision(half_size=[0.45, 0.34, 0.02])
    builder.add_box_visual(half_size=[0.45, 0.34, 0.02], material=material)
    builder.set_initial_pose(sapien.Pose([0.0, 0.0, 0.72]))
    return builder.build(name="table")


def _local_cylinder_label_mesh(
    center,
    radius,
    y_half,
    side="back",
    arc_deg=180.0,
    center_deg=None,
    flip_u=False,
    flip_v=False,
    theta_segments=64,
    y_segments=4,
):
    side_angle = np.pi if str(side).lower() == "back" else 0.0
    if center_deg is not None:
        side_angle = np.deg2rad(float(center_deg))
    arc = np.deg2rad(float(arc_deg))
    thetas = np.linspace(side_angle - arc * 0.5, side_angle + arc * 0.5, int(theta_segments) + 1)
    ys = np.linspace(float(center[1]) - float(y_half), float(center[1]) + float(y_half), int(y_segments) + 1)

    vertices = []
    normals = []
    uvs = []
    for j, y in enumerate(ys):
        v = j / max(1, int(y_segments))
        for i, theta in enumerate(thetas):
            u = i / max(1, int(theta_segments))
            uu = u
            vv = v
            if flip_u:
                uu = 1.0 - uu
            if flip_v:
                vv = 1.0 - vv
            normal = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32)
            vertices.append([
                float(center[0]) + float(radius) * float(normal[0]),
                float(y),
                float(center[2]) + float(radius) * float(normal[2]),
            ])
            normals.append(normal.tolist())
            uvs.append([float(uu), float(vv)])

    row = len(thetas)
    triangles = []
    for j in range(int(y_segments)):
        for i in range(int(theta_segments)):
            v00 = j * row + i
            v10 = j * row + i + 1
            v01 = (j + 1) * row + i
            v11 = (j + 1) * row + i + 1
            triangles.append([v00, v01, v10])
            triangles.append([v10, v01, v11])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(triangles, dtype=np.uint32),
        np.array(normals, dtype=np.float32),
        np.array(uvs, dtype=np.float32),
    )


def add_date_label_to_can(
    scene,
    can,
    texture_path,
    side="back",
    arc_deg=180.0,
    radial_offset=0.00015,
    label_radius=None,
    label_center_deg=None,
    flip_u=False,
    flip_v=False,
):
    material = sapien.render.RenderMaterial()
    material.set_base_color_texture(sapien.render.RenderTexture2D(str(texture_path)))
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.45
    material.metallic = 0.0

    scale = np.array(can.config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    center = np.array(can.config.get("center", [0.0, 0.0, 0.0]), dtype=np.float64) * scale
    extents = np.array(can.config.get("extents", [1.0, 1.0, 1.0]), dtype=np.float64) * scale

    radius = (
        float(label_radius)
        if label_radius is not None
        else float(min(extents[0], extents[2]) * 0.5 + float(radial_offset))
    )
    y_half = float(extents[1] * 0.38)
    vertices, triangles, normals, uvs = _local_cylinder_label_mesh(
        center=center,
        radius=radius,
        y_half=y_half,
        side=side,
        arc_deg=arc_deg,
        center_deg=label_center_deg,
        flip_u=flip_u,
        flip_v=flip_v,
    )
    label_shape = sapien.render.RenderShapeTriangleMesh(vertices, triangles, normals, uvs, material)

    # SAPIEN does not support attaching a new render shape to an entity after it
    # has been added to the scene. This actor is a static, visual-only overlay
    # with the same pose and cylindrical center as the can.
    entity = sapien.Entity()
    entity.set_name("can_production_date_label")
    entity.set_pose(can.get_pose())
    render_body = sapien.render.RenderBodyComponent()
    render_body.attach(label_shape)
    entity.add_component(render_body)
    scene.add_entity(entity)
    return {
        "center": center.tolist(),
        "radius": radius,
        "y_half": y_half,
        "arc_deg": float(arc_deg),
        "radial_offset": float(radial_offset),
        "label_radius": None if label_radius is None else float(label_radius),
        "label_center_deg": None if label_center_deg is None else float(label_center_deg),
        "flip_u": bool(flip_u),
        "flip_v": bool(flip_v),
    }


def add_flat_date_label_to_can(scene, can, texture_path, side="back"):
    material = sapien.render.RenderMaterial()
    material.set_base_color_texture(sapien.render.RenderTexture2D(str(texture_path)))
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.45
    material.metallic = 0.0

    scale = np.array(can.config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    center = np.array(can.config.get("center", [0.0, 0.0, 0.0]), dtype=np.float64) * scale
    extents = np.array(can.config.get("extents", [1.0, 1.0, 1.0]), dtype=np.float64) * scale
    label_half = np.array([0.001, 0.034, 0.019], dtype=np.float32)
    sign = -1.0 if str(side).lower() == "back" else 1.0
    offset = center.copy()
    offset[0] += sign * (float(extents[0]) * 0.5 + float(label_half[0]) + 0.001)
    quat = [0.0, 0.0, 0.0, 1.0] if sign < 0 else [1.0, 0.0, 0.0, 0.0]
    local_pose = sapien.Pose(offset.tolist(), quat)
    world_pose = sapien.Pose(
        (can.get_pose().to_transformation_matrix() @ local_pose.to_transformation_matrix())[:3, 3],
        t3d.quaternions.mat2quat((can.get_pose().to_transformation_matrix() @ local_pose.to_transformation_matrix())[:3, :3]),
    )
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_box_visual(half_size=label_half.tolist(), material=material)
    builder.set_initial_pose(world_pose)
    builder.set_name("can_production_date_label")
    builder.build()
    return {"center": center.tolist(), "flat_offset": offset.tolist()}


def load_can(scene, model_id):
    pose = sapien.Pose([0.0, 0.0, 0.755], [0.0, 0.0, 0.707, 0.707])
    can = create_actor(
        scene=scene,
        pose=pose,
        modelname=CAN_MODEL_NAME,
        convex=True,
        model_id=int(model_id),
    )
    if can is None:
        raise FileNotFoundError(f"Missing {CAN_MODEL_NAME}/base{model_id} asset")
    can.set_name(f"{CAN_MODEL_NAME}_base{model_id}_date_annotated")
    can.set_mass(0.05)
    return can


def open_viewer(scene, camera="back"):
    viewer = scene.create_viewer()
    viewer.set_scene(scene)
    if camera == "front":
        pose = sapien.Pose([0.32, -0.50, 0.93], [0.895, -0.328, 0.099, 0.286])
    else:
        pose = sapien.Pose([-0.38, 0.46, 0.93], [0.249, 0.076, 0.280, 0.924])
    viewer.set_camera_pose(pose)
    return viewer


def main():
    parser = argparse.ArgumentParser(description="View a 071_can asset with an injected production-date label.")
    parser.add_argument("--model-id", type=int, default=0, choices=CAN_MODEL_IDS)
    parser.add_argument("--production-date", default=None, help="ISO production date shown on the can label.")
    parser.add_argument("--expiry-date", default=None, help="Deprecated alias for --production-date.")
    parser.add_argument("--today", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--side", choices=("back", "front"), default="back")
    parser.add_argument("--camera", choices=("back", "front"), default="back")
    parser.add_argument("--arc-deg", type=float, default=180.0, help="Curved label angle around the can surface.")
    parser.add_argument("--label-radius", type=float, default=None, help="Absolute label cylinder radius in can-local meters.")
    parser.add_argument(
        "--label-center-deg",
        type=float,
        default=None,
        help="Absolute can-local cylinder angle for label center. Overrides --side. 0 is +x, 180 is -x.",
    )
    parser.add_argument("--flip-u", action="store_true", help="Flip the label texture horizontally on the curved surface.")
    parser.add_argument("--flip-v", action="store_true", help="Flip the label texture vertically on the curved surface.")
    parser.add_argument("--horizontal-texture", action="store_true", help="Use the old horizontal text direction.")
    parser.add_argument("--label-text", default="PROD DATE", help="Text shown above the numeric production date.")
    parser.add_argument("--label-font-size", type=int, default=76, help="Requested font size for the label line.")
    parser.add_argument("--date-font-size", type=int, default=150, help="Requested font size for the numeric date line.")
    parser.add_argument(
        "--radial-offset",
        type=float,
        default=0.00015,
        help="Meters outside the estimated can radius. Use a small negative value if the label still floats.",
    )
    parser.add_argument("--flat", action="store_true", help="Use the old flat label overlay instead of a curved label.")
    parser.add_argument(
        "--texture-out",
        default=str(Path(tempfile.gettempdir()) / "activearena_can_date_label" / "expiry_label.png"),
    )
    parser.add_argument("--texture-only", action="store_true", help="Only generate the label PNG, do not open SAPIEN.")
    args = parser.parse_args()

    production_date = args.production_date or args.expiry_date or "2026-05-01"
    texture_path = make_date_label_texture(
        args.texture_out,
        production_date,
        vertical=not args.horizontal_texture,
        label_text=args.label_text,
        label_font_size=args.label_font_size,
        date_font_size=args.date_font_size,
    )
    if args.texture_only:
        print(f"Date label texture: {texture_path}")
        return

    scene = setup_scene()
    make_table(scene)
    can = load_can(scene, args.model_id)
    label_info = (
        add_flat_date_label_to_can(scene, can, texture_path, side=args.side)
        if args.flat
        else add_date_label_to_can(
            scene,
            can,
            texture_path,
            side=args.side,
            arc_deg=args.arc_deg,
            radial_offset=args.radial_offset,
            label_radius=args.label_radius,
            label_center_deg=args.label_center_deg,
            flip_u=args.flip_u,
            flip_v=args.flip_v,
        )
    )

    print(f"Loaded {CAN_MODEL_NAME}/base{args.model_id}")
    print(f"Date label texture: {texture_path}")
    print(f"Production date: {production_date}")
    print(f"Label info: {label_info}")
    print("Close the SAPIEN viewer window to exit.")

    viewer = open_viewer(scene, camera=args.camera)
    while not viewer.closed:
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
