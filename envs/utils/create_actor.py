import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re

from .actor_utils import Actor, ArticulationActor


class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)


def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)


# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/textures/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


def _create_table_visual_material(texture_id=None, color=(1, 1, 1)):
    if texture_id is not None:
        texturepath = f"./assets/background_texture/{texture_id}.png"
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        return material
    return color


def _normalize_fan_table_geometry(
        outer_radius: float,
        inner_radius: float,
        angle_deg: float,
        center_deg: float,
        thickness: float,
):
    outer_radius = max(float(outer_radius), 0.15)
    inner_radius = float(max(0.0, min(inner_radius, outer_radius - 0.05)))
    thickness = max(float(thickness), 1e-3)
    angle_rad = float(np.deg2rad(np.clip(angle_deg, 10.0, 340.0)))
    center_rad = float(np.deg2rad(center_deg))
    theta_start = center_rad - angle_rad / 2.0
    theta_end = center_rad + angle_rad / 2.0
    return outer_radius, inner_radius, thickness, angle_rad, theta_start, theta_end


def _add_table_box(builder, scene, pose: sapien.Pose, half_size, material):
    builder.add_box_collision(
        pose=pose,
        half_size=half_size,
        material=scene.default_physical_material,
    )
    builder.add_box_visual(
        pose=pose,
        half_size=half_size,
        material=material,
    )


def _add_fan_table_surface(
        builder,
        scene,
        material,
        inner_radius: float,
        outer_radius: float,
        thickness: float,
        theta_start: float,
        angle_rad: float,
        radial_segments: int,
        min_theta_segments: int,
        theta_segments_per_meter: float,
        top_z: float = 0.0,
):
    r_edges = np.linspace(inner_radius, outer_radius, max(1, int(radial_segments)) + 1)
    for ridx in range(len(r_edges) - 1):
        r0, r1 = float(r_edges[ridx]), float(r_edges[ridx + 1])
        r_mid = 0.5 * (r0 + r1)
        radial_depth = max((r1 - r0) * 1.12, 1e-3)

        arc_len = max(r_mid * angle_rad, 1e-6)
        theta_seg = max(
            int(min_theta_segments),
            int(np.ceil(arc_len * max(float(theta_segments_per_meter), 1.0))),
        )
        dtheta = angle_rad / max(theta_seg, 1)

        for tidx in range(theta_seg):
            theta = theta_start + (tidx + 0.5) * dtheta
            tangential_len = max(r_mid * dtheta * 1.12, 1e-3)
            half_size = [tangential_len / 2.0, radial_depth / 2.0, thickness / 2.0]
            patch_pose = sapien.Pose(
                p=[r_mid * np.cos(theta), r_mid * np.sin(theta), top_z - thickness / 2.0],
                q=t3d.euler.euler2quat(0.0, 0.0, theta + np.pi / 2.0, axes="sxyz"),
            )
            _add_table_box(builder, scene, patch_pose, half_size, material)


def _add_fan_table_outer_and_inner_legs(
        builder,
        scene,
        material,
        inner_radius: float,
        outer_radius: float,
        height: float,
        thickness: float,
        theta_start: float,
        angle_rad: float,
        outer_leg_count: int,
):
    leg_half_h = max(height / 2.0 - 0.002, 1e-3)
    leg_pose_z = -height / 2.0 - 0.002
    leg_half_size = [thickness / 2.0, thickness / 2.0, leg_half_h]
    leg_margin = max(0.06, thickness * 1.2)

    leg_num = max(2, int(outer_leg_count))
    outer_leg_radius = max(inner_radius + leg_margin, outer_radius - leg_margin)
    outer_angles = np.linspace(theta_start + 0.08 * angle_rad, theta_start + 0.92 * angle_rad, leg_num)
    for theta in outer_angles:
        leg_pose = sapien.Pose(
            p=[outer_leg_radius * np.cos(theta), outer_leg_radius * np.sin(theta), leg_pose_z]
        )
        _add_table_box(builder, scene, leg_pose, leg_half_size, material)

    if inner_radius > leg_margin * 1.5:
        inner_leg_radius = inner_radius + leg_margin
        for theta in [theta_start + 0.06 * angle_rad, theta_start + 0.94 * angle_rad]:
            leg_pose = sapien.Pose(
                p=[inner_leg_radius * np.cos(theta), inner_leg_radius * np.sin(theta), leg_pose_z]
            )
            _add_table_box(builder, scene, leg_pose, leg_half_size, material)


def _add_fan_table_side_columns(
        builder,
        scene,
        material,
        lower_outer_radius: float,
        upper_outer_radius: float,
        height: float,
        thickness: float,
        theta_values,
        column_top_z: float,
):
    column_half_h = max((float(column_top_z) + float(height)) / 2.0, 1e-3)
    column_pose_z = 0.5 * (float(column_top_z) - float(height))
    column_half_size = [thickness / 2.0, thickness / 2.0, column_half_h]
    support_radius = min(float(lower_outer_radius), float(upper_outer_radius)) - thickness / 2.0
    if support_radius <= thickness / 2.0:
        raise ValueError("fan_double support radius is invalid; outer radius must exceed table thickness.")

    for theta in [float(theta) for theta in theta_values]:
        column_pose = sapien.Pose(
            p=[support_radius * np.cos(theta), support_radius * np.sin(theta), column_pose_z]
        )
        _add_table_box(builder, scene, column_pose, column_half_size, material)


def create_fan_table(
        scene,
        pose: sapien.Pose,
        outer_radius: float,
        height: float,
        inner_radius: float = 0.3,
        angle_deg: float = 200.0,
        center_deg: float = 90.0,
        thickness: float = 0.05,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
        radial_segments: int = 14,
        min_theta_segments: int = 24,
        theta_segments_per_meter: float = 18.0,
        outer_leg_count: int = 6,
) -> sapien.Entity:
    """
    Create an annular-sector ("fan") table.
    - The table center is at `pose.p[:2]`.
    - Sector points around `center_deg` in XY plane.
    - Top surface is approximated by many thin box patches.
    """
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    outer_radius, inner_radius, thickness, angle_rad, theta_start, theta_end = _normalize_fan_table_geometry(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        angle_deg=angle_deg,
        center_deg=center_deg,
        thickness=thickness,
    )
    table_mat = _create_table_visual_material(texture_id=texture_id, color=color)

    _add_fan_table_surface(
        builder=builder,
        scene=scene,
        material=table_mat,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        thickness=thickness,
        theta_start=theta_start,
        angle_rad=angle_rad,
        radial_segments=radial_segments,
        min_theta_segments=min_theta_segments,
        theta_segments_per_meter=theta_segments_per_meter,
    )
    _add_fan_table_outer_and_inner_legs(
        builder=builder,
        scene=scene,
        material=color,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        height=height,
        thickness=thickness,
        theta_start=theta_start,
        angle_rad=angle_rad,
        outer_leg_count=outer_leg_count,
    )

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


def create_fan_double_table(
        scene,
        pose: sapien.Pose,
        lower_outer_radius: float,
        height: float,
        lower_inner_radius: float = 0.3,
        upper_outer_radius: float | None = None,
        upper_inner_radius: float | None = None,
        angle_deg: float = 200.0,
        center_deg: float = 90.0,
        upper_theta_start_deg: float | None = None,
        upper_theta_end_deg: float | None = None,
        support_theta_deg: float | None = None,
        thickness: float = 0.05,
        layer_gap: float = 0.30,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
        radial_segments: int = 14,
        min_theta_segments: int = 24,
        theta_segments_per_meter: float = 18.0,
) -> sapien.Entity:
    """
    Create a two-layer annular-sector table.
    - The lower tabletop footprint is controlled by `lower_*_radius`.
    - The upper tabletop footprint is controlled by `upper_*_radius`.
    - If upper radii are omitted, they default to the lower-layer radii.
    - The upper tabletop can use its own angular span.
    - A single slim support column connects the outer edge of the upper layer to the lower layer.
    """
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    lower_outer_radius, lower_inner_radius, thickness, angle_rad, theta_start, theta_end = _normalize_fan_table_geometry(
        outer_radius=lower_outer_radius,
        inner_radius=lower_inner_radius,
        angle_deg=angle_deg,
        center_deg=center_deg,
        thickness=thickness,
    )
    if upper_outer_radius is None:
        upper_outer_radius = lower_outer_radius
    if upper_inner_radius is None:
        upper_inner_radius = lower_inner_radius
    upper_outer_radius, upper_inner_radius, _, _, _, _ = _normalize_fan_table_geometry(
        outer_radius=upper_outer_radius,
        inner_radius=upper_inner_radius,
        angle_deg=angle_deg,
        center_deg=center_deg,
        thickness=thickness,
    )
    if upper_theta_start_deg is None:
        upper_theta_start_deg = float(center_deg) - 30.0
    if upper_theta_end_deg is None:
        upper_theta_end_deg = float(center_deg) + 30.0
    upper_theta_start_deg = float(upper_theta_start_deg)
    upper_theta_end_deg = float(upper_theta_end_deg)
    if upper_theta_end_deg <= upper_theta_start_deg:
        upper_theta_end_deg = upper_theta_start_deg + 1.0
    upper_theta_start = float(np.deg2rad(upper_theta_start_deg))
    upper_theta_end = float(np.deg2rad(upper_theta_end_deg))
    upper_angle_rad = max(upper_theta_end - upper_theta_start, 1e-6)
    # Keep the white support column tied to the upper tabletop instead of
    # allowing it to be independently distributed.  The argument is retained
    # for backward-compatible call sites, but the task geometry requires the
    # support to sit at the center of the upper-layer angular span.
    support_theta_deg = 0.5 * (upper_theta_start_deg + upper_theta_end_deg)
    support_theta = float(np.deg2rad(support_theta_deg))
    layer_gap = max(float(layer_gap), thickness + 0.05)
    table_mat = _create_table_visual_material(texture_id=texture_id, color=color)

    _add_fan_table_surface(
        builder=builder,
        scene=scene,
        material=table_mat,
        inner_radius=lower_inner_radius,
        outer_radius=lower_outer_radius,
        thickness=thickness,
        theta_start=theta_start,
        angle_rad=angle_rad,
        radial_segments=radial_segments,
        min_theta_segments=min_theta_segments,
        theta_segments_per_meter=theta_segments_per_meter,
        top_z=0.0,
    )
    _add_fan_table_surface(
        builder=builder,
        scene=scene,
        material=table_mat,
        inner_radius=upper_inner_radius,
        outer_radius=upper_outer_radius,
        thickness=thickness,
        theta_start=upper_theta_start,
        angle_rad=upper_angle_rad,
        radial_segments=radial_segments,
        min_theta_segments=min_theta_segments,
        theta_segments_per_meter=theta_segments_per_meter,
        top_z=layer_gap,
    )
    _add_fan_table_side_columns(
        builder=builder,
        scene=scene,
        material=color,
        lower_outer_radius=lower_outer_radius,
        upper_outer_radius=upper_outer_radius,
        height=height,
        thickness=thickness,
        theta_values=[support_theta],
        column_top_z=layer_gap - thickness,
    )

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets/objects") / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor(mesh, model_data)


# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets") / "objects" / modelname
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            model_data["extents"] = (np.array(bounding_box["max"]) - np.array(bounding_box["min"])).tolist()
    object.set_name(modelname)
    return ArticulationActor(object, model_data)
