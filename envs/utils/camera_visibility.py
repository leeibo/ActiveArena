import numpy as np
import transforms3d as t3d


def _as_vec(vec, expected_dim: int, name: str):
    arr = np.array(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(expected_dim):
        raise ValueError(f"{name} must have shape ({expected_dim},), got {arr.shape}")
    return arr


def _normalize_aabb(aabb):
    arr = np.array(aabb, dtype=np.float64)
    if arr.shape != (2, 3):
        raise ValueError(f"aabb must have shape (2, 3), got {arr.shape}")
    mins = np.minimum(arr[0], arr[1])
    maxs = np.maximum(arr[0], arr[1])
    return np.stack([mins, maxs], axis=0)


def _combine_aabbs(aabbs):
    valid = []
    for aabb in aabbs:
        if aabb is None:
            continue
        try:
            valid.append(_normalize_aabb(aabb))
        except Exception:
            continue
    if len(valid) == 0:
        return None
    mins = np.min([aabb[0] for aabb in valid], axis=0)
    maxs = np.max([aabb[1] for aabb in valid], axis=0)
    return np.stack([mins, maxs], axis=0)


def _get_component_world_aabb(component):
    for name in ("compute_global_aabb_tight", "get_global_aabb_fast"):
        if not hasattr(component, name):
            continue
        try:
            return _normalize_aabb(getattr(component, name)())
        except Exception:
            continue
    return None


def _get_entity_world_aabb(entity):
    if entity is None:
        return None
    components = None
    if hasattr(entity, "get_components"):
        try:
            components = list(entity.get_components())
        except Exception:
            components = None
    elif hasattr(entity, "components"):
        try:
            components = list(entity.components)
        except Exception:
            components = None
    if components is None:
        return None
    return _combine_aabbs(_get_component_world_aabb(component) for component in components)


def _get_articulation_world_aabb(articulation):
    if articulation is None:
        return None
    links = None
    if hasattr(articulation, "get_links"):
        try:
            links = list(articulation.get_links())
        except Exception:
            links = None
    elif hasattr(articulation, "links"):
        try:
            links = list(articulation.links)
        except Exception:
            links = None
    if links is None:
        return None
    return _combine_aabbs(_get_component_world_aabb(link) for link in links)


def _get_object_world_aabb(obj):
    if obj is None:
        return None

    aabb = _get_component_world_aabb(obj)
    if aabb is not None:
        return aabb

    aabb = _get_entity_world_aabb(obj)
    if aabb is not None:
        return aabb

    aabb = _get_articulation_world_aabb(obj)
    if aabb is not None:
        return aabb

    wrapped = getattr(obj, "actor", None)
    if wrapped is not None and wrapped is not obj:
        aabb = _get_object_world_aabb(wrapped)
        if aabb is not None:
            return aabb

    return None


def _get_aabb_corners_world(aabb):
    mins, maxs = _normalize_aabb(aabb)
    corners = []
    for x in (mins[0], maxs[0]):
        for y in (mins[1], maxs[1]):
            for z in (mins[2], maxs[2]):
                corners.append([x, y, z])
    return np.array(corners, dtype=np.float64)


def _get_visible_uv_bounds(image_w, image_h, fovy_rad, horizontal_margin_rad=0.0, vertical_margin_rad=0.0):
    fovx_rad, fovy_rad = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    yaw_limit = max(0.0, 0.5 * fovx_rad - max(float(horizontal_margin_rad), 0.0))
    pitch_limit = max(0.0, 0.5 * float(fovy_rad) - max(float(vertical_margin_rad), 0.0))

    tan_half_fovx = float(np.tan(0.5 * fovx_rad))
    tan_half_fovy = float(np.tan(0.5 * float(fovy_rad)))
    tan_yaw_limit = float(np.tan(yaw_limit))
    tan_pitch_limit = float(np.tan(pitch_limit))

    u_half_span = 0.0 if tan_half_fovx <= 0.0 else min(tan_yaw_limit / tan_half_fovx, 1.0)
    v_half_span = 0.0 if tan_half_fovy <= 0.0 else min(tan_pitch_limit / tan_half_fovy, 1.0)

    return {
        "u_min": float(0.5 * (1.0 - u_half_span)),
        "u_max": float(0.5 * (1.0 + u_half_span)),
        "v_min": float(0.5 * (1.0 - v_half_span)),
        "v_max": float(0.5 * (1.0 + v_half_span)),
    }


def get_camera_fov_xy(image_w, image_h, fovy_rad):
    width = int(image_w)
    height = int(image_h)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: ({image_w}, {image_h})")

    fovy = float(fovy_rad)
    if not np.isfinite(fovy) or fovy <= 0.0 or fovy >= np.pi:
        raise ValueError(f"invalid vertical fov: {fovy_rad}")

    aspect = float(width) / float(height)
    fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)
    return float(fovx), float(fovy)


def image_u_to_yaw_error_rad(u_norm, image_w, image_h, fovy_rad):
    """
    Convert normalized image x position into a planar yaw error.

    Positive yaw error means the target lies to the camera's left and the
    camera should rotate left to center it.
    """
    u = float(u_norm)
    if not np.isfinite(u):
        raise ValueError(f"invalid normalized u value: {u_norm}")
    fovx_rad, _ = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    ndc_x = 2.0 * u - 1.0
    return float(np.arctan(-(ndc_x) * np.tan(0.5 * fovx_rad)))


def world_point_to_camera_local(world_point, camera_pose):
    """
    Transform a world-space point into the local frame of the camera.

    In this repository the camera local axes are:
      - +X: forward
      - +Y: left
      - +Z: up
    """
    point_world = _as_vec(world_point, 3, "world_point")
    cam_pos = _as_vec(camera_pose.p, 3, "camera_pose.p")
    cam_quat = _as_vec(camera_pose.q, 4, "camera_pose.q")
    cam_rot = t3d.quaternions.quat2mat(cam_quat)
    return cam_rot.T @ (point_world - cam_pos)


def project_world_point_to_image_uv(
    world_point,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    ret_debug=False,
):
    """
    Project a world-space point into image coordinates.

    Returns:
      - (u_norm, v_norm) when ret_debug is False.
      - ((u_norm, v_norm), debug) when ret_debug is True.

    Coordinate convention:
      - u_norm/v_norm are normalized to [0, 1] on the image plane.
      - u increases from left to right.
      - v increases from top to bottom.
    """
    cam_point = world_point_to_camera_local(world_point, camera_pose)
    cam_x, cam_y, cam_z = cam_point.tolist()

    fovx_rad, fovy_rad = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    tan_half_fovx = float(np.tan(0.5 * fovx_rad))
    tan_half_fovy = float(np.tan(0.5 * float(fovy_rad)))
    if tan_half_fovx <= 0.0 or tan_half_fovy <= 0.0:
        raise ValueError("invalid camera fov")

    inside, inside_debug = is_world_point_in_camera_fov(
        world_point=world_point,
        camera_pose=camera_pose,
        image_w=image_w,
        image_h=image_h,
        fovy_rad=fovy_rad,
        near_eps=near_eps,
        far=far,
        horizontal_margin_rad=horizontal_margin_rad,
        vertical_margin_rad=vertical_margin_rad,
        ret_debug=True,
    )

    if cam_x <= max(float(near_eps), 0.0):
        u_norm = np.nan
        v_norm = np.nan
    else:
        ndc_x = float(-cam_y / (cam_x * tan_half_fovx))
        ndc_y = float(-cam_z / (cam_x * tan_half_fovy))
        u_norm = float(0.5 * (ndc_x + 1.0))
        v_norm = float(0.5 * (ndc_y + 1.0))

    if not ret_debug:
        return u_norm, v_norm

    debug = dict(inside_debug)
    debug.update(
        {
            "inside": bool(inside),
            "camera_point": cam_point,
            "u_norm": u_norm,
            "v_norm": v_norm,
            "pixel_x": None if not np.isfinite(u_norm) else float(u_norm * (int(image_w) - 1)),
            "pixel_y": None if not np.isfinite(v_norm) else float(v_norm * (int(image_h) - 1)),
        }
    )
    return (u_norm, v_norm), debug


def is_world_point_in_camera_fov(
    world_point,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    ret_debug=False,
):
    """
    Check whether a world-space point lies inside a camera frustum.
    """
    cam_point = world_point_to_camera_local(world_point, camera_pose)
    cam_x, cam_y, cam_z = cam_point.tolist()

    fovx_rad, fovy_rad = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    yaw_limit = max(0.0, 0.5 * fovx_rad - max(float(horizontal_margin_rad), 0.0))
    pitch_limit = max(0.0, 0.5 * float(fovy_rad) - max(float(vertical_margin_rad), 0.0))

    yaw_err_rad = float(np.arctan2(cam_y, cam_x))
    pitch_err_rad = float(np.arctan2(cam_z, cam_x))
    inside_depth = bool(cam_x > max(float(near_eps), 0.0))
    if far is not None:
        inside_depth = bool(inside_depth and cam_x < float(far))
    inside_horizontal = bool(abs(yaw_err_rad) <= yaw_limit + 1e-12)
    inside_vertical = bool(abs(pitch_err_rad) <= pitch_limit + 1e-12)
    inside = bool(inside_depth and inside_horizontal and inside_vertical)

    if not ret_debug:
        return inside

    debug = {
        "camera_point": cam_point,
        "yaw_err_rad": yaw_err_rad,
        "pitch_err_rad": pitch_err_rad,
        "yaw_limit_rad": yaw_limit,
        "pitch_limit_rad": pitch_limit,
        "inside_depth": inside_depth,
        "inside_horizontal": inside_horizontal,
        "inside_vertical": inside_vertical,
    }
    return inside, debug


def _get_object_center_world(obj):
    if hasattr(obj, "get_pose"):
        pose = obj.get_pose()
    elif hasattr(obj, "pose"):
        pose = obj.pose
    else:
        raise TypeError("object must provide get_pose() or pose")

    if not hasattr(pose, "p"):
        raise TypeError("object pose must provide .p")
    return _as_vec(pose.p, 3, "object_pose.p")


def project_object_to_image_uv(
    obj,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    mode="center",
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    aabb_visible_ratio_threshold=0.0,
    ret_debug=False,
):
    mode = str(mode).lower()

    if mode == "center":
        world_point = _get_object_center_world(obj)
        projected = project_world_point_to_image_uv(
            world_point=world_point,
            camera_pose=camera_pose,
            image_w=image_w,
            image_h=image_h,
            fovy_rad=fovy_rad,
            near_eps=near_eps,
            far=far,
            horizontal_margin_rad=horizontal_margin_rad,
            vertical_margin_rad=vertical_margin_rad,
            ret_debug=True,
        )
        (u_norm, v_norm), debug = projected
        debug["world_point"] = world_point
        debug["mode"] = mode
        if not ret_debug:
            return u_norm, v_norm
        return (u_norm, v_norm), debug

    if mode != "aabb":
        raise NotImplementedError(f"unsupported visibility mode: {mode}")

    aabb = _get_object_world_aabb(obj)
    if aabb is None:
        # Fall back to center projection when collision AABB is unavailable.
        return project_object_to_image_uv(
            obj=obj,
            camera_pose=camera_pose,
            image_w=image_w,
            image_h=image_h,
            fovy_rad=fovy_rad,
            mode="center",
            near_eps=near_eps,
            far=far,
            horizontal_margin_rad=horizontal_margin_rad,
            vertical_margin_rad=vertical_margin_rad,
            aabb_visible_ratio_threshold=aabb_visible_ratio_threshold,
            ret_debug=ret_debug,
        )

    aabb = _normalize_aabb(aabb)
    world_point = 0.5 * (aabb[0] + aabb[1])
    corners = _get_aabb_corners_world(aabb)
    sample_points = np.concatenate([corners, world_point.reshape(1, 3)], axis=0)

    fovx_rad, fovy_rad = get_camera_fov_xy(image_w=image_w, image_h=image_h, fovy_rad=fovy_rad)
    tan_half_fovx = float(np.tan(0.5 * fovx_rad))
    tan_half_fovy = float(np.tan(0.5 * float(fovy_rad)))
    uv_bounds = _get_visible_uv_bounds(
        image_w=image_w,
        image_h=image_h,
        fovy_rad=fovy_rad,
        horizontal_margin_rad=horizontal_margin_rad,
        vertical_margin_rad=vertical_margin_rad,
    )

    projected_uvs = []
    valid_world_points = []
    valid_camera_points = []
    for point in sample_points:
        cam_point = world_point_to_camera_local(point, camera_pose)
        cam_x, cam_y, cam_z = cam_point.tolist()
        if cam_x <= max(float(near_eps), 0.0):
            continue
        if far is not None and cam_x >= float(far):
            continue

        ndc_x = float(-cam_y / (cam_x * tan_half_fovx))
        ndc_y = float(-cam_z / (cam_x * tan_half_fovy))
        u_norm = float(0.5 * (ndc_x + 1.0))
        v_norm = float(0.5 * (ndc_y + 1.0))
        projected_uvs.append([u_norm, v_norm])
        valid_world_points.append(np.array(point, dtype=np.float64))
        valid_camera_points.append(np.array(cam_point, dtype=np.float64))

    center_uv = project_world_point_to_image_uv(
        world_point=world_point,
        camera_pose=camera_pose,
        image_w=image_w,
        image_h=image_h,
        fovy_rad=fovy_rad,
        near_eps=near_eps,
        far=far,
        horizontal_margin_rad=horizontal_margin_rad,
        vertical_margin_rad=vertical_margin_rad,
        ret_debug=False,
    )

    if len(projected_uvs) == 0:
        if not ret_debug:
            return center_uv
        return center_uv, {
            "inside": False,
            "mode": mode,
            "world_point": world_point,
            "aabb": aabb.tolist(),
            "projected_bbox": None,
            "valid_projected_points": 0,
        }

    projected_uvs = np.array(projected_uvs, dtype=np.float64)
    u_min = float(np.min(projected_uvs[:, 0]))
    u_max = float(np.max(projected_uvs[:, 0]))
    v_min = float(np.min(projected_uvs[:, 1]))
    v_max = float(np.max(projected_uvs[:, 1]))

    clipped_u_min = max(u_min, uv_bounds["u_min"])
    clipped_u_max = min(u_max, uv_bounds["u_max"])
    clipped_v_min = max(v_min, uv_bounds["v_min"])
    clipped_v_max = min(v_max, uv_bounds["v_max"])
    bbox_w = max(float(u_max - u_min), 0.0)
    bbox_h = max(float(v_max - v_min), 0.0)
    bbox_area = bbox_w * bbox_h
    clipped_w = max(float(clipped_u_max - clipped_u_min), 0.0)
    clipped_h = max(float(clipped_v_max - clipped_v_min), 0.0)
    clipped_area = clipped_w * clipped_h
    visible_ratio = 0.0 if bbox_area <= 1e-12 else float(np.clip(clipped_area / bbox_area, 0.0, 1.0))
    ratio_threshold = float(np.clip(float(aabb_visible_ratio_threshold), 0.0, 1.0))

    has_overlap = bool(clipped_w > 0.0 and clipped_h > 0.0)
    inside = bool(has_overlap and visible_ratio >= ratio_threshold)

    if inside:
        u_norm = float(0.5 * (clipped_u_min + clipped_u_max))
        v_norm = float(0.5 * (clipped_v_min + clipped_v_max))
    else:
        u_norm, v_norm = center_uv

    if not ret_debug:
        return u_norm, v_norm

    debug = {
        "inside": inside,
        "mode": mode,
        "world_point": world_point,
        "aabb": aabb.tolist(),
        "projected_bbox": {
            "u_min": u_min,
            "u_max": u_max,
            "v_min": v_min,
            "v_max": v_max,
        },
        "visible_uv_bounds": uv_bounds,
        "visible_ratio": float(visible_ratio),
        "visible_ratio_threshold": float(ratio_threshold),
        "valid_projected_points": int(projected_uvs.shape[0]),
        "sample_world_points": [point.tolist() for point in valid_world_points],
        "sample_camera_points": [point.tolist() for point in valid_camera_points],
        "u_norm": float(u_norm) if np.isfinite(u_norm) else np.nan,
        "v_norm": float(v_norm) if np.isfinite(v_norm) else np.nan,
        "pixel_x": None if not np.isfinite(u_norm) else float(u_norm * (int(image_w) - 1)),
        "pixel_y": None if not np.isfinite(v_norm) else float(v_norm * (int(image_h) - 1)),
    }
    return (u_norm, v_norm), debug


def is_object_in_camera_fov(
    obj,
    camera_pose,
    image_w,
    image_h,
    fovy_rad,
    mode="center",
    near_eps=1e-4,
    far=None,
    horizontal_margin_rad=0.0,
    vertical_margin_rad=0.0,
    aabb_visible_ratio_threshold=0.0,
    ret_debug=False,
):
    """
    Check whether an object falls inside the camera frustum.
    """
    projected = project_object_to_image_uv(
        obj=obj,
        camera_pose=camera_pose,
        image_w=image_w,
        image_h=image_h,
        fovy_rad=fovy_rad,
        mode=mode,
        near_eps=near_eps,
        far=far,
        horizontal_margin_rad=horizontal_margin_rad,
        vertical_margin_rad=vertical_margin_rad,
        aabb_visible_ratio_threshold=aabb_visible_ratio_threshold,
        ret_debug=True,
    )

    (u_norm, v_norm), debug = projected
    inside = bool(debug.get("inside", np.isfinite(u_norm) and np.isfinite(v_norm)))
    if not ret_debug:
        return inside

    return inside, debug
