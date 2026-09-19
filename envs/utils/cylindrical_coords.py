import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .transforms import _toPose


def _to_flat_array(data, valid_dims=(3, 7), name="data") -> np.ndarray:
    if isinstance(data, sapien.Pose):
        arr = np.array(data.p.tolist() + data.q.tolist(), dtype=np.float64).reshape(-1)
    else:
        arr = np.array(data, dtype=np.float64).reshape(-1)
    if arr.shape[0] not in valid_dims:
        raise ValueError(f"{name} dim must be one of {valid_dims}, got {arr.shape}")
    return arr


def _normalize_root_xy(robot_root_xy) -> np.ndarray:
    root_xy = np.array(robot_root_xy, dtype=np.float64).reshape(-1)
    if root_xy.shape[0] != 2:
        raise ValueError(f"robot_root_xy must have shape (2,), got {root_xy.shape}")
    return root_xy


def _wrap_to_pi(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def _build_cyl_basis_world(theta: float, robot_yaw_rad: float = 0.0) -> np.ndarray:
    """
    Build world rotation matrix of cylindrical local basis:
      x: radial outward
      y: tangential (theta increasing)
      z: world z
    """
    phi = float(theta + robot_yaw_rad)
    c, s = np.cos(phi), np.sin(phi)
    e_r = np.array([c, s, 0.0], dtype=np.float64)
    e_t = np.array([-s, c, 0.0], dtype=np.float64)
    e_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.stack([e_r, e_t, e_z], axis=1)


def world_to_robot(world_pt, robot_root_xy, robot_yaw_rad=0):
    """
    Convert world Cartesian point/pose to robot-centered cylindrical point/pose.

    Args:
      world_pt: 3D [x,y,z] or 7D [x,y,z,qw,qx,qy,qz] or sapien.Pose.
      robot_root_xy: robot root xy center [cx, cy].
      robot_yaw_rad: robot heading yaw in world frame. theta=0 aligns with this heading.

    Returns:
      3D [r,theta,z] or 7D [r,theta,z,qw,qx,qy,qz].
      For 7D, quaternion is expressed in cylindrical local basis:
        x=radial outward, y=tangential, z=world z.
    """
    root_xy = _normalize_root_xy(robot_root_xy)
    world = _to_flat_array(world_pt, valid_dims=(3, 7), name="world_pt")

    dx = float(world[0] - root_xy[0])
    dy = float(world[1] - root_xy[1])
    r = float(np.hypot(dx, dy))
    if r < 1e-9:
        # r=0 is degenerate for polar angle: fix theta to 0 for deterministic output.
        theta = 0.0
    else:
        phi_world = float(np.arctan2(dy, dx))
        theta = _wrap_to_pi(phi_world - float(robot_yaw_rad))
    z = float(world[2])

    if world.shape[0] == 3:
        return [r, theta, z]

    # Orientation mapping: world frame -> cylindrical local basis.
    R_w_obj = t3d.quaternions.quat2mat(world[3:7])
    R_w_cyl = _build_cyl_basis_world(theta, robot_yaw_rad)
    R_cyl_obj = R_w_cyl.T @ R_w_obj
    q_cyl = t3d.quaternions.mat2quat(R_cyl_obj).tolist()
    return [r, theta, z] + q_cyl


def robot_to_world(robot_pt, robot_root_xy, robot_yaw_rad=0):
    """
    Convert robot-centered cylindrical point/pose to world Cartesian point/pose.

    Args:
      robot_pt: 3D [r,theta,z] or 7D [r,theta,z,qw,qx,qy,qz].
      robot_root_xy: robot root xy center [cx, cy].
      robot_yaw_rad: robot heading yaw in world frame. theta=0 aligns with this heading.

    Returns:
      3D [x,y,z] or 7D [x,y,z,qw,qx,qy,qz].
      For 7D input, quaternion is interpreted in cylindrical local basis:
        x=radial outward, y=tangential, z=world z.
    """
    root_xy = _normalize_root_xy(robot_root_xy)
    robot = _to_flat_array(robot_pt, valid_dims=(3, 7), name="robot_pt")

    r = float(robot[0])
    theta = float(robot[1])
    z = float(robot[2])
    if r < 0:
        r = -r
        theta = _wrap_to_pi(theta + np.pi)

    phi_world = theta + float(robot_yaw_rad)
    x = float(root_xy[0] + r * np.cos(phi_world))
    y = float(root_xy[1] + r * np.sin(phi_world))

    if robot.shape[0] == 3:
        return [x, y, z]

    # Orientation mapping: cylindrical local basis -> world frame.
    R_cyl_obj = t3d.quaternions.quat2mat(robot[3:7])
    R_w_cyl = _build_cyl_basis_world(theta, robot_yaw_rad)
    R_w_obj = R_w_cyl @ R_cyl_obj
    q_world = t3d.quaternions.mat2quat(R_w_obj).tolist()
    return [x, y, z] + q_world


def place_point_cyl(point_cyl, robot_root_xy, robot_yaw_rad=0, ret: str = "list"):
    """
    Place a point using cylindrical coordinates.
    Args:
      point_cyl: [r,theta,z]
      ret: "list" | "array" | "pose"
    """
    point_cyl = _to_flat_array(point_cyl, valid_dims=(3,), name="point_cyl")
    world_point = robot_to_world(point_cyl, robot_root_xy=robot_root_xy, robot_yaw_rad=robot_yaw_rad)
    if ret == "array":
        return np.array(world_point, dtype=np.float64)
    if ret == "pose":
        return _toPose(world_point)
    return world_point


def place_pose_cyl(
    pose_cyl,
    robot_root_xy,
    robot_yaw_rad=0,
    ret: str = "list",
    quat_frame: str = "cyl",
):
    """
    Place a pose using cylindrical coordinates.
    Args:
      pose_cyl: [r,theta,z,qw,qx,qy,qz].
      quat_frame:
        - "cyl" (default): quaternion is interpreted in cylindrical local basis, then rotated +90deg around world z.
        - "cyl_legacy": quaternion is interpreted in cylindrical local basis (no +90deg correction).
        - "world": quaternion is interpreted in world frame; only xyz uses cylindrical mapping.
      ret: "list" | "array" | "pose"
    """
    pose_cyl = _to_flat_array(pose_cyl, valid_dims=(7,), name="pose_cyl")
    if quat_frame == "world":
        world_xyz = robot_to_world(
            pose_cyl[:3],
            robot_root_xy=robot_root_xy,
            robot_yaw_rad=robot_yaw_rad,
        )
        world_quat = pose_cyl[3:].tolist()
    elif quat_frame in ("cyl", "cyl_legacy"):
        world_pose_raw = robot_to_world(
            pose_cyl,
            robot_root_xy=robot_root_xy,
            robot_yaw_rad=robot_yaw_rad,
        )
        world_xyz = world_pose_raw[:3]
        world_quat = world_pose_raw[3:]
        if quat_frame == "cyl":
            q_rot = t3d.euler.euler2quat(0, 0, np.pi / 2)
            world_quat = t3d.quaternions.qmult(q_rot, world_quat).tolist()
    else:
        raise ValueError(f"quat_frame must be 'world', 'cyl', or 'cyl_legacy', got {quat_frame}")

    world_pose = world_xyz + world_quat

    if ret == "array":
        return np.array(world_pose, dtype=np.float64)
    if ret == "pose":
        return _toPose(world_pose)
    return world_pose


def rand_pose_cyl(
    rlim: np.ndarray,
    thetalim: np.ndarray,
    zlim: np.ndarray = [0.741],
    robot_root_xy=(0.0, 0.0),
    robot_yaw_rad=0.0,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=[1, 0, 0, 0],
    quat_frame: str = "cyl",
) -> sapien.Pose:
    """
    Random pose sampler in cylindrical coordinates.

    Args:
      rlim/thetalim/zlim: [min, max] (if max < min or only one value -> fixed).
      qpos: base quaternion [qw,qx,qy,qz].
      rotate_rand: apply random Euler perturbation around qpos (in local basis).
      rotate_lim: Euler random limits [rx, ry, rz] in radians.
      quat_frame:
        - "cyl" (default): quaternion is interpreted in cylindrical local basis, then rotated +90deg around world z.
        - "cyl_legacy": quaternion is interpreted in cylindrical local basis (no +90deg correction).
        - "world": quaternion is interpreted in world frame; only xyz uses cylindrical mapping.
    Returns:
      sapien.Pose in world frame.
    """

    def _normalize_range(v):
        v = np.array(v, dtype=np.float64).reshape(-1)
        if v.shape[0] == 0:
            return np.array([0.0, 0.0], dtype=np.float64)
        if v.shape[0] == 1:
            return np.array([v[0], v[0]], dtype=np.float64)
        if v[1] < v[0]:
            return np.array([v[0], v[0]], dtype=np.float64)
        return np.array([v[0], v[1]], dtype=np.float64)

    rlim = _normalize_range(rlim)
    thetalim = _normalize_range(thetalim)
    zlim = _normalize_range(zlim)

    r = float(np.random.uniform(rlim[0], rlim[1]))
    theta = float(np.random.uniform(thetalim[0], thetalim[1]))
    z = float(np.random.uniform(zlim[0], zlim[1]))

    q_base = np.array(qpos, dtype=np.float64).reshape(-1)
    if q_base.shape[0] != 4:
        raise ValueError(f"qpos must have shape (4,), got {q_base.shape}")
    if rotate_rand:
        rot_lim = np.array(rotate_lim, dtype=np.float64).reshape(-1)
        if rot_lim.shape[0] != 3:
            raise ValueError(f"rotate_lim must have shape (3,), got {rot_lim.shape}")
        angles = np.array([
            np.random.uniform(-rot_lim[0], rot_lim[0]),
            np.random.uniform(-rot_lim[1], rot_lim[1]),
            np.random.uniform(-rot_lim[2], rot_lim[2]),
        ], dtype=np.float64)
        q_rand = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
        q_base = t3d.quaternions.qmult(q_base, q_rand)

    if quat_frame == "world":
        world_xyz = robot_to_world(
            [r, theta, z],
            robot_root_xy=robot_root_xy,
            robot_yaw_rad=robot_yaw_rad,
        )
        world_quat = q_base.tolist()
    elif quat_frame in ("cyl", "cyl_legacy"):
        world_pose_raw = robot_to_world(
            [r, theta, z, float(q_base[0]), float(q_base[1]), float(q_base[2]), float(q_base[3])],
            robot_root_xy=robot_root_xy,
            robot_yaw_rad=robot_yaw_rad,
        )
        world_xyz = world_pose_raw[:3]
        world_quat = world_pose_raw[3:]
        if quat_frame == "cyl":
            q_rot = t3d.euler.euler2quat(0, 0, -np.pi / 2)
            world_quat = t3d.quaternions.qmult(q_rot, world_quat).tolist()
    else:
        raise ValueError(f"quat_frame must be 'world', 'cyl', or 'cyl_legacy', got {quat_frame}")

    return _toPose(world_xyz + world_quat)
