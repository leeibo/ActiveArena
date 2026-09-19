"""Shared helpers for fan-double second-layer tasks.

This module centralizes layer sampling, discrete head-search helpers, and
upper/lower transfer primitives used by the fan_double ranking and basket
tasks.  Keep task files declarative; put reusable transfer/search logic here.
"""

from .utils import *

import numpy as np
import sapien
import transforms3d as t3d


def get_robot_root_xy_yaw(task):
    root_xy = task.robot.left_entity_origion_pose.p[:2].tolist()
    root_q = np.asarray(task.robot.left_entity_origion_pose.q, dtype=np.float64)
    yaw = float(t3d.euler.quat2euler(root_q)[2])
    return root_xy, yaw


def normalize_layer(layer_name):
    layer_name = str(layer_name).lower()
    if layer_name not in ("lower", "upper"):
        raise ValueError(f"Layer must be 'lower' or 'upper', got {layer_name}")
    return layer_name


def get_layer_top_z(task, layer_name):
    layer_name = normalize_layer(layer_name)
    top_z = float(getattr(task, "rotate_table_top_z", 0.74))
    if layer_name == "upper":
        top_z += float(getattr(task, "rotate_fan_double_layer_gap", 0.35))
    return top_z


def get_layer_spec(task, layer_name, spec_attr="LAYER_SPECS"):
    layer_name = normalize_layer(layer_name)
    if layer_name == "upper":
        inner_radius = float(getattr(task, "rotate_fan_double_upper_inner_radius", 0.6))
        outer_radius = float(getattr(task, "rotate_fan_double_upper_outer_radius", 0.8))
        theta_start = getattr(task, "rotate_fan_double_upper_theta_start_world_rad", None)
        theta_end = getattr(task, "rotate_fan_double_upper_theta_end_world_rad", None)
    else:
        inner_radius = float(getattr(task, "rotate_fan_double_lower_inner_radius", 0.3))
        outer_radius = float(getattr(task, "rotate_fan_double_lower_outer_radius", 0.9))
        theta_start = None
        theta_end = None

    layer_specs = getattr(task, spec_attr, {})
    local_spec = dict(layer_specs.get(layer_name, {})) if isinstance(layer_specs, dict) else {}
    if "r_min" in local_spec or "r_max" in local_spec:
        r_min = float(local_spec.get("r_min", inner_radius))
        r_max = float(local_spec.get("r_max", outer_radius))
    else:
        inner_margin = float(local_spec.get("inner_margin", 0.08 if layer_name == "upper" else 0.12))
        outer_margin = float(local_spec.get("outer_margin", 0.06 if layer_name == "upper" else 0.14))
        max_cyl_r = float(local_spec.get("max_cyl_r", outer_radius - outer_margin))
        r_min = min(max(inner_radius + inner_margin, inner_radius + 0.04), outer_radius - 0.06)
        r_max = max(r_min, min(max_cyl_r, outer_radius - outer_margin))

    if (
        layer_name == "upper"
        and theta_start is not None
        and theta_end is not None
        and hasattr(task, "robot_yaw")
    ):
        theta0 = float(task._wrap_to_pi(float(theta_start) - float(task.robot_yaw)))
        theta1 = float(task._wrap_to_pi(float(theta_end) - float(task.robot_yaw)))
        thetalim = [min(theta0, theta1), max(theta0, theta1)]
    else:
        theta_shrink = float(local_spec.get("theta_shrink", 0.92))
        theta_half = float(rotate_theta_half(task)) * theta_shrink
        if theta_half <= 1e-3:
            theta_half = float(getattr(task, "rotate_object_theta_half_rad", np.deg2rad(45.0))) * theta_shrink
        thetalim = [-theta_half, theta_half]

    if "theta_min_deg" in local_spec or "theta_max_deg" in local_spec:
        theta_min = float(local_spec.get("theta_min_deg", np.rad2deg(thetalim[0])))
        theta_max = float(local_spec.get("theta_max_deg", np.rad2deg(thetalim[1])))
        thetalim = [float(np.deg2rad(theta_min)), float(np.deg2rad(theta_max))]

    return {
        "layer": layer_name,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "rlim": [float(r_min), float(r_max)],
        "thetalim": thetalim,
        "top_z": get_layer_top_z(task, layer_name),
    }


def pose_from_cyl(task, layer_name, cyl_spec, default_qpos=None, ret="pose", quat_frame="cyl"):
    layer_name = normalize_layer(layer_name)
    if default_qpos is None:
        default_qpos = [1, 0, 0, 0]
    z = get_layer_top_z(task, layer_name) + float(cyl_spec.get("z_offset", 0.0))
    return place_pose_cyl(
        [
            float(cyl_spec.get("r", 0.55)),
            float(np.deg2rad(cyl_spec.get("theta_deg", 0.0))),
            z,
        ] + list(cyl_spec.get("qpos", default_qpos)),
        robot_root_xy=task.robot_root_xy,
        robot_yaw_rad=task.robot_yaw,
        ret=ret,
        quat_frame=quat_frame,
    )


def valid_spacing(new_pose, existing_pose_lst, min_dist_sq):
    for pose in existing_pose_lst:
        if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < float(min_dist_sq):
            return False
    return True


def far_from_xy(new_pose, avoid_xy_lst, min_dist_sq):
    for xy in avoid_xy_lst:
        if np.sum(np.square(new_pose.p[:2] - np.array(xy, dtype=np.float64))) < float(min_dist_sq):
            return False
    return True


def sample_pose_on_layer(
    task,
    layer_name,
    z_offset=0.0,
    qpos=None,
    existing_pose_lst=None,
    avoid_xy_lst=None,
    min_dist_sq=0.012,
    spec_attr="LAYER_SPECS",
    max_tries=120,
):
    if qpos is None:
        qpos = [1, 0, 0, 0]
    if existing_pose_lst is None:
        existing_pose_lst = []
    if avoid_xy_lst is None:
        avoid_xy_lst = []

    layer_spec = get_layer_spec(task, layer_name, spec_attr=spec_attr)
    for _ in range(int(max_tries)):
        pose = rand_pose_cyl(
            rlim=layer_spec["rlim"],
            thetalim=layer_spec["thetalim"],
            zlim=[layer_spec["top_z"] + float(z_offset), layer_spec["top_z"] + float(z_offset)],
            robot_root_xy=task.robot_root_xy,
            robot_yaw_rad=task.robot_yaw,
            qpos=qpos,
            rotate_rand=False,
        )
        if not valid_spacing(pose, existing_pose_lst, min_dist_sq):
            continue
        if not far_from_xy(pose, avoid_xy_lst, min_dist_sq):
            continue
        return pose

    fallback_theta = float(np.mean(layer_spec["thetalim"]))
    return rand_pose_cyl(
        rlim=[layer_spec["rlim"][0], layer_spec["rlim"][0]],
        thetalim=[fallback_theta, fallback_theta],
        zlim=[layer_spec["top_z"] + float(z_offset), layer_spec["top_z"] + float(z_offset)],
        robot_root_xy=task.robot_root_xy,
        robot_yaw_rad=task.robot_yaw,
        qpos=qpos,
        rotate_rand=False,
    )


def get_object_arm_tag(task, obj):
    if not bool(getattr(task, "need_plan", True)):
        left_remaining = len(getattr(task, "left_joint_path", []) or []) - int(getattr(task, "left_cnt", 0))
        right_remaining = len(getattr(task, "right_joint_path", []) or []) - int(getattr(task, "right_cnt", 0))
        if left_remaining > 0 and right_remaining <= 0:
            return ArmTag("left")
        if right_remaining > 0 and left_remaining <= 0:
            return ArmTag("right")

    obj_cyl = world_to_robot(obj.get_pose().p.tolist(), task.robot_root_xy, task.robot_yaw)
    return ArmTag("left" if obj_cyl[1] >= 0 else "right")


def planner_pose_from_tcp_pose(task, tcp_pose):
    tcp_pose = np.array(tcp_pose, dtype=np.float64).reshape(-1)
    backoff = float(getattr(task, "DIRECT_RELEASE_TCP_BACKOFF", 0.12))
    planner_pos = tcp_pose[:3] - t3d.quaternions.quat2mat(tcp_pose[3:]) @ np.array(
        [backoff, 0.0, 0.0],
        dtype=np.float64,
    )
    return planner_pos.tolist() + tcp_pose[3:].tolist()


def build_horizontal_tcp_pose(task, cyl_r, cyl_theta_rad, tcp_z, yaw):
    quat = t3d.euler.euler2quat(0.0, 0.0, float(yaw)).tolist()
    return place_pose_cyl(
        [float(cyl_r), float(cyl_theta_rad), float(tcp_z)] + quat,
        robot_root_xy=task.robot_root_xy,
        robot_yaw_rad=task.robot_yaw,
        ret="list",
        quat_frame="world",
    )


def direct_entry_r(task, target_layer, release_r):
    explicit = getattr(task, "DIRECT_RELEASE_ENTRY_TCP_CYL_R", None)
    if explicit is not None:
        return float(explicit)

    entry_r = float(release_r) - 0.15
    if normalize_layer(target_layer) == "upper":
        upper_inner = getattr(task, "rotate_fan_double_upper_inner_radius", None)
        if upper_inner is not None:
            margin = float(getattr(task, "DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER", 0.08))
            entry_r = float(upper_inner) - margin
    return float(max(0.35, min(entry_r, float(release_r) - 0.08)))


def build_direct_release_pose_candidates(task, target_world_point, target_layer, arm_tag=None):
    target_world_point = np.array(target_world_point, dtype=np.float64).reshape(-1)
    if target_world_point.shape[0] < 3:
        raise ValueError("target_world_point must contain xyz")
    target_cyl = world_to_robot(target_world_point[:3].tolist(), task.robot_root_xy, task.robot_yaw)
    target_xy = target_world_point[:2]
    root_xy = np.array(task.robot_root_xy, dtype=np.float64)
    outward_yaw = float(np.arctan2(target_xy[1] - root_xy[1], target_xy[0] - root_xy[0]))

    yaw_offsets = getattr(task, "DIRECT_RELEASE_YAW_OFFSETS_DEG", (0.0, 15.0, -15.0))
    yaw_candidates = [float(outward_yaw + np.deg2rad(offset)) for offset in yaw_offsets]
    r_offsets = getattr(task, "DIRECT_RELEASE_R_OFFSETS", (0.0, -0.03, 0.03))
    theta_offsets = getattr(task, "DIRECT_RELEASE_THETA_OFFSETS_DEG", (0.0, -3.0, 3.0))

    release_z = float(target_world_point[2] + getattr(task, "DIRECT_RELEASE_TCP_Z_OFFSET", 0.06))
    entry_z = max(float(target_world_point[2] + getattr(task, "DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET", 0.10)), release_z)
    approach_z = max(
        float(target_world_point[2] + getattr(task, "DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET", 0.10)),
        release_z,
    )

    candidates = []
    for r_offset in r_offsets:
        release_r = float(target_cyl[0] + float(r_offset))
        entry_r = direct_entry_r(task, target_layer, release_r)
        for theta_offset_deg in theta_offsets:
            release_theta = float(target_cyl[1] + np.deg2rad(float(theta_offset_deg)))
            for yaw in yaw_candidates:
                entry_tcp_pose = build_horizontal_tcp_pose(task, entry_r, target_cyl[1], entry_z, yaw)
                approach_tcp_pose = build_horizontal_tcp_pose(task, release_r, release_theta, approach_z, yaw)
                tcp_pose = build_horizontal_tcp_pose(task, release_r, release_theta, release_z, yaw)
                candidates.append(
                    {
                        "entry_tcp_pose": entry_tcp_pose,
                        "entry_planner_pose": planner_pose_from_tcp_pose(task, entry_tcp_pose),
                        "approach_tcp_pose": approach_tcp_pose,
                        "approach_planner_pose": planner_pose_from_tcp_pose(task, approach_tcp_pose),
                        "tcp_pose": tcp_pose,
                        "planner_pose": planner_pose_from_tcp_pose(task, tcp_pose),
                    }
                )
    return candidates


def expand_active_plan_qpos_to_entity_qpos(task, arm_tag, active_qpos):
    active_qpos = np.array(active_qpos, dtype=np.float64).reshape(-1)
    if ArmTag(arm_tag) == "left":
        entity = getattr(task.robot, "left_entity", None)
        planner = getattr(task.robot, "left_planner", None)
    else:
        entity = getattr(task.robot, "right_entity", None)
        planner = getattr(task.robot, "right_planner", None)
    if entity is None or planner is None:
        return None

    try:
        full_qpos = np.array(entity.get_qpos(), dtype=np.float64).reshape(-1)
    except Exception:
        return None

    active_joint_names = list(getattr(planner, "active_joints_name", []) or [])
    all_joint_names = list(getattr(planner, "all_joints", []) or [])
    if len(active_joint_names) != active_qpos.shape[0] or len(all_joint_names) == 0:
        return None

    for active_idx, joint_name in enumerate(active_joint_names):
        if joint_name not in all_joint_names:
            return None
        full_idx = all_joint_names.index(joint_name)
        if full_idx >= full_qpos.shape[0]:
            return None
        full_qpos[full_idx] = active_qpos[active_idx]
    return [float(value) for value in full_qpos.astype(np.float32).tolist()]


def plan_path_for_sequence_check(plan_path, planner_pose, last_entity_qpos=None):
    try:
        if last_entity_qpos is None:
            return plan_path(planner_pose)
        return plan_path(planner_pose, last_qpos=last_entity_qpos)
    except TypeError:
        try:
            return plan_path(planner_pose)
        except Exception as exc:
            return {"status": "Fail", "error": str(exc)}
    except (IndexError, RuntimeError) as exc:
        if last_entity_qpos is None:
            return {"status": "Fail", "error": str(exc)}
        try:
            return plan_path(planner_pose)
        except Exception as retry_exc:
            return {"status": "Fail", "error": str(retry_exc)}


def select_pose_sequence_candidate(task, arm_tag, candidates, pose_keys):
    if not bool(getattr(task, "need_plan", True)):
        for candidate in candidates:
            if all(candidate.get(key, None) is not None for key in pose_keys):
                return candidate
        return None

    plan_path = task.robot.left_plan_path if ArmTag(arm_tag) == "left" else task.robot.right_plan_path
    for candidate in candidates:
        last_entity_qpos = None
        ok = True
        for pose_key in pose_keys:
            planner_pose = candidate.get(pose_key, None)
            if planner_pose is None:
                ok = False
                break
            plan_res = plan_path_for_sequence_check(plan_path, planner_pose, last_entity_qpos=last_entity_qpos)
            if not (isinstance(plan_res, dict) and plan_res.get("status", None) == "Success"):
                ok = False
                break
            position = plan_res.get("position", None)
            if position is not None and len(position) > 0:
                last_entity_qpos = expand_active_plan_qpos_to_entity_qpos(task, arm_tag, position[-1])
        if ok:
            return candidate
    return None


def move_pose_sequence(task, arm_tag, candidate, pose_keys):
    for pose_key in pose_keys:
        if not task.move(task.move_to_pose(arm_tag=arm_tag, target_pose=candidate[pose_key])):
            return False
    return True


def pose_with_min_z(pose, min_z):
    if pose is None:
        return None
    pose_arr = np.array(pose, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] < 7:
        return None
    pose_arr = pose_arr[:7].copy()
    pose_arr[2] = max(float(pose_arr[2]), float(min_z))
    return pose_arr.tolist()


def append_unique_pose(pose_sequence, pose, pos_tol=1e-4, quat_tol=1e-4):
    if pose is None:
        return
    pose_arr = np.array(pose, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] < 7:
        return
    pose = pose_arr[:7].tolist()
    if len(pose_sequence) > 0:
        prev = np.array(pose_sequence[-1], dtype=np.float64).reshape(-1)
        if prev.shape[0] >= 7:
            pos_same = float(np.linalg.norm(prev[:3] - pose_arr[:3])) <= float(pos_tol)
            quat_same = float(np.linalg.norm(prev[3:7] - pose_arr[3:7])) <= float(quat_tol)
            quat_same = quat_same or float(np.linalg.norm(prev[3:7] + pose_arr[3:7])) <= float(quat_tol)
            if pos_same and quat_same:
                return
    pose_sequence.append(pose)


def get_upper_release_clear_z(task):
    clear_offset = float(
        getattr(
            task,
            "upper_place_return_curobo_clear_z_offset",
            getattr(task, "UPPER_PLACE_RETURN_CUROBO_CLEAR_Z_OFFSET", 0.18),
        )
    )
    return float(get_layer_top_z(task, "upper") + max(clear_offset, 0.0))


def build_reverse_release_return_poses(task, selected_release_candidate):
    if not bool(getattr(task, "upper_place_return_curobo_reverse_release_path", True)):
        return []
    if not isinstance(selected_release_candidate, dict):
        return []

    clear_z = get_upper_release_clear_z(task)
    poses = []
    for pose_key in ("approach_planner_pose", "entry_planner_pose"):
        append_unique_pose(
            poses,
            pose_with_min_z(selected_release_candidate.get(pose_key, None), clear_z),
        )
    return poses


def move_upper_release_return_to_entry(task, arm_tag, selected_release_candidate):
    pose_sequence = build_reverse_release_return_poses(task, selected_release_candidate)
    if len(pose_sequence) == 0:
        return True

    candidate = {}
    pose_keys = []
    for idx, pose in enumerate(pose_sequence):
        pose_key = f"return_pose_{idx}"
        candidate[pose_key] = pose
        pose_keys.append(pose_key)

    selected = select_pose_sequence_candidate(task, arm_tag, [candidate], tuple(pose_keys))
    if selected is None:
        task.plan_success = False
        return False
    return move_pose_sequence(task, arm_tag, selected, tuple(pose_keys))


def pose_like_to_matrix(pose_like):
    if isinstance(pose_like, sapien.Pose):
        return pose_like.to_transformation_matrix()
    pose_arr = np.array(pose_like, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] != 7:
        raise ValueError(f"pose_like must contain 7 values, got shape {pose_arr.shape}")
    return sapien.Pose(pose_arr[:3], pose_arr[3:]).to_transformation_matrix()


def matrix_to_pose_list(matrix):
    matrix = np.array(matrix, dtype=np.float64).reshape(4, 4)
    quat = t3d.quaternions.mat2quat(matrix[:3, :3])
    return matrix[:3, 3].tolist() + quat.tolist()


def build_upper_pick_pose_candidates(task, obj, arm_tag):
    obj_pos = np.array(obj.get_pose().p, dtype=np.float64).reshape(3)
    root_xy = np.array(task.robot_root_xy, dtype=np.float64)
    outward_yaw = float(np.arctan2(obj_pos[1] - root_xy[1], obj_pos[0] - root_xy[0]))
    yaw_offsets = getattr(task, "UPPER_PICK_YAW_OFFSETS_DEG", (0.0, 15.0, -15.0, 30.0, -30.0))

    candidates = []
    for yaw_offset_deg in yaw_offsets:
        yaw = float(outward_yaw + np.deg2rad(float(yaw_offset_deg)))
        quat = t3d.euler.euler2quat(0.0, 0.0, yaw).tolist()
        local_x = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)

        grasp_pos = np.array(obj_pos, dtype=np.float64)
        grasp_pos[2] += float(getattr(task, "UPPER_PICK_GRASP_Z_BIAS", 0.0))
        pre_grasp_pos = grasp_pos - float(getattr(task, "UPPER_PICK_PRE_GRASP_DIS", 0.10)) * local_x
        entry_pos = np.array(pre_grasp_pos, dtype=np.float64)
        entry_pos[2] += float(getattr(task, "UPPER_PICK_ENTRY_Z_OFFSET", 0.08))

        entry_tcp_pose = entry_pos.tolist() + quat
        pre_grasp_tcp_pose = pre_grasp_pos.tolist() + quat
        grasp_tcp_pose = grasp_pos.tolist() + quat
        candidates.append(
            {
                "entry_tcp_pose": entry_tcp_pose,
                "entry_planner_pose": planner_pose_from_tcp_pose(task, entry_tcp_pose),
                "pre_grasp_tcp_pose": pre_grasp_tcp_pose,
                "pre_grasp_planner_pose": planner_pose_from_tcp_pose(task, pre_grasp_tcp_pose),
                "grasp_tcp_pose": grasp_tcp_pose,
                "grasp_planner_pose": planner_pose_from_tcp_pose(task, grasp_tcp_pose),
            }
        )
    return candidates


def build_upper_to_lower_hover_release_pose_candidates(task, obj, arm_tag, target_pose):
    target_pose = np.array(target_pose, dtype=np.float64).reshape(-1)
    if target_pose.shape[0] < 3:
        raise ValueError("target_pose must contain xyz")

    actor_pose_mat = obj.get_pose().to_transformation_matrix()
    fp_pose_mat = np.array(obj.get_functional_point(0, "matrix"), dtype=np.float64).reshape(4, 4)
    ee_pose = np.array(task.get_arm_pose(arm_tag), dtype=np.float64).reshape(-1)
    ee_pose_mat = pose_like_to_matrix(ee_pose)

    actor_to_fp = np.linalg.inv(actor_pose_mat) @ fp_pose_mat
    actor_to_ee = np.linalg.inv(actor_pose_mat) @ ee_pose_mat
    actor_to_fp_inv = np.linalg.inv(actor_to_fp)

    base_x_axis = np.array(actor_pose_mat[:2, 0], dtype=np.float64).reshape(2)
    base_x_axis_norm = float(np.linalg.norm(base_x_axis))
    if base_x_axis_norm <= 1e-9:
        root_xy = np.array(task.robot_root_xy, dtype=np.float64).reshape(2)
        base_x_axis = np.array(target_pose[:2], dtype=np.float64).reshape(2) - root_xy
        base_x_axis_norm = float(np.linalg.norm(base_x_axis))
    if base_x_axis_norm <= 1e-9:
        base_yaw = float(task.robot_yaw)
    else:
        base_yaw = float(np.arctan2(base_x_axis[1], base_x_axis[0]))

    yaw_offsets_deg = tuple(getattr(task, "UPPER_TO_LOWER_DROP_YAW_OFFSETS_DEG", (0.0, 90.0, -90.0, 180.0)))
    hover_z_offsets = tuple(getattr(task, "UPPER_TO_LOWER_HOVER_Z_OFFSETS", (0.06, 0.08, 0.10)))

    candidates = []
    for yaw_offset_deg in yaw_offsets_deg:
        actor_yaw = float(base_yaw + np.deg2rad(float(yaw_offset_deg)))
        actor_rot = t3d.euler.euler2mat(0.0, 0.0, actor_yaw)
        fp_rot = actor_rot @ actor_to_fp[:3, :3]
        for hover_z in hover_z_offsets:
            target_fp_pose_mat = np.eye(4, dtype=np.float64)
            target_fp_pose_mat[:3, :3] = fp_rot
            target_fp_pose_mat[:3, 3] = np.array(target_pose[:3], dtype=np.float64) + np.array(
                [0.0, 0.0, float(hover_z)],
                dtype=np.float64,
            )
            hover_actor_pose_mat = target_fp_pose_mat @ actor_to_fp_inv
            hover_ee_pose = matrix_to_pose_list(hover_actor_pose_mat @ actor_to_ee)
            candidates.append(
                {
                    "hover_pose": hover_ee_pose,
                    "hover_z": float(hover_z),
                }
            )

    candidates.sort(key=lambda candidate: float(candidate.get("hover_z", 0.0)))
    return candidates


def pick_object(task, subtask_idx, object_key, obj, layer_name, arm_tag=None, lower_grasp_kwargs=None):
    object_key = str(object_key)
    layer_name = normalize_layer(layer_name)
    if arm_tag is None:
        arm_tag = get_object_arm_tag(task, obj)
    arm_tag = ArmTag(arm_tag)
    if lower_grasp_kwargs is None:
        lower_grasp_kwargs = {}

    task.enter_rotate_action_stage(subtask_idx, focus_object_key=object_key)
    if layer_name == "upper":
        candidates = build_upper_pick_pose_candidates(task, obj, arm_tag)
        selected = select_pose_sequence_candidate(
            task,
            arm_tag,
            candidates,
            ("entry_planner_pose", "pre_grasp_planner_pose", "grasp_planner_pose"),
        )
        if selected is None:
            task.plan_success = False
            return arm_tag
        if not task.move(task.open_gripper(arm_tag)):
            return arm_tag
        if not move_pose_sequence(
            task,
            arm_tag,
            selected,
            ("entry_planner_pose", "pre_grasp_planner_pose", "grasp_planner_pose"),
        ):
            return arm_tag
        if not task.move(task.close_gripper(arm_tag, pos=float(getattr(task, "UPPER_PICK_GRIPPER_POS", 0.0)))):
            return arm_tag
    else:
        if not task.move(task.grasp_actor(obj, arm_tag=arm_tag, **lower_grasp_kwargs)):
            return arm_tag

    task._set_carried_object_keys([object_key])
    lift_z = float(getattr(task, "PICK_LIFT_Z", 0.10))
    if lift_z > 1e-9 and not task.move(task.move_by_displacement(arm_tag=arm_tag, z=lift_z)):
        return arm_tag
    extra_lift = float(getattr(task, "POST_GRASP_EXTRA_LIFT_Z", 0.0))
    if extra_lift > 1e-9 and task.plan_success:
        task.move(task.move_by_displacement(arm_tag=arm_tag, z=extra_lift))
    task.complete_rotate_subtask(subtask_idx, carried_after=[object_key])
    return arm_tag


def release_object_at_point(
    task,
    arm_tag,
    target_world_point,
    target_layer,
    retreat_after_release=True,
    return_after_upper_release=True,
):
    arm_tag = ArmTag(arm_tag)
    target_layer = normalize_layer(target_layer)
    candidates = build_direct_release_pose_candidates(task, target_world_point, target_layer, arm_tag=arm_tag)
    pose_keys = ("entry_planner_pose", "approach_planner_pose", "planner_pose")
    selected = select_pose_sequence_candidate(task, arm_tag, candidates, pose_keys)
    if selected is None:
        task.plan_success = False
        return False
    if not move_pose_sequence(task, arm_tag, selected, pose_keys):
        return False
    if not task.move(task.open_gripper(arm_tag)):
        return False
    retreat_z = float(getattr(task, "DIRECT_RELEASE_RETREAT_Z", 0.06))
    if bool(retreat_after_release) and retreat_z > 1e-9:
        if not task.move(task.move_by_displacement(arm_tag=arm_tag, z=retreat_z, move_axis="world")):
            return False
    if target_layer == "upper" and bool(return_after_upper_release):
        if not move_upper_release_return_to_entry(task, arm_tag, selected):
            return False
        lateral_xy = get_upper_place_lateral_escape_xy(task, arm_tag)
        if lateral_xy is not None and (
            abs(float(lateral_xy[0])) > 1e-9 or abs(float(lateral_xy[1])) > 1e-9
        ):
            if not task.move(
                task.move_by_displacement(
                    arm_tag=arm_tag,
                    x=float(lateral_xy[0]),
                    y=float(lateral_xy[1]),
                    move_axis="world",
                )
            ):
                return False
    return True


def get_current_body_facing_yaw(task):
    joint_name = getattr(
        task,
        "UPPER_PLACE_BODY_JOINT_NAME",
        getattr(task, "SCAN_JOINT_NAME", "astribot_torso_joint_2"),
    )
    joint_idx = task._get_preferred_torso_joint_index(
        joint_name_prefer=joint_name
    )
    torso_joints = list(getattr(task.robot, "torso_joints", []) or [])
    if joint_idx is not None and 0 <= joint_idx < len(torso_joints):
        joint = torso_joints[joint_idx]
        body_link = None if joint is None else getattr(joint, "child_link", None)
        if body_link is not None:
            facing_yaw, _ = task._compute_link_planar_facing_yaw(body_link)
            if facing_yaw is not None and np.isfinite(float(facing_yaw)):
                return float(facing_yaw)
    return float(task.robot_yaw)


def get_upper_place_lateral_escape_xy(task, arm_tag):
    lateral_dis = float(getattr(task, "UPPER_PLACE_LATERAL_ESCAPE_DIS", 0.0))
    if lateral_dis <= 1e-9:
        return None

    body_yaw = get_current_body_facing_yaw(task)
    leftward_xy = np.array(
        [-np.sin(body_yaw), np.cos(body_yaw)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(leftward_xy))
    if norm <= 1e-9:
        return None
    leftward_xy /= norm
    if ArmTag(arm_tag) == "right":
        leftward_xy = -leftward_xy
    return (leftward_xy * lateral_dis).tolist()


def release_object_by_hover_drop(task, arm_tag, obj, target_pose):
    arm_tag = ArmTag(arm_tag)
    candidates = build_upper_to_lower_hover_release_pose_candidates(task, obj, arm_tag, target_pose)
    selected = select_pose_sequence_candidate(task, arm_tag, candidates, ("hover_pose",))
    if selected is None:
        task.plan_success = False
        return False
    if not task.move(task.move_to_pose(arm_tag=arm_tag, target_pose=selected["hover_pose"])):
        return False
    if not task.move(task.open_gripper(arm_tag)):
        return False
    release_delay_steps = int(max(getattr(task, "UPPER_TO_LOWER_RELEASE_DELAY_STEPS", 0), 0))
    if release_delay_steps > 0:
        task.delay(release_delay_steps)
    retreat_z = float(getattr(task, "UPPER_TO_LOWER_RELEASE_RETREAT_Z", 0.08))
    if retreat_z > 1e-9:
        return bool(task.move(task.move_by_displacement(arm_tag=arm_tag, z=retreat_z, move_axis="world")))
    return True


def place_object(
    task,
    subtask_idx,
    object_key,
    obj,
    arm_tag,
    target_pose,
    target_layer,
    place_kwargs=None,
    focus_object_key=None,
    retreat_after_release=True,
    return_after_upper_release=True,
):
    object_key = str(object_key)
    arm_tag = ArmTag(arm_tag)
    target_layer = normalize_layer(target_layer)
    if place_kwargs is None:
        place_kwargs = {}

    task.enter_rotate_action_stage(subtask_idx, focus_object_key=focus_object_key)
    object_layers = getattr(task, "object_layers", {}) or {}
    source_layer = object_layers.get(object_key, None)
    if source_layer is not None:
        source_layer = normalize_layer(source_layer)

    if (
        target_layer == "lower"
        and source_layer == "upper"
        and bool(getattr(task, "UPPER_TO_LOWER_USE_HOVER_DROP", False))
    ):
        if not release_object_by_hover_drop(task, arm_tag, obj, target_pose):
            return False
        task._set_carried_object_keys([])
    elif target_layer == "lower" and bool(getattr(task, "LOWER_PLACE_WITH_PLACE_ACTOR", True)):
        if not task.move(task.place_actor(obj, arm_tag=arm_tag, target_pose=target_pose, **place_kwargs)):
            return False
        task._set_carried_object_keys([])
        lift_z = float(getattr(task, "LOWER_PLACE_RETREAT_Z", getattr(task, "PLACE_RETREAT_Z", 0.08)))
        move_axis = str(getattr(task, "LOWER_PLACE_RETREAT_MOVE_AXIS", "arm"))
        if lift_z > 1e-9 and not task.move(task.move_by_displacement(arm_tag=arm_tag, z=lift_z, move_axis=move_axis)):
            return False
    else:
        target_point = np.array(target_pose[:3], dtype=np.float64).reshape(3)
        if not release_object_at_point(
            task,
            arm_tag,
            target_point,
            target_layer,
            retreat_after_release=retreat_after_release,
            return_after_upper_release=return_after_upper_release,
        ):
            return False
        task._set_carried_object_keys([])

    object_layers = getattr(task, "object_layers", None)
    if isinstance(object_layers, dict):
        object_layers[object_key] = target_layer
    mark_discovered = getattr(task, "_mark_registry_target_discovered", None)
    if callable(mark_discovered):
        mark_discovered(object_key)
    task.complete_rotate_subtask(subtask_idx, carried_after=[])
    if bool(getattr(task, "RETURN_TO_HOMESTATE_AFTER_PLACE", True)):
        task.move(task.back_to_origin("left"), task.back_to_origin("right"))
    return True


def reset_head(task):
    return task._reset_head_to_home_pose(save_freq=getattr(task, "HEAD_RESET_SAVE_FREQ", -1))


def clear_rotate_target_search_history(task, object_key):
    key = str(object_key)
    discovered_objects = getattr(task, "discovered_objects", None)
    if isinstance(discovered_objects, dict):
        state = discovered_objects.get(key, None)
        if state is not None:
            state.update(
                {
                    "discovered": False,
                    "visible_now": False,
                    "first_seen_frame": None,
                    "last_seen_frame": None,
                    "last_seen_subtask": 0,
                    "last_seen_stage": 0,
                    "last_uv_norm": None,
                    "last_world_point": None,
                }
            )
    visible_objects = getattr(task, "visible_objects", None)
    if isinstance(visible_objects, dict) and key in visible_objects:
        visible_objects[key] = False


def get_subtask_search_target_keys(task, subtask_idx):
    subtask_def = task._get_rotate_subtask_def(subtask_idx) or {}
    return [str(key) for key in subtask_def.get("search_target_keys", [])]


def get_subtask_upper_search_target_keys(task, subtask_idx):
    object_layers = getattr(task, "object_layers", {}) or {}
    upper_keys = []
    for key in get_subtask_search_target_keys(task, subtask_idx):
        layer_name = object_layers.get(key, None)
        if layer_name is None:
            continue
        if normalize_layer(layer_name) == "upper":
            upper_keys.append(str(key))
    return upper_keys


def should_search_lower_before_upper_for_subtask(task, subtask_idx):
    first_upper_idx = task._get_rotate_first_upper_search_state_index()
    if first_upper_idx is None:
        return False
    return bool(len(get_subtask_upper_search_target_keys(task, subtask_idx)) > 0)


def has_unfinished_lower_search_phase(task):
    first_upper_idx = task._get_rotate_first_upper_search_state_index()
    if first_upper_idx is None:
        return False

    state_idx = getattr(task, "search_cursor_state_index", None)
    if state_idx is None:
        return True
    try:
        state_idx = int(state_idx)
    except (TypeError, ValueError):
        return True
    return bool(state_idx < int(first_upper_idx))


def prepare_subtask_rotate_search(task, subtask_idx):
    return None


def get_subtask_search_layers(task, subtask_idx):
    subtask_def = task._get_rotate_subtask_def(subtask_idx) or {}
    search_target_keys = [str(key) for key in subtask_def.get("search_target_keys", [])]
    if len(search_target_keys) == 0:
        return None

    object_registry = getattr(task, "object_registry", {}) or {}
    object_layers = getattr(task, "object_layers", {}) or {}
    resolved_layers = set()
    for key in search_target_keys:
        if key not in object_registry:
            return None
        layer_name = object_layers.get(key, None)
        if layer_name is None:
            return None
        try:
            resolved_layers.add(normalize_layer(layer_name))
        except Exception:
            return None
    return resolved_layers if len(resolved_layers) > 0 else None


def subtask_requires_head_home_reset(task, subtask_idx, prev_subtask_idx=None):
    if bool(task._should_skip_rotate_head_home_reset(subtask_idx, prev_subtask_idx=prev_subtask_idx)):
        return False

    current_layers = get_subtask_search_layers(task, subtask_idx)
    if prev_subtask_idx is None or current_layers is None or len(current_layers) != 1:
        return True

    prev_layers = get_subtask_search_layers(task, prev_subtask_idx)
    if prev_layers is None or len(prev_layers) != 1:
        return True

    return next(iter(current_layers)) != next(iter(prev_layers))


def maybe_reset_head_for_subtask(task, subtask_idx, prev_subtask_idx=None):
    return True


def search_focus(task, subtask_idx):
    return task.search_and_focus_rotate_subtask(
        subtask_idx,
        scan_r=float(getattr(task, "SCAN_R", 0.62)),
        scan_z=float(getattr(task, "SCAN_Z_BIAS", 0.90)) + float(task.table_z_bias),
        joint_name_prefer=str(getattr(task, "SCAN_JOINT_NAME", "astribot_torso_joint_2")),
    )


def object_world_point(obj):
    return np.array(obj.get_pose().p, dtype=np.float64).reshape(3)


def pose_list_from_point(point, quat=None):
    if quat is None:
        quat = [0, 1, 0, 0]
    point = np.array(point, dtype=np.float64).reshape(3)
    return point.tolist() + list(quat)
