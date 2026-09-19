#!/usr/bin/env python3
"""Search CuRobo-feasible EEF poses for showing an object to the head camera.

The search is intentionally task-agnostic: it initializes the Astribot scene from
an existing config, samples candidate object centers in the head-camera frustum,
sets the gripper local +X (the grasp/approach/palm-normal direction used by
RobotWin grasp poses) to point back toward the head camera, and asks CuRobo for
left/right arm trajectories from homestate.

Outputs a JSON report and optional camera/observer screenshots under
outputs/eef_observation_pose_search/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml

sys.path.append("./")

from script.collect_data import apply_task_table_config, get_embodiment_config  # noqa: E402
from envs.count_target_press_button import count_target_press_button  # noqa: E402
from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


GRASP_CONTACT_OFFSET_M = 0.12
DEFAULT_OUTPUT_DIR = Path("outputs/eef_observation_pose_search")


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def _quat_to_mat(q) -> np.ndarray:
    return t3d.quaternions.quat2mat(np.asarray(q, dtype=np.float64).reshape(4))


def _mat_to_quat(R: np.ndarray) -> list[float]:
    q = t3d.quaternions.mat2quat(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if q[0] < 0:
        q = -q
    return q.astype(float).tolist()


def _make_frame_from_x(x_axis: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed rotation matrix whose local +X is x_axis."""
    x = _normalize(x_axis)
    if x is None:
        raise ValueError("invalid x axis")
    up = _normalize(up_hint)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z = up - float(np.dot(up, x)) * x
    z = _normalize(z)
    if z is None:
        for fallback in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
            z = np.asarray(fallback, dtype=np.float64) - float(np.dot(fallback, x)) * x
            z = _normalize(z)
            if z is not None:
                break
    if z is None:
        raise ValueError("could not build orthogonal frame")
    y = _normalize(np.cross(z, x))
    z = _normalize(np.cross(x, y))
    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0:
        y = -y
        R = np.column_stack([x, y, z])
    return R


def _make_frame_with_axis(axis_name: str, axis_vec: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a gripper frame whose requested local axis equals axis_vec."""
    axis_name = str(axis_name).lower()
    v = _normalize(axis_vec)
    if v is None:
        raise ValueError("invalid face axis")
    up = _normalize(up_hint)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if axis_name == "x":
        return _make_frame_from_x(v, up)

    if axis_name == "z":
        z = v
        x = up - float(np.dot(up, z)) * z
        x = _normalize(x)
        if x is None:
            for fallback in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
                x = np.asarray(fallback, dtype=np.float64) - float(np.dot(fallback, z)) * z
                x = _normalize(x)
                if x is not None:
                    break
        if x is None:
            raise ValueError("could not build z-constrained frame")
        y = _normalize(np.cross(z, x))
        # Ensure x cross y == z.
        y = _normalize(np.cross(z, x))
        R = np.column_stack([x, y, z])
        if np.linalg.det(R) < 0:
            y = -y
            R = np.column_stack([x, y, z])
        return R

    if axis_name == "y":
        y = v
        z = up - float(np.dot(up, y)) * y
        z = _normalize(z)
        if z is None:
            for fallback in ([0, 0, 1], [1, 0, 0], [0, 1, 0]):
                z = np.asarray(fallback, dtype=np.float64) - float(np.dot(fallback, y)) * y
                z = _normalize(z)
                if z is not None:
                    break
        if z is None:
            raise ValueError("could not build y-constrained frame")
        x = _normalize(np.cross(y, z))
        z = _normalize(np.cross(x, y))
        R = np.column_stack([x, y, z])
        if np.linalg.det(R) < 0:
            x = -x
            R = np.column_stack([x, y, z])
        return R

    raise ValueError(f"Unsupported axis_name={axis_name!r}")


def _roll_about_x(R: np.ndarray, angle_rad: float) -> np.ndarray:
    x = R[:, 0]
    return t3d.axangles.axangle2mat(x, float(angle_rad)) @ R


def _roll_about_axis(R: np.ndarray, axis_name: str, angle_rad: float) -> np.ndarray:
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(axis_name).lower()]
    axis = R[:, axis_idx]
    return t3d.axangles.axangle2mat(axis, float(angle_rad)) @ R


def _pose_axes(pose7: Iterable[float]) -> dict:
    arr = np.asarray(pose7, dtype=np.float64).reshape(7)
    R = _quat_to_mat(arr[3:])
    return {
        "x": R[:, 0].astype(float).tolist(),
        "y": R[:, 1].astype(float).tolist(),
        "z": R[:, 2].astype(float).tolist(),
    }


def _fmt_pose(pose7: Iterable[float]) -> str:
    arr = np.asarray(pose7, dtype=np.float64).reshape(7)
    return "[" + ", ".join(f"{v:+.6f}" for v in arr) + "]"


def _quat_distance_rad(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = float(abs(np.dot(q1, q2)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _load_args(task, config_name: str, output_dir: Path):
    with open(f"task_config/{config_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = "count_target_press_button"
    args = apply_task_table_config(task, args)

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    embodiment = args["embodiment"]

    def get_embodiment_file(embodiment_type):
        return embodiment_types[embodiment_type]["file_path"]

    args["left_robot_file"] = get_embodiment_file(embodiment[0])
    args["right_robot_file"] = get_embodiment_file(embodiment[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment[0])
    args["task_config"] = config_name
    args["difficulty_tag"] = "eef_observation_search"
    args["storage_setting"] = "eef_observation_search"
    args["save_path"] = str(output_dir / "tmp_scene")
    args["need_plan"] = True
    args["render_freq"] = 0
    args["save_freq"] = None
    args["save_data"] = False
    args["collect_data"] = False
    args["use_seed"] = False
    args["verbose_planner_log"] = False
    return args


def _sample_candidates(camera_pose7, arm: str, face_axis_options):
    cam_p = np.asarray(camera_pose7[:3], dtype=np.float64)
    cam_R = _quat_to_mat(camera_pose7[3:])
    cam_forward = cam_R[:, 0]  # SAPIEN cameras in this repo use local +X as optical forward.
    cam_left = cam_R[:, 1]
    cam_up = cam_R[:, 2]

    # Sample object centers in a conservative "in front of eyes" region.
    # The EEF origin is then placed behind the object along local +X.
    distance_values = np.linspace(0.28, 0.58, 7)
    lateral_values = np.linspace(-0.16, 0.16, 5)
    vertical_values = np.linspace(-0.08, 0.08, 5)
    roll_values = np.deg2rad(np.arange(-180, 180, 30))

    for d in distance_values:
        for lateral in lateral_values:
            for vertical in vertical_values:
                object_center = cam_p + d * cam_forward + lateral * cam_left + vertical * cam_up
                if object_center[2] < 0.78 or object_center[2] > 1.18:
                    continue
                if object_center[1] < -0.18 or object_center[1] > 0.34:
                    continue
                to_camera = _normalize(cam_p - object_center)
                if to_camera is None:
                    continue

                for face_axis, face_sign in face_axis_options:
                    # face_sign=+1 means local +axis points to camera;
                    # face_sign=-1 means local -axis points to camera.
                    axis_vec = float(face_sign) * to_camera
                    try:
                        base_R = _make_frame_with_axis(face_axis, axis_vec, cam_up)
                    except ValueError:
                        continue

                    for roll in roll_values:
                        R = _roll_about_axis(base_R, face_axis, roll)
                        # RobotWin grasp poses put the approximate grasped object/contact
                        # center along local +X from the gripper frame.
                        eef_p = object_center - GRASP_CONTACT_OFFSET_M * R[:, 0]

                        if eef_p[2] < 0.68 or eef_p[2] > 1.16:
                            continue
                        if abs(eef_p[0]) > 0.55 or eef_p[1] < -0.20 or eef_p[1] > 0.46:
                            continue
                        # Route left arm mostly to the left half, right arm to the right half,
                        # but keep a center overlap so both can be tested near the optical axis.
                        if arm == "left" and eef_p[0] > 0.16:
                            continue
                        if arm == "right" and eef_p[0] < -0.16:
                            continue

                        pose = eef_p.astype(float).tolist() + _mat_to_quat(R)
                        yield {
                            "arm": arm,
                            "pose": pose,
                            "object_center": object_center.astype(float).tolist(),
                            "camera_distance": float(d),
                            "camera_lateral": float(lateral),
                            "camera_vertical": float(vertical),
                            "roll_rad": float(roll),
                            "face_axis": str(face_axis),
                            "face_sign": int(face_sign),
                        }


def _plan_batch(robot, arm: str, candidates: list[dict], chunk_size: int = 10) -> list[dict]:
    plan_multi = robot.left_plan_multi_path if arm == "left" else robot.right_plan_multi_path
    successes = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start + chunk_size]
        if not chunk:
            continue
        # The batch MotionGen is warmed for CONFIGS.ROTATE_NUM; pad the final chunk
        # to avoid version-dependent batch-size assumptions.
        padded = list(chunk)
        while len(padded) < chunk_size:
            padded.append(chunk[-1])
        target_list = [c["pose"] for c in padded]
        try:
            res = plan_multi(target_list)
        except Exception as exc:
            print(f"[WARN] {arm} batch {start}: {type(exc).__name__}: {exc}")
            continue
        status = list(res.get("status", [])) if isinstance(res, dict) else []
        for local_i, c in enumerate(chunk):
            ok = local_i < len(status) and str(status[local_i]) == "Success"
            if not ok:
                continue
            item = dict(c)
            item["batch_index"] = int(start + local_i)
            try:
                if "position" in res:
                    item["batch_path_steps"] = int(np.asarray(res["position"])[local_i].shape[0])
            except Exception:
                pass
            successes.append(item)
    return successes


def _build_ik_solver(planner, num_seeds: int = 64):
    """Create a CuRobo IK solver sharing the same robot/world config as MotionGen."""
    import torch  # noqa: F401
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    # Reuse the resolved CuRobo yml and the planner's table/head collision world.
    world_model = planner._build_world_config()
    ik_config = IKSolverConfig.load_from_robot_config(
        planner.yml_path,
        world_model,
        num_seeds=int(num_seeds),
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=True,
    )
    return IKSolver(ik_config)


def _ik_filter_batch(task, arm: str, candidates: list[dict], chunk_size: int = 256, num_seeds: int = 64) -> list[dict]:
    """Fast coarse filter: CuRobo IK + collision, then full MotionGen only for top candidates."""
    import torch
    from curobo.types.math import Pose as CuroboPose

    robot = task.robot
    planner = robot.left_planner if arm == "left" else robot.right_planner
    ik_solver = _build_ik_solver(planner, num_seeds=num_seeds)
    successes = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start + chunk_size]
        if not chunk:
            continue
        pose_rows = []
        for c in chunk:
            endlink_pose = robot._trans_from_gripper_to_endlink(c["pose"], arm_tag=arm)
            p_base, q_base = planner._trans_world_to_curobo_frame(endlink_pose)
            pose_rows.append(list(p_base) + list(q_base))
        pose_tensor = torch.tensor(pose_rows, dtype=torch.float32, device="cuda")
        goal_pose = CuroboPose(pose_tensor[:, :3], pose_tensor[:, 3:])
        try:
            result = ik_solver.solve_batch(goal_pose)
        except Exception as exc:
            print(f"[WARN] {arm} IK batch {start}: {type(exc).__name__}: {exc}")
            continue
        success = result.success.detach().cpu().numpy().reshape(-1)
        solution = None
        try:
            solution = result.solution.detach().cpu().numpy()
        except Exception:
            solution = None
        for local_i, ok in enumerate(success[:len(chunk)]):
            if not bool(ok):
                continue
            item = dict(chunk[local_i])
            item["ik_batch_index"] = int(start + local_i)
            if solution is not None:
                sol = np.asarray(solution[local_i]).reshape(-1)
                item["ik_solution"] = sol.astype(float).tolist()
            successes.append(item)
    return successes


def _score_candidate(candidate: dict, camera_pose7) -> float:
    pose = np.asarray(candidate["pose"], dtype=np.float64)
    object_center = np.asarray(candidate["object_center"], dtype=np.float64)
    cam_p = np.asarray(camera_pose7[:3], dtype=np.float64)
    cam_R = _quat_to_mat(camera_pose7[3:])
    cam_forward = cam_R[:, 0]
    view_vec = _normalize(object_center - cam_p)
    palm_to_camera = _normalize(cam_p - object_center)
    axes = _pose_axes(pose)
    face_axis = str(candidate.get("face_axis", "x"))
    face_sign = float(candidate.get("face_sign", 1))
    face_vec = face_sign * np.asarray(axes[face_axis], dtype=np.float64)
    align = float(np.dot(face_vec, palm_to_camera)) if palm_to_camera is not None else -1.0
    optical = float(np.dot(view_vec, cam_forward)) if view_vec is not None else -1.0
    d = float(candidate["camera_distance"])
    lat = abs(float(candidate["camera_lateral"]))
    vert = abs(float(candidate["camera_vertical"]))
    # Prefer central, 35-45 cm object distance, and shorter planned paths.
    steps = float(candidate.get("single_path_steps", candidate.get("batch_path_steps", 999)))
    # Direct link-origin inspection shows the RobotWin EEF local +X axis is the
    # gripper/palm extension direction (base -> fingertips/object); local +/-Y
    # is the finger opening direction.  Therefore +X is the preferred "palm
    # extension toward camera" axis.  Other axes are kept only for diagnostics.
    axis_prior = 0.25 if face_axis == "x" else 0.0
    return 10.0 * align + 4.0 * optical + axis_prior - 2.0 * lat - 1.0 * vert - 2.0 * abs(d - 0.40) - 0.002 * steps


def _refine_single(robot, candidates: list[dict], camera_pose7, max_refine: int = 30) -> list[dict]:
    ranked = sorted(candidates, key=lambda c: _score_candidate(c, camera_pose7), reverse=True)
    refined = []
    for c in ranked[:max_refine]:
        plan = robot.left_plan_path(c["pose"]) if c["arm"] == "left" else robot.right_plan_path(c["pose"])
        if not isinstance(plan, dict) or plan.get("status") != "Success":
            continue
        item = dict(c)
        pos = np.asarray(plan.get("position", []), dtype=np.float64)
        vel = np.asarray(plan.get("velocity", []), dtype=np.float64)
        item["single_path_steps"] = int(pos.shape[0]) if pos.ndim >= 2 else 0
        item["final_arm_joints"] = pos[-1].astype(float).tolist() if pos.ndim == 2 and pos.shape[0] else []
        item["final_arm_velocity"] = vel[-1].astype(float).tolist() if vel.ndim == 2 and vel.shape[0] else []
        item["debug_target_tcp_pose"] = plan.get("debug_target_tcp_pose")
        item["debug_target_endlink_pose"] = plan.get("debug_target_endlink_pose")
        item["score"] = float(_score_candidate(item, camera_pose7))
        refined.append(item)
    return sorted(refined, key=lambda c: c["score"], reverse=True)


def _execute_final_pose_and_render(task, best: dict, output_dir: Path):
    arm = best["arm"]
    q = np.asarray(best["final_arm_joints"], dtype=np.float64)
    if q.shape[0] == 0:
        return {}
    if arm == "left":
        task.robot.set_arm_joints(q, np.zeros_like(q), "left")
    else:
        task.robot.set_arm_joints(q, np.zeros_like(q), "right")
    for _ in range(160):
        task.scene.step()
    actual_pose = (
        task.robot.get_left_ee_pose()
        if arm == "left"
        else task.robot.get_right_ee_pose()
    )
    actual_pose = np.asarray(actual_pose, dtype=np.float64).reshape(7)
    target_pose = np.asarray(best["pose"], dtype=np.float64).reshape(7)
    validation = {
        "executed_actual_eef_pose": actual_pose.astype(float).tolist(),
        "executed_position_error_m": float(np.linalg.norm(actual_pose[:3] - target_pose[:3])),
        "executed_quat_error_rad": _quat_distance_rad(actual_pose[3:], target_pose[3:]),
    }

    if imageio is None:
        return {"validation": validation, "images": {}}

    # Visualize the target EEF frame and the estimated object center as a small cube.
    marker_pose = sapien.Pose(np.asarray(best["pose"][:3], dtype=np.float64), np.asarray(best["pose"][3:], dtype=np.float64))
    task.robot._visualize_target(marker_pose, name=f"{arm}_observation_eef_target")

    mat = sapien.render.RenderMaterial()
    mat.set_base_color([0.1, 0.9, 0.2, 1.0])
    builder = task.scene.create_actor_builder()
    builder.add_box_visual(half_size=[0.025, 0.025, 0.025], material=mat)
    cube = builder.build_static(name="estimated_object_center_marker")
    cube.set_pose(sapien.Pose(np.asarray(best["object_center"], dtype=np.float64), [1, 0, 0, 0]))

    task._update_render()
    paths = {}
    for camera_name in ["camera_head", "observer_camera", "world_camera1"]:
        rgb = task.cameras.get_rgb_by_name(camera_name)
        if not rgb:
            continue
        path = output_dir / f"best_{arm}_{camera_name}.png"
        imageio.imwrite(path, rgb["rgb"])
        paths[camera_name] = str(path)
    return {"validation": validation, "images": paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="info_gathering_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-refine", type=int, default=30)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--coarse-mode", choices=["ik", "motion"], default="ik")
    parser.add_argument("--ik-chunk-size", type=int, default=256)
    parser.add_argument("--ik-num-seeds", type=int, default=64)
    parser.add_argument(
        "--face-options",
        default="x:+",
        help="Comma-separated local axes to point toward camera. Example: z:+,x:+ means local +Z and +X.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = count_target_press_button()
    setup_args = _load_args(task, args.config, output_dir)
    task.setup_demo(now_ep_num=0, seed=args.seed, **setup_args)

    try:
        head_pose = task.robot.head_camera.get_pose()
        camera_pose7 = np.asarray(list(head_pose.p) + list(head_pose.q), dtype=np.float64).tolist()
        camera_axes = _pose_axes(camera_pose7)
        print("head_camera_pose", _fmt_pose(camera_pose7))
        print("head_camera_forward(+X)", np.round(camera_axes["x"], 6).tolist())
        face_axis_options = []
        for item in str(args.face_options).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                axis, sign = item.split(":", 1)
            else:
                axis, sign = item, "+"
            axis = axis.strip().lower()
            if axis not in {"x", "y", "z"}:
                raise ValueError(f"invalid face axis option: {item!r}")
            face_axis_options.append((axis, 1 if sign.strip() != "-" else -1))
        if not face_axis_options:
            raise ValueError("No face axis options provided")
        print("face_axis_options", face_axis_options)

        all_successes = []
        stats = {}
        for arm in ["left", "right"]:
            candidates = list(_sample_candidates(camera_pose7, arm=arm, face_axis_options=face_axis_options))
            print(f"{arm}: sampled {len(candidates)} candidate EEF poses")
            if args.coarse_mode == "ik":
                successes = _ik_filter_batch(
                    task,
                    arm,
                    candidates,
                    chunk_size=max(int(args.ik_chunk_size), 1),
                    num_seeds=max(int(args.ik_num_seeds), 1),
                )
                print(f"{arm}: curobo IK success {len(successes)} / {len(candidates)}")
                stats[arm] = {"sampled": len(candidates), "ik_success": len(successes)}
            else:
                successes = _plan_batch(task.robot, arm, candidates, chunk_size=10)
                print(f"{arm}: curobo batch MotionGen success {len(successes)} / {len(candidates)}")
                stats[arm] = {"sampled": len(candidates), "batch_success": len(successes)}
            all_successes.extend(successes)

        refined = _refine_single(task.robot, all_successes, camera_pose7, max_refine=args.max_refine)
        print(f"single-plan verified successes: {len(refined)} / {min(len(all_successes), args.max_refine)} refined")
        if not refined:
            raise SystemExit("No verified observation EEF pose found")

        best = refined[0]
        best["pose_axes"] = _pose_axes(best["pose"])
        best["camera_pose"] = camera_pose7
        best["camera_axes"] = camera_axes
        best["grasp_contact_offset_m"] = GRASP_CONTACT_OFFSET_M
        best["local_x_semantics"] = (
            "RobotWin gripper local +X is the palm/finger extension and grasp/approach direction; "
            "the grasped object center is approximated as eef_p + 0.12 * local_x."
        )
        best["face_axis_semantics"] = (
            "face_sign * local face_axis is constrained to point from the estimated object center back to the head camera; "
            "for Astribot diagnostics, local +/-Y is the finger opening direction and local +/-Z is normal to the linkage plane."
        )
        best["alignment_dot_face_axis_to_camera"] = float(
            np.dot(
                float(best.get("face_sign", 1)) * np.asarray(best["pose_axes"][str(best.get("face_axis", "x"))]),
                _normalize(np.asarray(camera_pose7[:3]) - np.asarray(best["object_center"])),
            )
        )
        best["object_view_direction_dot_camera_forward"] = float(
            np.dot(_normalize(np.asarray(best["object_center"]) - np.asarray(camera_pose7[:3])), np.asarray(camera_axes["x"]))
        )

        render_payload = _execute_final_pose_and_render(task, best, output_dir) if args.visualize else {}
        rendered_paths = {}
        if render_payload:
            best.update(render_payload.get("validation", {}))
            rendered_paths = render_payload.get("images", {})
        if rendered_paths:
            best["rendered_paths"] = rendered_paths

        report = {
            "config": args.config,
            "seed": args.seed,
            "stats": stats,
            "best": best,
            "top_candidates": refined[:10],
        }
        report_path = output_dir / "observation_eef_pose_search.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nBEST")
        print(" arm:", best["arm"])
        print(" eef_pose_world [x,y,z,qw,qx,qy,qz]:", _fmt_pose(best["pose"]))
        print(" estimated_object_center_world:", [round(x, 6) for x in best["object_center"]])
        print(" local +X axis:", [round(x, 6) for x in best["pose_axes"]["x"]])
        print(f" face axis/sign: {best.get('face_axis')} {best.get('face_sign'):+d}")
        print(" local +Z axis:", [round(x, 6) for x in best["pose_axes"]["z"]])
        print(" alignment dot(face axis, object->camera):", round(best["alignment_dot_face_axis_to_camera"], 6))
        print(" camera optical dot:", round(best["object_view_direction_dot_camera_forward"], 6))
        print(" path steps:", best.get("single_path_steps"))
        print(" final_arm_joints:", [round(x, 6) for x in best.get("final_arm_joints", [])])
        if "executed_position_error_m" in best:
            print(" executed_position_error_m:", round(best["executed_position_error_m"], 6))
            print(" executed_quat_error_rad:", round(best["executed_quat_error_rad"], 6))
        print(" report:", report_path)
        for k, v in rendered_paths.items():
            print(f" {k} image: {v}")
    finally:
        task.close_env(clear_cache=False)


if __name__ == "__main__":
    main()
