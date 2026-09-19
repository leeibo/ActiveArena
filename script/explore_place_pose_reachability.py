import argparse
import itertools
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.utils import ArmTag, place_point_cyl, world_to_robot  # noqa: E402
from script.analyze_low_success_tasks import _load_args, _task_class  # noqa: E402


SCAN_R = 0.62
SCAN_Z_BASE = 0.88
SCAN_JOINT = "astribot_torso_joint_2"


def _float_list(text):
    return [float(item) for item in str(text).split(",") if str(item).strip()]


def _axis_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _as_status_list(status):
    if isinstance(status, np.ndarray):
        return [str(item) for item in status.tolist()]
    if isinstance(status, (list, tuple)):
        return [str(item) for item in status]
    return [str(status)]


def _status_is_success(status):
    return str(status).lower() == "success"


def _arm_qpos_full_after_plan(env, arm_tag, active_qpos):
    active_qpos = np.asarray(active_qpos, dtype=np.float64).reshape(-1)
    if arm_tag == "left":
        entity = env.robot.left_entity
        active_names = list(env.robot.left_arm_joints_name)
    else:
        entity = env.robot.right_entity
        active_names = list(env.robot.right_arm_joints_name)

    full_qpos = np.array(entity.get_qpos(), dtype=np.float64).reshape(-1)
    if active_qpos.shape[0] == full_qpos.shape[0]:
        return [float(value) for value in active_qpos.tolist()]

    all_names = [joint.get_name() for joint in entity.get_active_joints()]
    for idx, joint_name in enumerate(active_names[: active_qpos.shape[0]]):
        if joint_name in all_names:
            full_qpos[all_names.index(joint_name)] = active_qpos[idx]
    # cuRobo's current plan_path builds torch tensors without an explicit dtype.
    # Return plain Python floats so the tensor is float32-compatible instead of
    # inheriting numpy.float64 and failing with "expected Float but found Double".
    return [float(value) for value in full_qpos.tolist()]


def _plan_one(env, arm_tag, target_pose, last_qpos=None, constraint_pose=None):
    planner = env.robot.left_plan_path if arm_tag == "left" else env.robot.right_plan_path
    try:
        result = planner(target_pose, constraint_pose=constraint_pose, last_qpos=last_qpos)
    except Exception as exc:  # noqa: BLE001 - diagnostic script records planner exceptions.
        return {"status": "Exception", "exception": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        return {"status": "InvalidResult"}
    return result


def _plan_pre_batch(env, arm_tag, target_poses):
    if not target_poses:
        return []
    planner = env.robot.left_plan_multi_path if arm_tag == "left" else env.robot.right_plan_multi_path
    try:
        result = planner(target_poses)
    except Exception:
        # Some planner builds are less stable for batch calls; fall back to one
        # target at a time so the sweep can still complete.
        return [_plan_one(env, arm_tag, pose) for pose in target_poses]
    if not isinstance(result, dict):
        return [{"status": "InvalidResult"} for _ in target_poses]

    statuses = _as_status_list(result.get("status", []))
    if len(statuses) != len(target_poses):
        return [_plan_one(env, arm_tag, pose) for pose in target_poses]

    positions = result.get("position")
    split_positions = [None] * len(target_poses)
    if positions is not None:
        pos_arr = np.asarray(positions, dtype=np.float64)
        if pos_arr.ndim == 3 and pos_arr.shape[0] == len(target_poses):
            split_positions = [pos_arr[idx] for idx in range(len(target_poses))]

    return [
        {
            "status": statuses[idx],
            **({"position": split_positions[idx]} if split_positions[idx] is not None else {}),
        }
        for idx in range(len(target_poses))
    ]


def _prepare_a2b_pre_place(env, task_name):
    source_key = env.search_and_focus_rotate_subtask(
        1,
        scan_r=SCAN_R,
        scan_z=SCAN_Z_BASE + env.table_z_bias,
        joint_name_prefer=SCAN_JOINT,
    )
    object_theta = float(env._pose_to_cyl(env.object.get_pose())[1])
    arm_tag = ArmTag("left" if object_theta >= 0.0 else "right")
    env.enter_rotate_action_stage(1, focus_object_key=(source_key or "A"))
    env.move(env.grasp_actor(env.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
    env._set_carried_object_keys(["A"])
    env.move(env.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
    env.complete_rotate_subtask(1, carried_after=["A"])

    target_key = env.search_and_focus_rotate_subtask(
        2,
        scan_r=SCAN_R,
        scan_z=SCAN_Z_BASE + env.table_z_bias,
        joint_name_prefer=SCAN_JOINT,
    )
    env.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))

    to_left = task_name == "place_a2b_left_rotate_view"
    return {
        "actor": env.object,
        "arm_tag": str(arm_tag),
        "target_actor": env.target_object,
        "to_left": to_left,
    }


def _prepare_stapler_pre_place(env, _task_name):
    stapler_key = env.search_and_focus_rotate_subtask(
        1,
        scan_r=SCAN_R,
        scan_z=SCAN_Z_BASE + env.table_z_bias,
        joint_name_prefer=SCAN_JOINT,
    )
    stapler_cyl = world_to_robot(env.stapler.get_pose().p.tolist(), env.robot_root_xy, env.robot_yaw)
    arm_tag = ArmTag("left" if stapler_cyl[1] >= 0 else "right")
    env.enter_rotate_action_stage(1, focus_object_key=(stapler_key or "A"))
    env.move(env.grasp_actor(env.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1))
    env._set_carried_object_keys(["A"])
    env.move(env.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
    env.complete_rotate_subtask(1, carried_after=["A"])

    pad_key = env.search_and_focus_rotate_subtask(
        2,
        scan_r=SCAN_R,
        scan_z=SCAN_Z_BASE + env.table_z_bias,
        joint_name_prefer=SCAN_JOINT,
    )
    env.enter_rotate_action_stage(2, focus_object_key=(pad_key or "B"))
    return {
        "actor": env.stapler,
        "arm_tag": str(arm_tag),
        "target_actor": env.pad,
        "to_left": None,
    }


def _prepare_pre_place(env, task_name):
    if task_name in {"place_a2b_left_rotate_view", "place_a2b_right_rotate_view"}:
        return _prepare_a2b_pre_place(env, task_name)
    if task_name == "move_stapler_pad_rotate_view":
        return _prepare_stapler_pre_place(env, task_name)
    raise ValueError(f"Unsupported task for place-pose reachability exploration: {task_name}")


def _a2b_target_pose(env, prep, arc_dis, r_bias, tangent_bias, z_bias):
    target_cyl = world_to_robot(prep["target_actor"].get_pose().p.tolist(), env.robot_root_xy, env.robot_yaw)
    base_r = float(target_cyl[0])
    r = max(base_r + float(r_bias), 1e-6)
    sign = 1.0 if prep["to_left"] else -1.0
    theta = float(target_cyl[1]) + sign * float(arc_dis + tangent_bias) / max(r, 1e-6)
    z = float(target_cyl[2]) + float(z_bias)
    return place_point_cyl([r, theta, z], robot_root_xy=env.robot_root_xy, robot_yaw_rad=env.robot_yaw, ret="list")


def _stapler_target_pose(env, prep, arc_dis, r_bias, tangent_bias, z_bias):
    pad_pose = prep["target_actor"].get_pose()
    pad_cyl = world_to_robot(pad_pose.p.tolist(), env.robot_root_xy, env.robot_yaw)
    r = max(float(pad_cyl[0]) + float(r_bias), 1e-6)
    theta = float(pad_cyl[1]) + float(tangent_bias) / max(r, 1e-6)
    z = float(pad_cyl[2]) + float(z_bias)
    xyz = place_point_cyl([r, theta, z], robot_root_xy=env.robot_root_xy, robot_yaw_rad=env.robot_yaw, ret="list")
    # Keep the current stapler-on-pad orientation convention.
    return xyz + [0.707, 0.0, 0.0, 0.707]


def _target_pose(env, task_name, prep, candidate):
    if task_name in {"place_a2b_left_rotate_view", "place_a2b_right_rotate_view"}:
        return _a2b_target_pose(
            env,
            prep,
            candidate["arc_dis"],
            candidate["r_bias"],
            candidate["tangent_bias"],
            candidate["z_bias"],
        )
    return _stapler_target_pose(
        env,
        prep,
        candidate["arc_dis"],
        candidate["r_bias"],
        candidate["tangent_bias"],
        candidate["z_bias"],
    )


def _semantic_ok(env, task_name, prep, target_pose):
    xyz = np.array(target_pose[:3], dtype=np.float64)
    target_xyz = np.array(prep["target_actor"].get_pose().p.tolist(), dtype=np.float64)
    if task_name in {"place_a2b_left_rotate_view", "place_a2b_right_rotate_view"}:
        distance = float(np.linalg.norm(xyz[:2] - target_xyz[:2]))
        obj_cyl = world_to_robot(xyz.tolist(), env.robot_root_xy, env.robot_yaw)
        tgt_cyl = world_to_robot(target_xyz.tolist(), env.robot_root_xy, env.robot_yaw)
        theta_diff = float(env._wrap_to_pi(obj_cyl[1] - tgt_cyl[1]))
        radial_diff = abs(float(obj_cyl[0] - tgt_cyl[0]))
        if task_name == "place_a2b_left_rotate_view":
            side_ok = theta_diff > 0.02
        else:
            side_ok = theta_diff < -0.02
        return bool(distance < 0.2 and distance > 0.08 and side_ok and radial_diff < 0.08)

    eps = np.array([0.02, 0.02, 0.01], dtype=np.float64)
    return bool(np.all(np.abs(xyz - target_xyz) < eps))


def _build_candidates(task_name, args):
    if task_name in {"place_a2b_left_rotate_view", "place_a2b_right_rotate_view"}:
        arc_vals = _float_list(args.a2b_arc_dis)
        r_vals = _float_list(args.a2b_r_bias)
        tangent_vals = _float_list(args.a2b_tangent_bias)
        z_vals = _float_list(args.a2b_z_bias)
        pre_vals = _float_list(args.pre_dis)
        dis_vals = _float_list(args.dis)
        axes = _axis_list(args.pre_dis_axis)
        constrains = _axis_list(args.constrain)
        align_axes = _axis_list(args.align_axis)
    else:
        arc_vals = [0.0]
        r_vals = _float_list(args.stapler_r_bias)
        tangent_vals = _float_list(args.stapler_tangent_bias)
        z_vals = _float_list(args.stapler_z_bias)
        pre_vals = _float_list(args.pre_dis)
        dis_vals = _float_list(args.stapler_dis if args.stapler_dis else args.dis)
        axes = _axis_list(args.pre_dis_axis)
        constrains = _axis_list(args.constrain)
        align_axes = _axis_list(args.align_axis)

    candidates = []
    seen = set()
    for arc_dis, r_bias, tangent_bias, z_bias, pre_dis, dis, axis, constrain, align_axis in itertools.product(
        arc_vals,
        r_vals,
        tangent_vals,
        z_vals,
        pre_vals,
        dis_vals,
        axes,
        constrains,
        align_axes,
    ):
        # align_axis is ignored by get_place_pose(constrain="free").  Collapse
        # those duplicates so broad orientation sweeps stay tractable.
        normalized_align_axis = align_axis if str(constrain).lower() == "align" else "none"
        candidate = {
            "arc_dis": float(arc_dis),
            "r_bias": float(r_bias),
            "tangent_bias": float(tangent_bias),
            "z_bias": float(z_bias),
            "pre_dis": float(pre_dis),
            "dis": float(dis),
            "pre_dis_axis": axis,
            "constrain": constrain,
            "align_axis": normalized_align_axis,
        }
        key = _candidate_key(_short_candidate(candidate))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _world_align_axis(env, prep, name):
    name = str(name).strip().lower()
    if name in {"none", "default", ""}:
        return None
    if name in {"x", "+x", "world_x"}:
        return [1.0, 0.0, 0.0]
    if name in {"-x", "negx", "world_negx"}:
        return [-1.0, 0.0, 0.0]
    if name in {"y", "+y", "world_y"}:
        return [0.0, 1.0, 0.0]
    if name in {"-y", "negy", "world_negy"}:
        return [0.0, -1.0, 0.0]
    target_xyz = np.array(prep["target_actor"].get_pose().p.tolist(), dtype=np.float64)
    root = np.array(env.robot_root_xy, dtype=np.float64)
    radial = np.array([target_xyz[0] - root[0], target_xyz[1] - root[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(radial))
    if norm < 1e-9:
        radial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        radial /= norm
    tangent = np.array([-radial[1], radial[0], 0.0], dtype=np.float64)
    if name in {"radial", "+radial"}:
        return radial.tolist()
    if name in {"-radial", "negradial"}:
        return (-radial).tolist()
    if name in {"tangent", "+tangent"}:
        return tangent.tolist()
    if name in {"-tangent", "negtangent"}:
        return (-tangent).tolist()
    raise ValueError(f"Unsupported align axis: {name}")


def _candidate_key(candidate):
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def _short_candidate(candidate):
    return {
        key: (round(value, 4) if isinstance(value, float) else value)
        for key, value in candidate.items()
    }


def run_seed(task_name, config, seed, candidates, verbose_failure=False, use_pre_batch=False):
    cls = _task_class(task_name)
    env = cls()
    seed_result = {
        "task": task_name,
        "seed": int(seed),
        "pre_place_ready": False,
        "prepare_plan_success": False,
        "prepare_exception": None,
        "arm_tag": None,
        "candidate_results": [],
    }
    try:
        args = _load_args(task_name, config, env)
        env.setup_demo(now_ep_num=0, seed=int(seed), **args)
        prep = _prepare_pre_place(env, task_name)
        seed_result["prepare_plan_success"] = bool(getattr(env, "plan_success", False))
        seed_result["arm_tag"] = prep.get("arm_tag")
        if not seed_result["prepare_plan_success"]:
            return seed_result
        seed_result["pre_place_ready"] = True

        pre_targets = []
        candidate_runtime = []
        for candidate in candidates:
            target_pose = _target_pose(env, task_name, prep, candidate)
            semantic_ok = _semantic_ok(env, task_name, prep, target_pose)
            pre_pose = env.get_place_pose(
                prep["actor"],
                arm_tag=prep["arm_tag"],
                target_pose=target_pose,
                pre_dis=candidate["pre_dis"],
                pre_dis_axis=candidate["pre_dis_axis"],
                constrain=candidate["constrain"],
                align_axis=_world_align_axis(env, prep, candidate["align_axis"]),
            )
            final_pose = env.get_place_pose(
                prep["actor"],
                arm_tag=prep["arm_tag"],
                target_pose=target_pose,
                pre_dis=candidate["dis"],
                pre_dis_axis=candidate["pre_dis_axis"],
                constrain=candidate["constrain"],
                align_axis=_world_align_axis(env, prep, candidate["align_axis"]),
            )
            pre_targets.append(pre_pose)
            candidate_runtime.append(
                {
                    "candidate": candidate,
                    "target_pose": target_pose,
                    "semantic_ok": semantic_ok,
                    "pre_pose": pre_pose,
                    "final_pose": final_pose,
                }
            )

        if use_pre_batch:
            pre_results = _plan_pre_batch(env, prep["arm_tag"], pre_targets)
        else:
            pre_results = [_plan_one(env, prep["arm_tag"], pose) for pose in pre_targets]
        for idx, item in enumerate(candidate_runtime):
            candidate = item["candidate"]
            pre_result = pre_results[idx] if idx < len(pre_results) else {"status": "MissingPreResult"}
            pre_status = str(pre_result.get("status", "MissingPreStatus"))
            final_status = "SkippedPreFail"
            final_exception = None
            if _status_is_success(pre_status):
                position = pre_result.get("position")
                if position is not None:
                    pre_final_q = np.asarray(position, dtype=np.float64)[-1]
                    last_qpos = _arm_qpos_full_after_plan(env, prep["arm_tag"], pre_final_q)
                else:
                    # Fallback: re-plan the pre pose singly to obtain the final
                    # joint state for the final placement reachability check.
                    single_pre = _plan_one(env, prep["arm_tag"], item["pre_pose"])
                    if _status_is_success(single_pre.get("status")) and single_pre.get("position") is not None:
                        pre_final_q = np.asarray(single_pre["position"], dtype=np.float64)[-1]
                        last_qpos = _arm_qpos_full_after_plan(env, prep["arm_tag"], pre_final_q)
                    else:
                        last_qpos = None
                if last_qpos is not None:
                    final_result = _plan_one(env, prep["arm_tag"], item["final_pose"], last_qpos=last_qpos)
                    final_status = str(final_result.get("status", "MissingFinalStatus"))
                    final_exception = final_result.get("exception")

            seed_result["candidate_results"].append(
                {
                    "candidate": _short_candidate(candidate),
                    "semantic_ok": bool(item["semantic_ok"]),
                    "pre_status": pre_status,
                    "final_status": final_status,
                    "place_plan_success": bool(
                        item["semantic_ok"]
                        and _status_is_success(pre_status)
                        and _status_is_success(final_status)
                    ),
                    "final_exception": final_exception,
                }
            )
    except Exception as exc:  # noqa: BLE001
        seed_result["prepare_exception"] = f"{type(exc).__name__}: {exc}"
        if verbose_failure:
            traceback.print_exc()
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            if verbose_failure:
                traceback.print_exc()
    return seed_result


def summarize(task_name, candidates, seed_results):
    by_key = {}
    for candidate in candidates:
        by_key[_candidate_key(_short_candidate(candidate))] = {
            "task": task_name,
            "candidate": _short_candidate(candidate),
            "ready_seeds": 0,
            "semantic_ok": 0,
            "pre_success": 0,
            "final_success": 0,
            "place_plan_success": 0,
            "pre_fail_status": defaultdict(int),
            "final_fail_status": defaultdict(int),
        }

    for seed_result in seed_results:
        if not seed_result.get("pre_place_ready"):
            continue
        for row in seed_result.get("candidate_results", []):
            key = _candidate_key(row["candidate"])
            item = by_key[key]
            item["ready_seeds"] += 1
            if row["semantic_ok"]:
                item["semantic_ok"] += 1
            if _status_is_success(row["pre_status"]):
                item["pre_success"] += 1
            else:
                item["pre_fail_status"][row["pre_status"]] += 1
            if _status_is_success(row["final_status"]):
                item["final_success"] += 1
            else:
                item["final_fail_status"][row["final_status"]] += 1
            if row["place_plan_success"]:
                item["place_plan_success"] += 1

    summaries = []
    for item in by_key.values():
        item["pre_fail_status"] = dict(item["pre_fail_status"])
        item["final_fail_status"] = dict(item["final_fail_status"])
        summaries.append(item)
    summaries.sort(
        key=lambda row: (
            row["place_plan_success"],
            row["final_success"],
            row["pre_success"],
            -abs(row["candidate"].get("z_bias", 0.0)),
            -abs(row["candidate"].get("r_bias", 0.0)),
            -abs(row["candidate"].get("tangent_bias", 0.0)),
        ),
        reverse=True,
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Explore cuRobo reachable placement pose/action space.")
    parser.add_argument("--config", default="demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "place_a2b_left_rotate_view",
            "place_a2b_right_rotate_view",
            "move_stapler_pad_rotate_view",
        ],
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=8, help="exclusive upper seed bound")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--verbose-failure", action="store_true")
    parser.add_argument(
        "--use-pre-batch",
        action="store_true",
        help="use cuRobo batch planning for pre-place poses; disabled by default because some CUDA builds reject dynamic goal batches",
    )

    parser.add_argument("--a2b-arc-dis", default="0.09,0.11,0.13,0.15,0.17")
    parser.add_argument("--a2b-r-bias", default="-0.04,-0.02,0.0,0.02,0.04")
    parser.add_argument("--a2b-tangent-bias", default="0.0")
    parser.add_argument("--a2b-z-bias", default="0.0,0.015,0.03")

    parser.add_argument("--stapler-r-bias", default="-0.01,0.0,0.01")
    parser.add_argument("--stapler-tangent-bias", default="-0.01,0.0,0.01")
    parser.add_argument("--stapler-z-bias", default="0.0,0.004,0.008")

    parser.add_argument("--pre-dis", default="0.06,0.10")
    parser.add_argument("--dis", default="0.0,0.02")
    parser.add_argument("--stapler-dis", default="0.0")
    parser.add_argument("--pre-dis-axis", default="grasp,fp")
    parser.add_argument("--constrain", default="free")
    parser.add_argument("--align-axis", default="none")
    args = parser.parse_args()

    all_output = {}
    for task_name in args.tasks:
        candidates = _build_candidates(task_name, args)
        seed_results = []
        print(
            json.dumps(
                {
                    "task_start": {
                        "task": task_name,
                        "candidate_count": len(candidates),
                        "start": args.start,
                        "end": args.end,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for seed in range(args.start, args.end):
            result = run_seed(
                task_name,
                args.config,
                seed,
                candidates,
                verbose_failure=args.verbose_failure,
                use_pre_batch=args.use_pre_batch,
            )
            seed_results.append(result)
            compact = {
                "task": task_name,
                "seed": seed,
                "pre_place_ready": result.get("pre_place_ready"),
                "prepare_plan_success": result.get("prepare_plan_success"),
                "arm_tag": result.get("arm_tag"),
                "prepare_exception": result.get("prepare_exception"),
            }
            if result.get("pre_place_ready"):
                compact["candidate_success"] = sum(
                    1 for row in result.get("candidate_results", []) if row.get("place_plan_success")
                )
                compact["candidate_count"] = len(result.get("candidate_results", []))
            print(json.dumps({"seed_result": compact}, ensure_ascii=False), flush=True)

        summaries = summarize(task_name, candidates, seed_results)
        top = summaries[: max(0, int(args.top_k))]
        print(json.dumps({"task_summary": {"task": task_name, "top": top}}, ensure_ascii=False), flush=True)
        all_output[task_name] = {
            "candidates": [_short_candidate(candidate) for candidate in candidates],
            "seed_results": seed_results,
            "summary": summaries,
        }

    print("===PLACE_POSE_REACHABILITY_JSON===")
    print(json.dumps(all_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("ACTIVEARENA_SKIP_RENDER_TEST", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_ANNOTATED_VIDEO", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_INSTRUCTIONS", "1")
    main()
