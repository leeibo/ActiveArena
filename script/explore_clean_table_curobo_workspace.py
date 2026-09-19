import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import transforms3d as t3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs._base_task import Base_Task  # noqa: E402
from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402
from envs.utils import place_point_cyl, prepare_rotate_task_kwargs  # noqa: E402
from script.collect_data import apply_task_table_config, get_embodiment_config  # noqa: E402


class CleanTableCuroboWorkspace(Base_Task):
    """Minimal clean-table environment: table + robot + cameras, no task actors."""

    ROTATE_TABLE_SHAPE = "fan"

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()


def _load_args(config_name, env):
    config_path = Path("task_config") / f"{config_name}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args["task_name"] = "clean_table_curobo_workspace"
    args = apply_task_table_config(env, args)

    embodiment_type = args["embodiment"]
    embodiment_types = yaml.safe_load(Path(CONFIGS_PATH, "_embodiment_config.yml").read_text(encoding="utf-8"))

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment_type) != 1:
        raise ValueError("This workspace sweep expects one dual-arm embodiment entry")

    args["left_robot_file"] = embodiment_file(embodiment_type[0])
    args["right_robot_file"] = embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment_type[0])
    args["task_config"] = config_name
    args["save_path"] = "debug_data/clean_table_curobo_workspace"
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    # Force a truly clean, deterministic table even if the chosen config has
    # randomization enabled.
    args.setdefault("domain_randomization", {})
    args["domain_randomization"] = dict(args["domain_randomization"])
    args["domain_randomization"]["cluttered_table"] = False
    args["domain_randomization"]["random_background"] = False
    args["domain_randomization"]["random_table_height"] = 0
    args["domain_randomization"]["random_light"] = False
    return args


def _float_list(text):
    return [float(item) for item in str(text).split(",") if str(item).strip()]


def _str_list(text):
    return [str(item).strip() for item in str(text).split(",") if str(item).strip()]


def _safe_normalize(vec, fallback):
    arr = np.array(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        arr = np.array(fallback, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1e-9)


def _quat_from_xy(x_axis, y_hint):
    """Build a world-frame quaternion from desired local x and approximate y."""
    x_axis = _safe_normalize(x_axis, [1, 0, 0])
    y_hint = np.array(y_hint, dtype=np.float64).reshape(3)
    y_axis = y_hint - float(np.dot(y_hint, x_axis)) * x_axis
    y_axis = _safe_normalize(y_axis, [0, 1, 0] if abs(x_axis[1]) < 0.9 else [1, 0, 0])
    z_axis = _safe_normalize(np.cross(x_axis, y_axis), [0, 0, 1])
    y_axis = _safe_normalize(np.cross(z_axis, x_axis), [0, 1, 0])
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return t3d.quaternions.mat2quat(rot).tolist()


def _orientation_quat(mode, arm, theta_rad, robot_yaw, home_quat):
    """Return target TCP quaternion in world frame.

    Convention used by the frozen simulator actions: for top-down poses, local +x points
    downward so moving by -x lifts upward.
    """
    phi = float(theta_rad) + float(robot_yaw)
    radial = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=np.float64)
    tangent = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    down = -up

    mode = str(mode).lower()
    if mode == "home":
        return list(home_quat)
    if mode == "topdown_y_radial":
        return _quat_from_xy(down, radial)
    if mode == "topdown_y_inward":
        return _quat_from_xy(down, -radial)
    if mode == "topdown_y_tangent":
        return _quat_from_xy(down, tangent)
    if mode == "topdown_y_side":
        side_y = tangent if str(arm) == "left" else -tangent
        return _quat_from_xy(down, side_y)
    if mode == "horizontal_inward":
        return _quat_from_xy(-radial, up)
    if mode == "horizontal_outward":
        return _quat_from_xy(radial, up)
    if mode == "horizontal_side":
        side_x = -tangent if str(arm) == "left" else tangent
        return _quat_from_xy(side_x, up)
    raise ValueError(f"Unsupported orientation mode: {mode}")


def _plan_one(env, arm, target_pose):
    planner = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
    try:
        result = planner(target_pose)
    except Exception as exc:  # noqa: BLE001 - diagnostic script records all planner errors.
        return {
            "status": "Exception",
            "exception": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(result, dict):
        return {"status": "InvalidResult"}
    return {"status": str(result.get("status", "MissingStatus"))}


def _summarize(rows):
    total = len(rows)
    success = sum(1 for row in rows if row["success"])
    return {
        "total": total,
        "success": success,
        "success_rate": float(success / total) if total else 0.0,
    }


def _group_summary(rows, keys, top_k=20):
    groups = defaultdict(list)
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        groups[group_key].append(row)
    result = []
    for group_key, group_rows in groups.items():
        item = {key: group_key[idx] for idx, key in enumerate(keys)}
        item.update(_summarize(group_rows))
        result.append(item)
    result.sort(key=lambda item: (item["success_rate"], item["success"], -item["total"]), reverse=True)
    return result[:top_k]


def run_sweep(args):
    env = CleanTableCuroboWorkspace()
    output = {
        "config": args.config,
        "seed": int(args.seed),
        "table_height_bias": float(args.table_height_bias),
        "rows": [],
        "summary": {},
    }
    try:
        env_args = _load_args(args.config, env)
        env.setup_demo(
            now_ep_num=0,
            seed=int(args.seed),
            table_height_bias=float(args.table_height_bias),
            **env_args,
        )
        env.robot.move_to_homestate()
        env.robot.set_origin_endpose()
        env.robot_root_xy, env.robot_yaw = env._get_robot_root_xy_yaw()

        table_top_z = float(getattr(env, "rotate_table_top_z", 0.74))
        output["table_top_z"] = table_top_z
        r_values = _float_list(args.r_values)
        theta_deg_values = _float_list(args.theta_deg_values)
        z_offsets = _float_list(args.z_offsets)
        arms = _str_list(args.arms)
        modes = _str_list(args.orientation_modes)

        home_quat = {
            "left": list(env.robot.get_left_ee_pose()[-4:]),
            "right": list(env.robot.get_right_ee_pose()[-4:]),
        }

        total = len(r_values) * len(theta_deg_values) * len(z_offsets) * len(arms) * len(modes)
        done = 0
        for arm in arms:
            for mode in modes:
                for r in r_values:
                    for theta_deg in theta_deg_values:
                        theta_rad = np.deg2rad(float(theta_deg))
                        for z_offset in z_offsets:
                            z = table_top_z + float(z_offset)
                            xyz = place_point_cyl(
                                [float(r), float(theta_rad), z],
                                robot_root_xy=env.robot_root_xy,
                                robot_yaw_rad=env.robot_yaw,
                                ret="list",
                            )
                            quat = _orientation_quat(mode, arm, theta_rad, env.robot_yaw, home_quat[arm])
                            target_pose = xyz + quat
                            plan_result = _plan_one(env, arm, target_pose)
                            status = plan_result["status"]
                            row = {
                                "arm": arm,
                                "orientation": mode,
                                "r": round(float(r), 4),
                                "theta_deg": round(float(theta_deg), 4),
                                "z_offset": round(float(z_offset), 4),
                                "z": round(float(z), 4),
                                "status": status,
                                "success": status == "Success",
                            }
                            if "exception" in plan_result:
                                row["exception"] = plan_result["exception"]
                            output["rows"].append(row)
                            done += 1
                            if args.progress_every > 0 and done % int(args.progress_every) == 0:
                                print(
                                    json.dumps(
                                        {
                                            "progress": {
                                                "done": done,
                                                "total": total,
                                                "last": row,
                                            }
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )

        rows = output["rows"]
        output["summary"]["overall"] = _summarize(rows)
        output["summary"]["by_arm"] = _group_summary(rows, ["arm"], top_k=20)
        output["summary"]["by_orientation"] = _group_summary(rows, ["orientation"], top_k=50)
        output["summary"]["by_arm_orientation"] = _group_summary(rows, ["arm", "orientation"], top_k=100)
        output["summary"]["by_r"] = _group_summary(rows, ["r"], top_k=100)
        output["summary"]["by_theta"] = _group_summary(rows, ["theta_deg"], top_k=100)
        output["summary"]["by_z_offset"] = _group_summary(rows, ["z_offset"], top_k=100)
        output["summary"]["by_r_theta"] = _group_summary(rows, ["r", "theta_deg"], top_k=100)
        output["summary"]["by_r_z"] = _group_summary(rows, ["r", "z_offset"], top_k=100)
        output["summary"]["best_pose_bins"] = _group_summary(
            rows,
            ["arm", "orientation", "r", "theta_deg", "z_offset"],
            top_k=int(args.top_k),
        )
    except Exception as exc:  # noqa: BLE001
        output["exception"] = f"{type(exc).__name__}: {exc}"
        if args.verbose_failure:
            traceback.print_exc()
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            if args.verbose_failure:
                traceback.print_exc()
    return output


def main():
    parser = argparse.ArgumentParser(description="Sweep cuRobo reachability on a clean table.")
    parser.add_argument("--config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--table-height-bias",
        type=float,
        default=0.0,
        help="additive bias to the default table top height 0.74m; e.g. -0.04 tests top_z=0.70",
    )
    parser.add_argument("--arms", default="left,right")
    parser.add_argument(
        "--orientation-modes",
        default="home,topdown_y_radial,topdown_y_inward,topdown_y_tangent,topdown_y_side,horizontal_inward,horizontal_outward,horizontal_side",
    )
    parser.add_argument("--r-values", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--theta-deg-values", default="-70,-50,-30,-10,0,10,30,50,70")
    parser.add_argument("--z-offsets", default="0.04,0.08,0.12,0.16,0.22")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--verbose-failure", action="store_true")
    args = parser.parse_args()

    output = run_sweep(args)
    print("===CLEAN_TABLE_CUROBO_WORKSPACE_JSON===")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("ACTIVEARENA_SKIP_RENDER_TEST", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_ANNOTATED_VIDEO", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_INSTRUCTIONS", "1")
    main()
