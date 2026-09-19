import argparse
import ast
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from script.collect_data import apply_task_table_config, get_embodiment_config
from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def _load_args(task_name, task_config, env):
    config_path = Path("task_config") / f"{task_config}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args["task_name"] = task_name
    args = apply_task_table_config(env, args)

    embodiment_type = args["embodiment"]
    embodiment_types = yaml.safe_load(Path(CONFIGS_PATH, "_embodiment_config.yml").read_text(encoding="utf-8"))

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment_type) != 1:
        raise ValueError("This diagnostic currently expects one dual-arm embodiment entry")

    args["left_robot_file"] = embodiment_file(embodiment_type[0])
    args["right_robot_file"] = embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = str(embodiment_type[0])
    args["task_config"] = task_config
    args["save_path"] = "debug_data/low_success_diagnostics"
    args["need_plan"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    return args


def _task_class(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)


def _parse_attr_override(text):
    if "=" not in text:
        raise ValueError(f"Attribute override must be NAME=VALUE, got: {text}")
    name, value_text = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Attribute override has empty name: {text}")
    try:
        value = ast.literal_eval(value_text)
    except Exception:
        value = value_text
    return name, value


def _apply_attr_overrides(cls, attr_overrides):
    for attr_name, attr_value in attr_overrides or []:
        setattr(cls, attr_name, attr_value)


def _summarize_actions(actions_by_arm):
    if actions_by_arm is None:
        return None
    arm_tag, actions = actions_by_arm
    return {
        "arm": str(arm_tag),
        "actions": [getattr(action, "action", None) for action in (actions or [])],
    }


def _summarize_env_state(env):
    state = {
        "current_subtask_idx": getattr(env, "current_subtask_idx", None),
        "current_stage": getattr(env, "current_stage", None),
        "search_cursor_state_index": getattr(env, "search_cursor_state_index", None),
        "search_cursor_layer": getattr(env, "search_cursor_layer", None),
    }
    return {key: value for key, value in state.items() if value is not None}


def _safe_pose_xyz(actor):
    try:
        return [float(value) for value in actor.get_pose().p[:3].tolist()]
    except Exception:
        return None


def _safe_robot_cyl(env, xyz):
    if xyz is None:
        return None
    try:
        from envs.utils import world_to_robot

        return [
            float(value)
            for value in world_to_robot(
                xyz,
                getattr(env, "robot_root_xy", [0.0, 0.0]),
                getattr(env, "robot_yaw", 0.0),
            )
        ]
    except Exception:
        return None


def _summarize_objects(env):
    object_registry = getattr(env, "object_registry", {}) or {}
    object_layers = getattr(env, "object_layers", {}) or {}
    result = {}
    for key, actor in object_registry.items():
        xyz = _safe_pose_xyz(actor)
        result[str(key)] = {
            "layer": object_layers.get(str(key), None),
            "xyz": xyz,
            "robot_cyl": _safe_robot_cyl(env, xyz),
        }
    return result


def run_seed(
    task_name,
    task_config,
    seed,
    verbose_failure=False,
    trace_moves=False,
    trace_objects=False,
    attr_overrides=None,
):
    cls = _task_class(task_name)
    _apply_attr_overrides(cls, attr_overrides)
    env = cls()
    args = _load_args(task_name, task_config, env)
    result = {
        "task": task_name,
        "seed": int(seed),
        "ok": False,
        "plan_success": False,
        "check_success": False,
        "exception_type": None,
        "exception_message": None,
        "info": None,
        "move_count": 0,
        "first_failed_move": None,
        "final_state": None,
        "objects": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=int(seed), **args)
        if trace_moves:
            original_move = env.move

            def traced_move(actions_by_arm1, actions_by_arm2=None, save_freq=-1):
                result["move_count"] += 1
                move_index = result["move_count"]
                plan_before = bool(getattr(env, "plan_success", False))
                move_result = original_move(actions_by_arm1, actions_by_arm2, save_freq=save_freq)
                plan_after = bool(getattr(env, "plan_success", False))
                if result["first_failed_move"] is None and (not move_result or not plan_after):
                    result["first_failed_move"] = {
                        "index": move_index,
                        "plan_before": plan_before,
                        "plan_after": plan_after,
                        "move_result": bool(move_result),
                        "actions_by_arm1": _summarize_actions(actions_by_arm1),
                        "actions_by_arm2": _summarize_actions(actions_by_arm2),
                        "env_state": _summarize_env_state(env),
                    }
                return move_result

            env.move = traced_move
        info = env.play_once()
        result["plan_success"] = bool(getattr(env, "plan_success", False))
        result["check_success"] = bool(env.check_success()) if result["plan_success"] else False
        result["ok"] = bool(result["plan_success"] and result["check_success"])
        result["info"] = info.get("info", None) if isinstance(info, dict) else None
        if trace_moves and not result["ok"]:
            result["final_state"] = _summarize_env_state(env)
        if trace_objects:
            result["objects"] = _summarize_objects(env)
    except Exception as exc:  # noqa: BLE001 - diagnostic script must record all failure modes.
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        if verbose_failure:
            traceback.print_exc()
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            if verbose_failure:
                traceback.print_exc()
    return result


def main():
    parser = argparse.ArgumentParser(description="Sweep seeds for DYJ low-success task diagnostics.")
    parser.add_argument("--config", default="demo_randomized_r5_random_env_dyj_ep1_upper20_ep1cam_observer")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20, help="exclusive upper seed bound")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--verbose-failure", action="store_true")
    parser.add_argument("--trace-moves", action="store_true", help="record move count and the first failed move")
    parser.add_argument("--trace-objects", action="store_true", help="record object registry poses and declared layers")
    parser.add_argument(
        "--set-attr",
        action="append",
        default=[],
        help="temporarily set task class attribute, e.g. --set-attr A2B_RLIM='(0.35, 0.45)'",
    )
    args = parser.parse_args()
    attr_overrides = [_parse_attr_override(item) for item in args.set_attr]

    for task_name in args.tasks:
        first_success = None
        attempts = 0
        for seed in range(args.start, args.end):
            result = run_seed(
                task_name,
                args.config,
                seed,
                verbose_failure=args.verbose_failure,
                trace_moves=args.trace_moves,
                trace_objects=args.trace_objects,
                attr_overrides=attr_overrides,
            )
            attempts += 1
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["ok"]:
                first_success = seed
                if args.stop_on_success:
                    break
        summary = {
            "task": task_name,
            "start": args.start,
            "end": args.end,
            "attempts": attempts,
            "first_success": first_success,
        }
        print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # Keep renderer sanity and post-processing out of diagnostic sweeps unless the caller opts in elsewhere.
    os.environ.setdefault("ACTIVEARENA_SKIP_RENDER_TEST", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_ANNOTATED_VIDEO", "1")
    os.environ.setdefault("ACTIVEARENA_SKIP_INSTRUCTIONS", "1")
    main()
