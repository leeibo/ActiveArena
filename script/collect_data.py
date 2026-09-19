import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from contextlib import contextmanager
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

DEFAULT_MAX_SEED_TRIES = 1000
SEED_LIMIT_EXCEEDED_EXIT_CODE = 2
COLLECTION_FAILED_EXIT_CODE = 1


def _env_flag(name, default=False):
    raw = _runtime_env(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_env(name, default=None):
    """Read the ActiveArena name first, then the deprecated RoboTwin alias."""
    legacy = name.replace("ACTIVEARENA_", "ROBOTWIN_", 1)
    if name in os.environ:
        return os.environ[name]
    return os.environ.get(legacy, default)


def _debug_timing_enabled():
    return _env_flag("ACTIVEARENA_DEBUG_TIMING", default=True)


def _debug_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _debug_log(message, **fields):
    if not _debug_timing_enabled():
        return
    field_text = ""
    if fields:
        field_text = " " + " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[DebugTiming {_debug_now()}] {message}{field_text}", flush=True)


@contextmanager
def _debug_stage(stage, **fields):
    start = time.perf_counter()
    _debug_log(f"START {stage}", **fields)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        _debug_log(
            f"FAIL {stage}",
            elapsed=f"{elapsed:.3f}s",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **fields,
        )
        raise
    else:
        elapsed = time.perf_counter() - start
        _debug_log(f"END {stage}", elapsed=f"{elapsed:.3f}s", **fields)


def class_decorator(task_name):
    with _debug_stage("class_decorator.import_task", task=task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        with _debug_stage("class_decorator.instantiate_task", task=task_name):
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with _debug_stage("get_embodiment_config", path=robot_config_file):
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _sanitize_tag(text):
    return str(text).strip().replace(" ", "_")


def infer_difficulty_tag(args):
    custom_tag = args.get("difficulty_tag", None)
    if custom_tag is not None and str(custom_tag).strip():
        return _sanitize_tag(custom_tag)

    fan_angle_deg = args.get("fan_angle_deg", None)
    if fan_angle_deg is None:
        return "unknown"

    fan_angle = float(fan_angle_deg)
    fan_angle_int = int(round(fan_angle))
    if fan_angle <= 170.0:
        level = "easy"
    elif fan_angle <= 220.0:
        level = "medium"
    else:
        level = "hard"
    return f"{level}_fan{fan_angle_int}"


def apply_task_table_config(task, args):
    """Expand the task-selected base table profile into the runtime config."""
    args = dict(args)
    has_task_table_shape = hasattr(task, "ROTATE_TABLE_SHAPE")
    table_shape = str(getattr(task, "ROTATE_TABLE_SHAPE", args.get("table_shape", "rect"))).lower()
    profile_key = str(getattr(task, "ROTATE_TABLE_CONFIG_KEY", table_shape))
    profile_shape = "fan" if table_shape == "fan_double" and profile_key == "fan" else table_shape
    table_configs = args.get("task_table_configs", {}) or {}
    if not isinstance(table_configs, dict):
        raise TypeError("task_table_configs must be a mapping")

    table_config = table_configs.get(profile_key)
    if table_config is None and table_configs and has_task_table_shape:
        task_name = getattr(task, "task_name", None) or getattr(task, "__name__", task.__class__.__name__)
        raise KeyError(f"Missing task_table_configs.{profile_key} for {task_name}")
    if table_config is None:
        table_config = {}
    if not isinstance(table_config, dict):
        raise TypeError(f"task_table_configs.{profile_key} must be a mapping")

    config_shape = table_config.get("table_shape", None)
    if config_shape is not None and str(config_shape).lower() != profile_shape:
        raise ValueError(
            f"task_table_configs.{profile_key}.table_shape={config_shape!r} "
            f"does not match task base table shape {profile_shape!r}"
        )

    args.update(table_config)
    args["table_shape"] = table_shape
    args["task_table_config_key"] = profile_key
    return args


def resolve_max_seed_tries(args):
    raw_value = _runtime_env("ACTIVEARENA_MAX_SEED_TRIES", args.get("max_seed_tries", DEFAULT_MAX_SEED_TRIES))
    if raw_value is None:
        return DEFAULT_MAX_SEED_TRIES

    max_seed_tries = int(raw_value)
    if max_seed_tries < 0:
        return None
    return max_seed_tries


def get_collection_failure_report_path(save_path):
    return os.path.join(save_path, "collection_failure.json")


def write_collection_failure_report(save_path, payload):
    os.makedirs(save_path, exist_ok=True)
    report_path = get_collection_failure_report_path(save_path)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)
    return report_path


def clear_collection_failure_report(save_path):
    report_path = get_collection_failure_report_path(save_path)
    if os.path.exists(report_path):
        with _debug_stage("clear_collection_failure_report", path=report_path):
            os.remove(report_path)


def make_json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    try:
        import numpy as np  # local import to keep module init unchanged

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def safe_close_env(task_env, render_freq, clear_cache=False):
    start = time.perf_counter()
    _debug_log("START safe_close_env.close_env", clear_cache=clear_cache, render_freq=render_freq)
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        traceback.print_exc()
    finally:
        _debug_log(
            "END safe_close_env.close_env",
            elapsed=f"{time.perf_counter() - start:.3f}s",
            clear_cache=clear_cache,
            render_freq=render_freq,
        )

    if render_freq:
        viewer = getattr(task_env, "viewer", None)
        if viewer is not None:
            start = time.perf_counter()
            _debug_log("START safe_close_env.viewer_close")
            try:
                viewer.close()
            except Exception:
                traceback.print_exc()
            finally:
                _debug_log("END safe_close_env.viewer_close", elapsed=f"{time.perf_counter() - start:.3f}s")


def main(task_name=None, task_config=None):

    _debug_log("main.start", task=task_name, task_config=task_config, debug_timing=_debug_timing_enabled())
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with _debug_stage("main.load_task_config", path=config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    with _debug_stage("main.apply_task_table_config", task=task_name):
        args = apply_task_table_config(task, args)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with _debug_stage("main.load_embodiment_types", path=embodiment_config_path):
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    difficulty_tag = infer_difficulty_tag(args)
    storage_setting = f"{task_config}__{difficulty_tag}"
    max_seed_tries = resolve_max_seed_tries(args)
    print("\033[94mDifficulty Tag:\033[0m " + difficulty_tag)
    print("\033[94mData Setting:\033[0m " + storage_setting)
    print("\033[94mMax Seed Tries:\033[0m " + ("unlimited" if max_seed_tries is None else str(max_seed_tries)))
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["difficulty_tag"] = difficulty_tag
    args["max_seed_tries"] = max_seed_tries
    args["storage_setting"] = storage_setting
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), storage_setting)
    clear_collection_failure_report(args["save_path"])

    try:
        with _debug_stage("main.run", task=args["task_name"], save_path=args["save_path"]):
            return run(task, args)
    except Exception as exc:
        traceback.print_exc()
        report = {
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "storage_setting": args["storage_setting"],
            "save_path": args["save_path"],
            "status": "failed",
            "reason": "collection_exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        report_path = write_collection_failure_report(args["save_path"], report)
        print(f"[Error] collection failure report written to: {report_path}")
        return COLLECTION_FAILED_EXIT_CODE


def run(TASK_ENV, args):
    run_start = time.perf_counter()
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    last_failure = None
    max_seed_tries = args.get("max_seed_tries", DEFAULT_MAX_SEED_TRIES)
    skip_episode_instructions = _env_flag("ACTIVEARENA_SKIP_INSTRUCTIONS", default=False)

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    _debug_log(
        "run.start",
        task=args["task_name"],
        save_path=args["save_path"],
        episode_num=args["episode_num"],
        use_seed=args["use_seed"],
        collect_data=args["collect_data"],
        max_seed_tries=("unlimited" if max_seed_tries is None else max_seed_tries),
        skip_episode_instructions=skip_episode_instructions,
    )

    # =========== Collect Seed ===========
    with _debug_stage("run.ensure_save_path", path=args["save_path"]):
        os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            seed_path = os.path.join(args["save_path"], "seed.txt")
            with _debug_stage("seed_stage.read_seed_file", path=seed_path):
                with open(seed_path, "r") as file:
                    seed_list = file.read().split()
                    if len(seed_list) != 0:
                        seed_list = [int(i) for i in seed_list]
                        suc_num = len(seed_list)
                        epid = max(seed_list) + 1
                    elif _runtime_env("ACTIVEARENA_START_SEED") is not None:
                        epid = int(_runtime_env("ACTIVEARENA_START_SEED"))
            print(f"Exist seed file, Start from: {epid} / {suc_num}")
        elif _runtime_env("ACTIVEARENA_START_SEED") is not None:
            epid = int(_runtime_env("ACTIVEARENA_START_SEED"))
            print(f"Start from env seed: {epid}")

        while suc_num < args["episode_num"]:
            attempt_start = time.perf_counter()
            attempt_status = "unknown"
            _debug_log(
                "seed_attempt.begin",
                task=args["task_name"],
                episode=suc_num,
                seed=epid,
                success=suc_num,
                failed=fail_num,
            )
            if max_seed_tries is not None and epid > max_seed_tries:
                report = {
                    "task_name": args["task_name"],
                    "task_config": args["task_config"],
                    "storage_setting": args["storage_setting"],
                    "save_path": args["save_path"],
                    "status": "failed",
                    "reason": "seed_limit_exceeded",
                    "max_seed_tries": max_seed_tries,
                    "episode_num": args["episode_num"],
                    "success_episode_num": suc_num,
                    "failure_num": fail_num,
                    "next_seed_to_try": epid,
                    "last_attempted_seed": epid - 1 if epid > 0 else None,
                    "seed_list": seed_list,
                    "last_failure": last_failure,
                }
                report_path = write_collection_failure_report(args["save_path"], report)
                print(
                    f"[Error] stop seed collection for {args['task_name']}: "
                    f"next seed {epid} exceeds max_seed_tries={max_seed_tries}"
                )
                print(f"[Error] collection failure report written to: {report_path}")
                return SEED_LIMIT_EXCEEDED_EXIT_CODE

            try:
                with _debug_stage("seed_attempt.setup_demo", task=args["task_name"], episode=suc_num, seed=epid):
                    TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                with _debug_stage("seed_attempt.play_once", task=args["task_name"], episode=suc_num, seed=epid):
                    TASK_ENV.play_once()

                with _debug_stage(
                    "seed_attempt.check_success",
                    task=args["task_name"],
                    episode=suc_num,
                    seed=epid,
                    plan_success=TASK_ENV.plan_success,
                ):
                    episode_success = TASK_ENV.plan_success and TASK_ENV.check_success()

                if episode_success:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    with _debug_stage("seed_attempt.save_traj_data", task=args["task_name"], episode=suc_num, seed=epid):
                        TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                    last_failure = None
                    attempt_status = "success"
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1
                    last_failure = {
                        "type": "plan_or_success_check_failed",
                        "seed": epid,
                    }
                    attempt_status = "plan_or_success_check_failed"

                with _debug_stage("seed_attempt.safe_close_env", task=args["task_name"], seed=epid):
                    safe_close_env(TASK_ENV, args["render_freq"])
            except UnStableError as e:
                attempt_status = type(e).__name__
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                traceback.print_exc()
                print(" -------------")
                fail_num += 1
                last_failure = {
                    "type": type(e).__name__,
                    "seed": epid,
                    "message": str(e),
                }
                with _debug_stage("seed_attempt.safe_close_env_after_unstable", task=args["task_name"], seed=epid):
                    safe_close_env(TASK_ENV, args["render_freq"])
                with _debug_stage("seed_attempt.sleep_after_unstable", task=args["task_name"], seed=epid, seconds=0.3):
                    time.sleep(0.3)
            except Exception as e:
                attempt_status = type(e).__name__
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                traceback.print_exc()
                print(" -------------")
                fail_num += 1
                last_failure = {
                    "type": type(e).__name__,
                    "seed": epid,
                    "message": str(e),
                }
                with _debug_stage("seed_attempt.safe_close_env_after_exception", task=args["task_name"], seed=epid):
                    safe_close_env(TASK_ENV, args["render_freq"])
                with _debug_stage("seed_attempt.sleep_after_exception", task=args["task_name"], seed=epid, seconds=1):
                    time.sleep(1)
            finally:
                _debug_log(
                    "seed_attempt.end",
                    task=args["task_name"],
                    seed=epid,
                    status=attempt_status,
                    elapsed=f"{time.perf_counter() - attempt_start:.3f}s",
                    success=suc_num,
                    failed=fail_num,
                )

            epid += 1

            seed_path = os.path.join(args["save_path"], "seed.txt")
            with _debug_stage("seed_stage.write_seed_file", path=seed_path, seed_count=len(seed_list), next_seed=epid):
                with open(seed_path, "w") as file:
                    for sed in seed_list:
                        file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        seed_path = os.path.join(args["save_path"], "seed.txt")
        with _debug_stage("use_seed.read_seed_file", path=seed_path):
            with open(seed_path, "r") as file:
                seed_list = file.read().split()
                seed_list = [int(i) for i in seed_list]

    if len(seed_list) < args["episode_num"]:
        report = {
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "storage_setting": args["storage_setting"],
            "save_path": args["save_path"],
            "status": "failed",
            "reason": "insufficient_seed_list",
            "episode_num": args["episode_num"],
            "success_episode_num": len(seed_list),
            "seed_list": seed_list,
        }
        report_path = write_collection_failure_report(args["save_path"], report)
        print(
            f"[Error] insufficient seeds for {args['task_name']}: "
            f"need {args['episode_num']}, only found {len(seed_list)}"
        )
        print(f"[Error] collection failure report written to: {report_path}")
        return COLLECTION_FAILED_EXIT_CODE

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        with _debug_stage("data_stage.scan_existing_hdf5", save_path=args["save_path"]):
            while exist_hdf5(st_idx):
                st_idx += 1
        _debug_log("data_stage.start_index", st_idx=st_idx, episode_num=args["episode_num"])

        for episode_idx in range(st_idx, args["episode_num"]):
            episode_start = time.perf_counter()
            seed = seed_list[episode_idx]
            _debug_log("data_episode.begin", task=args["task_name"], episode=episode_idx, seed=seed)
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            with _debug_stage("data_episode.setup_demo", task=args["task_name"], episode=episode_idx, seed=seed):
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed, **args)

            with _debug_stage("data_episode.load_tran_data", task=args["task_name"], episode=episode_idx):
                traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            with _debug_stage(
                "data_episode.set_path_lst",
                task=args["task_name"],
                episode=episode_idx,
                left_path_len=len(args["left_joint_path"]),
                right_path_len=len(args["right_joint_path"]),
            ):
                TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with _debug_stage("data_episode.init_scene_info", path=info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

            with _debug_stage("data_episode.read_scene_info", path=info_file_path):
                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

            with _debug_stage("data_episode.play_once", task=args["task_name"], episode=episode_idx, seed=seed):
                info_raw = TASK_ENV.play_once()
            with _debug_stage("data_episode.make_json_safe", task=args["task_name"], episode=episode_idx):
                info = make_json_safe(info_raw)
            info_db[f"episode_{episode_idx}"] = info

            with _debug_stage("data_episode.write_scene_info", path=info_file_path, episode=episode_idx):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

            clear_cache_this_episode = ((episode_idx + 1) % clear_cache_freq == 0)
            with _debug_stage(
                "data_episode.close_env",
                task=args["task_name"],
                episode=episode_idx,
                clear_cache=clear_cache_this_episode,
            ):
                TASK_ENV.close_env(clear_cache=clear_cache_this_episode)
            with _debug_stage("data_episode.merge_pkl_to_hdf5_video", task=args["task_name"], episode=episode_idx):
                TASK_ENV.merge_pkl_to_hdf5_video()
            with _debug_stage("data_episode.remove_data_cache", task=args["task_name"], episode=episode_idx):
                TASK_ENV.remove_data_cache()
            with _debug_stage("data_episode.check_success", task=args["task_name"], episode=episode_idx):
                assert TASK_ENV.check_success(), "Collect Error"
            _debug_log(
                "data_episode.end",
                task=args["task_name"],
                episode=episode_idx,
                seed=seed,
                elapsed=f"{time.perf_counter() - episode_start:.3f}s",
            )

        if not skip_episode_instructions:
            command = (
                "cd description && bash gen_episode_instructions.sh "
                f"{args['task_name']} {args['storage_setting']} {args['language_num']} {args['task_config']}"
            )
            with _debug_stage("data_stage.generate_episode_instructions", command=command):
                instruction_exit_code = os.system(command)
            _debug_log("data_stage.generate_episode_instructions.exit", exit_code=instruction_exit_code)
        else:
            _debug_log("data_stage.generate_episode_instructions.skip")

    clear_collection_failure_report(args["save_path"])
    _debug_log("run.end", task=args["task_name"], elapsed=f"{time.perf_counter() - run_start:.3f}s")
    return 0


if __name__ == "__main__":
    if not _env_flag("ACTIVEARENA_SKIP_RENDER_TEST", default=False):
        from test_render import Sapien_TEST
        Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    raise SystemExit(main(task_name=task_name, task_config=task_config))
