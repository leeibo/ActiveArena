import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append("./")

import importlib

import numpy as np
import yaml


DEFAULT_OUTPUT_DIR = "eval_seed_lists"
DEFAULT_NUM_SEEDS = 100
DEFAULT_MAX_SEED_TRIES = 10000


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception:
        raise SystemExit(f"No such task: {task_name}")


def make_json_safe(value: Any):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def safe_close_env(task_env, clear_cache: bool = False) -> None:
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        traceback.print_exc()

    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            traceback.print_exc()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def get_embodiment_config(robot_file: str):
    return load_yaml(Path(robot_file) / "config.yml")


def build_task_args(task_name: str, task_config: str) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    args = load_yaml(Path("task_config") / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True

    embodiment_type = args.get("embodiment")
    embodiment_types = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    camera_config = load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    def get_embodiment_file(name: str):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file for {name}")
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
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def parse_task_names(raw_names: list[str] | None, whitelist_file: str | None) -> list[str]:
    tasks: list[str] = []
    for raw in raw_names or []:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                tasks.append(item)

    if whitelist_file:
        payload = yaml.safe_load(Path(whitelist_file).read_text(encoding="utf-8")) or []
        if not isinstance(payload, list):
            raise TypeError(f"Expected list in whitelist file: {whitelist_file}")
        tasks.extend(str(item).strip() for item in payload if str(item).strip())

    deduped: list[str] = []
    seen = set()
    for task in tasks:
        if task not in seen:
            deduped.append(task)
            seen.add(task)
    if not deduped:
        raise ValueError("No tasks specified. Use --task-name or --whitelist-file.")
    return deduped


def default_start_seed(seed_arg: int) -> int:
    return 100000 * (1 + int(seed_arg))


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_outputs(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)

    txt_path = path.with_suffix(".txt")
    txt_path.write_text("\n".join(str(seed) for seed in payload.get("seeds", [])) + "\n", encoding="utf-8")


def result_path(output_dir: Path, task_config: str, task_name: str) -> Path:
    return output_dir / task_config / f"{task_name}.json"


def collect_task(args, task_name: str) -> bool:
    from envs.utils.create_actor import UnStableError

    output_path = result_path(Path(args.output_dir), args.task_config, task_name)
    start_seed = int(args.start_seed) if args.start_seed is not None else default_start_seed(args.seed)
    max_seed_tries = None if args.max_seed_tries < 0 else int(args.max_seed_tries)

    existing = load_existing(output_path) if args.resume else {}
    entries = list(existing.get("entries", []))
    failures = list(existing.get("failures", []))
    last_attempted_seed = existing.get("last_attempted_seed", None)
    attempt_count = int(existing.get("attempt_count", 0))
    next_seed = max(start_seed, int(last_attempted_seed) + 1) if last_attempted_seed is not None else start_seed

    if len(entries) >= args.num_seeds:
        print(f"[skip] {task_name}: already has {len(entries)} seeds at {output_path}")
        return True

    task_args = build_task_args(task_name, args.task_config)
    task_args["render_freq"] = 0
    task_args["need_plan"] = True
    task_args["eval_mode"] = True

    task_env = class_decorator(task_name)
    start_time = time.time()

    print(
        f"[start] task={task_name} config={args.task_config} "
        f"need={args.num_seeds} existing={len(entries)} start_seed={next_seed}"
    )

    seed = next_seed
    while len(entries) < args.num_seeds:
        if max_seed_tries is not None and seed >= start_seed + max_seed_tries:
            print(
                f"[stop] {task_name}: reached max_seed_tries={max_seed_tries}; "
                f"success={len(entries)} last_seed={seed - 1}"
            )
            break

        clear_cache = bool(args.clear_cache_freq and attempt_count > 0 and attempt_count % args.clear_cache_freq == 0)
        try:
            task_env.setup_demo(now_ep_num=len(entries), seed=seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            ok = bool(task_env.plan_success and task_env.check_success())
            if ok:
                info = {}
                if isinstance(episode_info, dict):
                    info = episode_info.get("info", {}) or {}
                entries.append(
                    {
                        "seed": int(seed),
                        "info": make_json_safe(info),
                    }
                )
                print(f"[ok] {task_name}: {len(entries)}/{args.num_seeds} seed={seed}")
            else:
                failures.append({"seed": int(seed), "type": "plan_or_success_check_failed"})
                print(f"[fail] {task_name}: seed={seed}")
            safe_close_env(task_env, clear_cache=clear_cache)
        except UnStableError as exc:
            failures.append({"seed": int(seed), "type": type(exc).__name__, "message": str(exc)})
            print(f"[unstable] {task_name}: seed={seed} {exc}")
            safe_close_env(task_env, clear_cache=clear_cache)
        except Exception as exc:
            failures.append({"seed": int(seed), "type": type(exc).__name__, "message": str(exc)})
            print(f"[error] {task_name}: seed={seed} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            safe_close_env(task_env, clear_cache=clear_cache)

        if len(failures) > args.failure_log_limit:
            failures = failures[-args.failure_log_limit :]
        attempt_count += 1
        last_attempted_seed = int(seed)
        seed += 1

        payload = {
            "format_version": 1,
            "task_name": task_name,
            "task_config": args.task_config,
            "seed_arg": int(args.seed),
            "start_seed": int(start_seed),
            "requested_successes": int(args.num_seeds),
            "success_count": len(entries),
            "attempt_count": int(attempt_count),
            "last_attempted_seed": last_attempted_seed,
            "next_seed": int(seed),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - start_time, 3),
            "seeds": [int(entry["seed"]) for entry in entries],
            "entries": entries,
            "failures": failures,
        }
        write_outputs(output_path, payload)

    complete = len(entries) >= args.num_seeds
    print(
        f"[done] {task_name}: success={len(entries)}/{args.num_seeds} "
        f"attempts={attempt_count} output={output_path}"
    )
    return complete


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute feasible eval seeds for ActiveArena policy evaluation.")
    parser.add_argument("--task-config", required=True, help="Task config name under task_config/*.yml")
    parser.add_argument("--task-name", action="append", help="Task name. Can be repeated or comma-separated.")
    parser.add_argument("--whitelist-file", help="YAML list of task names, e.g. task_config/rotate_task_whitelist.yml")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed", type=int, default=0, help="Eval seed block id; seed=0 starts from 100000.")
    parser.add_argument("--start-seed", type=int, default=None, help="Override absolute start seed.")
    parser.add_argument("--max-seed-tries", type=int, default=DEFAULT_MAX_SEED_TRIES, help="-1 means unlimited.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clear-cache-freq", type=int, default=20)
    parser.add_argument("--failure-log-limit", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing scan progress.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.resume = not args.no_resume
    tasks = parse_task_names(args.task_name, args.whitelist_file)

    failed = []
    for task_name in tasks:
        if not collect_task(args, task_name):
            failed.append(task_name)

    if failed:
        print("[incomplete] " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
