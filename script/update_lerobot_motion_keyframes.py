#!/usr/bin/env python3
"""Recompute LeRobot motion_keyframe from grippers, torso yaw, and head rotation.

The detector is strictly causal: the output at frame t only depends on state
samples from frames 0..t. Persistent events are marked when they are confirmed
instead of being backdated to their first frame.
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "2026-07-12-v1-causal-gripper-rotation"
STATE_COLUMN = "observation.state"
MOTION_COLUMN = "motion_keyframe"
REQUIRED_STATE_NAMES = ("left_gripper", "right_gripper", "torso_yaw", "head_2")


@dataclass(frozen=True)
class DetectorConfig:
    gripper_close_threshold: float = 0.2
    gripper_open_threshold: float = 0.8
    min_gripper_state_len: int = 2
    rotation_delta_threshold: float = 0.005
    min_rotation_motion_len: int = 4
    rotation_merge_gap: int = 5
    dedup_window: int = 5
    mark_initial: bool = True


def causal_run_events(mask, min_len=1, merge_gap=0):
    """Yield online-confirmable (frame, event, run_start) tuples."""
    min_len = max(int(min_len), 1)
    merge_gap = max(int(merge_gap), 0)
    run_start = None
    false_count = 0
    confirmed = False

    for frame, active in enumerate(np.asarray(mask, dtype=bool)):
        if active:
            if run_start is None:
                run_start = frame
                confirmed = False
            false_count = 0
            if not confirmed and frame - run_start + 1 >= min_len:
                yield int(frame), "start", int(run_start)
                confirmed = True
            continue

        if run_start is None:
            continue
        false_count += 1
        if false_count > merge_gap:
            if confirmed:
                yield int(frame), "end", int(run_start)
            run_start = None
            false_count = 0
            confirmed = False


def mark_causal_boundaries(candidates, mask, min_len, merge_gap=0, mark_initial=False):
    mask = np.asarray(mask, dtype=bool)
    if mark_initial and len(mask) > 0 and mask[0]:
        candidates[0] = 1
    for frame, event, run_start in causal_run_events(mask, min_len=min_len, merge_gap=merge_gap):
        if event == "start":
            if run_start > 0:
                candidates[frame] = 1
        else:
            candidates[frame] = 1


def suppress_nearby(candidates, dedup_window):
    dedup_window = max(int(dedup_window), 0)
    if dedup_window == 0:
        return np.asarray(candidates, dtype=np.int64)
    kept = np.zeros(len(candidates), dtype=np.int64)
    last_kept = -10**9
    for frame in np.flatnonzero(candidates):
        frame = int(frame)
        if frame - last_kept > dedup_window:
            kept[frame] = 1
            last_kept = frame
    return kept


def compute_motion_keyframes(state, state_names, config):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError(f"{STATE_COLUMN} must be 2-D, got shape={state.shape}")
    if state.shape[1] != len(state_names):
        raise ValueError(f"state width={state.shape[1]} but names={len(state_names)}")

    name_to_index = {name: index for index, name in enumerate(state_names)}
    missing = [name for name in REQUIRED_STATE_NAMES if name not in name_to_index]
    if missing:
        raise KeyError(f"missing required state names: {missing}; available={state_names}")

    candidates = np.zeros(len(state), dtype=np.int64)
    if len(state) > 0 and config.mark_initial:
        candidates[0] = 1

    for name in ("left_gripper", "right_gripper"):
        values = state[:, name_to_index[name]]
        for mask in (
            values <= config.gripper_close_threshold,
            values >= config.gripper_open_threshold,
        ):
            mark_causal_boundaries(
                candidates,
                mask,
                min_len=config.min_gripper_state_len,
                merge_gap=0,
                mark_initial=config.mark_initial,
            )

    for name in ("torso_yaw", "head_2"):
        values = state[:, name_to_index[name]]
        delta = np.abs(np.diff(values, prepend=values[:1]))
        moving = delta > config.rotation_delta_threshold
        mark_causal_boundaries(
            candidates,
            moving,
            min_len=config.min_rotation_motion_len,
            merge_gap=config.rotation_merge_gap,
            mark_initial=config.mark_initial,
        )

    return suppress_nearby(candidates, config.dedup_window)


def flatten_feature_names(raw_names):
    names = raw_names
    while isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"invalid state feature names: {raw_names!r}")
    return names


def load_state_names(task_dir):
    info_path = task_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(STATE_COLUMN, {})
    return flatten_feature_names(feature.get("names"))


def episode_index_from_path(path):
    try:
        return int(path.stem.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"invalid episode parquet name: {path.name}") from exc


def parquet_compression(path):
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_row_groups == 0 or metadata.row_group(0).num_columns == 0:
        return "snappy"
    return metadata.row_group(0).column(0).compression.lower()


def rewrite_parquet(path, state_names, config, apply):
    path = Path(path)
    columns = pq.read_table(path, columns=[STATE_COLUMN, MOTION_COLUMN])
    state = np.asarray(columns[STATE_COLUMN].to_pylist(), dtype=np.float64)
    old_motion = columns[MOTION_COLUMN].to_numpy(zero_copy_only=False).astype(np.int64)
    new_motion = compute_motion_keyframes(state, state_names, config)
    if len(new_motion) != len(old_motion):
        raise RuntimeError(f"row count mismatch for {path}")

    result = {
        "path": str(path),
        "task": path.parents[2].name,
        "episode_index": episode_index_from_path(path),
        "frames": int(len(new_motion)),
        "old_keyframes": int(old_motion.sum()),
        "new_keyframes": int(new_motion.sum()),
        "changed_frames": int(np.count_nonzero(old_motion != new_motion)),
    }
    if not apply:
        return result

    table = pq.read_table(path)
    column_index = table.schema.get_field_index(MOTION_COLUMN)
    if column_index < 0:
        raise KeyError(f"{MOTION_COLUMN} not found in {path}")
    field = table.schema.field(column_index)
    replacement = pa.array(new_motion, type=field.type)
    updated = table.set_column(column_index, field, replacement)

    temp_path = path.with_name(f".{path.name}.motion-keyframe-{os.getpid()}.tmp")
    try:
        pq.write_table(
            updated,
            temp_path,
            compression=parquet_compression(path),
            version="2.6",
            use_dictionary=True,
            write_statistics=True,
        )
        check = pq.read_table(temp_path, columns=[MOTION_COLUMN])
        written = check[MOTION_COLUMN].to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.array_equal(written, new_motion):
            raise RuntimeError(f"written keyframes failed validation for {path}")
        if not pq.read_schema(temp_path).equals(table.schema, check_metadata=True):
            raise RuntimeError(f"schema or schema metadata changed for {path}")
        os.chmod(temp_path, path.stat().st_mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return result


def binary_stats(frames, keyframes):
    frames = int(frames)
    keyframes = int(keyframes)
    if frames <= 0:
        return {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
    mean = keyframes / frames
    return {
        "min": [int(keyframes == frames)],
        "max": [int(keyframes > 0)],
        "mean": [float(mean)],
        "std": [float(math.sqrt(mean * (1.0 - mean)))],
        "count": [frames],
    }


def binary_quantile(frames, keyframes, quantile):
    if frames <= 0:
        return 0.0
    zeros = frames - keyframes
    rank = (frames - 1) * float(quantile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    lower_value = float(lower >= zeros)
    upper_value = float(upper >= zeros)
    return lower_value * (upper - rank) + upper_value * (rank - lower) if upper > lower else lower_value


def update_task_metadata(task_dir, results, config):
    results_by_episode = {item["episode_index"]: item for item in results}
    stats_path = task_dir / "meta" / "episodes_stats.jsonl"
    records = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        episode_index = int(record["episode_index"])
        result = results_by_episode.get(episode_index)
        if result is None:
            raise KeyError(f"no parquet result for episode {episode_index} in {task_dir.name}")
        record.setdefault("stats", {})[MOTION_COLUMN] = binary_stats(
            result["frames"], result["new_keyframes"]
        )
    temp_stats = stats_path.with_suffix(".jsonl.tmp")
    temp_stats.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temp_stats, stats_path)

    total_frames = sum(item["frames"] for item in results)
    total_keyframes = sum(item["new_keyframes"] for item in results)
    global_path = task_dir / "meta" / "stats_gr00t.json"
    global_stats = json.loads(global_path.read_text(encoding="utf-8"))
    motion_stats = binary_stats(total_frames, total_keyframes)
    global_stats.setdefault("statistics", {})[MOTION_COLUMN] = {
        "mean": motion_stats["mean"],
        "std": motion_stats["std"],
        "min": [float(motion_stats["min"][0])],
        "max": [float(motion_stats["max"][0])],
        "q01": [binary_quantile(total_frames, total_keyframes, 0.01)],
        "q99": [binary_quantile(total_frames, total_keyframes, 0.99)],
    }
    temp_global = global_path.with_suffix(".json.tmp")
    temp_global.write_text(json.dumps(global_stats, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    os.replace(temp_global, global_path)

    provenance = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definition": "strictly causal motion keyframes from left/right gripper state boundaries and torso_yaw/head_2 motion boundaries",
        "state_column": STATE_COLUMN,
        "state_names": list(REQUIRED_STATE_NAMES),
        "parameters": asdict(config),
        "episodes": len(results),
        "frames": total_frames,
        "keyframes": total_keyframes,
    }
    provenance_path = task_dir / "meta" / "motion_keyframe_causal.json"
    temp_provenance = provenance_path.with_suffix(".json.tmp")
    temp_provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_provenance, provenance_path)


def discover_tasks(root, selected_tasks):
    root = Path(root).expanduser().resolve()
    tasks = sorted(path for path in root.iterdir() if path.is_dir() and (path / "meta/info.json").is_file())
    if selected_tasks:
        selected = set(selected_tasks)
        tasks = [path for path in tasks if path.name in selected]
        missing = selected - {path.name for path in tasks}
        if missing:
            raise FileNotFoundError(f"tasks not found: {sorted(missing)}")
    return root, tasks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root containing one LeRobot dataset directory per task.")
    parser.add_argument("--task", action="append", default=[], help="Process only this task; repeatable.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true", help="Atomically replace Parquet files and update metadata.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files for dry-run inspection only.")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.2)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.8)
    parser.add_argument("--min-gripper-state-len", type=int, default=2)
    parser.add_argument("--rotation-delta-threshold", type=float, default=0.005)
    parser.add_argument("--min-rotation-motion-len", type=int, default=4)
    parser.add_argument("--rotation-merge-gap", type=int, default=5)
    parser.add_argument("--dedup-window", type=int, default=5)
    parser.add_argument("--no-mark-initial", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.apply and args.max_files > 0:
        raise ValueError("--max-files cannot be combined with --apply because metadata would be incomplete")
    config = DetectorConfig(
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
        min_gripper_state_len=args.min_gripper_state_len,
        rotation_delta_threshold=args.rotation_delta_threshold,
        min_rotation_motion_len=args.min_rotation_motion_len,
        rotation_merge_gap=args.rotation_merge_gap,
        dedup_window=args.dedup_window,
        mark_initial=not args.no_mark_initial,
    )
    root, tasks = discover_tasks(args.root, args.task)
    state_names_by_task = {task.name: load_state_names(task) for task in tasks}
    jobs = []
    for task in tasks:
        for path in sorted(task.glob("data/*/*.parquet")):
            jobs.append((path, state_names_by_task[task.name]))
    if args.max_files > 0:
        jobs = jobs[: args.max_files]
    if not jobs:
        raise RuntimeError(f"no episode Parquet files found under {root}")

    print(
        f"mode={'apply' if args.apply else 'dry-run'} tasks={len(tasks)} episodes={len(jobs)} "
        f"workers={max(args.workers, 1)} version={VERSION}",
        flush=True,
    )
    results = []
    workers = max(int(args.workers), 1)
    if workers == 1:
        for index, (path, state_names) in enumerate(jobs, 1):
            results.append(rewrite_parquet(path, state_names, config, args.apply))
            if index % 50 == 0 or index == len(jobs):
                print(f"processed={index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(rewrite_parquet, path, state_names, config, args.apply): path
                for path, state_names in jobs
            }
            for future in as_completed(future_to_path):
                results.append(future.result())
                if len(results) % 50 == 0 or len(results) == len(jobs):
                    print(f"processed={len(results)}/{len(jobs)}", flush=True)

    results.sort(key=lambda item: (item["task"], item["episode_index"]))
    if args.apply:
        by_task = {}
        for result in results:
            by_task.setdefault(result["task"], []).append(result)
        for task in tasks:
            update_task_metadata(task, by_task[task.name], config)

    total_frames = sum(item["frames"] for item in results)
    old_keyframes = sum(item["old_keyframes"] for item in results)
    new_keyframes = sum(item["new_keyframes"] for item in results)
    changed_frames = sum(item["changed_frames"] for item in results)
    print(
        f"done frames={total_frames} old_keyframes={old_keyframes} "
        f"new_keyframes={new_keyframes} changed_frames={changed_frames}",
        flush=True,
    )
    for task in sorted({item["task"] for item in results}):
        task_results = [item for item in results if item["task"] == task]
        print(
            f"  {task}: episodes={len(task_results)} frames={sum(x['frames'] for x in task_results)} "
            f"old={sum(x['old_keyframes'] for x in task_results)} "
            f"new={sum(x['new_keyframes'] for x in task_results)} "
            f"changed={sum(x['changed_frames'] for x in task_results)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
