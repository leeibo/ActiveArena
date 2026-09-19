from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TASK_DIR_RE = re.compile(r"^\d+_(.+)$")
EPISODE_BEGIN_RE = re.compile(r"Base_Task\._init_task_env_\.begin.*?episode=(\d+).*?seed=(\d+)")
CLIENT_EPISODE_RE = re.compile(r"\[StarVLAOFTClient\]\s+ep=(\d+)")
SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+).*?current seed:\s*(\d+)")

TARGET_FIELD_NAMES = {
    "search_target_keys",
    "action_target_keys",
    "required_carried_keys",
    "carry_keys_after_done",
}

CSV_FIELDS = [
    "model",
    "run",
    "task",
    "episode",
    "seed",
    "success",
    "failure_hint",
    "failure_hint_confidence",
    "target_keys",
    "target_keys_source",
    "request_count",
    "last_env_step",
    "all_targets_discovered",
    "all_targets_discovery_step",
    "discovered_target_count",
    "target_count",
    "observed_object_keys",
    "observed_object_count",
    "discovery_auc",
    "final_context_complete",
    "final_context_target_count",
    "context_drop_event_count",
    "reacquisition_count",
    "evidence_available_fraction_after_discovery",
    "scan_degrees",
    "unique_heading_bins",
    "heading_revisit_ratio",
    "heading_direction_reversals",
    "post_discovery_chunks",
    "gripper_close_chunk_count",
    "planner_error_count",
    "planner_subtask_change_count",
    "response_error_count",
    "request_sec_total",
    "request_sec_mean",
    "request_sec_p95",
    "incomplete_episode",
]

TASK_SUMMARY_FIELDS = [
    "model",
    "run",
    "task",
    "episode_count",
    "labeled_episode_count",
    "success_count",
    "success_rate",
    "failure_count",
    "all_targets_discovery_rate",
    "failure_all_targets_discovered_count",
    "failure_all_targets_discovered_rate",
    "failure_final_context_complete_count",
    "failure_final_context_complete_rate",
    "success_median_all_targets_discovery_step",
    "failure_median_all_targets_discovery_step",
    "success_mean_request_count",
    "failure_mean_request_count",
    "success_mean_scan_degrees",
    "failure_mean_scan_degrees",
    "planner_error_episode_count",
    "incomplete_episode_count",
    "failure_hint_counts",
    "target_keys",
    "target_keys_source",
    "observed_registry_keys",
    "malformed_request_lines",
    "stdout_errors",
]


def _as_key_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if item is not None and str(item)}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(items) if items else None


def _median(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(items) if items else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    items = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def _wrapped_heading_delta(left: float, right: float) -> float:
    return (float(right) - float(left) + 180.0) % 360.0 - 180.0


def _task_name(task_dir: Path) -> str:
    match = TASK_DIR_RE.match(task_dir.name)
    return match.group(1) if match else task_dir.name


def _literal_string_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                keys.add(element.value)
    return keys


def infer_static_target_keys(task_name: str, repo_root: Path = REPO_ROOT) -> set[str]:
    source_path = repo_root / "envs" / f"{task_name}.py"
    if not source_path.exists():
        return set()
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError):
        return set()

    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                if key_node.value in TARGET_FIELD_NAMES:
                    result.update(_literal_string_keys(value_node))
                elif key_node.value == "key" and isinstance(value_node, ast.Constant):
                    if isinstance(value_node.value, str):
                        result.add(value_node.value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in TARGET_FIELD_NAMES:
                    result.update(_literal_string_keys(keyword.value))
                elif keyword.arg == "object_registry" and isinstance(keyword.value, ast.Dict):
                    for key_node in keyword.value.keys:
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            result.add(key_node.value)
    return result


def _parse_stdout(stdout_path: Path) -> tuple[dict[int, bool], dict[int, int], list[str]]:
    success_by_episode: dict[int, bool] = {}
    seed_by_episode: dict[int, int] = {}
    diagnostic_errors: list[str] = []
    if not stdout_path.exists():
        return success_by_episode, seed_by_episode, ["stdout_missing"]

    current_episode: int | None = None
    previous_success_count: int | None = None
    try:
        lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return success_by_episode, seed_by_episode, [f"stdout_read_error:{type(exc).__name__}"]

    for raw_line in lines:
        line = ANSI_RE.sub("", raw_line)
        begin_match = EPISODE_BEGIN_RE.search(line)
        if begin_match:
            current_episode = int(begin_match.group(1))
            seed_by_episode[current_episode] = int(begin_match.group(2))
        client_match = CLIENT_EPISODE_RE.search(line)
        if client_match:
            current_episode = int(client_match.group(1))

        success_match = SUCCESS_RE.search(line)
        if success_match:
            success_count, completed_count, seed = map(int, success_match.groups())
            episode = current_episode if current_episode is not None else completed_count - 1
            seed_by_episode.setdefault(episode, seed)
            if previous_success_count is not None:
                success_by_episode[episode] = success_count > previous_success_count
            elif completed_count == 1:
                success_by_episode[episode] = success_count == 1
            elif success_count == 0:
                success_by_episode[episode] = False
            elif success_count == completed_count:
                success_by_episode[episode] = True
            previous_success_count = success_count

        lowered = line.lower()
        if "traceback (most recent call last)" in lowered:
            diagnostic_errors.append("traceback")
        elif "connection refused" in lowered:
            diagnostic_errors.append("connection_refused")
        elif "cuda kernel build failed" in lowered:
            diagnostic_errors.append("cuda_kernel_build_failed")
        elif "timed out" in lowered and "planner" not in lowered:
            diagnostic_errors.append("timeout")

    return success_by_episode, seed_by_episode, sorted(set(diagnostic_errors))


def _response_actions(response: dict[str, Any]) -> list[list[float]]:
    actions = response.get("actions")
    if actions is None and isinstance(response.get("data"), dict):
        actions = response["data"].get("actions")
    while isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], list):
        first = actions[0]
        if first and isinstance(first[0], list):
            actions = first
        else:
            break
    if not isinstance(actions, list):
        return []
    rows: list[list[float]] = []
    for action in actions:
        if not isinstance(action, list):
            continue
        try:
            rows.append([float(value) for value in action])
        except (TypeError, ValueError):
            continue
    return rows


def _request_seconds(response: dict[str, Any]) -> float | None:
    value = response.get("request_sec")
    if value is None and isinstance(response.get("data"), dict):
        value = response["data"].get("request_sec")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _planner_state(annotations: list[dict[str, Any]]) -> tuple[str | None, bool]:
    for annotation in reversed(annotations):
        planner = annotation.get("subtask_planner")
        if not isinstance(planner, dict):
            continue
        subtask = planner.get("current_subtask")
        return (str(subtask) if subtask is not None else None, bool(planner.get("error")))
    return None, False


def _normalize_request_record(record: dict[str, Any]) -> dict[str, Any] | None:
    request = record.get("request")
    response = record.get("response")
    if not isinstance(request, dict):
        return None
    annotations = request.get("annotations") or []
    annotations = [item for item in annotations if isinstance(item, dict)]
    if not annotations:
        return None
    current = annotations[-1]
    prompt_visible: set[str] = set()
    for annotation in annotations:
        prompt_visible.update(_as_key_set(annotation.get("visible_object_keys")))
    response = response if isinstance(response, dict) else {}
    planner_subtask, planner_annotation_error = _planner_state(annotations)
    response_error = record.get("error") or response.get("error")
    actions = _response_actions(response)
    gripper_close = any(
        len(action) > 15 and (float(action[7]) < 0.5 or float(action[15]) < 0.5)
        for action in actions
    )
    try:
        heading = float(current.get("waist_heading_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        heading = 0.0
    try:
        env_step = int(record.get("env_step", current.get("take_action_cnt", 0)) or 0)
    except (TypeError, ValueError):
        env_step = 0
    return {
        "env_step": env_step,
        "current_visible": _as_key_set(current.get("visible_object_keys")),
        "discovered": _as_key_set(current.get("discovered_object_keys")),
        "prompt_visible": prompt_visible,
        "heading": heading,
        "request_sec": _request_seconds(response),
        "planner_subtask": planner_subtask,
        "planner_error": bool(planner_annotation_error or response_error),
        "response_error": bool(response_error or response.get("ok") is False),
        "gripper_close": gripper_close,
    }


def _read_episode_requests(request_path: Path) -> tuple[dict[int, list[dict[str, Any]]], set[str], int]:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    observed_keys: set[str] = set()
    malformed_lines = 0
    if not request_path.exists():
        return episodes, observed_keys, malformed_lines
    with request_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
                episode = int(raw.get("episode", 0))
                record = _normalize_request_record(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed_lines += 1
                continue
            if record is None:
                malformed_lines += 1
                continue
            episodes[episode].append(record)
            observed_keys.update(record["discovered"])
            observed_keys.update(record["current_visible"])
    for records in episodes.values():
        records.sort(key=lambda item: item["env_step"])
    return episodes, observed_keys, malformed_lines


def _episode_metrics(
    records: list[dict[str, Any]],
    target_keys: set[str],
    heading_bin_deg: float,
) -> dict[str, Any]:
    discovered_so_far: set[str] = set()
    first_discovery: dict[str, int] = {}
    all_discovery_step: int | None = None
    context_drop_events = 0
    reacquisition_count = 0
    previously_visible: dict[str, bool] = {key: False for key in target_keys}
    previously_ever_visible: set[str] = set()
    evidence_after_discovery: list[bool] = []
    discovery_fractions: list[float] = []
    headings: list[float] = []
    request_seconds: list[float] = []
    planner_errors = 0
    response_errors = 0
    gripper_close_chunks = 0
    planner_subtasks: list[str] = []
    observed_objects: set[str] = set()
    first_observed_object_steps: dict[str, int] = {}

    for record in records:
        discovered_so_far.update(record["discovered"])
        discovered_so_far.update(record["current_visible"])
        observed_objects.update(record["discovered"])
        observed_objects.update(record["current_visible"])
        for key in record["discovered"] | record["current_visible"]:
            first_observed_object_steps.setdefault(str(key), int(record["env_step"]))
        for key in target_keys:
            if key in discovered_so_far and key not in first_discovery:
                first_discovery[key] = int(record["env_step"])
            visible = key in record["current_visible"]
            if visible and key in previously_ever_visible and not previously_visible[key]:
                reacquisition_count += 1
            if visible:
                previously_ever_visible.add(key)
            previously_visible[key] = visible
            if key in discovered_so_far and key not in record["prompt_visible"]:
                context_drop_events += 1

        if target_keys and target_keys <= discovered_so_far and all_discovery_step is None:
            all_discovery_step = int(record["env_step"])
        if all_discovery_step is not None:
            evidence_after_discovery.append(target_keys <= record["prompt_visible"])
        if target_keys:
            discovery_fractions.append(len(target_keys & discovered_so_far) / len(target_keys))

        headings.append(float(record["heading"]))
        if record["request_sec"] is not None:
            request_seconds.append(float(record["request_sec"]))
        planner_errors += int(record["planner_error"])
        response_errors += int(record["response_error"])
        gripper_close_chunks += int(record["gripper_close"])
        if record["planner_subtask"] is not None:
            planner_subtasks.append(record["planner_subtask"])

    heading_deltas = [_wrapped_heading_delta(left, right) for left, right in zip(headings, headings[1:])]
    nontrivial_signs = [1 if value > 1.0 else -1 for value in heading_deltas if abs(value) > 1.0]
    direction_reversals = sum(left != right for left, right in zip(nontrivial_signs, nontrivial_signs[1:]))
    bins = {
        int(math.floor(((heading + 180.0) % 360.0) / float(heading_bin_deg)))
        for heading in headings
    }
    final_prompt_visible = records[-1]["prompt_visible"] if records else set()
    planner_changes = sum(left != right for left, right in zip(planner_subtasks, planner_subtasks[1:]))
    post_discovery_chunks = (
        sum(int(record["env_step"] >= all_discovery_step) for record in records) - 1
        if all_discovery_step is not None
        else None
    )
    return {
        "request_count": len(records),
        "last_env_step": int(records[-1]["env_step"]) if records else None,
        "all_targets_discovered": bool(target_keys and target_keys <= discovered_so_far),
        "all_targets_discovery_step": all_discovery_step,
        "first_target_discovery_steps": dict(sorted(first_discovery.items())),
        "discovered_target_keys": sorted(target_keys & discovered_so_far),
        "discovered_target_count": len(target_keys & discovered_so_far),
        "target_count": len(target_keys),
        "observed_object_keys": sorted(observed_objects),
        "observed_object_count": len(observed_objects),
        "first_observed_object_steps": dict(sorted(first_observed_object_steps.items())),
        "final_prompt_object_keys": sorted(final_prompt_visible),
        "discovery_auc": _round(_mean(discovery_fractions)),
        "final_context_complete": bool(target_keys and target_keys <= final_prompt_visible),
        "final_context_target_keys": sorted(target_keys & final_prompt_visible),
        "final_context_target_count": len(target_keys & final_prompt_visible),
        "context_drop_event_count": context_drop_events,
        "reacquisition_count": reacquisition_count,
        "evidence_available_fraction_after_discovery": _round(
            _mean(float(value) for value in evidence_after_discovery)
        ),
        "scan_degrees": _round(sum(abs(value) for value in heading_deltas)),
        "unique_heading_bins": len(bins),
        "heading_revisit_ratio": _round(1.0 - len(bins) / len(headings)) if headings else None,
        "heading_direction_reversals": direction_reversals,
        "post_discovery_chunks": post_discovery_chunks,
        "gripper_close_chunk_count": gripper_close_chunks,
        "planner_error_count": planner_errors,
        "planner_subtask_change_count": planner_changes,
        "response_error_count": response_errors,
        "request_sec_total": _round(sum(request_seconds)),
        "request_sec_mean": _round(_mean(request_seconds)),
        "request_sec_p95": _round(_percentile(request_seconds, 0.95)),
    }


def _failure_hint(row: dict[str, Any]) -> tuple[str, str]:
    success = row.get("success")
    if success is True:
        return "success", "high"
    if success is None:
        return "incomplete_or_unlabeled", "high"
    if row.get("response_error_count", 0) > 0 and row.get("request_count", 0) <= 1:
        return "request_or_planner_failure", "high"
    if row.get("target_count", 0) and not row.get("all_targets_discovered"):
        return "target_not_fully_discovered", "high"
    if row.get("planner_error_count", 0) > 0:
        return "planner_degraded_failure", "medium"
    partial_targets = row.get("target_keys_source") == "static_env_partial"
    if row.get("target_count", 0) and not row.get("final_context_complete"):
        if partial_targets:
            return "known_context_evidence_missing_at_end", "low"
        return "context_evidence_missing_at_end", "medium"
    if row.get("all_targets_discovered"):
        if partial_targets:
            return "known_evidence_available_but_task_failed", "low"
        return "evidence_available_but_task_failed", "medium"
    return "unclassified_task_failure", "low"


def analyze_task(task_dir_value: str, run_dir_value: str, heading_bin_deg: float) -> dict[str, Any]:
    task_dir = Path(task_dir_value)
    run_dir = Path(run_dir_value)
    task_name = _task_name(task_dir)
    model_name = run_dir.parent.name
    run_name = run_dir.name
    episodes, observed_keys, malformed_lines = _read_episode_requests(task_dir / "requests.jsonl")
    static_keys = infer_static_target_keys(task_name)
    target_keys = static_keys
    if static_keys:
        target_source = "static_env_partial" if observed_keys - static_keys else "static_env"
    else:
        target_source = "unknown"
    successes, seeds, stdout_errors = _parse_stdout(task_dir / "stdout.log")

    rows: list[dict[str, Any]] = []
    for episode, records in sorted(episodes.items()):
        row = {
            "model": model_name,
            "run": run_name,
            "task": task_name,
            "episode": episode,
            "seed": seeds.get(episode),
            "success": successes.get(episode),
            "target_keys": sorted(target_keys),
            "target_keys_source": target_source,
        }
        row.update(_episode_metrics(records, target_keys, heading_bin_deg))
        row["incomplete_episode"] = row["success"] is None
        row["failure_hint"], row["failure_hint_confidence"] = _failure_hint(row)
        rows.append(row)

    return {
        "task": task_name,
        "task_dir": str(task_dir),
        "model": model_name,
        "run": run_name,
        "rows": rows,
        "malformed_request_lines": malformed_lines,
        "stdout_errors": stdout_errors,
        "target_keys": sorted(target_keys),
        "target_keys_source": target_source,
        "observed_registry_keys": sorted(observed_keys),
    }


def _task_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    if not rows:
        return {
            "model": result["model"],
            "run": result["run"],
            "task": result["task"],
            "episode_count": 0,
            "labeled_episode_count": 0,
            "success_count": 0,
            "success_rate": None,
            "failure_count": 0,
            "all_targets_discovery_rate": None,
            "failure_all_targets_discovered_count": 0,
            "failure_all_targets_discovered_rate": None,
            "failure_final_context_complete_count": 0,
            "failure_final_context_complete_rate": None,
            "success_median_all_targets_discovery_step": None,
            "failure_median_all_targets_discovery_step": None,
            "success_mean_request_count": None,
            "failure_mean_request_count": None,
            "success_mean_scan_degrees": None,
            "failure_mean_scan_degrees": None,
            "planner_error_episode_count": 0,
            "incomplete_episode_count": 0,
            "failure_hint_counts": {"no_episode_requests": 1},
            "target_keys": result["target_keys"],
            "target_keys_source": result["target_keys_source"],
            "observed_registry_keys": result["observed_registry_keys"],
            "malformed_request_lines": result["malformed_request_lines"],
            "stdout_errors": result["stdout_errors"],
        }
    labeled = [row for row in rows if row["success"] is not None]
    successes = [row for row in labeled if row["success"] is True]
    failures = [row for row in labeled if row["success"] is False]
    discovered = [row for row in labeled if row["all_targets_discovered"]]
    failure_discovered = [row for row in failures if row["all_targets_discovered"]]
    failure_context = [row for row in failures if row["final_context_complete"]]
    hints = Counter(row["failure_hint"] for row in rows)
    targets_known = rows[0]["target_count"] > 0

    def ratio(numerator: int, denominator: int) -> float | None:
        return _round(numerator / denominator) if denominator else None

    return {
        "model": rows[0]["model"],
        "run": rows[0]["run"],
        "task": rows[0]["task"],
        "episode_count": len(rows),
        "labeled_episode_count": len(labeled),
        "success_count": len(successes),
        "success_rate": ratio(len(successes), len(labeled)),
        "failure_count": len(failures),
        "all_targets_discovery_rate": ratio(len(discovered), len(labeled)) if targets_known else None,
        "failure_all_targets_discovered_count": len(failure_discovered) if targets_known else None,
        "failure_all_targets_discovered_rate": (
            ratio(len(failure_discovered), len(failures)) if targets_known else None
        ),
        "failure_final_context_complete_count": len(failure_context) if targets_known else None,
        "failure_final_context_complete_rate": ratio(len(failure_context), len(failures)) if targets_known else None,
        "success_median_all_targets_discovery_step": _round(
            _median(row["all_targets_discovery_step"] for row in successes)
        ),
        "failure_median_all_targets_discovery_step": _round(
            _median(row["all_targets_discovery_step"] for row in failures)
        ),
        "success_mean_request_count": _round(_mean(row["request_count"] for row in successes)),
        "failure_mean_request_count": _round(_mean(row["request_count"] for row in failures)),
        "success_mean_scan_degrees": _round(_mean(row["scan_degrees"] for row in successes)),
        "failure_mean_scan_degrees": _round(_mean(row["scan_degrees"] for row in failures)),
        "planner_error_episode_count": sum(row["planner_error_count"] > 0 for row in rows),
        "incomplete_episode_count": sum(row["incomplete_episode"] for row in rows),
        "failure_hint_counts": dict(sorted(hints.items())),
        "target_keys": rows[0]["target_keys"],
        "target_keys_source": rows[0]["target_keys_source"],
        "observed_registry_keys": result["observed_registry_keys"],
        "malformed_request_lines": result["malformed_request_lines"],
        "stdout_errors": result["stdout_errors"],
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return _json_text(value)
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _discover_task_dirs(run_dir: Path) -> list[Path]:
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks directory does not exist: {tasks_dir}")
    return sorted(path for path in tasks_dir.iterdir() if path.is_dir())


def _analyze_run(run_dir: Path, workers: int, heading_bin_deg: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dirs = _discover_task_dirs(run_dir)
    results: list[dict[str, Any]] = []
    if workers <= 1 or len(task_dirs) <= 1:
        for task_dir in task_dirs:
            results.append(analyze_task(str(task_dir), str(run_dir), heading_bin_deg))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(task_dirs))) as executor:
            futures = {
                executor.submit(analyze_task, str(task_dir), str(run_dir), heading_bin_deg): task_dir
                for task_dir in task_dirs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task_dir = futures[future]
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(futures)}] analyzed {task_dir.name}", file=sys.stderr, flush=True)

    results.sort(key=lambda item: item["task"])
    episode_rows = [row for result in results for row in result["rows"]]
    episode_rows.sort(key=lambda row: (row["task"], row["episode"]))
    summaries = [_task_summary(result) for result in results]
    summaries.sort(key=lambda row: row["task"])
    return episode_rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine process metrics from existing ActiveArena evaluation logs."
    )
    parser.add_argument("run_dir", type=Path, help="run directory containing tasks/ and summary.tsv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: RUN_DIR/process_analysis",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, int(os.cpu_count() or 1))),
        help="number of task-level worker processes",
    )
    parser.add_argument("--heading-bin-deg", type=float, default=15.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "process_analysis")).resolve()
    workers = max(1, int(args.workers))
    if not math.isfinite(args.heading_bin_deg) or args.heading_bin_deg <= 0.0:
        parser.error("--heading-bin-deg must be positive")

    episode_rows, summaries = _analyze_run(run_dir, workers, float(args.heading_bin_deg))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "episode_process_metrics.csv", episode_rows, CSV_FIELDS)
    _write_jsonl(output_dir / "episode_process_metrics.jsonl", episode_rows)
    _write_csv(output_dir / "task_process_summary.csv", summaries, TASK_SUMMARY_FIELDS)
    _write_jsonl(output_dir / "task_process_summary.jsonl", summaries)

    print(f"episodes: {len(episode_rows)}")
    print(f"tasks: {len(summaries)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
