from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.analyze_eval_process_metrics import (
    CSV_FIELDS,
    TASK_SUMMARY_FIELDS,
    _mean,
    _round,
    _task_summary,
    _write_csv,
    _write_jsonl,
    analyze_task,
)


SELECTION_FIELDS = [
    "model",
    "task",
    "selected_run",
    "candidate_run_count",
    "episode_count",
    "labeled_episode_count",
    "request_count",
    "target_keys_source",
    "stdout_errors",
]

MODEL_SUMMARY_FIELDS = [
    "model",
    "task_count",
    "task_with_episode_count",
    "task_with_50_labeled_episodes_count",
    "episode_count",
    "labeled_episode_count",
    "success_count",
    "micro_success_rate",
    "macro_task_success_rate",
    "failure_count",
    "success_mean_request_count",
    "failure_mean_request_count",
    "success_mean_scan_degrees",
    "failure_mean_scan_degrees",
    "failure_mean_context_drop_event_count",
    "failure_with_exact_target_oracle_count",
    "exact_oracle_failure_all_targets_discovered_count",
    "exact_oracle_failure_all_targets_discovered_rate",
    "exact_oracle_failure_final_context_complete_count",
    "exact_oracle_failure_final_context_complete_rate",
    "failure_with_partial_target_oracle_count",
    "partial_oracle_failure_known_targets_discovered_count",
    "partial_oracle_failure_final_context_complete_count",
    "planner_error_episode_count",
    "incomplete_episode_count",
    "failure_hint_counts",
]


INVENTORY_FIELDS = TASK_SUMMARY_FIELDS + ["task_dir"]


def _discover_jobs(root: Path) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    for tasks_dir in sorted(root.glob("*/*/tasks")):
        if not tasks_dir.is_dir():
            continue
        run_dir = tasks_dir.parent
        for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
            jobs.append((task_dir, run_dir))
    return jobs


def _selection_score(result: dict[str, Any]) -> tuple[int, int, int, str]:
    rows = result["rows"]
    labeled = sum(row["success"] is not None for row in rows)
    requests = sum(int(row["request_count"]) for row in rows)
    return labeled, len(rows), requests, str(result["run"])


def _select_best_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        candidates[(result["model"], result["task"])].append(result)

    selected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for (model, task), items in sorted(candidates.items()):
        best = max(items, key=_selection_score)
        selected.append(best)
        rows = best["rows"]
        manifest.append(
            {
                "model": model,
                "task": task,
                "selected_run": best["run"],
                "candidate_run_count": len(items),
                "episode_count": len(rows),
                "labeled_episode_count": sum(row["success"] is not None for row in rows),
                "request_count": sum(int(row["request_count"]) for row in rows),
                "target_keys_source": best["target_keys_source"],
                "stdout_errors": best["stdout_errors"],
            }
        )
    return selected, manifest


def _ratio(numerator: int, denominator: int) -> float | None:
    return _round(numerator / denominator) if denominator else None


def _model_summaries(selected_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in selected_results:
        by_model[result["model"]].append(result)

    summaries: list[dict[str, Any]] = []
    for model, results in sorted(by_model.items()):
        rows = [row for result in results for row in result["rows"]]
        labeled = [row for row in rows if row["success"] is not None]
        successes = [row for row in labeled if row["success"] is True]
        failures = [row for row in labeled if row["success"] is False]
        exact = [row for row in failures if row["target_keys_source"] == "static_env"]
        partial = [row for row in failures if row["target_keys_source"] == "static_env_partial"]
        exact_discovered = [row for row in exact if row["all_targets_discovered"]]
        exact_context = [row for row in exact if row["final_context_complete"]]
        partial_discovered = [row for row in partial if row["all_targets_discovered"]]
        partial_context = [row for row in partial if row["final_context_complete"]]
        task_summaries = [_task_summary(result) for result in results]
        task_rates = [
            summary["success_rate"]
            for summary in task_summaries
            if summary["success_rate"] is not None
        ]
        hints = Counter(row["failure_hint"] for row in failures)
        summaries.append(
            {
                "model": model,
                "task_count": len(results),
                "task_with_episode_count": sum(bool(result["rows"]) for result in results),
                "task_with_50_labeled_episodes_count": sum(
                    sum(row["success"] is not None for row in result["rows"]) >= 50
                    for result in results
                ),
                "episode_count": len(rows),
                "labeled_episode_count": len(labeled),
                "success_count": len(successes),
                "micro_success_rate": _ratio(len(successes), len(labeled)),
                "macro_task_success_rate": _round(_mean(task_rates)),
                "failure_count": len(failures),
                "success_mean_request_count": _round(_mean(row["request_count"] for row in successes)),
                "failure_mean_request_count": _round(_mean(row["request_count"] for row in failures)),
                "success_mean_scan_degrees": _round(_mean(row["scan_degrees"] for row in successes)),
                "failure_mean_scan_degrees": _round(_mean(row["scan_degrees"] for row in failures)),
                "failure_mean_context_drop_event_count": _round(
                    _mean(row["context_drop_event_count"] for row in failures)
                ),
                "failure_with_exact_target_oracle_count": len(exact),
                "exact_oracle_failure_all_targets_discovered_count": len(exact_discovered),
                "exact_oracle_failure_all_targets_discovered_rate": _ratio(len(exact_discovered), len(exact)),
                "exact_oracle_failure_final_context_complete_count": len(exact_context),
                "exact_oracle_failure_final_context_complete_rate": _ratio(len(exact_context), len(exact)),
                "failure_with_partial_target_oracle_count": len(partial),
                "partial_oracle_failure_known_targets_discovered_count": len(partial_discovered),
                "partial_oracle_failure_final_context_complete_count": len(partial_context),
                "planner_error_episode_count": sum(row["planner_error_count"] > 0 for row in rows),
                "incomplete_episode_count": sum(row["incomplete_episode"] for row in rows),
                "failure_hint_counts": dict(sorted(hints.items())),
            }
        )
    return summaries


def _analyze_jobs(
    jobs: list[tuple[Path, Path]],
    workers: int,
    heading_bin_deg: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, (task_dir, run_dir) in enumerate(jobs, start=1):
            results.append(analyze_task(str(task_dir), str(run_dir), heading_bin_deg))
            if index % 50 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] task attempts analyzed", file=sys.stderr, flush=True)
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {
            executor.submit(analyze_task, str(task_dir), str(run_dir), heading_bin_deg): (task_dir, run_dir)
            for task_dir, run_dir in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] task attempts analyzed", file=sys.stderr, flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and deduplicate every model/run under an evaluation log root."
    )
    parser.add_argument("root", type=Path, help="evaluation log root directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, int(os.cpu_count() or 1))),
        help="number of task-level worker processes",
    )
    parser.add_argument("--heading-bin-deg", type=float, default=15.0)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "process_analysis_all")).resolve()
    if not root.is_dir():
        parser.error(f"root directory does not exist: {root}")
    if not math.isfinite(args.heading_bin_deg) or args.heading_bin_deg <= 0.0:
        parser.error("--heading-bin-deg must be positive")

    jobs = _discover_jobs(root)
    if not jobs:
        parser.error(f"no task directories found below: {root}")
    run_count = len({str(run_dir) for _, run_dir in jobs})
    print(f"discovered {len(jobs)} task attempts across {run_count} runs", file=sys.stderr)
    results = _analyze_jobs(jobs, max(1, int(args.workers)), float(args.heading_bin_deg))
    results.sort(key=lambda item: (item["model"], item["run"], item["task"]))

    inventory = []
    for result in results:
        row = _task_summary(result)
        row["task_dir"] = result["task_dir"]
        inventory.append(row)

    selected, manifest = _select_best_results(results)
    selected_rows = [row for result in selected for row in result["rows"]]
    selected_rows.sort(key=lambda row: (row["model"], row["task"], row["episode"]))
    selected_task_summaries = [_task_summary(result) for result in selected]
    selected_task_summaries.sort(key=lambda row: (row["model"], row["task"]))
    model_summaries = _model_summaries(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_run_task_inventory.csv", inventory, INVENTORY_FIELDS)
    _write_jsonl(output_dir / "all_run_task_inventory.jsonl", inventory)
    _write_csv(output_dir / "selected_task_sources.csv", manifest, SELECTION_FIELDS)
    _write_jsonl(output_dir / "selected_task_sources.jsonl", manifest)
    _write_csv(output_dir / "episode_process_metrics.csv", selected_rows, CSV_FIELDS)
    _write_jsonl(output_dir / "episode_process_metrics.jsonl", selected_rows)
    _write_csv(output_dir / "task_process_summary.csv", selected_task_summaries, TASK_SUMMARY_FIELDS)
    _write_jsonl(output_dir / "task_process_summary.jsonl", selected_task_summaries)
    _write_csv(output_dir / "model_process_summary.csv", model_summaries, MODEL_SUMMARY_FIELDS)
    _write_jsonl(output_dir / "model_process_summary.jsonl", model_summaries)

    print(f"runs: {run_count}")
    print(f"models: {len(model_summaries)}")
    print(f"selected model/tasks: {len(selected)}")
    print(f"selected episodes: {len(selected_rows)}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
