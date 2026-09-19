#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import yaml


def load_excluded_tasks(fixing_file: Path) -> set[str]:
    if not fixing_file.exists():
        raise FileNotFoundError(f"fixing task file not found: {fixing_file}")

    data = yaml.safe_load(fixing_file.read_text(encoding="utf-8"))
    if data is None:
        return set()

    if isinstance(data, dict):
        for key in ("tasks", "exclude", "task_list", "fixing_tasks"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            f"unsupported fixing_task format in {fixing_file}. "
            "Expected a YAML list or a dict containing one of: tasks/exclude/task_list/fixing_tasks"
        )

    excluded = set()
    for item in data:
        if item is None:
            continue
        task_name = str(item).strip()
        if task_name:
            excluded.add(task_name)
    return excluded


def discover_tasks(env_dir: Path, scope: str) -> list[str]:
    tasks = []
    for py_file in sorted(env_dir.glob("*.py")):
        task_name = py_file.stem
        if task_name.startswith("_"):
            continue
        if scope == "rotate_view" and not task_name.endswith("_rotate_view"):
            continue
        tasks.append(task_name)
    return tasks


def run_collect(root_dir: Path, task_name: str, task_config: str, gpu_id: int) -> int:
    cmd = ["bash", "collect_data.sh", task_name, task_config, str(gpu_id)]
    print(f"[Collect] {task_name} | config={task_config} | gpu={gpu_id}")
    return subprocess.run(cmd, cwd=root_dir).returncode


def write_collection_summary_report(
    report_dir: Path,
    scope: str,
    task_config: str,
    gpu_id: int,
    selected_tasks: list[str],
    succeeded: list[str],
    failed: list[dict],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"collect_all_tasks_except_fixing__{scope}__{task_config}.json"
    payload = {
        "scope": scope,
        "task_config": task_config,
        "gpu_id": gpu_id,
        "selected_task_num": len(selected_tasks),
        "succeeded_task_num": len(succeeded),
        "failed_task_num": len(failed),
        "selected_tasks": selected_tasks,
        "succeeded_tasks": succeeded,
        "failed_tasks": failed,
        "per_task_failure_report": "./data/<task_name>/*/collection_failure.json",
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequentially collect tasks except those listed in fixing_task.yml"
    )
    parser.add_argument("--task-config", default="demo_clean", help="task_config yaml name without suffix")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA_VISIBLE_DEVICES id passed to collect_data.sh")
    parser.add_argument(
        "--fixing-file",
        default="task_config/fixing_task.yml",
        help="path to fixing task yaml (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--scope",
        choices=["rotate_view", "all"],
        default="rotate_view",
        help="task discovery scope",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue collecting next tasks when one task fails",
    )
    parser.add_argument("--dry-run", action="store_true", help="only print task list without collecting")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    env_dir = root_dir / "envs"

    fixing_file = Path(args.fixing_file)
    if not fixing_file.is_absolute():
        fixing_file = root_dir / fixing_file

    excluded = load_excluded_tasks(fixing_file)
    all_tasks = discover_tasks(env_dir, args.scope)
    selected_tasks = [t for t in all_tasks if t not in excluded]
    report_dir = root_dir / "data" / "collection_reports"

    print(f"[Info] scope={args.scope}")
    print(f"[Info] discovered={len(all_tasks)} | excluded={len(excluded)} | to_collect={len(selected_tasks)}")
    print("[Info] tasks:")
    for idx, task_name in enumerate(selected_tasks, 1):
        print(f"  {idx:03d}. {task_name}")

    if args.dry_run:
        return 0

    succeeded = []
    failed = []
    report_path = write_collection_summary_report(
        report_dir,
        args.scope,
        args.task_config,
        args.gpu_id,
        selected_tasks,
        succeeded,
        failed,
    )

    for task_name in selected_tasks:
        ret = run_collect(root_dir, task_name, args.task_config, args.gpu_id)
        if ret == 0:
            succeeded.append(task_name)
            report_path = write_collection_summary_report(
                report_dir,
                args.scope,
                args.task_config,
                args.gpu_id,
                selected_tasks,
                succeeded,
                failed,
            )
            continue

        failed.append({"task_name": task_name, "exit_code": ret})
        print(f"[Error] task failed: {task_name} (exit={ret})")
        report_path = write_collection_summary_report(
            report_dir,
            args.scope,
            args.task_config,
            args.gpu_id,
            selected_tasks,
            succeeded,
            failed,
        )
        if not args.continue_on_error:
            break

    print("\n[Summary]")
    print(f"  succeeded: {len(succeeded)}")
    print(f"  failed:    {len(failed)}")
    print(f"  summary_report: {report_path}")
    if failed:
        print("  failed_tasks:")
        for item in failed:
            print(f"    - {item['task_name']} (exit={item['exit_code']})")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
