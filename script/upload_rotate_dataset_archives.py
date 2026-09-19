from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_HOST = os.environ.get("ACTIVEARENA_REMOTE_HOST", "")
DEFAULT_REMOTE_PORT = int(os.environ.get("ACTIVEARENA_REMOTE_PORT", "22"))
DEFAULT_REMOTE_DATA_ROOT = os.environ.get("ACTIVEARENA_REMOTE_DATA_ROOT", "")
DEFAULT_WHITELIST_FILE = "task_config/rotate_task_whitelist.yml"
DEFAULT_DATA_ROOT = "data"
DEFAULT_ARCHIVE_CACHE_DIR = "data/.upload_archive_cache"
DEFAULT_CODEC = "zstd"
DEFAULT_ZSTD_LEVEL = 6
DEFAULT_GZIP_LEVEL = 6
DEFAULT_GZIP_THREADS = max(1, min(16, int(os.cpu_count() or 1)))


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        text=True,
        capture_output=capture_output,
        check=check,
    )


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required command not found in PATH: {name}")


def _load_whitelist(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("tasks", "include", "task_list", "whitelist_tasks", "selected_tasks"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise SystemExit(f"unsupported whitelist format: {path}")
    tasks: list[str] = []
    seen: set[str] = set()
    for item in payload:
        task = str(item).strip()
        if not task or task in seen:
            continue
        seen.add(task)
        tasks.append(task)
    return tasks


def _discover_task_dir(data_root: Path, task_name: str, save_dir_name: str | None, task_config: str | None) -> Path | None:
    task_root = data_root / task_name
    if not task_root.exists():
        return None
    if save_dir_name:
        target = task_root / save_dir_name
        return target if target.exists() else None
    candidates = sorted(path for path in task_root.iterdir() if path.is_dir())
    if task_config:
        prefix = f"{task_config}__"
        candidates = [path for path in candidates if path.name.startswith(prefix)]
    if not candidates:
        return None
    return candidates[-1]


def _source_signature(source_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    latest_mtime_ns = 0

    for path in sorted(source_dir.rglob("*")):
        rel = path.relative_to(source_dir)
        if path.is_symlink():
            target = os.readlink(path)
            stat = path.lstat()
            entry = f"L\t{rel.as_posix()}\t{target}\t{stat.st_mtime_ns}\n"
        elif path.is_dir():
            stat = path.stat()
            entry = f"D\t{rel.as_posix()}\t{stat.st_mtime_ns}\n"
        elif path.is_file():
            stat = path.stat()
            file_count += 1
            total_bytes += int(stat.st_size)
            latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
            entry = f"F\t{rel.as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n"
        else:
            continue
        latest_mtime_ns = max(latest_mtime_ns, int(path.lstat().st_mtime_ns))
        digest.update(entry.encode("utf-8"))

    return {
        "signature": digest.hexdigest(),
        "file_count": int(file_count),
        "total_bytes": int(total_bytes),
        "latest_mtime_ns": int(latest_mtime_ns),
    }


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _archive_suffix(codec: str) -> str:
    if codec == "zstd":
        return ".tar.zst"
    if codec == "gzip":
        return ".tar.gz"
    raise ValueError(f"unsupported codec: {codec}")


def _build_archive(
    *,
    source_dir: Path,
    archive_path: Path,
    codec: str,
    zstd_level: int,
    gzip_level: int,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    parent = source_dir.parent
    base_name = source_dir.name
    if codec == "zstd":
        cmd = [
            "tar",
            "-I",
            f"zstd -T0 -{int(zstd_level)}",
            "-cf",
            str(tmp_path),
            base_name,
        ]
    elif codec == "gzip":
        pigz_path = shutil.which("pigz")
        if pigz_path:
            cmd = [
                "tar",
                "-I",
                f"{pigz_path} -p {int(DEFAULT_GZIP_THREADS)} -{int(gzip_level)}",
                "-cf",
                str(tmp_path),
                base_name,
            ]
        else:
            cmd = [
                "tar",
                f"-czf",
                str(tmp_path),
                base_name,
            ]
            if int(gzip_level) != DEFAULT_GZIP_LEVEL:
                cmd = [
                    "tar",
                    "-I",
                    f"gzip -{int(gzip_level)}",
                    "-cf",
                    str(tmp_path),
                    base_name,
                ]
    else:
        raise ValueError(f"unsupported codec: {codec}")

    _run(cmd, cwd=parent)
    tmp_path.replace(archive_path)


def _load_archive_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _prepare_archive(
    *,
    source_dir: Path,
    cache_dir: Path,
    task_name: str,
    codec: str,
    zstd_level: int,
    gzip_level: int,
    force_rebuild: bool,
) -> dict[str, Any]:
    task_cache_dir = cache_dir / source_dir.name / task_name
    archive_name = f"{task_name}{_archive_suffix(codec)}"
    archive_path = task_cache_dir / archive_name
    manifest_path = task_cache_dir / f"{archive_name}.manifest.json"
    task_cache_dir.mkdir(parents=True, exist_ok=True)

    source_sig = _source_signature(source_dir)
    manifest = _load_archive_manifest(manifest_path)
    can_reuse = (
        not force_rebuild
        and manifest is not None
        and archive_path.exists()
        and manifest.get("source_signature") == source_sig["signature"]
        and manifest.get("codec") == codec
        and int(manifest.get("zstd_level", DEFAULT_ZSTD_LEVEL)) == int(zstd_level)
        and int(manifest.get("gzip_level", DEFAULT_GZIP_LEVEL)) == int(gzip_level)
    )

    rebuilt = False
    if not can_reuse:
        started_at = time.time()
        _build_archive(
            source_dir=source_dir,
            archive_path=archive_path,
            codec=codec,
            zstd_level=zstd_level,
            gzip_level=gzip_level,
        )
        elapsed_seconds = round(time.time() - started_at, 3)
        sha256 = _sha256_file(archive_path)
        manifest = {
            "task_name": task_name,
            "source_dir": str(source_dir),
            "source_signature": source_sig["signature"],
            "source_file_count": source_sig["file_count"],
            "source_total_bytes": source_sig["total_bytes"],
            "source_latest_mtime_ns": source_sig["latest_mtime_ns"],
            "codec": codec,
            "zstd_level": int(zstd_level),
            "gzip_level": int(gzip_level),
            "archive_path": str(archive_path),
            "archive_bytes": int(archive_path.stat().st_size),
            "archive_sha256": sha256,
            "build_elapsed_seconds": elapsed_seconds,
            "built_at_epoch": time.time(),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        rebuilt = True
    else:
        assert manifest is not None
        if not manifest.get("archive_sha256"):
            manifest["archive_sha256"] = _sha256_file(archive_path)
            manifest["archive_bytes"] = int(archive_path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "task_name": task_name,
        "source_dir": str(source_dir),
        "archive_path": str(archive_path),
        "manifest_path": str(manifest_path),
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": int(manifest["archive_bytes"]),
        "rebuilt": bool(rebuilt),
        "source_file_count": int(source_sig["file_count"]),
        "source_total_bytes": int(source_sig["total_bytes"]),
    }


def _ssh_base_cmd(host: str, port: int) -> list[str]:
    return ["ssh", "-p", str(port), "-C", host]


def _rsync_ssh_cmd(host: str, port: int) -> str:
    return f"ssh -p {int(port)} -C"


def _ssh_bash_lc(host: str, port: int, script: str) -> list[str]:
    return _ssh_base_cmd(host, port) + [f"bash -lc {shlex.quote(script)}"]


def _remote_preflight(host: str, port: int, codec: str) -> None:
    required = ["tar", "sha256sum", "bash"]
    if codec == "zstd":
        required.append("zstd")
    script = " && ".join(f"command -v {shlex.quote(cmd)} >/dev/null 2>&1" for cmd in required)
    _run(_ssh_bash_lc(host, port, script))


def _ensure_remote_dirs(host: str, port: int, remote_dirs: list[str]) -> None:
    quoted = " ".join(shlex.quote(path) for path in remote_dirs)
    _run(_ssh_bash_lc(host, port, f"mkdir -p {quoted}"))


def _rsync_archive(host: str, port: int, archive_path: Path, remote_archive_path: str) -> None:
    _run(
        [
            "rsync",
            "-a",
            "--info=progress2",
            "--human-readable",
            "--partial",
            "--append-verify",
            "-e",
            _rsync_ssh_cmd(host, port),
            str(archive_path),
            f"{host}:{remote_archive_path}",
        ]
    )


def _remote_extract_archive(
    *,
    host: str,
    port: int,
    codec: str,
    archive_sha256: str,
    remote_archive_path: str,
    remote_target_parent: str,
    remote_target_dir: str,
    remote_marker_path: str,
    remote_tmp_dir: str,
    save_dir_name: str,
    force_remote_replace: bool,
) -> str:
    extract_cmd = (
        f"tar -C {shlex.quote(remote_tmp_dir)} -I zstd -xf {shlex.quote(remote_archive_path)}"
        if codec == "zstd"
        else f"tar -C {shlex.quote(remote_tmp_dir)} -xzf {shlex.quote(remote_archive_path)}"
    )
    force_clause = ""
    if force_remote_replace:
        force_clause = (
            f"rm -rf {shlex.quote(remote_target_dir)} {shlex.quote(remote_tmp_dir)}; "
            f"rm -f {shlex.quote(remote_marker_path)};"
        )
    script = f"""
set -euo pipefail
archive_path={shlex.quote(remote_archive_path)}
expected_sha={shlex.quote(archive_sha256)}
target_parent={shlex.quote(remote_target_parent)}
target_dir={shlex.quote(remote_target_dir)}
marker_path={shlex.quote(remote_marker_path)}
tmp_dir={shlex.quote(remote_tmp_dir)}
save_dir_name={shlex.quote(save_dir_name)}
mkdir -p "$target_parent" "$(dirname "$marker_path")"
actual_sha=$(sha256sum "$archive_path" | awk '{{print $1}}')
if [ "$actual_sha" != "$expected_sha" ]; then
  echo "remote archive sha256 mismatch: $actual_sha != $expected_sha" >&2
  exit 21
fi
if [ -f "$marker_path" ] && [ -d "$target_dir" ] && [ "$(cat "$marker_path")" = "$expected_sha" ]; then
  echo "already_extracted"
  exit 0
fi
{force_clause}
if [ -e "$target_dir" ]; then
  echo "remote target already exists and marker mismatch: $target_dir" >&2
  exit 22
fi
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"
{extract_cmd}
if [ ! -d "$tmp_dir/$save_dir_name" ]; then
  echo "archive did not unpack expected top-level dir: $save_dir_name" >&2
  exit 23
fi
mv "$tmp_dir/$save_dir_name" "$target_dir"
rm -rf "$tmp_dir"
printf '%s\\n' "$expected_sha" > "$marker_path"
echo "extracted"
"""
    result = _run(_ssh_bash_lc(host, port, script), capture_output=True)
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compress rotate-view dataset save dirs into resumable per-task archives, "
            "upload via rsync, and extract on a remote machine."
        )
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--whitelist-file", type=str, default=DEFAULT_WHITELIST_FILE)
    parser.add_argument("--save-dir-name", type=str, default=None, help="Exact save-dir name, e.g. demo_randomized_easy_ep200_r5__easy_fan150")
    parser.add_argument("--task-config", type=str, default=None, help="Fallback prefix when --save-dir-name is not provided")
    parser.add_argument("--archive-cache-dir", type=str, default=DEFAULT_ARCHIVE_CACHE_DIR)
    parser.add_argument("--codec", type=str, choices=("zstd", "gzip"), default=DEFAULT_CODEC)
    parser.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL)
    parser.add_argument("--gzip-level", type=int, default=DEFAULT_GZIP_LEVEL)
    parser.add_argument("--remote-host", type=str, default=DEFAULT_REMOTE_HOST, help="SSH host; may also be set with ACTIVEARENA_REMOTE_HOST")
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT, help="SSH port (default: 22)")
    parser.add_argument("--remote-data-root", type=str, default=DEFAULT_REMOTE_DATA_ROOT, help="Remote data root; may also be set with ACTIVEARENA_REMOTE_DATA_ROOT")
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild local archives even if cached manifests match")
    parser.add_argument("--force-remote-replace", action="store_true", help="Replace existing remote target dirs when marker/checksum mismatches")
    parser.add_argument("--strict", action="store_true", help="Fail if any whitelist task is missing locally")
    parser.add_argument("--plan-only", action="store_true", help="Only build the upload plan; do not compress or upload")
    args = parser.parse_args()

    if not args.save_dir_name and not args.task_config:
        raise SystemExit("either --save-dir-name or --task-config is required")
    if not args.plan_only and (not args.remote_host or not args.remote_data_root):
        raise SystemExit(
            "remote upload requires --remote-host and --remote-data-root "
            "(or ACTIVEARENA_REMOTE_HOST / ACTIVEARENA_REMOTE_DATA_ROOT)"
        )

    _require_command("tar")
    _require_command("rsync")
    _require_command("ssh")
    if args.codec == "zstd":
        _require_command("zstd")

    data_root = (REPO_ROOT / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    whitelist_file = (REPO_ROOT / args.whitelist_file).resolve() if not Path(args.whitelist_file).is_absolute() else Path(args.whitelist_file)
    archive_cache_dir = (REPO_ROOT / args.archive_cache_dir).resolve() if not Path(args.archive_cache_dir).is_absolute() else Path(args.archive_cache_dir)
    whitelist = _load_whitelist(whitelist_file)

    task_entries: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    for task_name in whitelist:
        source_dir = _discover_task_dir(
            data_root=data_root,
            task_name=task_name,
            save_dir_name=args.save_dir_name,
            task_config=args.task_config,
        )
        if source_dir is None:
            missing_tasks.append(task_name)
            continue
        task_entries.append(
            {
                "task_name": task_name,
                "source_dir": source_dir,
                "save_dir_name": source_dir.name,
            }
        )

    if args.strict and missing_tasks:
        raise SystemExit(f"missing local task dirs: {', '.join(missing_tasks)}")
    if not task_entries:
        raise SystemExit("no matching local task dirs found")

    summary: dict[str, Any] = {
        "status": "planned" if args.plan_only else "ok",
        "data_root": str(data_root),
        "whitelist_file": str(whitelist_file),
        "save_dir_name": args.save_dir_name,
        "task_config": args.task_config,
        "codec": args.codec,
        "remote_host": args.remote_host,
        "remote_port": int(args.remote_port),
        "remote_data_root": str(args.remote_data_root),
        "task_count": len(task_entries),
        "missing_tasks": missing_tasks,
        "tasks": {},
    }

    if args.plan_only:
        for entry in task_entries:
            task_name = str(entry["task_name"])
            source_dir = Path(entry["source_dir"])
            archive_path = archive_cache_dir / source_dir.name / task_name / f"{task_name}{_archive_suffix(args.codec)}"
            summary["tasks"][task_name] = {
                "source_dir": str(source_dir),
                "archive_path": str(archive_path),
                "remote_target_dir": str(Path(args.remote_data_root) / task_name / source_dir.name),
            }
        if args.summary_path:
            summary_path = (REPO_ROOT / args.summary_path).resolve() if not Path(args.summary_path).is_absolute() else Path(args.summary_path)
            _write_summary(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    _remote_preflight(args.remote_host, args.remote_port, args.codec)

    for entry in task_entries:
        task_name = str(entry["task_name"])
        source_dir = Path(entry["source_dir"])
        save_dir_name = str(entry["save_dir_name"])

        print(f"[prepare] {task_name}: building/reusing archive for {source_dir}")
        archive_info = _prepare_archive(
            source_dir=source_dir,
            cache_dir=archive_cache_dir,
            task_name=task_name,
            codec=args.codec,
            zstd_level=int(args.zstd_level),
            gzip_level=int(args.gzip_level),
            force_rebuild=bool(args.force_rebuild),
        )

        archive_path = Path(archive_info["archive_path"])
        remote_cache_dir = str(Path(args.remote_data_root) / ".upload_cache" / save_dir_name)
        remote_state_dir = str(Path(args.remote_data_root) / ".upload_state" / save_dir_name)
        remote_tmp_root = str(Path(args.remote_data_root) / ".extract_tmp" / save_dir_name)
        remote_target_parent = str(Path(args.remote_data_root) / task_name)
        remote_target_dir = str(Path(args.remote_data_root) / task_name / save_dir_name)
        remote_archive_path = str(Path(remote_cache_dir) / archive_path.name)
        remote_marker_path = str(Path(remote_state_dir) / f"{task_name}.sha256")
        remote_tmp_dir = str(Path(remote_tmp_root) / task_name)

        _ensure_remote_dirs(
            args.remote_host,
            args.remote_port,
            [remote_cache_dir, remote_state_dir, remote_tmp_root, remote_target_parent],
        )

        print(
            f"[upload] {task_name}: {archive_path.name} "
            f"({archive_info['archive_bytes']} bytes, rebuilt={archive_info['rebuilt']})"
        )
        _rsync_archive(args.remote_host, args.remote_port, archive_path, remote_archive_path)

        print(f"[extract] {task_name}: remote -> {remote_target_dir}")
        remote_result = _remote_extract_archive(
            host=args.remote_host,
            port=args.remote_port,
            codec=args.codec,
            archive_sha256=str(archive_info["archive_sha256"]),
            remote_archive_path=remote_archive_path,
            remote_target_parent=remote_target_parent,
            remote_target_dir=remote_target_dir,
            remote_marker_path=remote_marker_path,
            remote_tmp_dir=remote_tmp_dir,
            save_dir_name=save_dir_name,
            force_remote_replace=bool(args.force_remote_replace),
        )

        summary["tasks"][task_name] = {
            "source_dir": str(source_dir),
            "archive_path": str(archive_path),
            "archive_sha256": str(archive_info["archive_sha256"]),
            "archive_bytes": int(archive_info["archive_bytes"]),
            "source_file_count": int(archive_info["source_file_count"]),
            "source_total_bytes": int(archive_info["source_total_bytes"]),
            "remote_archive_path": remote_archive_path,
            "remote_target_dir": remote_target_dir,
            "remote_marker_path": remote_marker_path,
            "remote_result": remote_result or "unknown",
        }

    if args.summary_path:
        summary_path = (REPO_ROOT / args.summary_path).resolve() if not Path(args.summary_path).is_absolute() else Path(args.summary_path)
        _write_summary(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
