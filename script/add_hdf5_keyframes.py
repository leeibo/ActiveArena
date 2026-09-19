import argparse
import bisect
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


VERSION = "2026-07-12-v4-causal"


def odd_window(value):
    value = max(int(value), 1)
    return value if value % 2 == 1 else value + 1


def causal_moving_average(values, window):
    """Smooth with a trailing window so output[t] only depends on values[:t + 1]."""
    window = odd_window(window)
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float64, copy=True)
    pad = window - 1
    padded = np.pad(values, [(pad, 0)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    flat = padded.reshape(padded.shape[0], -1)
    smoothed = np.stack(
        [np.convolve(flat[:, i], kernel, mode="valid") for i in range(flat.shape[1])],
        axis=1,
    )
    return smoothed.reshape(values.shape)


def mark(reason_sets, keyframes, frames, reason):
    t = len(keyframes)
    for frame in frames:
        if 0 <= frame < t:
            keyframes[frame] = 1
            reason_sets[frame].add(reason)


def causal_run_events(mask, min_len=1, merge_gap=0):
    """Return online-confirmable (frame, event, run_start) tuples.

    A start is emitted on the frame where the minimum run length is first
    confirmed, never retroactively at ``run_start``. An end is emitted only
    after more than ``merge_gap`` false samples have been observed.
    """
    min_len = max(int(min_len), 1)
    merge_gap = max(int(merge_gap), 0)
    events = []
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
                events.append((int(frame), "start", int(run_start)))
                confirmed = True
            continue

        if run_start is None:
            continue
        false_count += 1
        if false_count > merge_gap:
            if confirmed:
                events.append((int(frame), "end", int(run_start)))
            run_start = None
            false_count = 0
            confirmed = False

    return events


def mark_causal_run_boundaries(
    reason_sets,
    keyframes,
    mask,
    reason_prefix,
    min_len=1,
    mark_start=True,
    mark_end=True,
    mark_initial=False,
    merge_gap=0,
):
    mask = np.asarray(mask, dtype=bool)
    if mark_initial and len(mask) > 0 and mask[0]:
        mark(reason_sets, keyframes, [0], f"{reason_prefix}_start")
    for frame, event, run_start in causal_run_events(mask, min_len=min_len, merge_gap=merge_gap):
        if event == "start" and mark_start and run_start > 0:
            mark(reason_sets, keyframes, [frame], f"{reason_prefix}_start")
        elif event == "end" and mark_end:
            mark(reason_sets, keyframes, [frame], f"{reason_prefix}_end")


def suppress_nearby_frames(keyframes, reason_sets, dedup_window):
    dedup_window = int(dedup_window)
    if dedup_window <= 0:
        return keyframes
    kept = np.zeros_like(keyframes)
    last_kept = -10**9
    for frame in np.flatnonzero(keyframes):
        frame = int(frame)
        if frame - last_kept > dedup_window:
            kept[frame] = 1
            last_kept = frame
        else:
            reason_sets[frame].clear()
    return kept


def causal_effective_thresholds(values, absolute, percentile, scale):
    """Compute thresholds from preceding samples only.

    The threshold used for frame ``t`` is calculated from frames ``0..t-1``;
    the current sample is inserted afterwards for use by later frames.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    thresholds = np.full(len(values), float(absolute), dtype=np.float64)
    history = []
    quantile = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    for frame, value in enumerate(values):
        if history:
            rank = (len(history) - 1) * quantile
            lower = int(math.floor(rank))
            upper = int(math.ceil(rank))
            fraction = rank - lower
            adaptive = history[lower] * (1.0 - fraction) + history[upper] * fraction
            thresholds[frame] = max(float(absolute), float(adaptive) * float(scale))
        if np.isfinite(value) and value > 1e-12:
            bisect.insort(history, float(value))
    return thresholds


def subtask_keyframes(subtask, mark_first=True):
    subtask = np.asarray(subtask)
    keyframes = np.zeros(len(subtask), dtype=np.int8)
    if len(subtask) == 0:
        return keyframes
    if mark_first:
        keyframes[0] = 1
    keyframes[1:] = (subtask[1:] != subtask[:-1]).astype(np.int8)
    return keyframes


def compute_arm_motion(f, arm, opts, reason_sets, keyframes):
    pose_key = f"endpose/{arm}_endpose"
    if pose_key not in f:
        return {}

    pos = np.asarray(f[pose_key][:, :3], dtype=np.float64)
    if len(pos) == 0:
        return {}
    velocity = np.diff(pos, axis=0, prepend=pos[:1])
    velocity = causal_moving_average(velocity, opts["smooth_window"])
    speed = np.linalg.norm(velocity, axis=1)

    direction_speed = causal_effective_thresholds(
        speed,
        opts["speed_min"],
        opts["adaptive_speed_percentile"],
        opts["adaptive_speed_scale"],
    )
    stop_speed = causal_effective_thresholds(
        speed,
        opts["stop_speed"],
        opts["adaptive_stop_percentile"],
        opts["adaptive_stop_scale"],
    )

    if len(speed) > 1:
        prev_v = velocity[:-1]
        cur_v = velocity[1:]
        prev_speed = speed[:-1]
        cur_speed = speed[1:]
        valid = (prev_speed > direction_speed[:-1]) & (cur_speed > direction_speed[1:])
        denom = np.maximum(prev_speed * cur_speed, 1e-12)
        cos_angle = np.sum(prev_v * cur_v, axis=1) / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        direction_frames = np.flatnonzero(valid & (angles > opts["angle_deg"])) + 1
        mark(reason_sets, keyframes, direction_frames, f"{arm}_ee_direction_change")

    stopped = speed < stop_speed
    for frame, event, start in causal_run_events(
        stopped,
        min_len=opts["min_stop_len"],
        merge_gap=opts["stop_merge_gap"],
    ):
        if event != "start":
            continue
        if start == 0 and not opts["mark_motion_initial"]:
            continue
        if start == 0:
            continue
        if opts["min_move_before_stop"] > 0 and start > 0:
            lookback_start = max(1, start - opts["stop_lookback"])
            recent_path_len = float(np.sum(speed[lookback_start : start + 1]))
            if recent_path_len < opts["min_move_before_stop"]:
                continue
        mark(reason_sets, keyframes, [frame], f"{arm}_ee_stop_confirmed")
    return {
        f"{arm}_final_direction_speed_threshold": float(direction_speed[-1]),
        f"{arm}_final_stop_speed_threshold": float(stop_speed[-1]),
    }


def compute_joint_motion(f, joint_key, label, opts, reason_sets, keyframes):
    if joint_key not in f:
        return
    values = np.asarray(f[joint_key][:], dtype=np.float64).reshape(len(f[joint_key]), -1)
    if len(values) == 0:
        return
    delta = np.diff(values, axis=0, prepend=values[:1])
    delta_norm = np.linalg.norm(delta, axis=1)
    moving = delta_norm > opts["joint_threshold"]
    mark_causal_run_boundaries(
        reason_sets,
        keyframes,
        moving,
        f"{label}_motion",
        min_len=opts["min_joint_motion_len"],
        mark_start=True,
        mark_end=True,
        mark_initial=opts["mark_motion_initial"],
        merge_gap=opts["joint_merge_gap"],
    )


def compute_gripper_motion(f, arm, opts, reason_sets, keyframes):
    candidates = [f"endpose/{arm}_gripper", f"joint_action/{arm}_gripper"]
    gripper_key = next((key for key in candidates if key in f), None)
    if gripper_key is None:
        return
    values = np.asarray(f[gripper_key][:], dtype=np.float64)
    if len(values) == 0:
        return

    closed = values <= opts["gripper_close_threshold"]
    opened = values >= opts["gripper_open_threshold"]
    mark_causal_run_boundaries(
        reason_sets,
        keyframes,
        closed,
        f"{arm}_gripper_closed",
        min_len=opts["min_gripper_state_len"],
        mark_start=True,
        mark_end=True,
        mark_initial=opts["mark_motion_initial"],
    )
    mark_causal_run_boundaries(
        reason_sets,
        keyframes,
        opened,
        f"{arm}_gripper_open",
        min_len=opts["min_gripper_state_len"],
        mark_start=True,
        mark_end=True,
        mark_initial=opts["mark_motion_initial"],
    )

    if not opts["gripper_state_only"]:
        delta = np.abs(np.diff(values, prepend=values[:1]))
        moving = delta > opts["gripper_delta_threshold"]
        mark_causal_run_boundaries(
            reason_sets,
            keyframes,
            moving,
            f"{arm}_gripper_motion",
            min_len=opts["min_gripper_motion_len"],
            mark_start=True,
            mark_end=True,
            mark_initial=opts["mark_motion_initial"],
        )


def motion_keyframes(f, opts):
    t = len(f["subtask"])
    keyframes = np.zeros(t, dtype=np.int8)
    reason_sets = [set() for _ in range(t)]
    stats = {}
    if t > 0 and opts["mark_motion_initial"]:
        mark(reason_sets, keyframes, [0], "initial_frame")

    for arm in ("left", "right"):
        stats.update(compute_arm_motion(f, arm, opts, reason_sets, keyframes))
        compute_gripper_motion(f, arm, opts, reason_sets, keyframes)

    compute_joint_motion(f, "joint_action/head", "head", opts, reason_sets, keyframes)
    compute_joint_motion(f, "joint_action/torso", "torso", opts, reason_sets, keyframes)

    keyframes = suppress_nearby_frames(keyframes, reason_sets, opts["dedup_window"])
    return keyframes.astype(np.int8), reason_sets, stats


def decode_rgb(rgb_bytes):
    import cv2

    if isinstance(rgb_bytes, np.void):
        rgb_bytes = bytes(rgb_bytes)
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode RGB bytes")
    return image


def draw_label(image, lines):
    import cv2

    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_height = 18
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 12
    height = line_height * len(lines) + 8
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, out.shape[1]), min(height, out.shape[0])), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, line in enumerate(lines):
        y = 16 + i * line_height
        cv2.putText(out, line, (6, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def make_contact_sheet(image_paths, out_path, thumb_width=220, cols=5):
    import cv2

    if not image_paths:
        return
    images = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = thumb_width / float(img.shape[1])
        thumb = cv2.resize(img, (thumb_width, max(1, int(img.shape[0] * scale))))
        images.append(thumb)
    if not images:
        return
    thumb_height = max(img.shape[0] for img in images)
    cols = max(int(cols), 1)
    rows = int(math.ceil(len(images) / float(cols)))
    sheet = np.full((rows * thumb_height, cols * thumb_width, 3), 240, dtype=np.uint8)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = row * thumb_height
        x = col * thumb_width
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
    cv2.imwrite(str(out_path), sheet)


def export_keyframes(f, h5_path, out_dir, camera, subtask_kf, motion_kf, reason_sets, opts):
    out_dir = Path(out_dir)
    if opts["export_per_file_subdir"]:
        out_dir = out_dir / Path(h5_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_key = f"observation/{camera}/rgb"
    if rgb_key not in f:
        cameras = [
            name
            for name in f.get("observation", {}).keys()
            if f"observation/{name}/rgb" in f
        ]
        raise KeyError(f"{rgb_key} not found; available cameras: {cameras}")

    export_sets = []
    if opts["export_mode"] in {"all", "subtask"}:
        export_sets.append(("subtask", "subtask_keyframes", np.flatnonzero(subtask_kf)))
    if opts["export_mode"] in {"all", "motion"}:
        export_sets.append(("motion", "motion_keyframes", np.flatnonzero(motion_kf)))

    export_info = {"export_dir": str(out_dir)}
    for field_name, subdir_name, frames in export_sets:
        field_dir = out_dir / subdir_name
        field_dir.mkdir(parents=True, exist_ok=True)
        csv_path = field_dir / "keyframes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame", "subtask_keyframe", "motion_keyframe", "subtask", "reasons"])
            subtask = f["subtask"][:]
            for frame in frames:
                writer.writerow(
                    [
                        int(frame),
                        int(subtask_kf[frame]),
                        int(motion_kf[frame]),
                        int(subtask[frame]),
                        ";".join(sorted(reason_sets[int(frame)])),
                    ]
                )

        saved_paths = []
        max_export = opts["max_export"]
        selected = frames if max_export <= 0 else frames[:max_export]
        for frame in selected:
            frame = int(frame)
            reasons = sorted(reason_sets[frame])
            label_lines = [
                f"{Path(h5_path).name} frame={frame}",
                f"field={field_name} subtask={int(f['subtask'][frame])}",
                f"subtask_kf={int(subtask_kf[frame])} motion_kf={int(motion_kf[frame])}",
            ]
            if reasons:
                reason_text = ";".join(reasons)
                label_lines.extend([reason_text[i : i + 80] for i in range(0, len(reason_text), 80)])
            image = decode_rgb(f[rgb_key][frame])
            image = draw_label(image, label_lines)
            image_path = field_dir / f"frame_{frame:05d}.png"
            import cv2

            cv2.imwrite(str(image_path), image)
            saved_paths.append(image_path)

        make_contact_sheet(saved_paths, field_dir / "contact_sheet.png", cols=opts["contact_sheet_cols"])
        export_info[f"{field_name}_csv_path"] = str(csv_path)
        export_info[f"{field_name}_export_dir"] = str(field_dir)
        export_info[f"{field_name}_exported_images"] = len(saved_paths)
        export_info[f"{field_name}_selected_keyframes"] = int(len(frames))
    return export_info


def write_dataset(f, name, values, attrs, overwrite):
    if name in f:
        if not overwrite:
            raise RuntimeError(f"{name} already exists; use --overwrite or --skip-existing")
        del f[name]
    ds = f.create_dataset(name, data=values.astype(np.int8))
    for key, value in attrs.items():
        ds.attrs[key] = value


def process_one(path, opts):
    path = str(path)
    mode = "r" if opts["dry_run"] else "r+"
    with h5py.File(path, mode) as f:
        if "subtask" not in f:
            raise KeyError("missing required dataset: subtask")
        t = len(f["subtask"])
        if opts["skip_existing"] and "subtask_keyframe" in f and "motion_keyframe" in f:
            return {"path": path, "status": "skipped", "T": t}
        existing = [name for name in ("subtask_keyframe", "motion_keyframe") if name in f]
        if existing and not opts["overwrite"] and not opts["dry_run"]:
            raise RuntimeError(f"{existing} already exist; use --overwrite or --skip-existing")

        subtask_kf = subtask_keyframes(f["subtask"][:], mark_first=opts["mark_subtask_initial"])
        motion_kf, reason_sets, stats = motion_keyframes(f, opts)

        now = datetime.now().isoformat(timespec="seconds")
        common_attrs = {
            "generated_by": "script/add_hdf5_keyframes.py",
            "version": VERSION,
            "generated_at": now,
        }
        if not opts["dry_run"]:
            write_dataset(
                f,
                "subtask_keyframe",
                subtask_kf,
                {
                    **common_attrs,
                    "definition": "1 when subtask changes; optionally frame 0",
                    "mark_initial": int(opts["mark_subtask_initial"]),
                },
                opts["overwrite"],
            )
            motion_attrs = {
                **common_attrs,
                "definition": "strictly causal motion events from EE direction, confirmed stops, head/torso motion, and gripper state boundaries",
                "causal": 1,
                "causal_semantics": "frame t depends only on samples 0..t; persistent events are marked when confirmed, never backdated",
            }
            for key, value in opts.items():
                if isinstance(value, (bool, int, float, str)):
                    motion_attrs[f"param_{key}"] = value
            for key, value in stats.items():
                motion_attrs[key] = value
            write_dataset(f, "motion_keyframe", motion_kf, motion_attrs, opts["overwrite"])

        export_info = {}
        if opts["export_dir"]:
            export_info = export_keyframes(
                f,
                path,
                opts["export_dir"],
                opts["export_camera"],
                subtask_kf,
                motion_kf,
                reason_sets,
                opts,
            )

    reason_counts = {}
    for reasons in reason_sets:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "path": path,
        "status": "dry_run" if opts["dry_run"] else "written",
        "T": t,
        "subtask_keyframes": int(subtask_kf.sum()),
        "motion_keyframes": int(motion_kf.sum()),
        "reason_counts": reason_counts,
        **stats,
        **export_info,
    }


def collect_hdf5(paths):
    files = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix in {".hdf5", ".h5"}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.hdf5")))
            files.extend(sorted(path.rglob("*.h5")))
        else:
            raise FileNotFoundError(str(path))
    deduped = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def parse_args():
    parser = argparse.ArgumentParser(description="Add subtask_keyframe and motion_keyframe datasets to ActiveArena HDF5 episodes.")
    parser.add_argument("paths", nargs="+", help="HDF5 files or directories to process.")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing keyframe datasets.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have both keyframe datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and export without writing HDF5 datasets.")

    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--speed-min", type=float, default=0.003)
    parser.add_argument("--stop-speed", type=float, default=0.002)
    parser.add_argument("--adaptive-speed-percentile", type=float, default=20.0)
    parser.add_argument("--adaptive-speed-scale", type=float, default=0.5)
    parser.add_argument("--adaptive-stop-percentile", type=float, default=15.0)
    parser.add_argument("--adaptive-stop-scale", type=float, default=0.5)
    parser.add_argument("--angle-deg", type=float, default=75.0)
    parser.add_argument("--min-stop-len", type=int, default=5)
    parser.add_argument("--stop-merge-gap", type=int, default=2)
    parser.add_argument("--stop-lookback", type=int, default=12)
    parser.add_argument("--min-move-before-stop", type=float, default=0.02)
    parser.add_argument("--joint-threshold", type=float, default=0.005)
    parser.add_argument("--min-joint-motion-len", type=int, default=4)
    parser.add_argument("--joint-merge-gap", type=int, default=5)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.2)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.8)
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.03)
    parser.add_argument("--min-gripper-state-len", type=int, default=2)
    parser.add_argument("--min-gripper-motion-len", type=int, default=2)
    parser.add_argument("--gripper-state-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedup-window", type=int, default=5, help="Drop later motion candidates within this many frames of the previous kept frame.")
    parser.add_argument("--min-gap", type=int, default=None, help="Deprecated alias for old spacing semantics; --min-gap 6 is equivalent to --dedup-window 5.")
    parser.add_argument("--no-mark-subtask-initial", action="store_true")
    parser.add_argument("--no-mark-motion-initial", action="store_true")

    parser.add_argument("--export-dir", type=str, default="")
    parser.add_argument("--export-camera", type=str, default="camera_head")
    parser.add_argument("--export-mode", choices=["all", "motion", "subtask"], default="all")
    parser.add_argument("--max-export", type=int, default=200, help="Maximum PNG frames to export per file; <=0 exports all.")
    parser.add_argument("--no-export-per-file-subdir", action="store_true")
    parser.add_argument("--contact-sheet-cols", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    files = collect_hdf5(args.paths)
    if not files:
        raise RuntimeError("no HDF5 files found")

    opts = vars(args).copy()
    opts["mark_subtask_initial"] = not args.no_mark_subtask_initial
    opts["mark_motion_initial"] = not args.no_mark_motion_initial
    opts["export_per_file_subdir"] = not args.no_export_per_file_subdir
    if args.min_gap is not None:
        opts["dedup_window"] = max(int(args.min_gap) - 1, 0)
    opts.pop("paths", None)
    opts.pop("no_mark_subtask_initial", None)
    opts.pop("no_mark_motion_initial", None)
    opts.pop("no_export_per_file_subdir", None)

    workers = max(int(args.workers), 1)
    print(f"Found {len(files)} HDF5 file(s); workers={workers}; dry_run={args.dry_run}")

    results = []
    if workers == 1 or len(files) == 1:
        for path in files:
            results.append(process_one(path, opts))
            print_result(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {executor.submit(process_one, path, opts): path for path in files}
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                print_result(result)

    written = sum(1 for result in results if result.get("status") == "written")
    skipped = sum(1 for result in results if result.get("status") == "skipped")
    print(f"Done. written={written} skipped={skipped} total={len(results)}")


def print_result(result):
    status = result.get("status", "unknown")
    path = result.get("path", "")
    subtask_count = result.get("subtask_keyframes", "-")
    motion_count = result.get("motion_keyframes", "-")
    print(f"[{status}] {path} T={result.get('T', '-')} subtask={subtask_count} motion={motion_count}")
    if result.get("export_dir"):
        print(f"  export_dir={result['export_dir']}")
        for field_name in ("subtask", "motion"):
            field_dir = result.get(f"{field_name}_export_dir")
            if field_dir:
                print(
                    f"  {field_name}: dir={field_dir} images={result.get(f'{field_name}_exported_images', 0)} "
                    f"csv={result.get(f'{field_name}_csv_path')}"
                )


if __name__ == "__main__":
    main()
