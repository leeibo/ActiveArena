import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
from functools import lru_cache
import shutil
from PIL import Image, ImageDraw, ImageFont
from .images_to_video import images_to_video


_CAMERA_ALIASES = {
    "camera_head": ["head_camera"],
    "head_camera": ["camera_head"],
}

_ANNOTATION_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
)

_STAGE_LABELS = {
    0: "idle",
    1: "scan",
    2: "focus",
    3: "act",
}


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")

def _resolve_camera_key(observation_dict, camera_name):
    if not isinstance(observation_dict, Mapping):
        return None
    candidates = [camera_name] + _CAMERA_ALIASES.get(camera_name, [])
    for key in candidates:
        cam_data = observation_dict.get(key, None)
        if isinstance(cam_data, Mapping) and "rgb" in cam_data:
            return key
    return None


@lru_cache(maxsize=16)
def _load_annotation_font(font_size):
    font_size = max(12, int(font_size))
    for font_path in _ANNOTATION_FONT_CANDIDATES:
        if not os.path.exists(font_path):
            continue
        try:
            return ImageFont.truetype(font_path, size=font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _normalize_rgb_frame(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {img.shape}")
    return img


def _draw_panel(draw, box, radius, fill):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def _text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
    return right - left, bottom - top


def _split_long_token(draw, token, font, max_width):
    if token == "":
        return [""]
    chunks = []
    current = ""
    for ch in token:
        trial = current + ch
        if current and _text_size(draw, trial, font)[0] > max_width:
            chunks.append(current)
            current = ch
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks or [token]


def _wrap_text(draw, text, font, max_width):
    text = str(text).strip()
    if text == "":
        return ["-"]

    wrapped_lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split(" ")
        current_line = ""
        for raw_word in words:
            word = raw_word.strip()
            if word == "":
                continue
            if _text_size(draw, word, font)[0] > max_width:
                if current_line:
                    wrapped_lines.append(current_line)
                    current_line = ""
                wrapped_lines.extend(_split_long_token(draw, word, font, max_width))
                continue

            trial = word if current_line == "" else f"{current_line} {word}"
            if current_line and _text_size(draw, trial, font)[0] > max_width:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = trial

        if current_line:
            wrapped_lines.append(current_line)
        elif len(words) == 0:
            wrapped_lines.append("")
    return wrapped_lines or ["-"]


def _resolve_annotation_instruction(annotation, subtask_instruction_map):
    if not isinstance(annotation, Mapping):
        return "idle"
    instruction_idx = annotation.get("subtask_instruction_idx", annotation.get("subtask", 0))
    try:
        instruction_idx = int(instruction_idx)
    except Exception:
        instruction_idx = 0
    if isinstance(subtask_instruction_map, Mapping):
        return str(
            subtask_instruction_map.get(
                str(instruction_idx),
                subtask_instruction_map.get(instruction_idx, "idle"),
            )
        )
    return "idle"


def _resolve_found_object_names(annotation, object_key_to_name):
    if not isinstance(annotation, Mapping):
        return []
    object_keys = annotation.get("discovered_object_keys", []) or []
    resolved_names = []
    for key in object_keys:
        norm_key = str(key)
        name = object_key_to_name.get(norm_key, norm_key) if isinstance(object_key_to_name, Mapping) else norm_key
        name = str(name)
        if name not in resolved_names:
            resolved_names.append(name)
    return resolved_names


def _annotate_video_frames(imgs, annotation_metadata):
    imgs = np.asarray(imgs)
    if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
        raise ValueError(f"Invalid RGB tensor for annotation with shape {imgs.shape}")

    frame_annotations = []
    subtask_instruction_map = {}
    object_key_to_name = {}
    if isinstance(annotation_metadata, Mapping):
        frame_annotations = list(annotation_metadata.get("frame_annotations", []) or [])
        subtask_instruction_map = annotation_metadata.get("subtask_instruction_map", {}) or {}
        object_key_to_name = annotation_metadata.get("object_key_to_name", {}) or {}

    total_frames = int(imgs.shape[0])
    annotated_frames = np.empty((imgs.shape[0], imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)

    for frame_idx in range(total_frames):
        img = _normalize_rgb_frame(imgs[frame_idx])
        annotation = frame_annotations[frame_idx] if frame_idx < len(frame_annotations) else {}

        canvas = Image.fromarray(img).convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = canvas.size
        margin_x = max(12, int(width * 0.03))
        margin_y = max(12, int(height * 0.03))
        pad_x = max(10, int(width * 0.016))
        pad_y = max(8, int(height * 0.014))
        panel_gap = max(8, int(height * 0.02))
        panel_radius = max(8, int(height * 0.02))

        stats_font = _load_annotation_font(height * 0.034)
        body_font = _load_annotation_font(height * 0.036)
        stats_line_height = _text_size(draw, "Ag", stats_font)[1]
        body_line_height = _text_size(draw, "Ag", body_font)[1]
        line_gap = max(4, int(height * 0.008))

        subtask_idx = annotation.get("subtask", 0) if isinstance(annotation, Mapping) else 0
        stage = annotation.get("stage", 0) if isinstance(annotation, Mapping) else 0
        try:
            stage = int(stage)
        except Exception:
            stage = 0
        stats_lines = [
            f"frame: {frame_idx} / {max(total_frames - 1, 0)}",
            f"subtask: {subtask_idx} | stage: {stage} ({_STAGE_LABELS.get(stage, 'unknown')})",
        ]
        waist_heading_deg = annotation.get("waist_heading_deg", None) if isinstance(annotation, Mapping) else None
        if waist_heading_deg is None:
            stats_lines.append("waist angle: n/a")
        else:
            try:
                waist_heading_deg = float(waist_heading_deg)
            except Exception:
                waist_heading_deg = np.nan
            if np.isfinite(waist_heading_deg):
                stats_lines.append(f"waist angle: {waist_heading_deg:+.1f} deg")
            else:
                stats_lines.append("waist angle: n/a")

        instruction_text = _resolve_annotation_instruction(annotation, subtask_instruction_map)
        found_object_names = _resolve_found_object_names(annotation, object_key_to_name)
        found_objects_text = ", ".join(found_object_names) if found_object_names else "none"

        detail_width = width - (2 * margin_x) - (2 * pad_x)
        detail_lines = []
        detail_lines.extend(_wrap_text(draw, f"sub task instruction: {instruction_text}", body_font, detail_width))
        detail_lines.extend(_wrap_text(draw, f"found objects: {found_objects_text}", body_font, detail_width))

        stats_width = max(_text_size(draw, line, stats_font)[0] for line in stats_lines)
        stats_panel_w = stats_width + (2 * pad_x)
        stats_panel_h = (len(stats_lines) * stats_line_height) + ((len(stats_lines) - 1) * line_gap) + (2 * pad_y)
        stats_box = (margin_x, margin_y, margin_x + stats_panel_w, margin_y + stats_panel_h)
        _draw_panel(draw, stats_box, panel_radius, fill=(12, 18, 28, 200))

        cursor_y = margin_y + pad_y
        for line in stats_lines:
            draw.text((margin_x + pad_x, cursor_y), line, font=stats_font, fill=(245, 248, 252, 255))
            cursor_y += stats_line_height + line_gap

        detail_panel_h = (len(detail_lines) * body_line_height) + ((len(detail_lines) - 1) * line_gap) + (2 * pad_y)
        detail_box_top = height - margin_y - detail_panel_h
        min_detail_top = stats_box[3] + panel_gap
        if detail_box_top < min_detail_top:
            detail_box_top = min_detail_top
        detail_box = (margin_x, detail_box_top, width - margin_x, detail_box_top + detail_panel_h)
        _draw_panel(draw, detail_box, panel_radius, fill=(18, 24, 35, 192))

        cursor_y = detail_box_top + pad_y
        for line in detail_lines:
            draw.text((margin_x + pad_x, cursor_y), line, font=body_font, fill=(255, 255, 255, 255))
            cursor_y += body_line_height + line_gap

        annotated_frames[frame_idx] = np.array(Image.alpha_composite(canvas, overlay).convert("RGB"))

    return annotated_frames


def _dump_videos(
    data_list,
    video_path=None,
    video_camera_names=None,
    video_path_map=None,
    main_video_camera="camera_head",
    annotated_video_path=None,
    annotated_video_camera="camera_head",
    annotated_video_metadata=None,
):
    if video_path is None and not video_path_map:
        if annotated_video_path is None:
            return
    observation_dict = data_list.get("observation", {})
    if not isinstance(observation_dict, Mapping):
        print("[pkl2hdf5] observation is missing, skip video export")
        return

    video_jobs = []

    # Main legacy video path for compatibility.
    if video_path is not None:
        main_key = _resolve_camera_key(observation_dict, main_video_camera)
        if main_key is None:
            main_key = _resolve_camera_key(observation_dict, "camera_head")
        if main_key is None:
            main_key = _resolve_camera_key(observation_dict, "head_camera")
        if main_key is None and video_camera_names:
            for name in video_camera_names:
                main_key = _resolve_camera_key(observation_dict, name)
                if main_key is not None:
                    break
        if main_key is not None:
            video_jobs.append((main_key, video_path))
        else:
            print(f"[pkl2hdf5] no RGB camera found for legacy video path: {video_path}")

    # Multi-view camera videos.
    if video_path_map:
        for camera_name, out_path in video_path_map.items():
            cam_key = _resolve_camera_key(observation_dict, camera_name)
            if cam_key is None:
                print(f"[pkl2hdf5] skip video for {camera_name}: camera stream not found")
                continue
            video_jobs.append((cam_key, out_path))
    elif video_path is not None and video_camera_names:
        base, ext = os.path.splitext(video_path)
        if ext == "":
            ext = ".mp4"
        for camera_name in video_camera_names:
            cam_key = _resolve_camera_key(observation_dict, camera_name)
            if cam_key is None:
                print(f"[pkl2hdf5] skip video for {camera_name}: camera stream not found")
                continue
            video_jobs.append((cam_key, f"{base}_{camera_name}{ext}"))

    # Deduplicate by output path.
    dedup = {}
    for cam_key, out_path in video_jobs:
        dedup[out_path] = cam_key

    for out_path, cam_key in dedup.items():
        imgs = np.array(observation_dict[cam_key]["rgb"])
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            print(f"[pkl2hdf5] invalid RGB tensor for {cam_key}, skip {out_path}")
            continue
        images_to_video(imgs, out_path=out_path)

    if annotated_video_path is not None:
        annotated_key = _resolve_camera_key(observation_dict, annotated_video_camera)
        if annotated_key is None:
            annotated_key = _resolve_camera_key(observation_dict, "camera_head")
        if annotated_key is None:
            annotated_key = _resolve_camera_key(observation_dict, "head_camera")
        if annotated_key is None:
            print(f"[pkl2hdf5] no RGB camera found for annotated video path: {annotated_video_path}")
            return
        imgs = np.array(observation_dict[annotated_key]["rgb"])
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            print(f"[pkl2hdf5] invalid RGB tensor for annotated video on {annotated_key}, skip {annotated_video_path}")
            return
        annotated_imgs = _annotate_video_frames(imgs, annotated_video_metadata)
        images_to_video(annotated_imgs, out_path=annotated_video_path)


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path=None,
    video_camera_names=None,
    video_path_map=None,
    main_video_camera="camera_head",
    annotated_video_path=None,
    annotated_video_camera="camera_head",
    annotated_video_metadata=None,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    _dump_videos(
        data_list=data_list,
        video_path=video_path,
        video_camera_names=video_camera_names,
        video_path_map=video_path_map,
        main_video_camera=main_video_camera,
        annotated_video_path=annotated_video_path,
        annotated_video_camera=annotated_video_camera,
        annotated_video_metadata=annotated_video_metadata,
    )

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path=None,
    video_camera_names=None,
    video_path_map=None,
    main_video_camera="camera_head",
    annotated_video_path=None,
    annotated_video_camera="camera_head",
    annotated_video_metadata=None,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path=video_path,
        video_camera_names=video_camera_names,
        video_path_map=video_path_map,
        main_video_camera=main_video_camera,
        annotated_video_path=annotated_video_path,
        annotated_video_camera=annotated_video_camera,
        annotated_video_metadata=annotated_video_metadata,
    )
