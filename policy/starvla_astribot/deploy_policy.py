from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .motion_keyframes import CausalMotionKeyframeDetector, MotionKeyframeConfig
except ImportError:
    from motion_keyframes import CausalMotionKeyframeDetector, MotionKeyframeConfig

try:
    from .subtask_planner import (
        DEFAULT_QWEN3_30B_PATH,
        SubtaskPlannerError,
        build_subtask_planner,
        extract_subtask_text,
    )
except ImportError:
    from subtask_planner import (
        DEFAULT_QWEN3_30B_PATH,
        SubtaskPlannerError,
        build_subtask_planner,
        extract_subtask_text,
    )


ACTIVEARENA_VLA_ROOT = Path(
    os.environ.get(
        "ACTIVEARENA_VLA_ROOT",
        os.environ.get(
            "STARVLA_REPO_ROOT",  # deprecated compatibility alias
            Path(__file__).resolve().parents[2] / "../ActiveArena-VLA",
        ),
    )
).expanduser()
# Internal alias retained because older checkpoint adapters import this name.
STARVLA_ROOT = ACTIVEARENA_VLA_ROOT
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

ACTION_ORDER = "left_arm7,left_gripper,right_arm7,right_gripper,torso,head2"
ROBOTWIN_ORDER = "left_arm7,left_gripper,right_arm7,right_gripper,head1_delta,head2_delta,torso_delta"
HISTORY_NONE = {"", "none", "no", "disabled", "current", "current_only"}
KEYFRAME_HISTORY_MODES = {
    "motion_keyframe",
    "subtask_keyframe",
    "subtask_window_memory",
}
SUBTASK_KEYFRAME_HISTORY_MODES = {
    "subtask_keyframe",
    "subtask_window_memory",
}
INVALID_FAST_ACTION_ERROR_MARKER = "QwenFast generation produced invalid FAST action token sequence"


@dataclass
class FrameRecord:
    step: int
    image: np.ndarray
    state: np.ndarray
    annotation: dict[str, Any]
    subtask_keyframe: bool = False
    motion_keyframe: bool = False


def _is_invalid_fast_action_error(error: BaseException) -> bool:
    return INVALID_FAST_ACTION_ERROR_MARKER in str(error)


def _hold_current_state_actions(records: list[FrameRecord], action_steps: int) -> np.ndarray:
    if not records:
        raise ValueError("Cannot build hold actions without a current frame record")
    state = np.asarray(records[-1].state, dtype=np.float64).reshape(-1)
    if state.shape != (18,):
        raise ValueError(f"Expected current Astribot state shape (18,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("Current Astribot state contains non-finite values")
    return np.repeat(state[None, :], max(int(action_steps), 1), axis=0)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "none", ""}
    return bool(value)


def _as_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, "", "none", "None", "null", "Null", -1, "-1"):
        return default
    return int(value)


def _as_str_list(value: Any, default: Optional[list[str]] = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _parse_image_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"image_size string must be 'width,height', got {value!r}")
        return parts[0], parts[1]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"image_size must have two values, got {value!r}")


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _as_1d(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if arr.shape[0] >= length:
        return arr[:length].astype(np.float32, copy=False)
    out = np.full(length, default, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def _scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value if value is not None else [default], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def build_astribot_18d_state(
    observation: dict[str, Any],
    torso_index: int = 0,
    head2_index: int = 1,
) -> np.ndarray:
    joint_action = observation.get("joint_action", {})
    left_arm = _as_1d(joint_action.get("left_arm"), 7)
    right_arm = _as_1d(joint_action.get("right_arm"), 7)
    left_gripper = _scalar(joint_action.get("left_gripper"), 1.0)
    right_gripper = _scalar(joint_action.get("right_gripper"), 1.0)
    torso = _as_1d(joint_action.get("torso"), max(1, int(torso_index) + 1), default=0.0)
    head = _as_1d(joint_action.get("head"), max(2, int(head2_index) + 1), default=1.0)
    return np.concatenate(
        [
            left_arm,
            np.array([left_gripper], dtype=np.float32),
            right_arm,
            np.array([right_gripper], dtype=np.float32),
            np.array([torso[int(torso_index)], head[int(head2_index)]], dtype=np.float32),
        ]
    )


def encode_obs(observation: dict[str, Any]) -> tuple[np.ndarray, str]:
    obs = observation["observation"]
    for camera_key in ("camera_head", "head_camera", "camera_front", "cam_head"):
        if camera_key in obs and "rgb" in obs[camera_key]:
            return obs[camera_key]["rgb"], camera_key
    raise KeyError(f"Observation has no head RGB camera. Available cameras: {list(obs.keys())}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_name_part(value: Any) -> str:
    text = "unknown" if value is None else str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    return val if np.isfinite(val) else None


def _rounded_float(value: Any, digits: int = 4) -> float | None:
    val = _safe_float(value)
    return None if val is None else round(float(val), int(digits))


def _current_joint_value(values: Any, default: float = 0.0, index: int = 0) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == 0:
        return float(default)
    index = min(max(int(index), 0), values.shape[0] - 1)
    return float(values[index])


def adapt_model_action_to_robotwin(action: np.ndarray, task_env) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] == 19:
        return action
    if action.shape[0] != 18:
        raise ValueError(f"Expected StarVLA Astribot action dim 18 or 19, got {action.shape[0]}")

    head_now = task_env._get_head_joint_state_now()
    torso_now = task_env._get_torso_joint_state_now()
    head2_now = _current_joint_value(head_now, index=1)
    torso_now = _current_joint_value(torso_now, index=0)
    torso_target = float(action[16])
    head2_target = float(action[17])

    env_action = np.zeros(19, dtype=np.float64)
    env_action[:16] = action[:16]
    env_action[16] = 0.0
    env_action[17] = head2_target - head2_now
    env_action[18] = torso_target - torso_now
    return env_action


class StarVLAOFTClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7980,
        unnorm_key: Optional[str] = None,
        image_size: tuple[int, int] = (224, 224),
        action_steps: Optional[int] = None,
        action_type: str = "qpos",
        history_mode: str = "action_keyframe",
        history_frames: Optional[int] = 12,
        history_stride: Optional[int] = None,
        send_state: bool = False,
        swap_rgb_channels: bool = False,
        request_log_path: Optional[str] = None,
        request_image_dir: Optional[str] = None,
        motion_min_interval: int = 16,
        motion_state_delta_threshold: float = 0.05,
        motion_keyframe_strategy: str = "legacy",
        motion_gripper_close_threshold: float = 0.2,
        motion_gripper_open_threshold: float = 0.8,
        motion_min_gripper_state_len: int = 2,
        motion_rotation_delta_threshold: float = 0.005,
        motion_min_rotation_motion_len: int = 4,
        motion_rotation_merge_gap: int = 5,
        motion_dedup_window: int = 5,
        motion_mark_initial: bool = True,
        subtask_fallback_interval: int = 0,
        clip_grippers: bool = True,
        torso_index: int = 0,
        head2_index: int = 1,
        log_chunk_timing: bool = True,
        return_vlm_text: bool = False,
        vlm_text_max_new_tokens: int = 128,
        subtask_planner_enabled: bool = False,
        subtask_planner_backend: str = "http",
        subtask_planner_url: str = "http://127.0.0.1:7991/classify",
        subtask_planner_model_path: str = DEFAULT_QWEN3_30B_PATH,
        subtask_planner_timeout: float = 30.0,
        subtask_planner_device_map: str = "auto",
        subtask_planner_dtype: str = "bfloat16",
        subtask_planner_attn_implementation: str = "sdpa",
        subtask_planner_max_new_tokens: int = 192,
        subtask_planner_model_name: str = "qwen3.5-9b",
        subtask_planner_api_key: Optional[str] = None,
        subtask_planner_disable_reasoning: bool = True,
        subtask_planner_output_keys: Optional[list[str]] = None,
        subtask_planner_use_annotation: bool = False,
        subtask_planner_fail_open: bool = True,
        hold_on_invalid_fast_action: bool = False,
    ) -> None:
        try:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        except ModuleNotFoundError as exc:
            if exc.name == "websockets":
                raise ModuleNotFoundError(
                    "The ActiveArena websocket client requires `websockets`; run ActiveArena eval "
                    "in an environment with StarVLA dependencies."
                ) from exc
            raise

        self.client = WebsocketClientPolicy(host=host, port=port)
        self.server_metadata = self.client.get_server_metadata()
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.action_type = str(action_type)
        self.action_chunk_size = int(self.server_metadata.get("action_chunk_size", action_steps or 16))
        self.action_steps = int(action_steps or self.action_chunk_size)
        self.history_mode = str(history_mode or "none").strip().lower()
        self.max_history_frames = history_frames
        if self.max_history_frames is not None:
            self.max_history_frames = max(int(self.max_history_frames), 0)
        self.history_stride = int(history_stride or self.action_chunk_size)
        if self.history_stride <= 0:
            raise ValueError(f"history_stride must be positive, got {self.history_stride}")
        self.send_state = bool(send_state)
        self.swap_rgb_channels = bool(swap_rgb_channels)
        self.motion_min_interval = max(int(motion_min_interval), 0)
        self.motion_state_delta_threshold = float(motion_state_delta_threshold)
        self.motion_keyframe_strategy = str(motion_keyframe_strategy).strip().lower()
        if self.motion_keyframe_strategy not in {"legacy", "causal_gripper_rotation"}:
            raise ValueError(
                "motion_keyframe_strategy must be 'legacy' or "
                f"'causal_gripper_rotation', got {motion_keyframe_strategy!r}"
            )
        self.motion_keyframe_detector = None
        if self.motion_keyframe_strategy == "causal_gripper_rotation":
            self.motion_keyframe_detector = CausalMotionKeyframeDetector(
                MotionKeyframeConfig(
                    gripper_close_threshold=float(motion_gripper_close_threshold),
                    gripper_open_threshold=float(motion_gripper_open_threshold),
                    min_gripper_state_len=int(motion_min_gripper_state_len),
                    rotation_delta_threshold=float(motion_rotation_delta_threshold),
                    min_rotation_motion_len=int(motion_min_rotation_motion_len),
                    rotation_merge_gap=int(motion_rotation_merge_gap),
                    dedup_window=int(motion_dedup_window),
                    mark_initial=bool(motion_mark_initial),
                    left_gripper_index=7,
                    right_gripper_index=15,
                    torso_yaw_index=16,
                    head_2_index=17,
                )
            )
        self.subtask_fallback_interval = max(int(subtask_fallback_interval), 0)
        self.clip_grippers = bool(clip_grippers)
        self.torso_index = int(torso_index)
        self.head2_index = int(head2_index)
        self.log_chunk_timing_enabled = bool(log_chunk_timing)
        self.return_vlm_text = bool(return_vlm_text)
        self.vlm_text_max_new_tokens = int(vlm_text_max_new_tokens)
        self.subtask_planner_enabled = bool(subtask_planner_enabled)
        self.hold_on_invalid_fast_action = bool(hold_on_invalid_fast_action)
        self.subtask_planner_fail_open = bool(subtask_planner_fail_open)
        self.subtask_planner_use_annotation = bool(subtask_planner_use_annotation)
        self.subtask_planner_output_keys = list(
            subtask_planner_output_keys
            or ["subtask", "current_subtask", "subtask_instruction", "vlm_text", "text", "output", "generated_text"]
        )
        self.subtask_planner = None
        if self.subtask_planner_enabled:
            try:
                self.subtask_planner = build_subtask_planner(
                    subtask_planner_backend,
                    url=subtask_planner_url,
                    model_path=subtask_planner_model_path,
                    timeout=float(subtask_planner_timeout),
                    device_map=subtask_planner_device_map,
                    dtype=subtask_planner_dtype,
                    attn_implementation=subtask_planner_attn_implementation,
                    max_new_tokens=int(subtask_planner_max_new_tokens),
                    model_name=subtask_planner_model_name,
                    api_key=subtask_planner_api_key,
                    disable_reasoning=subtask_planner_disable_reasoning,
                )
            except Exception:
                if not self.subtask_planner_fail_open:
                    raise
                self.subtask_planner_enabled = False
                print("[StarVLAOFTClient] subtask planner init failed; disabled with fail_open=True", flush=True)

        self.request_log_path = Path(request_log_path).expanduser() if request_log_path else None
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.request_image_dir = Path(request_image_dir).expanduser() if request_image_dir else None
        if self.request_image_dir is not None:
            self.request_image_dir.mkdir(parents=True, exist_ok=True)

        self.language: Optional[str] = None
        self.records: list[FrameRecord] = []
        self.request_seed = None
        self.episode_idx = -1
        self.chunk_idx = 0
        self.last_request_id = None
        self.last_request_sec = None
        self.last_chunk_size = None
        self.last_subtask_signature = None
        self.last_planner_subtask_text: Optional[str] = None
        self.last_motion_signature = None
        self.last_motion_keyframe_step: Optional[int] = None
        self.last_motion_keyframe_state: Optional[np.ndarray] = None
        self.last_subtask_keyframe_step: Optional[int] = None
        self.current_candidate_subtasks: list[str] = []

        print(
            "[StarVLAOFTClient] connected: "
            f"host={host}, port={port}, action_chunk_size={self.action_chunk_size}, "
            f"action_steps={self.action_steps}, history_mode={self.history_mode}, "
            f"history_frames={self.max_history_frames}, send_state={self.send_state}, "
            f"motion_keyframe_strategy={self.motion_keyframe_strategy}, "
            f"swap_rgb_channels={self.swap_rgb_channels}, unnorm_key={self.unnorm_key}, "
            f"return_vlm_text={self.return_vlm_text}, "
            f"subtask_planner_enabled={self.subtask_planner_enabled}, "
            f"hold_on_invalid_fast_action={self.hold_on_invalid_fast_action}, "
            f"metadata={self.server_metadata}",
            flush=True,
        )

    def reset(self) -> None:
        self.episode_idx += 1
        self.chunk_idx = 0
        self.language = None
        self.records.clear()
        self.last_subtask_signature = None
        self.last_planner_subtask_text = None
        self.last_motion_signature = None
        self.last_motion_keyframe_step = None
        self.last_motion_keyframe_state = None
        self.last_subtask_keyframe_step = None
        if self.motion_keyframe_detector is not None:
            self.motion_keyframe_detector.reset()
        self.last_request_id = None
        self.last_request_sec = None
        self.last_chunk_size = None
        self.current_candidate_subtasks = []

    def set_request_seed(self, seed: Any) -> None:
        self.request_seed = seed

    def close(self) -> None:
        self.client.close()

    def _annotation_from_env(self, task_env, observation: dict[str, Any]) -> dict[str, Any]:
        annotation: dict[str, Any] = {}
        latest = getattr(task_env, "_latest_frame_annotation", None)
        if isinstance(latest, dict):
            annotation.update(latest)
        for key in (
            "subtask",
            "stage",
            "subtask_instruction_idx",
            "focus_object_key",
            "search_target_keys",
            "action_target_keys",
            "camera_target_theta",
            "waist_heading_deg",
            "target_uv_norm",
            "camera_mode",
        ):
            if key in observation:
                annotation[key] = _jsonable(observation[key])
        annotation.setdefault("subtask", _safe_int(getattr(task_env, "current_subtask_idx", 0), 0))
        annotation.setdefault("stage", _safe_int(getattr(task_env, "current_stage", 0), 0))
        annotation.setdefault("subtask_instruction_idx", _safe_int(getattr(task_env, "current_instruction_idx", 0), 0))
        annotation["take_action_cnt"] = _safe_int(getattr(task_env, "take_action_cnt", 0), 0)
        annotation["frame_idx"] = _safe_int(getattr(task_env, "FRAME_IDX", annotation["take_action_cnt"]), annotation["take_action_cnt"])
        return annotation

    def _annotation_is_reliable(self, annotation: dict[str, Any]) -> bool:
        return bool(_safe_int(annotation.get("subtask", 0), 0) > 0 or _safe_int(annotation.get("stage", 0), 0) > 0)

    def _subtask_signature(self, annotation: dict[str, Any]) -> tuple[int, int]:
        return (
            _safe_int(annotation.get("subtask", 0), 0),
            _safe_int(annotation.get("subtask_instruction_idx", 0), 0),
        )

    def _motion_signature(self, annotation: dict[str, Any]) -> tuple[Any, ...]:
        stage = _safe_int(annotation.get("stage", 0), 0)
        if stage == 3:
            theta = None
        else:
            theta = _rounded_float(annotation.get("camera_target_theta", None), digits=4)
        return (
            _safe_int(annotation.get("subtask", 0), 0),
            stage,
            theta,
            annotation.get("focus_object_key", None),
            tuple(annotation.get("search_target_keys", []) or []),
            tuple(annotation.get("action_target_keys", []) or []),
        )

    def _subtask_instruction(self, task_env, annotation: dict[str, Any]) -> Optional[str]:
        instruction_idx = _safe_int(annotation.get("subtask_instruction_idx", annotation.get("subtask", 0)), 0)
        candidates = [
            getattr(task_env, "subtask_instruction_map", None),
            getattr(task_env, "resolved_subtask_instruction_map", None),
        ]
        for mapping in candidates:
            if not isinstance(mapping, dict):
                continue
            for key in (instruction_idx, str(instruction_idx), _safe_int(annotation.get("subtask", 0), 0)):
                value = mapping.get(key, None)
                if value:
                    return str(value)
        return None

    def _candidate_subtask_instructions(self, task_env) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for mapping in (
            getattr(task_env, "subtask_instruction_map", None),
            getattr(task_env, "resolved_subtask_instruction_map", None),
        ):
            if not isinstance(mapping, dict):
                continue
            for _, value in sorted(mapping.items(), key=lambda item: str(item[0])):
                text = str(value).strip()
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
        return texts

    def _find_response_field(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for nested in value.values():
                found = self._find_response_field(nested, key)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found = self._find_response_field(nested, key)
                if found is not None:
                    return found
        return None

    def _current_subtask_from_response(
        self,
        response: dict[str, Any],
        record: FrameRecord,
        fallback_subtask_instruction: Optional[str],
    ) -> Optional[str]:
        data = response.get("data", {}) if isinstance(response, dict) else {}
        for key in self.subtask_planner_output_keys:
            found = self._find_response_field(data, key)
            text = extract_subtask_text(found)
            if text:
                return text
        if self.subtask_planner_use_annotation and self._annotation_is_reliable(record.annotation):
            text = extract_subtask_text(fallback_subtask_instruction)
            if text:
                return text
        return None

    def _planner_controls_subtask_keyframes(self) -> bool:
        return bool(
            self.history_mode in SUBTASK_KEYFRAME_HISTORY_MODES
            and self.subtask_planner_enabled
            and self.subtask_planner is not None
        )

    def _update_subtask_keyframe_from_planner(
        self,
        record: FrameRecord,
        response: dict[str, Any],
        instruction: str,
        subtask_instruction: Optional[str],
    ) -> None:
        if not self._planner_controls_subtask_keyframes():
            return

        current_text = self._current_subtask_from_response(response, record, subtask_instruction)
        planner_note: dict[str, Any] = {
            "enabled": True,
            "current_subtask": current_text,
            "previous_accepted_subtask": self.last_planner_subtask_text,
        }
        if not current_text:
            planner_note["skipped"] = "no_current_subtask_text"
            record.subtask_keyframe = False
            record.annotation["subtask_keyframe_reason"] = "external_subtask_planner_no_text"
            record.annotation["subtask_planner"] = planner_note
            return

        if self.last_planner_subtask_text is None:
            record.subtask_keyframe = True
            self.last_planner_subtask_text = current_text
            self.last_subtask_keyframe_step = int(record.step)
            planner_note.update({"same_as_previous": False, "reason": "first_planner_subtask"})
            record.annotation["subtask_keyframe_reason"] = "external_subtask_planner_first"
            record.annotation["subtask_planner"] = planner_note
            return

        try:
            decision = self.subtask_planner.compare(
                self.last_planner_subtask_text,
                current_text,
                task_instruction=instruction,
                candidate_subtasks=self.current_candidate_subtasks,
            )
        except Exception as exc:
            planner_note["error"] = repr(exc)
            record.subtask_keyframe = False
            record.annotation["subtask_keyframe_reason"] = "external_subtask_planner_error"
            record.annotation["subtask_planner"] = planner_note
            if not self.subtask_planner_fail_open:
                raise
            return

        planner_note.update(
            {
                "same_as_previous": bool(decision.same),
                "confidence": float(decision.confidence),
                "normalized_previous": decision.normalized_previous,
                "normalized_current": decision.normalized_current,
                "reason": decision.reason,
                "latency_sec": round(float(decision.latency_sec), 6),
            }
        )
        if not decision.same:
            record.subtask_keyframe = True
            self.last_planner_subtask_text = current_text
            self.last_subtask_keyframe_step = int(record.step)
            record.annotation["subtask_keyframe_reason"] = "external_subtask_planner"
        else:
            record.subtask_keyframe = False
            record.annotation["subtask_keyframe_reason"] = "external_subtask_planner_same"
        record.annotation["subtask_planner"] = planner_note

    def _register_frame(
        self,
        env_step: int,
        image: np.ndarray,
        state: np.ndarray,
        annotation: dict[str, Any],
    ) -> FrameRecord:
        env_step = int(env_step)
        if self.records and self.records[-1].step == env_step:
            return self.records[-1]

        subtask_signature = self._subtask_signature(annotation)
        motion_signature = self._motion_signature(annotation)
        reliable = self._annotation_is_reliable(annotation)
        planner_controls_subtask_keyframes = self._planner_controls_subtask_keyframes()

        subtask_keyframe = False
        subtask_reason = None
        if planner_controls_subtask_keyframes:
            subtask_reason = "pending_external_subtask_planner"
        elif not self.records:
            subtask_keyframe = True
            subtask_reason = "first_frame"
        elif reliable and subtask_signature != self.last_subtask_signature:
            subtask_keyframe = True
            subtask_reason = "annotation_subtask_change"
        elif (
            not reliable
            and self.subtask_fallback_interval > 0
            and self.last_subtask_keyframe_step is not None
            and env_step - self.last_subtask_keyframe_step >= self.subtask_fallback_interval
        ):
            subtask_keyframe = True
            subtask_reason = "interval"

        motion_keyframe = False
        motion_reason = None
        motion_detail = None
        if self.motion_keyframe_detector is not None:
            decision = self.motion_keyframe_detector.update(state)
            motion_keyframe = bool(decision.is_keyframe)
            motion_reason = ",".join(decision.reasons) if decision.reasons else None
            motion_detail = {
                "strategy": self.motion_keyframe_strategy,
                "detector_frame_index": decision.frame_index,
                "candidate": decision.is_candidate,
                "reasons": list(decision.reasons),
                "suppressed_by_frame": decision.suppressed_by_frame,
            }
        else:
            if not self.records:
                motion_keyframe = True
                motion_reason = "first_frame"
            elif reliable and motion_signature != self.last_motion_signature:
                motion_keyframe = True
                motion_reason = "annotation_segment_change"
            elif self.last_motion_keyframe_state is not None and self.motion_state_delta_threshold > 0:
                state_delta = float(
                    np.max(
                        np.abs(
                            np.asarray(state, dtype=np.float32)
                            - self.last_motion_keyframe_state
                        )
                    )
                )
                if state_delta >= self.motion_state_delta_threshold:
                    motion_keyframe = True
                    motion_reason = "state_delta"

            if not motion_keyframe and self.motion_min_interval > 0 and (
                self.last_motion_keyframe_step is None
                or env_step - self.last_motion_keyframe_step >= self.motion_min_interval
            ):
                motion_keyframe = True
                motion_reason = "interval"

        record_annotation = {
            **annotation,
            "subtask_keyframe_reason": subtask_reason,
            "motion_keyframe_reason": motion_reason,
            "motion_keyframe_strategy": self.motion_keyframe_strategy,
        }
        if motion_detail is not None:
            record_annotation["motion_keyframe_detector"] = motion_detail

        record = FrameRecord(
            step=env_step,
            image=np.asarray(image),
            state=np.asarray(state, dtype=np.float32),
            annotation=record_annotation,
            subtask_keyframe=bool(subtask_keyframe),
            motion_keyframe=bool(motion_keyframe),
        )
        self.records.append(record)
        if not planner_controls_subtask_keyframes:
            self.last_subtask_signature = subtask_signature
        self.last_motion_signature = motion_signature
        if subtask_keyframe:
            self.last_subtask_keyframe_step = env_step
        if motion_keyframe:
            self.last_motion_keyframe_step = env_step
            self.last_motion_keyframe_state = np.asarray(state, dtype=np.float32).copy()
        return record

    def _limit_history(self, records: list[FrameRecord]) -> list[FrameRecord]:
        if self.max_history_frames is None:
            return list(records)
        if self.max_history_frames <= 0:
            return []
        return list(records[-self.max_history_frames :])

    def _select_action_history(self, current: FrameRecord) -> list[FrameRecord]:
        if self.max_history_frames == 0:
            return [current]
        by_step = {record.step: record for record in self.records}
        selected: list[FrameRecord] = []
        step = current.step - self.history_stride
        while step >= 0:
            record = by_step.get(step)
            if record is not None:
                selected.append(record)
                if self.max_history_frames is not None and len(selected) >= self.max_history_frames:
                    break
            step -= self.history_stride
        selected = list(reversed(selected))
        if not selected and self.max_history_frames not in (0, None):
            selected = self._limit_history([record for record in self.records[:-1] if record.step < current.step])
        return selected + [current]

    def _select_keyframe_history(self, current: FrameRecord, key: str) -> list[FrameRecord]:
        if self.max_history_frames == 0:
            return [current]
        prior = [record for record in self.records[:-1] if record.step < current.step and bool(getattr(record, key))]
        prior = self._limit_history(prior)
        return prior + [current]

    def _select_history(self, current: FrameRecord) -> list[FrameRecord]:
        mode = self.history_mode
        if mode in HISTORY_NONE:
            return [current]
        if mode == "action_keyframe":
            return self._select_action_history(current)
        if mode == "motion_keyframe":
            return self._select_keyframe_history(current, "motion_keyframe")
        if mode in SUBTASK_KEYFRAME_HISTORY_MODES:
            return self._select_keyframe_history(current, "subtask_keyframe")
        raise ValueError(f"Unsupported history_mode={self.history_mode!r}")

    def _resize_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        width, height = self.image_size
        resized = []
        for image in images:
            arr = _as_uint8_rgb(image)
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
            if self.swap_rgb_channels:
                arr = arr[..., ::-1]
            resized.append(np.ascontiguousarray(arr))
        return resized

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if self.request_log_path is None:
            return
        with self.request_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def _save_request_image(self, image: np.ndarray, frame_step: int, prompt_index: int) -> dict[str, Any]:
        if self.request_image_dir is None:
            return {"path": None, "saved": False}
        arr = _as_uint8_rgb(image)
        seed_part = _safe_name_part(self.request_seed)
        path = self.request_image_dir / (
            f"episode_{self.episode_idx:04d}_chunk_{self.chunk_idx:05d}_"
            f"prompt_{int(prompt_index):02d}_step_{int(frame_step):06d}_seed_{seed_part}.png"
        )
        ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write request image: {path}")
        return {"path": str(path), "saved": True}

    def _image_summaries(self, images: list[np.ndarray], frame_indices: list[int]) -> list[dict[str, Any]]:
        summaries = []
        for prompt_index, image in enumerate(images):
            arr = np.asarray(image)
            frame_step = frame_indices[prompt_index] if prompt_index < len(frame_indices) else -1
            summaries.append(
                {
                    "prompt_index": int(prompt_index),
                    "frame_step": int(frame_step),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "mean": float(arr.mean()) if arr.size else None,
                    "min": int(arr.min()) if arr.size else None,
                    "max": int(arr.max()) if arr.size else None,
                    **self._save_request_image(arr, frame_step=frame_step, prompt_index=prompt_index),
                }
            )
        return summaries

    def _build_example(
        self,
        records: list[FrameRecord],
        resized_images: list[np.ndarray],
        instruction: str,
        subtask_instruction: Optional[str],
    ) -> dict[str, Any]:
        frame_indices = [int(record.step) for record in records]
        example: dict[str, Any] = {
            "image": resized_images,
            "lang": str(instruction),
            "task_lang": str(instruction),
            "robot_tag": "activearena_astribot",
            "history_mode": self.history_mode,
            "history_frame_indices": frame_indices,
            "num_frames": len(resized_images),
            "num_history_frames": max(len(resized_images) - 1, 0),
        }
        if subtask_instruction:
            example["subtask_lang"] = str(subtask_instruction)
            example["subtask_instruction"] = str(subtask_instruction)
        if self.send_state:
            state_history = np.stack([record.state for record in records], axis=0).astype(np.float32)
            example["state_history"] = state_history
            example["state"] = state_history
        return example

    def _extract_actions(self, response: dict[str, Any]) -> np.ndarray:
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA server inference failed: {response.get('error', response)}")
        data = response.get("data", {})
        if "actions" not in data:
            raise KeyError(f"StarVLA response has no actions: {response}")
        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"Expected a single action batch, got {actions.shape}")
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk shape (T, D), got {actions.shape}")
        if actions.shape[0] == 0 or actions.shape[1] not in (18, 19):
            raise ValueError(f"Expected a non-empty Astribot action chunk with 18 or 19 dimensions, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("StarVLA action chunk contains non-finite values")
        if self.action_steps > 0:
            actions = actions[: self.action_steps]
        if self.clip_grippers and actions.shape[1] >= 16:
            actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
            actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
        return actions

    def _response_extras(self, response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data", {}) if isinstance(response, dict) else {}
        if not isinstance(data, dict):
            return {}
        return {key: value for key, value in data.items() if key != "actions"}

    def _request_actions(
        self,
        records: list[FrameRecord],
        instruction: str,
        camera_key: str,
        subtask_instruction: Optional[str],
    ) -> np.ndarray:
        resized_images = self._resize_images([record.image for record in records])
        frame_indices = [int(record.step) for record in records]
        example = self._build_example(records, resized_images, instruction, subtask_instruction)
        request_id = (
            f"starvla_ep{self.episode_idx:04d}_chunk{self.chunk_idx:05d}_"
            f"step{frame_indices[-1] if frame_indices else 0:06d}"
        )
        payload: dict[str, Any] = {
            "type": "infer",
            "request_id": request_id,
            "examples": [example],
            "do_sample": False,
        }
        if self.return_vlm_text:
            payload["return_vlm_text"] = True
            payload["vlm_text_max_new_tokens"] = int(self.vlm_text_max_new_tokens)
        if self.unnorm_key is not None:
            payload["unnorm_key"] = self.unnorm_key

        request_time = time.time()
        start = time.perf_counter()
        response: dict[str, Any] | None = None
        try:
            response = self.client.predict_action(payload)
            request_sec = time.perf_counter() - start
            actions = self._extract_actions(response)
            extras = self._response_extras(response)
            if records:
                self._update_subtask_keyframe_from_planner(records[-1], response, instruction, subtask_instruction)
        except Exception as exc:
            request_sec = time.perf_counter() - start
            use_hold_fallback = self.hold_on_invalid_fast_action and _is_invalid_fast_action_error(exc)
            self._write_jsonl(
                {
                    "timestamp": request_time,
                    "episode": int(self.episode_idx),
                    "chunk": int(self.chunk_idx),
                    "request_id": request_id,
                    "env_step": int(frame_indices[-1] if frame_indices else 0),
                    "instruction": str(instruction),
                    "subtask_instruction": subtask_instruction,
                    "camera_key": camera_key,
                    "request": {
                        "example_fields": {
                            key: value
                            for key, value in example.items()
                            if key not in {"image", "state", "state_history"}
                        },
                        "state_history": example.get("state_history", None),
                        "send_state": self.send_state,
                        "image_channel_order": "RGB_swapped" if self.swap_rgb_channels else "RGB",
                        "image_history": self._image_summaries(resized_images, frame_indices),
                        "annotations": [record.annotation for record in records],
                        "subtask_keyframes": [bool(record.subtask_keyframe) for record in records],
                        "motion_keyframes": [bool(record.motion_keyframe) for record in records],
                        "unnorm_key": self.unnorm_key,
                        "do_sample": False,
                        "return_vlm_text": self.return_vlm_text,
                    },
                    "response": response,
                    "error": repr(exc),
                    "fallback": "hold_current_state" if use_hold_fallback else None,
                }
            )
            if not use_hold_fallback:
                raise
            actions = _hold_current_state_actions(records, self.action_steps)
            self.last_request_id = request_id
            self.last_request_sec = request_sec
            self.last_chunk_size = int(actions.shape[0])
            print(
                f"[StarVLAOFTClient] invalid FAST action tokens; holding current state "
                f"for {actions.shape[0]} steps request_id={request_id}",
                flush=True,
            )
            return actions

        self.last_request_id = request_id
        self.last_request_sec = request_sec
        self.last_chunk_size = int(actions.shape[0])
        self._write_jsonl(
            {
                "timestamp": request_time,
                "episode": int(self.episode_idx),
                "chunk": int(self.chunk_idx),
                "request_id": request_id,
                "env_step": int(frame_indices[-1] if frame_indices else 0),
                "instruction": str(instruction),
                "subtask_instruction": subtask_instruction,
                "camera_key": camera_key,
                "request": {
                    "example_fields": {key: value for key, value in example.items() if key not in {"image", "state", "state_history"}},
                    "state_history": example.get("state_history", None),
                    "send_state": self.send_state,
                    "image_channel_order": "RGB_swapped" if self.swap_rgb_channels else "RGB",
                    "image_history": self._image_summaries(resized_images, frame_indices),
                    "annotations": [record.annotation for record in records],
                    "subtask_keyframes": [bool(record.subtask_keyframe) for record in records],
                    "motion_keyframes": [bool(record.motion_keyframe) for record in records],
                    "unnorm_key": self.unnorm_key,
                    "do_sample": False,
                    "return_vlm_text": self.return_vlm_text,
                },
                "response": {
                    "ok": bool(response.get("ok", False)),
                    "status": response.get("status"),
                    "type": response.get("type"),
                    "request_sec": round(float(request_sec), 6),
                    "actions_shape": list(actions.shape),
                    "actions": actions,
                    "extra_outputs": extras,
                },
                "action_mapping": {
                    "model_order": ACTION_ORDER,
                    "robotwin_order": ROBOTWIN_ORDER,
                    "extra_conversion": "model torso/head2 absolute targets are converted to RobotWin deltas at execution time",
                },
            }
        )
        print(
            f"[StarVLAOFTClient] ep={self.episode_idx} chunk={self.chunk_idx} "
            f"step={frame_indices[-1] if frame_indices else 0} frames={frame_indices} "
            f"actions={actions.shape} request_sec={request_sec:.3f} "
            f"extras={list(extras.keys())}",
            flush=True,
        )
        return actions

    def get_actions(self, task_env, observation: dict[str, Any]) -> np.ndarray:
        instruction = str(task_env.get_instruction())
        if self.language is not None and instruction != self.language:
            self.reset()
        self.language = instruction
        self.current_candidate_subtasks = self._candidate_subtask_instructions(task_env)

        current, camera_key, annotation = self.observe_frame(task_env, observation)
        records = self._select_history(current)
        subtask_instruction = self._subtask_instruction(task_env, annotation)
        return self._request_actions(records, instruction, camera_key, subtask_instruction)

    def observe_frame(self, task_env, observation: dict[str, Any]) -> tuple[FrameRecord, str, dict[str, Any]]:
        env_step = _safe_int(getattr(task_env, "take_action_cnt", 0), 0)
        image, camera_key = encode_obs(observation)
        state = build_astribot_18d_state(
            observation,
            torso_index=self.torso_index,
            head2_index=self.head2_index,
        )
        annotation = self._annotation_from_env(task_env, observation)
        record = self._register_frame(env_step, image, state, annotation)
        return record, camera_key, annotation

    def log_chunk_timing(
        self,
        *,
        start_step: int,
        end_step: int,
        executed_steps: int,
        sim_sec: float,
        success: bool,
        step_limit: int,
    ) -> None:
        if self.log_chunk_timing_enabled:
            total_sec = sim_sec + self.last_request_sec if self.last_request_sec is not None else None
            total_text = "nan" if total_sec is None else f"{total_sec:.3f}"
            request_text = "nan" if self.last_request_sec is None else f"{self.last_request_sec:.3f}"
            print(
                f"[StarVLAOFTClient][chunk_timing] ep={self.episode_idx} chunk={self.chunk_idx} "
                f"request_id={self.last_request_id} start_step={start_step} end_step={end_step} "
                f"executed_steps={executed_steps} chunk_actions={self.last_chunk_size} "
                f"request_sec={request_text} sim_sec={sim_sec:.3f} total_sec={total_text} "
                f"success={bool(success)} step_limit={step_limit}",
                flush=True,
            )
        self.chunk_idx += 1


def _arg(usr_args: dict[str, Any], name: str, default: Any) -> Any:
    value = usr_args.get(name, None)
    return default if value is None else value


def get_model(usr_args: dict[str, Any]):
    policy_name = str(_arg(usr_args, "policy_name", "activearena_astribot"))
    default_log = f"policy/{policy_name}/logs/requests.jsonl"
    default_images = f"policy/{policy_name}/logs/images"
    history_frames = usr_args["history_frames"] if "history_frames" in usr_args else 12
    history_frames = None if history_frames is None else _as_optional_int(history_frames, default=None)
    return StarVLAOFTClient(
        host=str(os.environ.get("ACTIVEARENA_POLICY_HOST", os.environ.get("STARVLA_HOST", _arg(usr_args, "host", "127.0.0.1")))),
        port=int(os.environ.get("ACTIVEARENA_POLICY_PORT", os.environ.get("STARVLA_PORT", _arg(usr_args, "port", 7980)))),
        unnorm_key=_arg(usr_args, "unnorm_key", None),
        image_size=_parse_image_size(_arg(usr_args, "image_size", [224, 224])),
        action_steps=int(_arg(usr_args, "action_steps", 16)),
        action_type=str(_arg(usr_args, "action_type", "qpos")),
        history_mode=str(_arg(usr_args, "history_mode", "action_keyframe")),
        history_frames=history_frames,
        history_stride=_arg(usr_args, "history_stride", None),
        send_state=_as_bool(_arg(usr_args, "send_state", False)),
        swap_rgb_channels=_as_bool(_arg(usr_args, "swap_rgb_channels", False)),
        request_log_path=_arg(usr_args, "request_log_path", default_log),
        request_image_dir=_arg(usr_args, "request_image_dir", default_images),
        motion_min_interval=int(_arg(usr_args, "motion_min_interval", 16)),
        motion_state_delta_threshold=float(_arg(usr_args, "motion_state_delta_threshold", 0.05)),
        motion_keyframe_strategy=str(_arg(usr_args, "motion_keyframe_strategy", "legacy")),
        motion_gripper_close_threshold=float(
            _arg(usr_args, "motion_gripper_close_threshold", 0.2)
        ),
        motion_gripper_open_threshold=float(
            _arg(usr_args, "motion_gripper_open_threshold", 0.8)
        ),
        motion_min_gripper_state_len=int(
            _arg(usr_args, "motion_min_gripper_state_len", 2)
        ),
        motion_rotation_delta_threshold=float(
            _arg(usr_args, "motion_rotation_delta_threshold", 0.005)
        ),
        motion_min_rotation_motion_len=int(
            _arg(usr_args, "motion_min_rotation_motion_len", 4)
        ),
        motion_rotation_merge_gap=int(
            _arg(usr_args, "motion_rotation_merge_gap", 5)
        ),
        motion_dedup_window=int(_arg(usr_args, "motion_dedup_window", 5)),
        motion_mark_initial=_as_bool(
            _arg(usr_args, "motion_mark_initial", True), default=True
        ),
        subtask_fallback_interval=int(_arg(usr_args, "subtask_fallback_interval", 0)),
        clip_grippers=_as_bool(_arg(usr_args, "clip_grippers", True), default=True),
        torso_index=int(_arg(usr_args, "torso_index", 0)),
        head2_index=int(_arg(usr_args, "head2_index", 1)),
        log_chunk_timing=_as_bool(_arg(usr_args, "log_chunk_timing", True), default=True),
        return_vlm_text=_as_bool(_arg(usr_args, "return_vlm_text", False), default=False),
        vlm_text_max_new_tokens=int(_arg(usr_args, "vlm_text_max_new_tokens", 128)),
        subtask_planner_enabled=_as_bool(_arg(usr_args, "subtask_planner_enabled", False), default=False),
        subtask_planner_backend=str(_arg(usr_args, "subtask_planner_backend", "http")),
        subtask_planner_url=str(_arg(usr_args, "subtask_planner_url", "http://127.0.0.1:7991/classify")),
        subtask_planner_model_path=str(_arg(usr_args, "subtask_planner_model_path", DEFAULT_QWEN3_30B_PATH)),
        subtask_planner_timeout=float(_arg(usr_args, "subtask_planner_timeout", 30.0)),
        subtask_planner_device_map=str(_arg(usr_args, "subtask_planner_device_map", "auto")),
        subtask_planner_dtype=str(_arg(usr_args, "subtask_planner_dtype", "bfloat16")),
        subtask_planner_attn_implementation=str(_arg(usr_args, "subtask_planner_attn_implementation", "sdpa")),
        subtask_planner_max_new_tokens=int(_arg(usr_args, "subtask_planner_max_new_tokens", 192)),
        subtask_planner_model_name=str(_arg(usr_args, "subtask_planner_model_name", "qwen3.5-9b")),
        subtask_planner_api_key=_arg(usr_args, "subtask_planner_api_key", None),
        subtask_planner_disable_reasoning=_as_bool(
            _arg(usr_args, "subtask_planner_disable_reasoning", True), default=True
        ),
        subtask_planner_output_keys=_as_str_list(
            _arg(usr_args, "subtask_planner_output_keys", None),
            default=["subtask", "current_subtask", "subtask_instruction", "vlm_text", "text", "output", "generated_text"],
        ),
        subtask_planner_use_annotation=_as_bool(_arg(usr_args, "subtask_planner_use_annotation", False), default=False),
        subtask_planner_fail_open=_as_bool(_arg(usr_args, "subtask_planner_fail_open", True), default=True),
        hold_on_invalid_fast_action=_as_bool(
            _arg(usr_args, "hold_on_invalid_fast_action", False),
            default=False,
        ),
    )


def eval(TASK_ENV, model: StarVLAOFTClient, observation: dict[str, Any]):
    if hasattr(model, "set_request_seed"):
        model.set_request_seed(getattr(TASK_ENV, "eval_seed", getattr(TASK_ENV, "seed", None)))

    actions = model.get_actions(TASK_ENV, observation)
    start_step = int(TASK_ENV.take_action_cnt)
    sim_start = time.perf_counter()
    executed_steps = 0
    for action in actions:
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_done:
            break
        env_action = adapt_model_action_to_robotwin(action, TASK_ENV)
        TASK_ENV.take_action(env_action, action_type=model.action_type)
        executed_steps += 1
        if TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_done:
            observation = TASK_ENV.get_obs()
            model.observe_frame(TASK_ENV, observation)
    sim_sec = time.perf_counter() - sim_start
    model.log_chunk_timing(
        start_step=start_step,
        end_step=int(TASK_ENV.take_action_cnt),
        executed_steps=executed_steps,
        sim_sec=sim_sec,
        success=bool(TASK_ENV.eval_success),
        step_limit=int(TASK_ENV.step_lim),
    )
    return observation


def reset_model(model: StarVLAOFTClient):
    model.reset()
