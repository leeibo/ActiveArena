from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionKeyframeConfig:
    gripper_close_threshold: float = 0.2
    gripper_open_threshold: float = 0.8
    min_gripper_state_len: int = 2
    rotation_delta_threshold: float = 0.005
    min_rotation_motion_len: int = 4
    rotation_merge_gap: int = 5
    dedup_window: int = 5
    mark_initial: bool = True
    left_gripper_index: int = 7
    right_gripper_index: int = 15
    torso_yaw_index: int = 16
    head_2_index: int = 17


@dataclass(frozen=True)
class MotionKeyframeDecision:
    frame_index: int
    is_keyframe: bool
    is_candidate: bool
    reasons: tuple[str, ...]
    suppressed_by_frame: int | None = None


class _CausalRunTracker:
    def __init__(self, min_len: int, merge_gap: int = 0) -> None:
        self.min_len = max(int(min_len), 1)
        self.merge_gap = max(int(merge_gap), 0)
        self.reset()

    def reset(self) -> None:
        self.run_start: int | None = None
        self.false_count = 0
        self.confirmed = False

    def update(self, frame: int, active: bool) -> tuple[str, int] | None:
        if active:
            if self.run_start is None:
                self.run_start = int(frame)
                self.confirmed = False
            self.false_count = 0
            if not self.confirmed and frame - self.run_start + 1 >= self.min_len:
                self.confirmed = True
                return "start", self.run_start
            return None

        if self.run_start is None:
            return None
        self.false_count += 1
        if self.false_count <= self.merge_gap:
            return None

        event = ("end", self.run_start) if self.confirmed else None
        self.reset()
        return event


class CausalMotionKeyframeDetector:
    """Online equivalent of script/update_lerobot_motion_keyframes.py."""

    def __init__(self, config: MotionKeyframeConfig | None = None) -> None:
        self.config = config or MotionKeyframeConfig()
        self._required_width = max(
            self.config.left_gripper_index,
            self.config.right_gripper_index,
            self.config.torso_yaw_index,
            self.config.head_2_index,
        ) + 1
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self._frame_index = 0
        self._last_kept_frame: int | None = None
        self._previous_rotation: dict[str, float] = {}
        self._trackers = {
            "left_gripper_closed": _CausalRunTracker(cfg.min_gripper_state_len),
            "left_gripper_open": _CausalRunTracker(cfg.min_gripper_state_len),
            "right_gripper_closed": _CausalRunTracker(cfg.min_gripper_state_len),
            "right_gripper_open": _CausalRunTracker(cfg.min_gripper_state_len),
            "torso_yaw_motion": _CausalRunTracker(
                cfg.min_rotation_motion_len, cfg.rotation_merge_gap
            ),
            "head_2_motion": _CausalRunTracker(
                cfg.min_rotation_motion_len, cfg.rotation_merge_gap
            ),
        }

    def _update_tracker(
        self,
        name: str,
        frame: int,
        active: bool,
        reasons: list[str],
    ) -> None:
        event = self._trackers[name].update(frame, active)
        if event is None:
            return
        event_name, run_start = event
        if event_name == "start" and run_start == 0:
            return
        reasons.append(f"{name}_{event_name}")

    def update(self, state: np.ndarray) -> MotionKeyframeDecision:
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.shape[0] < self._required_width:
            raise ValueError(
                f"motion keyframe state requires at least {self._required_width} values, "
                f"got {values.shape[0]}"
            )

        cfg = self.config
        frame = self._frame_index
        reasons: list[str] = []
        if frame == 0 and cfg.mark_initial:
            reasons.append("initial_frame")

        for side, index in (
            ("left", cfg.left_gripper_index),
            ("right", cfg.right_gripper_index),
        ):
            gripper = float(values[index])
            self._update_tracker(
                f"{side}_gripper_closed",
                frame,
                gripper <= cfg.gripper_close_threshold,
                reasons,
            )
            self._update_tracker(
                f"{side}_gripper_open",
                frame,
                gripper >= cfg.gripper_open_threshold,
                reasons,
            )

        for name, index in (
            ("torso_yaw", cfg.torso_yaw_index),
            ("head_2", cfg.head_2_index),
        ):
            current = float(values[index])
            previous = self._previous_rotation.get(name, current)
            moving = abs(current - previous) > cfg.rotation_delta_threshold
            self._previous_rotation[name] = current
            self._update_tracker(f"{name}_motion", frame, moving, reasons)

        is_candidate = bool(reasons)
        suppressed_by = None
        is_keyframe = False
        if is_candidate:
            if (
                self._last_kept_frame is None
                or frame - self._last_kept_frame > cfg.dedup_window
            ):
                is_keyframe = True
                self._last_kept_frame = frame
            else:
                suppressed_by = self._last_kept_frame

        self._frame_index += 1
        return MotionKeyframeDecision(
            frame_index=frame,
            is_keyframe=is_keyframe,
            is_candidate=is_candidate,
            reasons=tuple(reasons),
            suppressed_by_frame=suppressed_by,
        )
