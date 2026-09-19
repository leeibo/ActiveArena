from pathlib import Path

import numpy as np
import sapien.core as sapien

from .utils import (
    Action,
    ArmTag,
    Actor,
    create_box,
    create_sapien_urdf_obj,
    place_point_cyl,
    place_pose_cyl,
    rand_pose_cyl,
    preprocess,
    rotate_theta_half,
    world_to_robot,
)
from ._GLOBAL_CONFIGS import left_check_pose


RMBENCH_BUTTON_MODEL_NAME = "005_button"
RMBENCH_CHECK_BUTTON_MODEL_NAME = "006_check_button"
RMBENCH_BUTTON_MODEL_ID = 10124


def ensure_rmbench_button_assets():
    """Fail early with a useful message if the RMBench button assets are absent."""
    missing = []
    for model_name in (RMBENCH_BUTTON_MODEL_NAME, RMBENCH_CHECK_BUTTON_MODEL_NAME):
        model_dir = Path("assets") / "objects" / model_name / str(RMBENCH_BUTTON_MODEL_ID)
        for rel in ("mobility.urdf", "model_data.json"):
            path = model_dir / rel
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing RMBench button asset files: "
            + ", ".join(missing)
            + ". Run `python script/download_rmbench_info_assets.py` from the repo root."
        )


class RMBenchButtonMixin:
    """Shared helpers for tasks that use the RMBench prismatic button asset."""

    BUTTON_JOINT_NAME = "button_joint"
    BUTTON_PRESS_THRESHOLD = -0.005
    BUTTON_RESET_THRESHOLD = -0.001
    BUTTON_PRESS_DOWN_Z = -0.045
    BUTTON_PRESS_UP_Z = 0.045
    # Move from a hover pose to the actual cap before pressing.  Keeping
    # pre_grasp_dis == grasp_dis leaves the gripper in the air.

    BUTTON_PRE_GRASP_DIS = 0.05
    BUTTON_GRASP_DIS = 0.05
    BUTTON_CONTACT_POINT_ID = 0
    BUTTON_GRIPPER_POS = -0.1
    BUTTON_CAP_MASS = 0.0001

    def _create_rmbench_button(
        self,
        r,
        theta,
        model_name=RMBENCH_BUTTON_MODEL_NAME,
        model_id=RMBENCH_BUTTON_MODEL_ID,
        z=0.741,
        qpos=(1, 0, 0, 0),
        name=None,
    ):
        ensure_rmbench_button_assets()
        point = place_point_cyl(
            [float(r), float(theta), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        button = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose(point, list(qpos)),
            modelname=str(model_name),
            modelid=int(model_id),
            fix_root_link=True,
        )
        if name is not None:
            button.set_name(str(name))
        button.set_mass(float(self.BUTTON_CAP_MASS), ["button_cap"])
        self._set_button_unpressed(button)
        return button

    @staticmethod
    def _button_articulation(button_actor):
        return button_actor.actor if hasattr(button_actor, "actor") else button_actor

    def _button_joint_index(self, button_actor, joint_name=None):
        art = self._button_articulation(button_actor)
        joint_name = str(joint_name or self.BUTTON_JOINT_NAME)
        joints = art.get_active_joints()
        joint_names = [joint.get_name() for joint in joints]
        return joint_names.index(joint_name)

    def _get_button_qpos(self, button_actor, joint_name=None):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        return float(art.get_qpos()[idx])

    def _set_button_unpressed(self, button_actor, joint_name=None, target=0.0):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        qpos = art.get_qpos()
        qpos[idx] = float(target)
        art.set_qpos(qpos)
        joints = art.get_active_joints()
        joints[idx].set_drive_target(float(target))

    def _set_button_pressed(self, button_actor, joint_name=None, target=None):
        art = self._button_articulation(button_actor)
        idx = self._button_joint_index(button_actor, joint_name=joint_name)
        if target is None:
            try:
                lower = float(art.get_qlimits()[idx][0])
            except Exception:
                lower = -0.006
            target = min(lower, float(self.BUTTON_PRESS_THRESHOLD) - 1e-3)
        qpos = art.get_qpos()
        qpos[idx] = float(target)
        art.set_qpos(qpos)
        joints = art.get_active_joints()
        joints[idx].set_drive_target(float(target))

    def _is_button_pressed(self, button_actor, joint_name=None, threshold=None):
        threshold = float(self.BUTTON_PRESS_THRESHOLD if threshold is None else threshold)
        return bool(self._get_button_qpos(button_actor, joint_name=joint_name) < threshold)

    def _get_pressed_button_values(self, buttons):
        return [
            int(value)
            for value, button in buttons.items()
            if self._is_button_pressed(button)
        ]

    def _get_wrong_button_failure(
        self,
        buttons,
        target_value,
        pressed_value_attr="pressed_button_value",
    ):
        try:
            target_value = int(target_value)
        except (TypeError, ValueError):
            return None

        pressed_values = self._get_pressed_button_values(buttons)
        wrong_values = [value for value in pressed_values if value != target_value]
        if not wrong_values:
            return None

        setattr(self, str(pressed_value_attr), int(wrong_values[0]))
        return {
            "reason": "wrong_button_pressed",
            "target_button": target_value,
            "pressed_buttons": pressed_values,
            "wrong_buttons": wrong_values,
        }

    def _check_matching_button_success(
        self,
        buttons,
        target_value,
        pressed_value_attr="pressed_button_value",
    ):
        """Check scripted or physical button presses against the target value."""
        try:
            target_value = int(target_value)
        except (TypeError, ValueError):
            return False

        physically_pressed_values = self._get_pressed_button_values(buttons)
        if physically_pressed_values:
            pressed_value = (
                target_value
                if target_value in physically_pressed_values
                else physically_pressed_values[0]
            )
            setattr(self, str(pressed_value_attr), pressed_value)
        else:
            pressed_value = getattr(self, str(pressed_value_attr), None)

        try:
            return int(pressed_value) == target_value
        except (TypeError, ValueError):
            return False

    def _update_button_reset_flag(self, button_actor, flag_attr, joint_name=None, threshold=None):
        threshold = float(self.BUTTON_RESET_THRESHOLD if threshold is None else threshold)
        if self._get_button_qpos(button_actor, joint_name=joint_name) > threshold:
            setattr(self, str(flag_attr), False)

    def _update_button_press_count(self, button_actor, flag_attr, count_attr):
        if self._is_button_pressed(button_actor) and not bool(getattr(self, str(flag_attr))):
            setattr(self, str(flag_attr), True)
            setattr(self, str(count_attr), int(getattr(self, str(count_attr))) + 1)

    def _grasp_button_for_press(
        self,
        button_actor,
        arm_tag="left",
        language_annotation="Press the button once.",
    ):
        arm_tag = ArmTag(arm_tag)
        del language_annotation  # retained in signature for call-site readability.
        self.move(
            self.grasp_actor(
                button_actor,
                arm_tag=arm_tag,
                pre_grasp_dis=float(self.BUTTON_PRE_GRASP_DIS),
                grasp_dis=float(self.BUTTON_GRASP_DIS),
                gripper_pos=float(self.BUTTON_GRIPPER_POS),
                contact_point_id=int(self.BUTTON_CONTACT_POINT_ID),
            )
        )
        if not self.plan_success:
            return False
        return True

    def _press_button_cycle_after_grasp(
        self,
        button_actor,
        arm_tag="left",
        flag_attr="button_press_flag",
        count_attr="button_press_count",
        language_annotation="Press the button once.",
    ):
        arm_tag = ArmTag(arm_tag)
        del language_annotation
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=float(self.BUTTON_PRESS_DOWN_Z),
            )
        )
        self._update_button_press_count(button_actor, flag_attr, count_attr)
        if not self.plan_success:
            return False
        if not self._is_button_pressed(button_actor):
            return False

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=float(self.BUTTON_PRESS_UP_Z),
            )
        )
        if not self.plan_success:
            return False

        self._set_button_unpressed(button_actor)
        self.delay(2)
        self._update_button_reset_flag(button_actor, flag_attr)
        return True

    def _press_button_once(
        self,
        button_actor,
        arm_tag="left",
        flag_attr="button_press_flag",
        count_attr="button_press_count",
        language_annotation="Press the button once.",
    ):
        if not self._grasp_button_for_press(
            button_actor,
            arm_tag=arm_tag,
            language_annotation=language_annotation,
        ):
            return False
        return self._press_button_cycle_after_grasp(
            button_actor,
            arm_tag=arm_tag,
            flag_attr=flag_attr,
            count_attr=count_attr,
            language_annotation=language_annotation,
        )

    def _soft_reset_button_for_success_check(self, button_actor):
        # Match RMBench's gradual reset logic so a held-down button cannot be
        # counted repeatedly in one physical press.
        current = self._get_button_qpos(button_actor)
        self._set_button_unpressed(button_actor, target=min(0.0, current + 0.002))


class PolarCountLayoutMixin:
    """Shared polar layout and fixed-head scanning for count-style info tasks."""

    COUNT_OBJECT_RLIM = (0.46, 0.66)
    COUNT_OBJECT_THETA_RATIO = 0.58
    COUNT_SLOT_COUNT = 7
    COUNT_OBJECT_MIN_XY_DISTANCE = 0.075
    COUNT_OBJECT_BUTTON_AVOID_DISTANCE = 0.12
    COUNT_OBJECT_SAMPLE_TRIES = 400
    COUNT_OBJECT_ROTATE_RAND = False

    def _pose_from_cyl(self, r, theta, z, qpos=None, rotate_rand=False, rotate_lim=None):
        if qpos is None:
            qpos = [1, 0, 0, 0]
        if rotate_lim is None:
            rotate_lim = [0, 0, 0]
        return rand_pose_cyl(
            rlim=[float(r), float(r)],
            thetalim=[float(theta), float(theta)],
            zlim=[float(z), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=qpos,
            rotate_rand=rotate_rand,
            rotate_lim=rotate_lim,
        )

    def _block_pose_from_cyl(self, r, theta):
        z = 0.74 + float(self.BLOCK_HALF_SIZE) + 0.002
        return self._pose_from_cyl(
            r,
            theta,
            z,
            rotate_rand=bool(getattr(self, "COUNT_OBJECT_ROTATE_RAND", False)),
            rotate_lim=[0.0, 0.0, 0.75],
        )

    def _polar_count_slots(self, total=None):
        total = int(self.COUNT_SLOT_COUNT if total is None else total)
        total = max(total, 1)
        r_min, r_max = [float(v) for v in self.COUNT_OBJECT_RLIM]
        theta_half = float(rotate_theta_half(self)) * float(self.COUNT_OBJECT_THETA_RATIO)
        theta_count = max(1, int(np.ceil(total / 2)))
        theta_values = np.linspace(-theta_half, theta_half, theta_count)
        r_values = np.linspace(r_min, r_max, 2 if total > 1 else 1)
        slots = [(float(r), float(theta)) for theta in theta_values for r in r_values]
        return slots[:total]

    def _sample_polar_count_slots(self, total):
        """Sample count objects as cluttered tabletop points, not a visible grid."""
        total = int(total)
        r_min, r_max = [float(v) for v in self.COUNT_OBJECT_RLIM]
        theta_half = float(rotate_theta_half(self)) * float(self.COUNT_OBJECT_THETA_RATIO)
        min_dis = float(getattr(self, "COUNT_OBJECT_MIN_XY_DISTANCE", 0.075))
        button_avoid = float(getattr(self, "COUNT_OBJECT_BUTTON_AVOID_DISTANCE", 0.0))

        def xy_from_slot(slot):
            return np.array(
                place_point_cyl(
                    [float(slot[0]), float(slot[1]), 0.74],
                    robot_root_xy=self.robot_root_xy,
                    robot_yaw_rad=self.robot_yaw,
                    ret="list",
                )[:2],
                dtype=np.float64,
            )

        avoid_xy = []
        if button_avoid > 1e-9 and hasattr(self, "BUTTON_R") and hasattr(self, "BUTTON_THETAS"):
            for theta in getattr(self, "BUTTON_THETAS"):
                avoid_xy.append(xy_from_slot((float(getattr(self, "BUTTON_R")), float(theta))))

        slots = []
        slots_xy = []
        for _ in range(int(getattr(self, "COUNT_OBJECT_SAMPLE_TRIES", 400))):
            if len(slots) >= total:
                break
            slot = (
                float(np.random.uniform(r_min, r_max)),
                float(np.random.uniform(-theta_half, theta_half)),
            )
            xy = xy_from_slot(slot)
            if any(float(np.linalg.norm(xy - old_xy)) < min_dis for old_xy in slots_xy):
                continue
            if any(float(np.linalg.norm(xy - old_xy)) < button_avoid for old_xy in avoid_xy):
                continue
            slots.append(slot)
            slots_xy.append(xy)

        if len(slots) < total:
            # Fallback is still shuffled but only used when a very dense sample fails.
            fallback = self._polar_count_slots(total)
            order = np.random.permutation(len(fallback)).tolist()
            for idx in order:
                if len(slots) >= total:
                    break
                slots.append(fallback[idx])
        return slots[:total]

    def _scan_count_objects_fixed_head(self, subtask_idx, target_keys, all_keys, objects):
        self.begin_rotate_subtask(subtask_idx)
        self._hold_head_home_pose_without_recording()
        scan_thetas = self._get_scan_thetas_from_object_list(
            objects,
            fallback_thetas=getattr(self, "ROTATE_SCAN_SCENE_FALLBACK_THETAS", (0.72, 0.24, -0.24, -0.72)),
        )
        for theta in scan_thetas:
            self._set_rotate_subtask_state(
                subtask_idx=subtask_idx,
                stage=1,
                focus_object_key=None,
                search_target_keys=all_keys,
                action_target_keys=[],
                info_complete=0,
                camera_mode=1,
                camera_target_theta=float(theta),
            )
            self._move_scan_camera_to_theta(
                float(theta),
                scan_r=float(self.SCAN_R),
                scan_z=float(self.SCAN_Z_BIAS) + float(self.table_z_bias),
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            self._hold_head_home_pose_without_recording()
            self._refresh_rotate_discovery_from_current_view()
            self.delay(2)

        self.counted_target_count = int(self.target_count)
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=2,
            focus_object_key=None,
            search_target_keys=target_keys,
            action_target_keys=[],
            info_complete=1,
            camera_mode=2,
            camera_target_theta=np.nan,
        )
        self.complete_rotate_subtask(subtask_idx, carried_after=[])

    def _focus_lower_object_fixed_head(self, subtask_idx, object_key, obj, search_target_keys, action_target_keys):
        world_point = np.array(obj.get_pose().p, dtype=np.float64).reshape(3)
        theta = float(world_to_robot(world_point.tolist(), self.robot_root_xy, self.robot_yaw)[1])
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=2,
            focus_object_key=str(object_key),
            search_target_keys=search_target_keys,
            action_target_keys=action_target_keys,
            info_complete=1,
            camera_mode=2,
            camera_target_theta=theta,
        )
        self._move_scan_camera_to_theta(
            theta,
            scan_r=float(self.SCAN_R),
            scan_z=float(self.SCAN_Z_BIAS) + float(self.table_z_bias),
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        self._hold_head_home_pose_without_recording()
        self._refresh_rotate_discovery_from_current_view()
        return str(object_key)


BACKSIDE_RGB_COLOR_SPECS = (
    ("red", (0.90, 0.20, 0.20)),
    ("green", (0.10, 0.75, 0.20)),
    ("blue", (0.20, 0.45, 0.92)),
)


INFO_COLOR_SPECS = (
    ("blue", (0.10, 0.25, 0.95), (24, 86, 235, 255)),
    ("green", (0.05, 0.72, 0.22), (28, 175, 78, 255)),
    ("yellow", (0.92, 0.74, 0.18), (235, 196, 44, 255)),
    ("red", (0.90, 0.18, 0.14), (230, 45, 36, 255)),
    ("purple", (0.52, 0.28, 0.86), (132, 72, 219, 255)),
    ("orange", (0.95, 0.46, 0.12), (242, 117, 31, 255)),
)
INFO_COLOR_RGB_MAP = {label: rgb for label, rgb, _ in INFO_COLOR_SPECS}
INFO_COLOR_RGBA_MAP = {label: rgba for label, _, rgba in INFO_COLOR_SPECS}


def sample_info_color_specs(count, required_label=None):
    count = int(count)
    if count <= 0:
        return ()
    if count > len(INFO_COLOR_SPECS):
        raise ValueError(f"Cannot sample {count} distinct info colors from {len(INFO_COLOR_SPECS)} options")

    labels = [label for label, _, _ in INFO_COLOR_SPECS]
    if required_label is not None:
        required_label = str(required_label).strip().lower()
        if required_label not in INFO_COLOR_RGB_MAP:
            raise ValueError(f"Unknown info color label: {required_label}")
        remaining = [label for label in labels if label != required_label]
        sampled = [required_label]
        if count > 1:
            sampled.extend([str(label) for label in np.random.choice(remaining, size=count - 1, replace=False)])
        np.random.shuffle(sampled)
    else:
        sampled = [str(label) for label in np.random.choice(labels, size=count, replace=False)]

    return tuple((label, INFO_COLOR_RGB_MAP[label]) for label in sampled)


class BacksidePatchBlockMixin:
    """Small helpers for left-hand backside-information block tasks."""

    ARM = "left"
    OUTER_COLOR = (0.72, 0.72, 0.72)
    BLOCK_HALF_SIZE = 0.024
    BACKSIDE_PATCH_HALF_SIZE = (0.0015, 0.012, 0.012)
    BACKSIDE_PATCH_AXIS = "x"
    BACKSIDE_PATCH_SIGN = 1.0
    BACKSIDE_PATCH_PROTRUSION = 0.00005
    PAD_HALF_SIZE = (0.060, 0.045, 0.004)
    PAD_ANCHOR_HALF_SIZE = (0.006, 0.006, 0.004)
    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    BLOCK_RLIM = (0.44, 0.50)
    BLOCK_THETA_RATIO = 0.82
    BLOCK_MIN_ABS_THETA = 0.0
    BLOCK_MIN_XY_DISTANCE = 0.095
    BLOCK_SAMPLE_TRIES = 300
    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_CONTACT_POINT_ID = 1
    LIFT_Z = 0.08
    INSPECT_HOLD_STEPS = 8
    RETURN_AFTER_INSPECT = True
    PLACE_PRE_DIS = 0.09
    PLACE_DIS = 0.02
    PLACE_RELEASE_CLEARANCE = 0.012
    PLACE_RETREAT_Z = 0.06
    SUCCESS_XY_TOL = 0.075

    def _point_from_cyl(self, cyl, z=None):
        r, theta = cyl
        if z is None:
            z = 0.74 + float(self.BLOCK_HALF_SIZE) + 0.002
        return np.array(
            place_point_cyl(
                [float(r), float(theta), float(z)],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="list",
            ),
            dtype=np.float64,
        ).reshape(3)

    def _pose_from_cyl(self, cyl, z=None, qpos=None, quat_frame="cyl_legacy", rotate_rand=False, rotate_lim=None):
        r, theta = cyl
        if z is None:
            z = 0.74 + float(self.BLOCK_HALF_SIZE) + 0.002
        if qpos is None:
            qpos = [1.0, 0.0, 0.0, 0.0]
        if rotate_lim is None:
            rotate_lim = [0.0, 0.0, 0.75]
        return rand_pose_cyl(
            rlim=[float(r), float(r)],
            thetalim=[float(theta), float(theta)],
            zlim=[float(z), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=list(qpos),
            rotate_rand=bool(rotate_rand),
            rotate_lim=rotate_lim,
            quat_frame=quat_frame,
        )

    def _sample_block_cylinders(self, count, avoid_points=()):
        """Sample graspable block positions using the same r band as check_block_color."""
        count = int(count)
        r_min, r_max = [float(v) for v in getattr(self, "BLOCK_RLIM", (0.44, 0.50))]
        theta_half = float(rotate_theta_half(self)) * float(getattr(self, "BLOCK_THETA_RATIO", 0.82))
        min_abs_theta = max(0.0, float(getattr(self, "BLOCK_MIN_ABS_THETA", 0.0)))
        min_dis = float(getattr(self, "BLOCK_MIN_XY_DISTANCE", 0.095))
        avoid_xy = [np.array(p, dtype=np.float64).reshape(-1)[:2] for p in avoid_points]
        slots, slots_xy = [], []

        for _ in range(int(getattr(self, "BLOCK_SAMPLE_TRIES", 300))):
            if len(slots) >= count:
                break
            theta = float(np.random.uniform(-theta_half, theta_half))
            if min_abs_theta > 0.0 and abs(theta) <= min_abs_theta:
                continue
            slot = (
                float(np.random.uniform(r_min, r_max)),
                theta,
            )
            xy = self._point_from_cyl(slot)[:2]
            if any(float(np.linalg.norm(xy - old_xy)) < min_dis for old_xy in slots_xy):
                continue
            if any(float(np.linalg.norm(xy - old_xy)) < min_dis for old_xy in avoid_xy):
                continue
            slots.append(slot)
            slots_xy.append(xy)

        if len(slots) < count:
            if min_abs_theta > 0.0 and theta_half > min_abs_theta:
                theta_candidates = []
                for theta in np.linspace(theta_half, min_abs_theta, max(count * 2, 4)):
                    theta_candidates.extend([float(theta), float(-theta)])
                r_candidates = np.linspace(r_max, r_min, max(count, 1))
                for theta in theta_candidates:
                    if len(slots) >= count:
                        break
                    for r in r_candidates:
                        slot = (float(r), float(theta))
                        xy = self._point_from_cyl(slot)[:2]
                        if any(float(np.linalg.norm(xy - old_xy)) < min_dis for old_xy in slots_xy):
                            continue
                        if any(float(np.linalg.norm(xy - old_xy)) < min_dis for old_xy in avoid_xy):
                            continue
                        slots.append(slot)
                        slots_xy.append(xy)
                        break
                if len(slots) < count:
                    for theta in theta_candidates:
                        if len(slots) >= count:
                            break
                        slots.append((float((r_min + r_max) * 0.5), float(theta)))
            else:
                # Deterministic but still spread fallback across the same r band.
                fallback_thetas = np.linspace(theta_half, -theta_half, count)
                fallback_rs = np.linspace(r_max, r_min, count)
                slots = [(float(r), float(theta)) for r, theta in zip(fallback_rs, fallback_thetas)]
        return slots[:count]

    def _target_theta(self, point):
        local = world_to_robot(np.array(point, dtype=np.float64).reshape(3).tolist(), self.robot_root_xy, self.robot_yaw)
        return float(local[1])

    def _focus_world_point(self, point, subtask_idx, stage=1, focus_object_key=None, search_keys=None, action_keys=None, info_complete=0):
        point = np.array(point, dtype=np.float64).reshape(3)
        theta = self._target_theta(point)
        self._set_rotate_subtask_state(
            subtask_idx=subtask_idx,
            stage=stage,
            focus_object_key=focus_object_key,
            search_target_keys=[] if search_keys is None else search_keys,
            action_target_keys=[] if action_keys is None else action_keys,
            info_complete=info_complete,
            camera_mode=1 if stage <= 2 else 0,
            camera_target_theta=theta,
        )
        scan_point = point.copy()
        scan_point[2] = float(self.SCAN_Z_BIAS) + float(self.table_z_bias)
        self.face_world_point_with_torso(scan_point, joint_name_prefer=self.SCAN_JOINT_NAME)
        self._refresh_rotate_discovery_from_current_view()

    def _make_backside_patch_block(self, pose, patch_color, name="backside_patch_block"):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name(str(name))
        entity.set_pose(pose)

        half = float(self.BLOCK_HALF_SIZE)
        half_vec = np.array([half, half, half], dtype=np.float32)
        rigid_component = sapien.physx.PhysxRigidDynamicComponent()
        rigid_component.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half_vec,
                material=scene.default_physical_material,
            )
        )

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(
            sapien.render.RenderShapeBox(
                half_vec,
                sapien.render.RenderMaterial(base_color=[*self.OUTER_COLOR[:3], 1.0]),
            )
        )

        patch_half = np.array(self.BACKSIDE_PATCH_HALF_SIZE, dtype=np.float32)
        axis = str(getattr(self, "BACKSIDE_PATCH_AXIS", "x")).lower()
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        sign = 1.0 if float(getattr(self, "BACKSIDE_PATCH_SIGN", 1.0)) >= 0.0 else -1.0
        offset = np.zeros(3, dtype=np.float64)
        offset[axis_idx] = sign * (half + float(patch_half[axis_idx]) - float(self.BACKSIDE_PATCH_PROTRUSION))
        patch = sapien.render.RenderShapeBox(
            patch_half,
            sapien.render.RenderMaterial(base_color=[*tuple(float(v) for v in patch_color[:3]), 1.0]),
        )
        patch.set_local_pose(sapien.Pose(offset.tolist(), [1.0, 0.0, 0.0, 0.0]))
        render_component.attach(patch)

        entity.add_component(rigid_component)
        entity.add_component(render_component)
        scene.add_entity(entity)

        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [half, half, half],
            "scale": [half, half, half],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point of the colored patch on the block backside."],
        }
        return Actor(entity, data)

    def _make_static_pad(self, key, label, color, cyl, half_size=None):
        half_size = tuple(half_size or self.PAD_HALF_SIZE)
        pose = self._pose_from_cyl(cyl, z=0.741, quat_frame="world")
        pad = create_box(
            scene=self,
            pose=pose,
            half_size=half_size,
            color=tuple(float(v) for v in color),
            name=f"{label}_pad",
            is_static=True,
        )
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = f"{label} pad"
        self.add_prohibit_area(pad, padding=0.08)
        return pad

    def _remember_backside_inspection_return_state(self, key):
        store = getattr(self, "_backside_inspection_return_joint_states", None)
        if store is None:
            store = {}
            self._backside_inspection_return_joint_states = store
        store[str(key)] = np.array(self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64).tolist()
        return store[str(key)]

    def _get_backside_inspection_return_state(self, key):
        store = getattr(self, "_backside_inspection_return_joint_states", {}) or {}
        return store.get(str(key), None)

    def _pick_block_for_inspection(self, block, key, subtask_idx):
        if int(getattr(self, "current_subtask_idx", 0)) != int(subtask_idx):
            self.begin_rotate_subtask(subtask_idx)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=key)
        if not self.move(
            self.grasp_actor(
                block,
                arm_tag=self.ARM,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                contact_point_id=int(self.PICK_CONTACT_POINT_ID),
            )
        ):
            self.plan_success = False
            return False
        self._set_carried_object_keys([key])
        if not self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(self.LIFT_Z), move_axis="world")):
            self.plan_success = False
            return False
        self._remember_backside_inspection_return_state(key)
        self.complete_rotate_subtask(subtask_idx, carried_after=[key])
        return True

    def _inspect_carried_block_backside(self, block, key, subtask_idx):
        del block
        if int(getattr(self, "current_subtask_idx", 0)) != int(subtask_idx):
            self.begin_rotate_subtask(subtask_idx)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=key)
        if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_pose=left_check_pose)])):
            self.plan_success = False
            return False
        self.delay(int(self.INSPECT_HOLD_STEPS), save_freq=1)
        if bool(getattr(self, "RETURN_AFTER_INSPECT", False)):
            return_state = self._get_backside_inspection_return_state(key)
            if return_state is not None:
                if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_joint_pos=return_state)])):
                    self.plan_success = False
                    return False
        self.complete_rotate_subtask(subtask_idx, carried_after=[key])
        return True

    def _pick_inspect_block(self, block, key, subtask_idx):
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=key)
        if not self.move(
            self.grasp_actor(
                block,
                arm_tag=self.ARM,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                contact_point_id=int(self.PICK_CONTACT_POINT_ID),
            )
        ):
            self.plan_success = False
            return False
        self._set_carried_object_keys([key])
        if not self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(self.LIFT_Z), move_axis="world")):
            self.plan_success = False
            return False
        left_state = np.array(self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64).tolist()
        if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_pose=left_check_pose)])):
            self.plan_success = False
            return False
        self.delay(int(self.INSPECT_HOLD_STEPS), save_freq=1)
        if bool(getattr(self, "RETURN_AFTER_INSPECT", False)):
            if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_joint_pos=left_state)])):
                self.plan_success = False
                return False
        self.complete_rotate_subtask(subtask_idx, carried_after=[key])
        return True

    def _place_block_at(self, block, key, point, subtask_idx, focus_key=None):
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=focus_key or key)
        point = np.array(point, dtype=np.float64).reshape(3)
        ee_pose = np.array(self.robot.get_left_ee_pose(), dtype=np.float64)
        block_pos = np.array(block.get_pose().p, dtype=np.float64)
        ee_to_block = block_pos - ee_pose[:3]
        target_center = point.copy()
        target_center[2] = (
            float(getattr(self, "rotate_table_top_z", 0.74 + float(getattr(self, "table_z_bias", 0.0))))
            + float(self.BLOCK_HALF_SIZE)
            + float(getattr(self, "PLACE_RELEASE_CLEARANCE", 0.012))
        )
        release_pose = ee_pose.copy()
        release_pose[:3] = target_center - ee_to_block
        if not self.move(self.move_to_pose(self.ARM, release_pose)):
            self.plan_success = False
            return False
        if not self.move(self.open_gripper(self.ARM)):
            self.plan_success = False
            return False
        self._set_carried_object_keys([])
        self.delay(10)
        self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(getattr(self, "PLACE_RETREAT_Z", 0.06)), move_axis="world"))
        self.move(self.back_to_origin(self.ARM))
        place_ok = getattr(self, "_backside_place_xy_ok", None)
        if place_ok is None:
            self._backside_place_xy_ok = {}
            place_ok = self._backside_place_xy_ok
        block_xy = np.array(block.get_pose().p[:2], dtype=np.float64)
        target_xy = np.array(point[:2], dtype=np.float64)
        place_ok[str(key)] = bool(float(np.linalg.norm(block_xy - target_xy)) <= float(self.SUCCESS_XY_TOL))
        self.complete_rotate_subtask(subtask_idx, carried_after=[])
        return True
