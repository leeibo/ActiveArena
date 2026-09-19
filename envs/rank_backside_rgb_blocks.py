import numpy as np
import transforms3d as t3d

from ._base_task import Base_Task
from ._info_task_helpers import BACKSIDE_RGB_COLOR_SPECS, BacksidePatchBlockMixin
from .utils import create_box, prepare_rotate_task_kwargs


class rank_backside_rgb_blocks(BacksidePatchBlockMixin, Base_Task):
    """Inspect hidden backside colors, then arrange the gray blocks red-green-blue."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    BLOCK_RLIM = (0.41, 0.41)
    BLOCK_THETA_RATIO = 0.82
    BLOCK_MIN_ABS_THETA = 0.20
    BLOCK_MIN_XY_DISTANCE = 0.105
    PRE_INSPECT_DELAY = 2

    SORT_R = 0.4
    SORT_THETA_STEP = 0.18
    SORT_TARGETS = {
        "red": (SORT_R, SORT_THETA_STEP),
        "green": (SORT_R, 0.0),
        "blue": (SORT_R, -SORT_THETA_STEP),
    }
    PAD_KEYS = {
        "red": "RED_PAD",
        "green": "GREEN_PAD",
        "blue": "BLUE_PAD",
    }
    PAD_COLOR = (0.82, 0.84, 0.78)
    PAD_HALF_SIZE = (0.040, 0.040, 0.0005)
    PAD_Z = 0.741
    PAD_FOOTPRINT_TOL = 0.005
    PLACE_TARGET_QUAT = (0.0, 1.0, 0.0, 0.0)
    PLACE_PRE_DIS = 0.1
    PLACE_DIS = 0.02
    PLACE_RETREAT_Z = 0.05

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _sort_targets(self):
        return {
            label: self._point_from_cyl(cyl, z=float(self.PAD_Z))
            for label, cyl in self.SORT_TARGETS.items()
        }

    def _make_sort_pad(self, label, cyl):
        pose = self._pose_from_cyl(
            cyl,
            z=float(self.PAD_Z),
            qpos=[1.0, 0.0, 0.0, 0.0],
            quat_frame="cyl",
            rotate_rand=False,
            rotate_lim=[0.0, 0.0, 0.0],
        )
        pad = create_box(
            scene=self,
            pose=pose,
            half_size=self.PAD_HALF_SIZE,
            color=self.PAD_COLOR,
            name=f"{label}_rank_position_pad",
            is_static=True,
        )
        pad_key = self.PAD_KEYS[label]
        self.object_layers[pad_key] = "lower"
        self.object_labels[pad_key] = f"{label} target position pad"
        self.add_prohibit_area(pad, padding=0.07)
        return pad

    @staticmethod
    def _ordered_block_samples(color_order, cylinders):
        samples = list(zip(color_order, cylinders))
        samples.sort(key=lambda item: float(item[1][1]), reverse=True)
        return samples

    @staticmethod
    def _subtask_ids(block_idx):
        pick_id = (block_idx - 1) * 3 + 1
        return pick_id, pick_id + 1, pick_id + 2

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers, self.object_labels = {}, {}
        self.sort_targets = self._sort_targets()
        self.sort_pads = {
            label: self._make_sort_pad(label, cyl)
            for label, cyl in self.SORT_TARGETS.items()
        }
        self.pad_centers = {
            label: np.array(pad.get_pose().p, dtype=np.float64).reshape(3)
            for label, pad in self.sort_pads.items()
        }

        color_order = list(BACKSIDE_RGB_COLOR_SPECS)
        np.random.shuffle(color_order)
        block_slots = self._sample_block_cylinders(
            len(color_order),
            avoid_points=self.sort_targets.values(),
        )

        self.blocks, self.block_colors, self.place_targets = {}, {}, {}
        self.placed_keys = set()
        for idx, ((label, color), cyl) in enumerate(self._ordered_block_samples(color_order, block_slots), start=1):
            key = f"B{idx}"
            pose = self._pose_from_cyl(cyl, rotate_rand=False, rotate_lim=[0.0, 0.0, 0.0])
            block = self._make_backside_patch_block(
                pose,
                color,
                name="a_grey_block",
            )
            block.set_mass(0.035)

            self.blocks[key] = block
            self.block_colors[key] = label
            self.place_targets[key] = self.sort_targets[label].copy()
            self.object_layers[key] = "lower"
            self.object_labels[key] = "a grey block"
            self.add_prohibit_area(block, padding=0.055)

        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        subtasks = []
        subtask_instruction_map = {}
        total = len(self.blocks)
        for idx, key in enumerate(self.blocks, start=1):
            label = self.block_colors[key]
            pad_key = self.PAD_KEYS[label]
            pick_id, inspect_id, place_id = self._subtask_ids(idx)
            subtask_instruction_map[pick_id] = "pick up a grey block with the left arm"
            subtask_instruction_map[inspect_id] = "inspect a grey block's backside color"
            subtask_instruction_map[place_id] = "place a grey block on its RGB-order pad"
            next_pick_id = self._subtask_ids(idx + 1)[0] if idx < total else -1
            subtasks.extend(
                [
                    dict(
                        id=pick_id,
                        name="pick_a_grey_block",
                        instruction_idx=pick_id,
                        search_target_keys=[key],
                        action_target_keys=[key],
                        required_carried_keys=[],
                        carry_keys_after_done=[key],
                        allow_stage2_from_memory=True,
                        done_when=f"{key}_grasped",
                        next_subtask_id=inspect_id,
                    ),
                    dict(
                        id=inspect_id,
                        name="inspect_a_grey_block_backside_color",
                        instruction_idx=inspect_id,
                        search_target_keys=[],
                        action_target_keys=[key],
                        required_carried_keys=[key],
                        carry_keys_after_done=[key],
                        allow_stage2_from_memory=False,
                        done_when=f"{key}_backside_seen",
                        next_subtask_id=place_id,
                    ),
                    dict(
                        id=place_id,
                        name="place_a_grey_block_on_rgb_rank_pad",
                        instruction_idx=place_id,
                        search_target_keys=[pad_key],
                        action_target_keys=[key, pad_key],
                        required_carried_keys=[key],
                        carry_keys_after_done=[],
                        allow_stage2_from_memory=True,
                        done_when=f"{key}_placed_by_backside_color",
                        next_subtask_id=next_pick_id,
                    ),
                ]
            )

        self.configure_rotate_subtask_plan(
            object_registry={
                **self.blocks,
                **{self.PAD_KEYS[label]: pad for label, pad in self.sort_pads.items()},
            },
            subtask_defs=subtasks,
            subtask_instruction_map=subtask_instruction_map,
            subtask_instruction_template_map={},
            task_instruction=(
                "Inspect the hidden backside colors of three gray blocks and arrange "
                "them on the three position pads in red, green, blue order."
            ),
        )

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _place_block_at(self, block, key, point, subtask_idx, focus_key=None):
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=focus_key or key)
        point = np.array(point, dtype=np.float64).reshape(3)

        table_top_z = float(getattr(self, "rotate_table_top_z", 0.74 + float(getattr(self, "table_z_bias", 0.0))))
        target_pose = point.copy()
        target_pose[2] = max(table_top_z, float(point[2]) + float(self.PAD_HALF_SIZE[2]))
        target_pose = target_pose.tolist() + list(self.PLACE_TARGET_QUAT)

        # if not self.move(
        #     self.place_actor(
        #         block,
        #         target_pose=target_pose,
        #         arm_tag=self.ARM,
        #         functional_point_id=0,
        #         pre_dis=float(self.PLACE_PRE_DIS),
        #         dis=float(self.PLACE_DIS),
        #         constrain="free",
        #     )
        # ):
        #     self.plan_success = False
        #     return False

        self.move(
            self.move_by_displacement(
                arm_tag=self.ARM,
                z=-0.05,
                move_axis="arm",
            )
        )
        self.move(
            self.open_gripper(
                arm_tag=self.ARM,
            )
        )
        self._set_carried_object_keys([])
        self.move(
            self.move_by_displacement(
                arm_tag=self.ARM,
                z=float(self.PLACE_RETREAT_Z),
                move_axis="arm",
            )
        )
        self.complete_rotate_subtask(subtask_idx, carried_after=[])
        return True

    def _run_one_block(self, idx, key, block):
        pick_id, inspect_id, place_id = self._subtask_ids(idx)
        if int(getattr(self, "current_subtask_idx", 0)) != pick_id:
            self.begin_rotate_subtask(pick_id)

        self._focus_world_point(
            block.get_pose().p,
            pick_id,
            focus_object_key=key,
            search_keys=[key],
            action_keys=[key],
        )

        if not self._pick_block_for_inspection(block, key, pick_id):
            self.plan_success = False
            return False

        if not self._inspect_carried_block_backside(block, key, inspect_id):
            self.plan_success = False
            return False

        label = self.block_colors[key]
        pad_key = self.PAD_KEYS[label]
        self._focus_world_point(
            self.pad_centers[label],
            place_id,
            stage=2,
            focus_object_key=pad_key,
            search_keys=[pad_key],
            action_keys=[key, pad_key],
            info_complete=1,
        )
        if not self._place_block_at(block, key, self.place_targets[key], place_id, focus_key=pad_key):
            return False

        self.placed_keys.add(key)
        return True

    def play_once(self):
        for idx, (key, block) in enumerate(self.blocks.items(), start=1):
            if not self._run_one_block(idx, key, block):
                break

        self.info["info"] = {
            "{A}": "three gray backside-marked blocks",
            "{B}": "three pale ordering pads",
            "{a}": str(self.ARM),
        }
        return self.info

    def _block_center_on_pad(self, block, label):
        pad_pose = self.sort_pads[label].get_pose()
        block_pos = np.array(block.get_pose().p, dtype=np.float64).reshape(3)
        pad_pos = np.array(pad_pose.p, dtype=np.float64).reshape(3)
        pad_rot = t3d.quaternions.quat2mat(np.array(pad_pose.q, dtype=np.float64))
        local_xy = (pad_rot.T @ (block_pos - pad_pos))[:2]
        half_xy = np.array(self.PAD_HALF_SIZE[:2], dtype=np.float64)
        tol = float(self.PAD_FOOTPRINT_TOL)
        return bool(np.all(np.abs(local_xy) <= half_xy + tol))

    def check_success(self):
        return bool(
            all(
                self._block_center_on_pad(block, self.block_colors[key])
                for key, block in self.blocks.items()
            )
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
