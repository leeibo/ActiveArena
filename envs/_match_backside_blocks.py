import numpy as np
import transforms3d as t3d

from ._base_task import Base_Task
from ._info_task_helpers import BacksidePatchBlockMixin, sample_info_color_specs
from .utils import create_box, prepare_rotate_task_kwargs, rotate_theta_half


class MatchBacksideBlocksBase(BacksidePatchBlockMixin, Base_Task):
    """Shared flow for backside-color block-to-pad matching tasks."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    BLOCK_COUNT = 2
    COLOR_SPECS = ()
    COLOR_SAMPLE_COUNT = None
    UNIQUE_LABEL_RATE = None
    MAX_LABEL_REPEAT = None

    # Same polar pad description style as check_block_color.
    PAD_RLIM = (0.43, 0.43)
    # With rand_pose_cyl(..., quat_frame="cyl"), pad local y follows the radial axis.
    PAD_HALF_SIZE = (0.045, 0.105, 0.004)
    PAD_THETA_RATIO = 0.55
    PAD_Z = 0.741
    PAD_QUAT_FRAME = "cyl"
    PAD_SLOT_R_OFFSETS = (-0.04, 0.01)
    PAD_FOOTPRINT_TOL = 0.005

    BLOCK_RLIM = (0.42, 0.42)
    BLOCK_THETA_RATIO = 0.82
    BLOCK_MIN_XY_DISTANCE = 0.105
    PRE_INSPECT_DELAY = 0

    TASK_INSTRUCTION = ""
    INFO_BLOCKS = "gray backside-marked blocks"
    INFO_PADS = "matching color pads"
    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(**kwargs)

    def _get_color_specs(self):
        specs = getattr(self, "active_color_specs", None)
        if specs is not None:
            return tuple(specs)
        sample_count = getattr(self, "COLOR_SAMPLE_COUNT", None)
        if sample_count is None:
            specs = tuple(self.COLOR_SPECS)
        else:
            specs = sample_info_color_specs(int(sample_count))
        self.active_color_specs = tuple(specs)
        return self.active_color_specs

    def _color_labels(self):
        return [label for label, _ in self._get_color_specs()]

    def _color_map(self):
        return dict(self._get_color_specs())

    def _pad_layout(self):
        theta_half = rotate_theta_half(self)
        theta_abs = float(theta_half * self.PAD_THETA_RATIO)
        color_specs = self._get_color_specs()
        pad_thetas = np.linspace(-theta_abs, theta_abs, len(color_specs))
        pad_r = float(np.mean(self.PAD_RLIM))
        return {
            label: (pad_r, float(theta))
            for (label, _), theta in zip(color_specs, pad_thetas)
        }

    def _sample_block_labels(self):
        labels = self._color_labels()
        block_count = int(self.BLOCK_COUNT)
        unique_rate = getattr(self, "UNIQUE_LABEL_RATE", None)
        max_repeat = getattr(self, "MAX_LABEL_REPEAT", None)

        if unique_rate is not None and block_count == len(labels):
            if np.random.rand() < float(unique_rate):
                sampled = labels[:]
                np.random.shuffle(sampled)
                return sampled
            duplicate = str(np.random.choice(labels))
            remaining = [label for label in labels if label != duplicate]
            extra_count = block_count - 2
            extras = list(np.random.choice(remaining, size=extra_count, replace=False))
            sampled = [duplicate, duplicate] + [str(label) for label in extras]
            np.random.shuffle(sampled)
            return sampled

        for _ in range(100):
            sampled = [str(np.random.choice(labels)) for _ in range(block_count)]
            if max_repeat is None or max(sampled.count(label) for label in labels) <= int(max_repeat):
                return sampled

        sampled = labels[:]
        while len(sampled) < block_count:
            sampled.append(str(np.random.choice(labels)))
        np.random.shuffle(sampled)
        return sampled[:block_count]

    def _ordered_block_samples(self, labels, cylinders):
        samples = list(zip(labels, cylinders))
        samples.sort(key=lambda item: float(item[1][1]), reverse=True)
        return samples

    def _pad_slot_centers(self, label):
        r, theta = self.pad_cylinders[label]
        return [
            self._point_from_cyl((r + float(offset), theta), z=float(self.PAD_Z))
            for offset in self.PAD_SLOT_R_OFFSETS
        ]

    def _place_target(self, label, repeat_idx):
        slot_idx = min(int(repeat_idx), len(self.pad_slots[label]) - 1)
        return self.pad_slots[label][slot_idx].copy()

    def _make_pad(self, key, label, color, cyl):
        pose = self._pose_from_cyl(
            cyl,
            z=float(self.PAD_Z),
            qpos=[1.0, 0.0, 0.0, 0.0],
            quat_frame=str(self.PAD_QUAT_FRAME),
            rotate_rand=False,
            rotate_lim=[0.0, 0.0, 0.0],
        )
        pad = create_box(
            scene=self,
            pose=pose,
            half_size=self.PAD_HALF_SIZE,
            color=tuple(float(v) for v in color),
            name=f"{label}_pad",
            is_static=True,
        )
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = f"{label} pad"
        self.add_prohibit_area(pad, padding=0.08)
        return pad

    @staticmethod
    def _pad_key(label):
        return str(label).upper()

    @staticmethod
    def _subtask_ids(block_idx):
        pick_id = (block_idx - 1) * 3 + 1
        return pick_id, pick_id + 1, pick_id + 2

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers, self.object_labels = {}, {}
        self.pads, self.pad_centers, self.pad_cylinders = {}, {}, self._pad_layout()
        self.pad_slots = {}

        for label, color in self._get_color_specs():
            pad = self._make_pad(self._pad_key(label), label, color, self.pad_cylinders[label])
            self.pads[label] = pad
            self.pad_centers[label] = np.array(pad.get_pose().p, dtype=np.float64)
            self.pad_slots[label] = self._pad_slot_centers(label)

        sampled_labels = self._sample_block_labels()
        block_slots = self._sample_block_cylinders(
            len(sampled_labels),
            avoid_points=self.pad_centers.values(),
        )

        self.blocks, self.block_colors, self.place_targets = {}, {}, {}
        self.placed_keys = set()
        repeat_counts = {label: 0 for label in self._color_labels()}
        color_map = self._color_map()

        for idx, (label, cyl) in enumerate(self._ordered_block_samples(sampled_labels, block_slots), start=1):
            key = f"B{idx}"
            pose = self._pose_from_cyl(cyl, rotate_rand=False, rotate_lim=[0.0, 0.0, 0.0])
            block = self._make_backside_patch_block(
                pose,
                color_map[label],
                name="a_grey_block",
            )
            block.set_mass(0.035)

            self.blocks[key] = block
            self.block_colors[key] = label
            self.place_targets[key] = self._place_target(label, repeat_counts[label])
            repeat_counts[label] += 1

            self.object_layers[key] = "lower"
            self.object_labels[key] = "a grey block"
            self.add_prohibit_area(block, padding=0.055)

        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        registry = {**self.blocks}
        registry.update({self._pad_key(label): pad for label, pad in self.pads.items()})

        subtasks = []
        subtask_instruction_map = {}
        total = len(self.blocks)
        for idx, key in enumerate(self.blocks, start=1):
            label = self.block_colors[key]
            pad_key = self._pad_key(label)
            pick_id, inspect_id, place_id = self._subtask_ids(idx)
            subtask_instruction_map[pick_id] = "pick up a grey block with the left arm"
            subtask_instruction_map[inspect_id] = "inspect a grey block's backside color"
            subtask_instruction_map[place_id] = "place a grey block on the matching pad"
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
                        name="place_a_grey_block_on_matching_pad",
                        instruction_idx=place_id,
                        search_target_keys=[pad_key],
                        action_target_keys=[key, pad_key],
                        required_carried_keys=[key],
                        carry_keys_after_done=[],
                        allow_stage2_from_memory=True,
                        done_when=f"{key}_matched_to_pad",
                        next_subtask_id=next_pick_id,
                    ),
                ]
            )

        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=subtasks,
            subtask_instruction_map=subtask_instruction_map,
            subtask_instruction_template_map={},
            task_instruction=self.TASK_INSTRUCTION,
        )

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _block_center_on_pad(self, block, label):
        pad_pose = self.pads[label].get_pose()
        block_pos = np.array(block.get_pose().p, dtype=np.float64).reshape(3)
        pad_pos = np.array(pad_pose.p, dtype=np.float64).reshape(3)
        pad_rot = t3d.quaternions.quat2mat(np.array(pad_pose.q, dtype=np.float64))
        local_xy = (pad_rot.T @ (block_pos - pad_pos))[:2]
        half_xy = np.array(self.PAD_HALF_SIZE[:2], dtype=np.float64)
        tol = float(getattr(self, "PAD_FOOTPRINT_TOL", 0.0))
        return bool(np.all(np.abs(local_xy) <= half_xy + tol))

    def _block_center_on_matching_pad(self, key, block):
        return self._block_center_on_pad(block, self.block_colors[key])

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
        self.move(
            self.move_by_displacement(
                arm_tag=self.ARM,
                z=float(getattr(self, "PLACE_RETREAT_Z", 0.06)),
                move_axis="world",
            )
        )

        place_ok = getattr(self, "_backside_place_xy_ok", None)
        if place_ok is None:
            self._backside_place_xy_ok = {}
            place_ok = self._backside_place_xy_ok
        place_ok[str(key)] = self._block_center_on_matching_pad(key, block)

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
        if int(getattr(self, "PRE_INSPECT_DELAY", 0)) > 0:
            self.delay(int(self.PRE_INSPECT_DELAY))

        if not self._pick_block_for_inspection(block, key, pick_id):
            self.plan_success = False
            return False

        if not self._inspect_carried_block_backside(block, key, inspect_id):
            self.plan_success = False
            return False

        label = self.block_colors[key]
        pad_key = self._pad_key(label)
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
            "{A}": self.INFO_BLOCKS,
            "{B}": self.INFO_PADS,
            "{C}": ", ".join(self.block_colors[key] for key in self.blocks),
            "{a}": str(self.ARM),
        }
        return self.info

    def check_success(self):
        return bool(
            all(
                self._block_center_on_matching_pad(key, block)
                for key, block in self.blocks.items()
            )
        )
