import numpy as np

from ._base_task import Base_Task
from ._info_task_helpers import PolarCountLayoutMixin, RMBenchButtonMixin
from .utils import *


class count_target_press_button(RMBenchButtonMixin, PolarCountLayoutMixin, Base_Task):
    """Count green target blocks, then press the left-to-right 1/2/3 button."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    TARGET_COLOR = (0.10, 0.80, 0.20)
    DISTRACTOR_COLORS = ((0.90, 0.20, 0.20), (0.20, 0.45, 0.92), (0.92, 0.74, 0.18))
    BLOCK_HALF_SIZE = 0.022
    TARGET_COUNT_RANGE = (1, 3)
    DISTRACTOR_COUNT = 3
    COUNT_OBJECT_RLIM = (0.46, 0.74)
    COUNT_OBJECT_THETA_RATIO = 1.0
    COUNT_SLOT_COUNT = 7
    COUNT_OBJECT_MIN_XY_DISTANCE = 0.075
    COUNT_OBJECT_BUTTON_AVOID_DISTANCE = 0.14
    COUNT_OBJECT_ROTATE_RAND = True

    BUTTON_R = 0.40
    BUTTON_THETAS = (0.58, 0.0, -0.58)  # left, middle, right in robot/table view.
    BUTTON_VALUES = (1, 2, 3)

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.72, 0.24, -0.24, -0.72)

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        override = kwargs.get("count_target_count_override", None)
        self.count_target_count_override = None if override is None else int(override)
        super()._init_task_env_(**kwargs)

    def _make_block(self, key, pose, color, label):
        block = create_box(
            scene=self,
            pose=pose,
            half_size=(self.BLOCK_HALF_SIZE,) * 3,
            color=tuple(float(v) for v in color),
            name=str(key),
        )
        block.set_mass(0.03)
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = str(label)
        self.add_prohibit_area(block, padding=0.04)
        return block

    def _make_buttons(self):
        self.buttons = {}
        for value, theta in zip(self.BUTTON_VALUES, self.BUTTON_THETAS):
            key = f"BTN{value}"
            button = self._create_rmbench_button(
                r=float(self.BUTTON_R),
                theta=float(theta),
                name=f"number_{value}_button",
            )
            self.buttons[int(value)] = button
            self.object_layers[key] = "lower"
            self.object_labels[key] = f"number {value} button"
            self.add_prohibit_area(button, padding=0.055)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers, self.object_labels = {}, {}
        self.button_press_flag = False
        self.button_press_count = 0
        self.pressed_button_value = None
        if self.count_target_count_override is None:
            self.target_count = int(np.random.randint(1, 4))
        else:
            self.target_count = int(np.clip(self.count_target_count_override, 1, 3))

        slots = self._sample_polar_count_slots(self.target_count + self.DISTRACTOR_COUNT)
        self.target_blocks, self.distractor_blocks = {}, {}
        for idx, (r, theta) in enumerate(slots[: self.target_count], start=1):
            self.target_blocks[f"T{idx}"] = self._make_block(
                f"T{idx}", self._block_pose_from_cyl(r, theta), self.TARGET_COLOR, "green target block"
            )
        for idx, (r, theta) in enumerate(slots[self.target_count :], start=1):
            color = self.DISTRACTOR_COLORS[(idx - 1) % len(self.DISTRACTOR_COLORS)]
            self.distractor_blocks[f"D{idx}"] = self._make_block(
                f"D{idx}", self._block_pose_from_cyl(r, theta), color, "distractor block"
            )
        self._make_buttons()
        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        registry = {**self.target_blocks, **self.distractor_blocks}
        registry.update({f"BTN{value}": button for value, button in self.buttons.items()})
        target_button_key = f"BTN{int(self.target_count)}"
        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                dict(
                    id=1,
                    name="count_green_target_blocks",
                    instruction_idx=1,
                    search_target_keys=list(self.target_blocks) + list(self.distractor_blocks),
                    action_target_keys=[],
                    required_carried_keys=[],
                    carry_keys_after_done=[],
                    allow_stage2_from_memory=False,
                    done_when="target_count_known",
                    next_subtask_id=2,
                ),
                dict(
                    id=2,
                    name="press_matching_number_button",
                    instruction_idx=2,
                    search_target_keys=[target_button_key],
                    action_target_keys=[target_button_key],
                    required_carried_keys=[],
                    carry_keys_after_done=[],
                    allow_stage2_from_memory=True,
                    done_when="matching_number_button_pressed",
                    next_subtask_id=-1,
                ),
            ],
            task_instruction=(
                "Count the green blocks, then press the matching button; "
                "from left to right, the buttons represent 1, 2, and 3."
            ),
        )

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _scan_all_blocks_for_count(self):
        objects = list(self.target_blocks.values()) + list(self.distractor_blocks.values())
        self._scan_count_objects_fixed_head(
            subtask_idx=1,
            target_keys=list(self.target_blocks),
            all_keys=list(self.target_blocks) + list(self.distractor_blocks),
            objects=objects,
        )

    def _press_matching_button(self):
        value = int(self.counted_target_count)
        button = self.buttons[value]
        key = f"BTN{value}"
        self._focus_lower_object_fixed_head(2, key, button, [key], [key])
        theta = world_to_robot(button.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)[1]
        arm_tag = ArmTag("left" if float(theta) >= 0.0 else "right")
        self.button_arm_tag = arm_tag
        self.enter_rotate_action_stage(2, focus_object_key=key)
        if self._press_button_once(button, arm_tag=arm_tag, flag_attr="button_press_flag", count_attr="button_press_count"):
            self.pressed_button_value = value
        else:
            self.plan_success = False
            return
        self.complete_rotate_subtask(2, carried_after=[])

    def play_once(self):
        self._scan_all_blocks_for_count()
        if self.plan_success:
            self._press_matching_button()
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": "green target blocks",
            "{B}": f"number {int(getattr(self, 'target_count', 0))} button",
            "{a}": str(getattr(self, "button_arm_tag", "left/right")),
            "{x}": int(getattr(self, "target_count", 0)),
        }

    def check_success(self):
        return self._check_matching_button_success(self.buttons, self.target_count)

    def check_failure(self):
        return self._get_wrong_button_failure(self.buttons, self.target_count)
