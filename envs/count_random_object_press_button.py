import numpy as np

from ._base_task import Base_Task
from ._info_task_helpers import PolarCountLayoutMixin, RMBenchButtonMixin
from .utils import *


class count_random_object_press_button(RMBenchButtonMixin, PolarCountLayoutMixin, Base_Task):
    """Count a randomly selected object type, then press the left-to-right 1/2/3 button."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    OBJECT_TYPES = (
        dict(
            label="cans",
            object_label="can",
            modelname="071_can",
            model_ids=(0, 1, 2, 3, 5, 6),
            z=0.755,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=False,
            rotate_lim=(0.0, 0.0, 0.0),
            padding=0.055,
        ),
        dict(
            label="pill bottles",
            object_label="pill bottle",
            modelname="080_pillbottle",
            model_ids=(1, 2, 3, 4, 5),
            z=0.741,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=False,
            rotate_lim=(0.0, 0.0, 0.0),
            padding=0.055,
        ),
        dict(
            label="mice",
            object_label="mouse",
            modelname="047_mouse",
            model_ids=(0, 1, 2),
            z=0.741,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=True,
            rotate_lim=(0.0, np.pi, 0.0),
            padding=0.055,
        ),
        dict(
            label="staplers",
            object_label="stapler",
            modelname="048_stapler",
            model_ids=(0, 1, 2, 3, 4, 5, 6),
            z=0.741,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=True,
            rotate_lim=(0.0, np.pi, 0.0),
            padding=0.06,
        ),
        dict(
            label="bells",
            object_label="bell",
            modelname="050_bell",
            model_ids=(0, 1),
            z=0.741,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=False,
            rotate_lim=(0.0, 0.0, 0.0),
            padding=0.075,
        ),
        dict(
            label="alarm clocks",
            object_label="alarm clock",
            modelname="046_alarm-clock",
            model_ids=(1, 3),
            z=0.741,
            qpos=(0.5, 0.5, 0.5, 0.5),
            rotate_rand=True,
            rotate_lim=(0.0, np.pi, 0.0),
            padding=0.065,
        ),
    )
    TARGET_COUNT_RANGE = (1, 3)
    DISTRACTOR_COUNT = 3
    COUNT_OBJECT_RLIM = (0.46, 0.74)
    COUNT_OBJECT_THETA_RATIO = 1.0
    COUNT_SLOT_COUNT = 7
    COUNT_OBJECT_MIN_XY_DISTANCE = 0.13
    COUNT_OBJECT_BUTTON_AVOID_DISTANCE = 0.16

    BUTTON_R = 0.40
    BUTTON_THETAS = (0.58, 0.0, -0.58)
    BUTTON_VALUES = (1, 2, 3)

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.72, 0.24, -0.24, -0.72)

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        override = kwargs.pop("count_random_object_count_override", None)
        self.target_count_override = None if override is None else int(override)
        super()._init_task_env_(**kwargs)

    def _pose_for_object(self, r, theta, spec):
        return rand_pose_cyl(
            rlim=[float(r), float(r)],
            thetalim=[float(theta), float(theta)],
            zlim=[float(spec["z"]), float(spec["z"])],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=list(spec.get("qpos", (1, 0, 0, 0))),
            rotate_rand=bool(spec.get("rotate_rand", False)),
            rotate_lim=list(spec.get("rotate_lim", (0.0, 0.0, 0.0))),
        )

    def _make_object(self, key, pose, spec):
        model_id = int(np.random.choice(list(spec["model_ids"])))
        obj = create_actor(
            scene=self,
            pose=pose,
            modelname=str(spec["modelname"]),
            convex=True,
            model_id=model_id,
            is_static=True,
        )
        if obj is None:
            raise RuntimeError(f"Failed to create count object {spec['modelname']} model_id={model_id}")
        obj.actor.set_name(str(key))
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = str(spec["object_label"])
        self.add_prohibit_area(obj, padding=float(spec.get("padding", 0.06)))
        return obj

    def _make_buttons(self):
        self.buttons = {}
        for value, theta in zip(self.BUTTON_VALUES, self.BUTTON_THETAS):
            key = f"BTN{value}"
            button = self._create_rmbench_button(r=float(self.BUTTON_R), theta=float(theta), name=f"number_{value}_button")
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
        count_min, count_max = [int(v) for v in self.TARGET_COUNT_RANGE]
        self.target_count = int(
            np.random.randint(count_min, count_max + 1)
            if self.target_count_override is None
            else np.clip(self.target_count_override, count_min, count_max)
        )

        target_idx = int(np.random.randint(len(self.OBJECT_TYPES)))
        target_spec = self.OBJECT_TYPES[target_idx]
        self.target_type_label = str(target_spec["label"])
        distractor_specs = [spec for idx, spec in enumerate(self.OBJECT_TYPES) if idx != target_idx]
        distractor_indices = np.random.choice(
            len(distractor_specs),
            size=int(self.DISTRACTOR_COUNT),
            replace=int(self.DISTRACTOR_COUNT) > len(distractor_specs),
        )
        slots = self._sample_polar_count_slots(self.target_count + self.DISTRACTOR_COUNT)

        self.target_objects, self.distractor_objects = {}, {}
        for idx, (r, theta) in enumerate(slots[: self.target_count], start=1):
            key = f"T{idx}"
            self.target_objects[key] = self._make_object(
                key,
                self._pose_for_object(r, theta, target_spec),
                target_spec,
            )
        for idx, ((r, theta), spec_idx) in enumerate(zip(slots[self.target_count :], distractor_indices), start=1):
            spec = distractor_specs[int(spec_idx)]
            key = f"D{idx}"
            self.distractor_objects[key] = self._make_object(
                key,
                self._pose_for_object(r, theta, spec),
                spec,
            )
        self._make_buttons()
        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        registry = {**self.target_objects, **self.distractor_objects}
        registry.update({f"BTN{value}": button for value, button in self.buttons.items()})
        target_button_key = f"BTN{int(self.target_count)}"
        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                dict(id=1, name="count_random_target_objects", instruction_idx=1,
                     search_target_keys=list(self.target_objects) + list(self.distractor_objects), action_target_keys=[],
                     required_carried_keys=[], carry_keys_after_done=[], allow_stage2_from_memory=False,
                     done_when="target_object_count_known", next_subtask_id=2),
                dict(id=2, name="press_matching_number_button", instruction_idx=2,
                     search_target_keys=[target_button_key], action_target_keys=[target_button_key],
                     required_carried_keys=[], carry_keys_after_done=[], allow_stage2_from_memory=True,
                     done_when="matching_number_button_pressed", next_subtask_id=-1),
            ],
            task_instruction=(
                f"Count the {self.target_type_label}, then press the matching button; "
                "from left to right, the buttons represent 1, 2, and 3."
            ),
        )

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _scan_all_objects_for_count(self):
        objects = list(self.target_objects.values()) + list(self.distractor_objects.values())
        self._scan_count_objects_fixed_head(
            subtask_idx=1,
            target_keys=list(self.target_objects),
            all_keys=list(self.target_objects) + list(self.distractor_objects),
            objects=objects,
        )

    def _press_matching_button(self):
        value = int(self.counted_target_count)
        key = f"BTN{value}"
        button = self.buttons[value]
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
        self._scan_all_objects_for_count()
        if self.plan_success:
            self._press_matching_button()
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": self.target_type_label,
            "{B}": f"number {int(getattr(self, 'target_count', 0))} button",
            "{a}": str(getattr(self, "button_arm_tag", "left/right")),
            "{x}": int(getattr(self, "target_count", 0)),
        }

    def check_success(self):
        return self._check_matching_button_success(self.buttons, self.target_count)

    def check_failure(self):
        return self._get_wrong_button_failure(self.buttons, self.target_count)
