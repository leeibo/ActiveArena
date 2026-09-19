import numpy as np
from pathlib import Path
from PIL import Image

from .check_cola_date import check_cola_date
from ._info_task_helpers import INFO_COLOR_RGBA_MAP, sample_info_color_specs


class check_cola_color(check_cola_date):
    """Simpler cola backside task: inspect a pure color label and sort by color."""

    LABEL_COLOR_OPTIONS = tuple(INFO_COLOR_RGBA_MAP)
    LABEL_TEXTURE_COLORS = INFO_COLOR_RGBA_MAP
    PAD_CYLS = ((0.42, 0.4), (0.42, -0.4))

    def setup_demo(self, **kwargs):
        self.label_color_override = kwargs.pop("can_label_color_override", None)
        super().setup_demo(**kwargs)

    @classmethod
    def _make_date_label_texture(cls, path, production_date):
        label_color = str(production_date).strip().lower()
        rgba = cls.LABEL_TEXTURE_COLORS.get(label_color)
        if rgba is None:
            rgba = next(iter(cls.LABEL_TEXTURE_COLORS.values()))
        image = Image.new("RGBA", (512, 768), rgba)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def _sample_color_specs(self):
        required = None
        if self.label_color_override is not None:
            required = str(self.label_color_override).strip().lower()
        return sample_info_color_specs(len(self.PAD_CYLS), required_label=required)

    def _sample_label_color(self, labels):
        if self.label_color_override is not None:
            color = str(self.label_color_override).strip().lower()
            if color not in labels:
                raise ValueError(f"Invalid can_label_color_override: {self.label_color_override}")
            return color
        return str(np.random.choice(list(labels)))

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.target_pads = {}
        self.pad_centers = {}
        self.active_color_specs = self._sample_color_specs()
        self.PAD_TARGETS = {
            label: cyl
            for (label, _), cyl in zip(self.active_color_specs, self.PAD_CYLS)
        }
        self.PAD_COLORS = dict(self.active_color_specs)
        self.PAD_KEYS = {label: label.upper() for label, _ in self.active_color_specs}
        for label, cyl in self.PAD_TARGETS.items():
            key = self.PAD_KEYS[label]
            pad = self._make_target_pad(label, cyl)
            self.target_pads[label] = pad
            self.pad_centers[label] = np.array(pad.get_pose().p, dtype=np.float64).reshape(3)
            self.object_layers[key] = "lower"
            self.object_labels[key] = f"{label} area"

        labels = [label for label, _ in self.active_color_specs]
        self.label_color = self._sample_label_color(labels)
        self.target_label = self.label_color
        self.target_pad_key = self.PAD_KEYS[self.target_label]
        self.can_backside_inspected = False
        self.can_placed = False
        self.can_id = int(self.CAN_MODEL_ID)
        self.can = self._make_can_asset(
            self._pose_from_cyl(
                self.CAN_CYL,
                z=float(self.CAN_Z),
                qpos=list(self.CAN_QPOS),
                quat_frame="cyl_legacy",
                rotate_rand=False,
                rotate_lim=[0.0, 0.0, 0.0],
            ),
            self.label_color,
        )
        self.object_layers["A"] = "lower"
        self.object_labels["A"] = f"071 can with {self.label_color} backside label"
        self.initial_can_z = float(self.can.get_pose().p[2])
        self.add_prohibit_area(self.can, padding=0.06)
        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        pad_registry = {
            self.PAD_KEYS[label]: self.target_pads[label]
            for label, _ in self.active_color_specs
        }
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.can,
                **pad_registry,
            },
            subtask_defs=[
                dict(id=1, name="pick_cola_can", instruction_idx=1,
                     search_target_keys=["A"], action_target_keys=["A"], required_carried_keys=[],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=True,
                     done_when="cola_can_grasped", next_subtask_id=2),
                dict(id=2, name="inspect_cola_backside_color", instruction_idx=2,
                     search_target_keys=[], action_target_keys=["A"], required_carried_keys=["A"],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=False,
                     done_when="cola_backside_color_seen", next_subtask_id=3),
                dict(id=3, name="restore_cola_after_inspection", instruction_idx=3,
                     search_target_keys=[], action_target_keys=["A"], required_carried_keys=["A"],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=False,
                     done_when="cola_pose_restored_after_color_inspection", next_subtask_id=4),
                dict(id=4, name="place_can_on_matching_color_area", instruction_idx=4,
                     search_target_keys=[self.target_pad_key], action_target_keys=["A", self.target_pad_key],
                     required_carried_keys=["A"], carry_keys_after_done=[], allow_stage2_from_memory=True,
                     done_when="cola_sorted_by_color", next_subtask_id=-1),
            ],
        )

    def play_once(self):
        found = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=self.SCAN_Z_BIAS + self.table_z_bias,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if found is None or not self._pick_and_lift_can(1):
            self.plan_success = False
        else:
            if not self._inspect_can_backside(2):
                self.plan_success = False
            if self.plan_success and not self._restore_can_after_inspection(3):
                self.plan_success = False
        if self.plan_success:
            self.search_and_focus_rotate_subtask(
                4,
                scan_r=self.SCAN_R,
                scan_z=self.SCAN_Z_BIAS + self.table_z_bias,
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if not self._place_can_in_target_area(4):
                self.plan_success = False
        self.info["info"] = {
            "{A}": self._natural_model_label(self.CAN_MODEL_NAME, fallback="cola can"),
            "{B}": f"{self.active_color_specs[0][0]} area",
            "{C}": f"{self.active_color_specs[1][0]} area",
            "{F}": self.label_color,
            "{a}": str(self.ARM),
        }
        return self.info
