from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np


class place_fan_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.fan,
                "B": self.pad,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_fan",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "fan_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_fan_on_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "fan_on_pad",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, is_test=False, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(is_test=is_test, **kwargs)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        fan_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.0, 0.0, 0.707, 0.707],
            rotate_rand=True,
            rotate_lim=[0, 2 * np.pi, 0],
        )
        self.fan_id = int(np.random.choice([4, 5]))
        self.fan = create_actor(
            scene=self,
            pose=fan_pose,
            modelname="099_fan",
            convex=True,
            model_id=self.fan_id,
        )
        self.fan.set_mass(0.01)

        pad_side = 1 if self.fan.get_pose().p[0] > 0 else 1
        pad_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_side(self, side=pad_side),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }
        color_items = list(colors.items())
        idx = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[idx]

        self.pad = create_box(
            scene=self,
            pose=pad_pose,
            half_size=(0.05, 0.05, 0.001),
            color=self.color_value,
            name="box",
        )
        self.pad.set_mass(1)
        self.add_prohibit_area(self.fan, padding=0.07)
        self.prohibited_area.append([
            pad_pose.p[0] - 0.15,
            pad_pose.p[1] - 0.15,
            pad_pose.p[0] + 0.15,
            pad_pose.p[1] + 0.15,
        ])
        target_pose = self.pad.get_pose().p
        self.target_pose = target_pose.tolist() + [1, 0, 0, 0]
        self._configure_rotate_subtask_plan()

    def play_once(self):
        fan_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.fan.get_pose().p[0] > 0 else "left")
        self.enter_rotate_action_stage(1, focus_object_key=(fan_key or "A"))
        self.move(self.grasp_actor(self.fan, arm_tag=arm_tag, pre_grasp_dis=0.05))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.complete_rotate_subtask(1, carried_after=["A"])

        pad_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(pad_key or "B"))
        self.move(
            self.place_actor(
                self.fan,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",  # Orientation is explicitly checked in this task.
                pre_dis=0.04,
                dis=0.005,
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "fan",
            "{B}": self.color_name,
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        fan_qpose = self.fan.get_pose().q
        fan_pose = self.fan.get_pose().p

        target_pose = self.target_pose[:3]
        target_qpose = np.array([0.707, 0.707, 0.0, 0.0])

        if fan_qpose[0] < 0:
            fan_qpose *= -1

        eps = np.array([0.05, 0.05, 0.05, 0.05])

        return np.all(abs(fan_pose - target_pose) < np.array([0.04, 0.04, 0.04]))
