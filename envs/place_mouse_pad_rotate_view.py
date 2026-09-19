from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class place_mouse_pad_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.mouse,
                "B": self.target,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_mouse",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "mouse_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_mouse_on_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "mouse_on_pad",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        while True:
            mouse_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            c = world_to_robot(mouse_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(c[1]) < 0.2:
                continue
            break

        self.mouse_id = int(np.random.choice([0, 1, 2], 1)[0])
        self.mouse = create_actor(
            scene=self,
            pose=mouse_pose,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
        )
        self.mouse.set_mass(0.05)

        side = 1.0 if mouse_pose.p[0] > 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_side(self, side=side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - mouse_pose.p[:2]) < 0.1:
                continue
            break

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }
        color_items = list(colors.items())
        color_index = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[color_index]

        self.target = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=[0.035, 0.035, 0.0005],
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.target, padding=0.12)
        self.add_prohibit_area(self.mouse, padding=0.03)
        self.target_pose = self.target.get_pose().p.tolist() + [0, 0, 0, 1]
        self._configure_rotate_subtask_plan()

    def play_once(self):
        mouse_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")
        self.enter_rotate_action_stage(1, focus_object_key=(mouse_key or "A"))
        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["A"])

        target_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",  # Mouse orientation is part of success condition.
                pre_dis=0.07,
                dis=0.005,
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "mouse",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        mouse_pose = self.mouse.get_pose().p
        target_pose = self.target.get_pose().p
        eps = np.array([0.015, 0.012])
        return np.all(abs(mouse_pose[:2] - target_pose[:2]) < eps)
