from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import time
import numpy as np


class stamp_seal_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_Z_BIAS = 0.90

    def check_success(self):
        seal_pose = self.seal.get_pose().p
        target_pos = self.target.get_pose().p
        eps1 = 0.03

        return bool(
            np.all(abs(seal_pose[:2] - target_pos[:2]) < np.array([eps1, eps1]))
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.seal,
                "B": self.target,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_seal",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "seal_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "stamp_target_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "target_stamped",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = rotate_theta_side(self, side=side)
        target_lim = rotate_theta_side(self, side=-side)
        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=theta_lim,

                zlim=[0.741, 0.741],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            seal_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(seal_cyl[1]) < 0.25:
                continue
            break

        self.seal_id = int(np.random.choice([0, 2, 3, 4, 6], 1)[0])
        self.seal = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.45, 0.5],
                thetalim=target_lim,

                zlim=[0.741, 0.741],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1:
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

        half_size = [0.035, 0.035, 0.0005]
        self.target = create_visual_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)
        self.target_pose = self.target.get_pose()
        self._configure_rotate_subtask_plan()

    def play_once(self):
        seal_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.seal.get_pose().p[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(seal_key or "A"))
        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=[4, 5, 6, 7]))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.complete_rotate_subtask(1, carried_after=["A"])

        target_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.target.get_pose()
        self.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))
        self.move(
            self.place_actor(
                self.seal,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.1,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "seal",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info
