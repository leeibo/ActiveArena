from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class move_pillbottle_pad_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def check_success(self):
        pillbottle_pos = self.pillbottle.get_pose().p
        target_pos = self.pad.get_pose().p
        eps1 = 0.03
        return (np.all(abs(pillbottle_pos[:2] - target_pos[:2]) < np.array([eps1, eps1]))
                and np.abs(self.pillbottle.get_pose().p[2] - (0.741 + self.table_z_bias)) < 0.005
                and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.pillbottle,
                "B": self.pad,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_pillbottle",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "pillbottle_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_pillbottle_on_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "pillbottle_on_pad",
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
            rand_pos = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.pillbottle_id = int(np.random.choice([1, 2, 3, 4, 5], 1)[0])
        self.pillbottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="080_pillbottle",
            convex=True,
            model_id=self.pillbottle_id,
        )
        self.pillbottle.set_mass(0.05)

        pill_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
        same_side = 1.0 if pill_cyl[1] >= 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=[0.35, 0.45],
                thetalim=rotate_theta_mixed(self, side=same_side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < 0.1:
                continue
            break

        self.pad = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=[0.04, 0.04, 0.0005],
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.pillbottle, padding=0.05)
        self.add_prohibit_area(self.pad, padding=0.1)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        bottle_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")
        self.enter_rotate_action_stage(1, focus_object_key=(bottle_key or "A"))
        self.move(
            self.grasp_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                pre_grasp_dis=0.12,
                gripper_pos=0.3,
                grasp_dis=-0.02,
            )
        )
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.complete_rotate_subtask(1, carried_after=["A"])

        pad_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.pad.get_functional_point(1)
        self.enter_rotate_action_stage(2, focus_object_key=(pad_key or "B"))
        self.move(
            self.place_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.05,
                dis=0,
                functional_point_id=0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "pill bottle",
            "{a}": str(arm_tag),
        }
        return self.info
