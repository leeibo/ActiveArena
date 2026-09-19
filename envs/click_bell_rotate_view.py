from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class click_bell_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    BELL_RLIM = (0.40, 0.45)

    def check_success(self):
        if self.stage_success_tag:
            return True
        if not self.check_arm_function():
            return False
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        eps = [0.025, 0.025]
        for position in positions:
            if (np.all(np.abs(position[:2] - bell_pose[:2]) < eps) and abs(position[2] - bell_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.bell,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "_click_bell",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "bell_pressed",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        while True:
            rand_pos = rand_pose_cyl(
                rlim=list(self.BELL_RLIM),
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.bell_id = int(np.random.choice([0, 1], 1)[0])
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (
            self.is_left_gripper_close
            if self.bell.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )
        self._configure_rotate_subtask_plan()

    def play_once(self):
        bell_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        bell_cyl = world_to_robot(self.bell.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if bell_cyl[1] >= 0 else "right")

        self.enter_rotate_action_stage(1, focus_object_key=(bell_key or "A"))
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                gripper_pos=-0.1,
                contact_point_id=0,
            )
        )

        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.check_success()
        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.check_success()
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {"{A}": "bell", "{a}": str(arm_tag)}
        return self.info
