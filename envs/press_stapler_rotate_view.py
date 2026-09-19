from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import numpy as np


class press_stapler_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def check_success(self):
        if self.stage_success_tag:
            return True
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps = [0.03, 0.03]
        for position in positions:
            if (np.all(np.abs(position[:2] - stapler_pose[:2]) < eps) and abs(position[2] - stapler_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.stapler,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "_press_stapler",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "stapler_pressed",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        rand_pos = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_center(self),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )
        self.stapler_id = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0])
        self.stapler = create_actor(
            self,
            pose=rand_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )
        self.add_prohibit_area(self.stapler, padding=0.05)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        stapler_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")
        self.enter_rotate_action_stage(1, focus_object_key=(stapler_key or "A"))
        self.move(
            self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.1, contact_point_id=2)
        )
        self.move(self.close_gripper(arm_tag=arm_tag))

        self.move(
            self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.02, grasp_dis=0.02, contact_point_id=2)
        )
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {"{A}": "stapler", "{a}": str(arm_tag)}
        return self.info
