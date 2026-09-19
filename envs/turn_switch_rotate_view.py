from ._base_task import Base_Task
from .utils import *
import numpy as np


class turn_switch_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_Z_BIAS = 0.92

    def check_success(self):
        limit = self.switch.get_qlimits()[0]
        return self.switch.get_qpos()[0] >= limit[1] - 0.05

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.switch,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "_turn_switch",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "switch_triggered",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, is_test=False, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(is_test=is_test, **kwargs)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        self.model_name = "056_switch"
        self.model_id = int(np.random.randint(0, 8))
        while True:
            switch_pose = rand_pose_cyl(
                rlim=[0.46, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.81, 0.84],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
                qpos=[0.704141, 0, 0, 0.71006],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            switch_cyl = world_to_robot(switch_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(switch_cyl[1]) < 0.2:
                continue
            break

        self.switch = create_sapien_urdf_obj(
            scene=self,
            pose=switch_pose,
            modelname=self.model_name,
            modelid=self.model_id,
            fix_root_link=True,
        )
        self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2])
        self._configure_rotate_subtask_plan()

    def play_once(self):
        focus_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.92 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        switch_pose = self.switch.get_pose()
        face_dir = -switch_pose.to_transformation_matrix()[:3, 0]
        arm_tag = ArmTag("right" if face_dir[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(focus_key or "A"))
        self.move(self.close_gripper(arm_tag=arm_tag, pos=-0.1))
        self.move(self.grasp_actor(self.switch, arm_tag=arm_tag, pre_grasp_dis=0.04))
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {"{A}": "switch", "{a}": str(arm_tag)}
        return self.info
