from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class open_laptop_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def check_success(self, target=0.4):
        limit = self.laptop.get_qlimits()[0]
        qpos = self.laptop.get_qpos()
        rotate_pose = self.laptop.get_contact_point(1)
        tip_pose = (self.robot.get_left_tcp_pose() if self.arm_tag == "left" else self.robot.get_right_tcp_pose())
        dis = np.sqrt(np.sum((np.array(tip_pose[:3]) - np.array(rotate_pose[:3]))**2))
        return qpos[0] >= limit[0] + (limit[1] - limit[0]) * target and dis < 0.1

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.laptop,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "_open_laptop",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "laptop_opened",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, is_test=False, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(is_test=is_test, **kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        self.model_name = "015_laptop"
        # self.model_id = int(np.random.randint(0, 11))
        self.model_id = 1
        laptop_pose = rand_pose_cyl(
            rlim=[0.58, 0.58],
            thetalim=[-0.1,0.1],

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.7, 0, 0, 0.7],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 3],
        )
        self.laptop = create_sapien_urdf_obj(
            scene=self,
            pose=laptop_pose,
            modelname=self.model_name,
            modelid=self.model_id,
            fix_root_link=True,
        )
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.2])
        self.laptop.set_mass(0.01)
        self.laptop.set_properties(1, 0)
        self.add_prohibit_area(self.laptop, padding=0.1)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        laptop_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        face_prod = get_face_prod(self.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        arm_tag = ArmTag("left" if face_prod > 0 else "right")
        self.arm_tag = arm_tag

        self.enter_rotate_action_stage(1, focus_object_key=(laptop_key or "A"))
        self.move(self.grasp_actor(self.laptop, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))
        for _ in range(15):
            self.move(
                self.grasp_actor(
                    self.laptop,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.0,
                    grasp_dis=-0.01,
                    contact_point_id=1,
                )
            )
            if not self.plan_success:
                break
            if self.check_success(target=0.2):
                break
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {
            "{A}": self._natural_model_label(self.model_name),
            "{a}": str(arm_tag),
        }
        return self.info
