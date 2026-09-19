from ._base_task import Base_Task
from .utils import *
import math
import sapien
import numpy as np


class place_shoe_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.shoe,
                "B": self.target_block,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_shoe",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "shoe_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_shoe_on_mat",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "shoe_on_mat",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, is_test=False, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(is_test=is_test, **kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        target_pose = place_pose_cyl(
            [0.45, 0.0, 0.74, 1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="pose",
        )
        self.target_block = create_box(
            scene=self,
            pose=target_pose,
            half_size=(0.08, 0.08, 0.0005),
            color=(0, 0, 1),
            is_static=True,
            name="box",
        )
        self.target_block.config["functional_matrix"] = [[
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ], [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]]

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = rotate_theta_side(self, side=side)
        while True:
            shoe_pose = rand_pose_cyl(
                rlim=[0.45, 0.45],
                thetalim=theta_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
            shoe_cyl = world_to_robot(shoe_pose.get_p().tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(shoe_cyl[1]) < 0.35:
                continue
            if np.sum((shoe_pose.get_p()[:2] - self.target_block.get_pose().p[:2])**2) < 0.03:
                continue
            break

        self.shoe_id = int(np.random.choice([i for i in range(10)]))
        self.shoe = create_actor(
            scene=self,
            pose=shoe_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        self.add_prohibit_area(self.target_block, padding=0.08)
        self.add_prohibit_area(self.shoe, padding=0.1)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        shoe_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        shoe_pose = self.shoe.get_pose().p
        arm_tag = ArmTag("left" if shoe_pose[0] < 0 else "right")

        self.enter_rotate_action_stage(1, focus_object_key=(shoe_key or "A"))
        self.move(self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=0.07, grasp_dis=-0.01,gripper_pos=0.2))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self.complete_rotate_subtask(1, carried_after=["A"])

        target_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.target_block.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(target_key or "B"))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.07))
        # self.move(
        #     self.place_actor(
        #         self.shoe,
        #         arm_tag=arm_tag,
        #         target_pose=target_pose,
        #         functional_point_id=0,
        #         pre_dis=0.12,
        #         constrain="free",  # Shoe placement needs orientation alignment on the target pad.
        #     )
        # )
        self._set_carried_object_keys([])
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {"{A}": "shoe", "{a}": str(arm_tag)}
        return self.info
    def check_success(self):
        shoe_pose = self.shoe.get_pose().p
        target_pose = self.target_block.get_pose().p
        eps = np.array([0.05, 0.05, 0.05])
        return np.all(abs(shoe_pose - target_pose) < eps)
