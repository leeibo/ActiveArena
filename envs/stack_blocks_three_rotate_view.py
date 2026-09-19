from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class stack_blocks_three_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_R = 0.64
    ROTATE_SCAN_SCENE_Z_BIAS = 0.90
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (1.00, -1.00)

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        eps = [0.025, 0.025, 0.012]

        return (np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2] + 0.05])) < eps)
                and np.all(abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2] + 0.05])) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.block1,
                "B": self.block2,
                "C": self.block3,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_green_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "stack_green_on_red",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "green_block_stacked",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_blue_block",
                    "instruction_idx": 3,
                    "search_target_keys": ["C"],
                    "action_target_keys": ["C"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["C"],
                    "allow_stage2_from_memory": True,
                    "done_when": "blue_block_grasped",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "stack_blue_on_green",
                    "instruction_idx": 4,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B", "C"],
                    "required_carried_keys": ["C"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "blue_block_stacked",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    @staticmethod
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq=0.01):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < min_dist_sq:
                return False
        return True

    def _sample_block_pose(self, block_half_size, existing_pose_lst):
        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_center(self),
                zlim=[0.741 + block_half_size, 0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
            )
            if world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)[0] < 0.4:
                continue
            if not self._valid_spacing(pose, existing_pose_lst):
                continue
            return deepcopy(pose)

        fallback_theta = float(
            np.clip(
                0.35 * rotate_theta_half(self) * (1.0 - len(existing_pose_lst)),
                -rotate_theta_half(self),
                rotate_theta_half(self),
            )
        )
        return rand_pose_cyl(
            rlim=[0.45, 0.45],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + block_half_size, 0.741 + block_half_size],
            qpos=[1, 0, 0, 0],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            rotate_rand=False,
        )

    def _get_block_arm_tag(self, block):
        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if block_cyl[1] >= 0 else "right")

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        block_half_size = 0.025
        block_pose_lst = []
        while len(block_pose_lst) < 3:
            block_pose_lst.append(self._sample_block_pose(block_half_size, block_pose_lst))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.block3 = create_block(block_pose_lst[2], (0, 0, 1))
        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
        self._configure_rotate_subtask_plan()

    def pick_and_place_block(self, block: Actor):
        arm_tag = self._get_block_arm_tag(block)

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        target_pose = self.last_actor.get_functional_point(1)
        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def play_once(self):
        block2_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        self.last_gripper = None
        anchor_arm = self._get_block_arm_tag(self.block1)

        arm_tag2 = self._get_block_arm_tag(self.block2)
        self.enter_rotate_action_stage(1, focus_object_key=(block2_key or "B"))
        self.move(self.grasp_actor(self.block2, arm_tag=arm_tag2, pre_grasp_dis=0.09))
        self._set_carried_object_keys(["B"])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["B"])

        block1_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.block1.get_functional_point(1)
        self.enter_rotate_action_stage(2, focus_object_key=(block1_key or "A"))
        self.move(
            self.place_actor(
                self.block2,
                target_pose=target_pose,
                arm_tag=arm_tag2,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.07))
        self.complete_rotate_subtask(2, carried_after=[])
        self.last_gripper = arm_tag2

        block3_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        arm_tag3 = self._get_block_arm_tag(self.block3)
        self.enter_rotate_action_stage(3, focus_object_key=(block3_key or "C"))
        if self.last_gripper is not None and self.last_gripper != arm_tag3:
            self.move(self.back_to_origin(arm_tag=arm_tag3.opposite))
        self.move(self.grasp_actor(self.block3, arm_tag=arm_tag3, pre_grasp_dis=0.09))
        self._set_carried_object_keys(["C"])
        self.move(self.move_by_displacement(arm_tag=arm_tag3, z=0.1))
        self.complete_rotate_subtask(3, carried_after=["C"])

        block2_anchor_key = self.search_and_focus_rotate_subtask(
            4,
            scan_r=0.64,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.block2.get_functional_point(1)
        self.enter_rotate_action_stage(4, focus_object_key=(block2_anchor_key or "B"))
        self.move(
            self.place_actor(
                self.block3,
                target_pose=target_pose,
                arm_tag=arm_tag3,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag3, z=0.07))
        self.complete_rotate_subtask(4, carried_after=[])
        self.last_gripper = arm_tag3

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(anchor_arm),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
        }
        return self.info
