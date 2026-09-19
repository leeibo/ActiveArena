from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class blocks_ranking_size_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_R = 0.64
    BLOCK_RLIM = (0.40, 0.45)

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
                    "name": "pick_medium_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "medium_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_medium_block_right_of_large",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "medium_block_placed",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_small_block",
                    "instruction_idx": 3,
                    "search_target_keys": ["C"],
                    "action_target_keys": ["C"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["C"],
                    "allow_stage2_from_memory": True,
                    "done_when": "small_block_grasped",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "place_small_block_right_of_medium",
                    "instruction_idx": 4,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B", "C"],
                    "required_carried_keys": ["C"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "small_block_placed",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    @staticmethod
    def _valid_spacing(new_pose, existing_pose_lst, min_dist_sq=0.014):
        for pose in existing_pose_lst:
            if np.sum(np.square(new_pose.p[:2] - pose.p[:2])) < min_dist_sq:
                return False
        return True

    @staticmethod
    def _far_from_target_band(new_pose, target_xy_lst, min_dist_sq=0.014):
        for xy in target_xy_lst:
            if np.sum(np.square(new_pose.p[:2] - xy)) < min_dist_sq:
                return False
        return True

    def _sample_target_gaps(self, halfsize_lst):
        gap12 = max(float(np.random.uniform(0.16, 0.2)), (halfsize_lst[0] + halfsize_lst[1] + 0.02) / 0.45)
        gap23 = max(float(np.random.uniform(0.16, 0.2)), (halfsize_lst[1] + halfsize_lst[2] + 0.02) / 0.45)
        return gap12, gap23

    def _sample_anchor_pose(self, size, total_gap):
        theta_half = rotate_theta_half(self)
        theta_lo = max(0.15, -theta_half + total_gap + 0.15)
        theta_hi = theta_half - 0.12
        if theta_lo > theta_hi:
            theta_lo = max(-theta_half + total_gap + 0.05, 0.0)
            theta_hi = max(theta_lo, theta_half - 0.05)

        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=list(self.BLOCK_RLIM),
                thetalim=[theta_lo, theta_hi],
                zlim=[0.741 + size, 0.741 + size],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            if world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)[0] < 0.4:
                continue
            return deepcopy(pose)

        fallback_theta = max(theta_lo, min(theta_hi, max(0.15, total_gap + 0.1)))
        return rand_pose_cyl(
            rlim=[float(np.mean(self.BLOCK_RLIM)), float(np.mean(self.BLOCK_RLIM))],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + size, 0.741 + size],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def _sample_block_pose(self, size, existing_pose_lst, avoid_xy_lst):
        for _ in range(120):
            pose = rand_pose_cyl(
                rlim=list(self.BLOCK_RLIM),
                thetalim=rotate_theta_center(self),

                zlim=[0.741 + size, 0.741 + size],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if cyl[0] < 0.4:
                continue
            if not self._valid_spacing(pose, existing_pose_lst):
                continue
            if not self._far_from_target_band(pose, avoid_xy_lst):
                continue
            return pose
        theta_half = rotate_theta_half(self)
        fallback_theta = float(np.clip(-0.35 * theta_half, -theta_half, theta_half))
        return rand_pose_cyl(
            rlim=[self.BLOCK_RLIM[1], self.BLOCK_RLIM[1]],
            thetalim=[fallback_theta, fallback_theta],
            zlim=[0.741 + size, 0.741 + size],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def _get_block_arm_tag(self, block):
        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if block_cyl[1] >= 0 else "right")

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        color_lst = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(3)]
        halfsize_lst = [
            float(np.random.uniform(0.03, 0.033)),
            float(np.random.uniform(0.024, 0.027)),
            float(np.random.uniform(0.018, 0.021)),
        ]
        gap12, gap23 = self._sample_target_gaps(halfsize_lst)
        anchor_pose = self._sample_anchor_pose(size=halfsize_lst[0], total_gap=gap12 + gap23)
        anchor_cyl = world_to_robot(anchor_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
        target_theta_lst = [anchor_cyl[1] - gap12, anchor_cyl[1] - gap12 - gap23]
        target_xy_lst = [
            place_point_cyl(
                [anchor_cyl[0], theta, anchor_pose.p[2]],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                ret="array",
            )[:2]
            for theta in target_theta_lst
        ]
        block_pose_lst = [anchor_pose]
        for i in range(1, 3):
            block_pose_lst.append(
                self._sample_block_pose(
                    size=halfsize_lst[i],
                    existing_pose_lst=block_pose_lst,
                    avoid_xy_lst=target_xy_lst,
                )
            )

        def create_block(block_pose, size, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(size, size, size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], halfsize_lst[0], color_lst[0])
        self.block2 = create_block(block_pose_lst[1], halfsize_lst[1], color_lst[1])
        self.block3 = create_block(block_pose_lst[2], halfsize_lst[2], color_lst[2])

        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.add_prohibit_area(self.block3, padding=0.1)
        target_z = 0.74 + self.table_z_bias
        target_quat = [0, 1, 0, 0]

        self.block2_target_pose = place_pose_cyl(
            [anchor_cyl[0], target_theta_lst[0], target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self.block3_target_pose = place_pose_cyl(
            [anchor_cyl[0], target_theta_lst[1], target_z] + target_quat,
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        self._configure_rotate_subtask_plan()

    def play_once(self):
        self.last_gripper = None
        block2_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        anchor_arm = self._get_block_arm_tag(self.block1)
        arm_tag2 = self._get_block_arm_tag(self.block2)
        self.enter_rotate_action_stage(1, focus_object_key=(block2_key or "B"))
        self.move(self.grasp_actor(self.block2, arm_tag=arm_tag2, pre_grasp_dis=0.09))
        self._set_carried_object_keys(["B"])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.12))
        self.complete_rotate_subtask(1, carried_after=["B"])

        block1_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(block1_key or "A"))
        self.move(
            self.place_actor(
                self.block2,
                target_pose=self.block2_target_pose,
                arm_tag=arm_tag2,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag2, z=0.07, move_axis="arm"))
        self.complete_rotate_subtask(2, carried_after=[])
        self.last_gripper = arm_tag2

        block3_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        arm_tag3 = ArmTag(arm_tag2)
        self.enter_rotate_action_stage(3, focus_object_key=(block3_key or "C"))
        if self.last_gripper is not None and self.last_gripper != arm_tag3:
            self.move(self.back_to_origin(arm_tag=arm_tag3.opposite))
        self.move(self.grasp_actor(self.block3, arm_tag=arm_tag3, pre_grasp_dis=0.09))
        self._set_carried_object_keys(["C"])
        self.move(self.move_by_displacement(arm_tag=arm_tag3, z=0.12))
        self.complete_rotate_subtask(3, carried_after=["C"])

        block2_anchor_key = self.search_and_focus_rotate_subtask(
            4,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(4, focus_object_key=(block2_anchor_key or "B"))
        self.move(
            self.place_actor(
                self.block3,
                target_pose=self.block3_target_pose,
                arm_tag=arm_tag3,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag3, z=0.07, move_axis="arm"))
        self.complete_rotate_subtask(4, carried_after=[])
        self.last_gripper = arm_tag3

        self.info["info"] = {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": str(anchor_arm),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
        }
        return self.info

    def pick_and_place_block(self, block, target_pose):


        block_cyl = world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))
        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12))

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        block1_pose = self.block1.get_pose().p.tolist()
        block2_pose = self.block2.get_pose().p.tolist()
        block3_pose = self.block3.get_pose().p.tolist()

        c1 = world_to_robot(block1_pose, self.robot_root_xy, self.robot_yaw)
        c2 = world_to_robot(block2_pose, self.robot_root_xy, self.robot_yaw)
        c3 = world_to_robot(block3_pose, self.robot_root_xy, self.robot_yaw)

        theta_eps = 0.35
        radius_eps = 0.14
        same_arc = abs(c1[0] - c2[0]) < radius_eps and abs(c2[0] - c3[0]) < radius_eps
        ordered = c1[1] > c2[1] > c3[1]
        compact = abs(c1[1] - c2[1]) < theta_eps and abs(c2[1] - c3[1]) < theta_eps
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return same_arc and ordered and compact and gripper_open
