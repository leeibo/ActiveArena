from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class place_cans_plasticbox_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    CAN_BOX_MIN_DISTANCE = 0.13
    CAN_BOX_SAMPLE_TRIES = 120

    def check_success(self):
        plasticbox_functional_points_0 = self.plasticbox.get_functional_point(0)[0:2]
        plasticbox_functional_points_1 = self.plasticbox.get_functional_point(1)[0:2]
        dis1 = min(np.linalg.norm(self.object1.get_pose().p[0:2] - plasticbox_functional_points_0),
                   np.linalg.norm(self.object1.get_pose().p[0:2] - plasticbox_functional_points_1))
        dis2 = min(np.linalg.norm(self.object2.get_pose().p[0:2] - plasticbox_functional_points_0),
                   np.linalg.norm(self.object2.get_pose().p[0:2] - plasticbox_functional_points_1))
        threshold = 0.04
        return dis1 < threshold and dis2 < threshold and self.is_left_gripper_open() and self.is_right_gripper_open()

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object1,
                "B": self.plasticbox,
                "C": self.object2,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_first_can",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "first_can_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_first_can_in_box",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "first_can_in_box",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_second_can",
                    "instruction_idx": 3,
                    "search_target_keys": ["C"],
                    "action_target_keys": ["C"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["C"],
                    "allow_stage2_from_memory": True,
                    "done_when": "second_can_grasped",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "place_second_can_in_box",
                    "instruction_idx": 4,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B", "C"],
                    "required_carried_keys": ["C"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "second_can_in_box",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def _can_pose_far_from_box(self, pose):
        can_xy = np.array(pose.p[:2], dtype=np.float64)
        box_points = [np.array(self.plasticbox.get_pose().p[:2], dtype=np.float64)]
        for point_id in (0, 1):
            try:
                box_points.append(np.array(self.plasticbox.get_functional_point(point_id)[:2], dtype=np.float64))
            except Exception:
                pass
        min_distance = float(self.CAN_BOX_MIN_DISTANCE)
        return all(float(np.linalg.norm(can_xy - point)) >= min_distance for point in box_points)

    def _sample_can_pose_away_from_box(self, side):
        for _ in range(int(self.CAN_BOX_SAMPLE_TRIES)):
            pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=rotate_theta_side(self, side=side),
                zlim=[0.755, 0.755],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
                rotate_lim=[0, 0, 0],
            )
            if self._can_pose_far_from_box(pose):
                return pose
        raise RuntimeError("Failed to sample can pose outside the plastic box footprint")

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        box_pose = rand_pose_cyl(
            rlim=[0.4, 0.5],
            thetalim=rotate_theta_side(self, side=1),

            zlim=[0.741, 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.plasticbox_id = int(np.random.choice([3, 5], 1)[0])
        self.plasticbox = create_actor(
            scene=self,
            pose=box_pose,
            modelname="062_plasticbox",
            convex=True,
            model_id=self.plasticbox_id,
            is_static=True,
        )
        self.plasticbox.set_mass(0.05)

        obj1_pose = self._sample_can_pose_away_from_box(side=1)
        self.object1_id = int(np.random.choice([0, 1, 2, 3, 5, 6], 1)[0])
        self.object1 = create_actor(
            scene=self,
            pose=obj1_pose,
            modelname="071_can",
            convex=True,
            model_id=self.object1_id,
        )
        self.object1.set_mass(0.05)

        obj2_pose = self._sample_can_pose_away_from_box(side=-1)
        self.object2_id = int(np.random.choice([0, 1, 2, 3, 5, 6], 1)[0])
        self.object2 = create_actor(
            scene=self,
            pose=obj2_pose,
            modelname="071_can",
            convex=True,
            model_id=self.object2_id,
        )
        self.object2.set_mass(0.05)

        self.add_prohibit_area(self.plasticbox, padding=0.1)
        self.add_prohibit_area(self.object1, padding=0.05)
        self.add_prohibit_area(self.object2, padding=0.05)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        object1_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")
        t1 = self.plasticbox.get_functional_point(1)
        t0 = self.plasticbox.get_functional_point(0)
        self.enter_rotate_action_stage(1, focus_object_key=(object1_key or "A"))
        self.move(self.grasp_actor(self.object1, arm_tag=arm_tag_left, pre_grasp_dis=0.07))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.1))
        self.complete_rotate_subtask(1, carried_after=["A"])

        box_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(box_key or "B"))
        self.move(
            self.place_actor(
                self.object1,
                arm_tag=arm_tag_left,
                target_pose=t1,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self.move(self.back_to_origin(arm_tag=arm_tag_left))
        self.complete_rotate_subtask(2, carried_after=[])

        object2_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(3, focus_object_key=(object2_key or "C"))
        self.move(self.grasp_actor(self.object2, arm_tag=arm_tag_right, pre_grasp_dis=0.07))
        self._set_carried_object_keys(["C"])
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.1))
        self.complete_rotate_subtask(3, carried_after=["C"])

        box_key = self.search_and_focus_rotate_subtask(
            4,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(4, focus_object_key=(box_key or "B"))
        self.move(
            self.place_actor(
                self.object2,
                arm_tag=arm_tag_right,
                target_pose=t0,
                constrain="free",
            ),
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))
        self.complete_rotate_subtask(4, carried_after=[])

        self.info["info"] = {
            "{A}": "can",
            "{B}": "plastic box",
            "{C}": "can",
        }
        return self.info
