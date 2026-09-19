from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class move_stapler_pad_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    STAPLER_RLIM = (0.40, 0.47)
    PAD_RLIM = (0.32, 0.42)
    STAPLER_MIN_ABS_THETA = 0.2
    PAD_MIN_DISTANCE = 0.1
    STAPLER_PLACE_PRE_DIS = 0.10
    STAPLER_PLACE_DIS = 0.0
    STAPLER_PLACE_PRE_DIS_AXIS = "grasp"
    STAPLER_PLACE_CONSTRAIN = "free"

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.stapler,
                "B": self.pad,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_stapler",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "stapler_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_stapler_on_pad",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "stapler_on_pad",
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
                rlim=list(self.STAPLER_RLIM),
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < float(self.STAPLER_MIN_ABS_THETA):
                continue
            break

        self.stapler_id = int(np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0])
        self.stapler = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
        )

        stapler_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
        same_side = 1.0 if stapler_cyl[1] >= 0 else -1.0
        while True:
            target_rand_pose = rand_pose_cyl(
                rlim=list(self.PAD_RLIM),
                thetalim=rotate_theta_side(self, side=-same_side),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            if np.linalg.norm(target_rand_pose.p[:2] - rand_pos.p[:2]) < float(self.PAD_MIN_DISTANCE):
                continue
            break

        half_size = [0.03, 0.03, 0.0005]
        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }
        color_items = list(colors.items())
        color_index = int(np.random.choice(len(color_items)))
        self.color_name, self.color_value = color_items[color_index]

        self.pad = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.stapler, padding=0.1)
        self.add_prohibit_area(self.pad, padding=0.15)
        self.pad_pose = self.pad.get_pose().p.tolist() + [0.707, 0, 0, 0.707]
        self._configure_rotate_subtask_plan()

    def play_once(self):
        stapler_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        stapler_cyl = world_to_robot(self.stapler.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if stapler_cyl[1] >= 0 else "right")
        self.enter_rotate_action_stage(1, focus_object_key=(stapler_key or "A"))
        self.move(self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["A"])

        pad_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        self.enter_rotate_action_stage(2, focus_object_key=(pad_key or "B"))
        self.move(
            self.place_actor(
                self.stapler,
                target_pose=self.pad_pose,
                arm_tag=arm_tag,
                pre_dis=float(self.STAPLER_PLACE_PRE_DIS),
                dis=float(self.STAPLER_PLACE_DIS),
                pre_dis_axis=self.STAPLER_PLACE_PRE_DIS_AXIS,
                constrain=self.STAPLER_PLACE_CONSTRAIN,  # Success check requires orientation consistency on the pad.
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "stapler",
            "{B}": self.color_name,
            "{a}": str(arm_tag),
        }
        return self.info
    def check_success(self):
        stapler_pose = self.stapler.get_pose().p
        target_pos = self.pad.get_pose().p
        eps = [0.02, 0.02, 0.01]
        return (np.all(abs(stapler_pose - target_pos) < np.array(eps))) and (self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
