from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class place_empty_cup_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def check_success(self):
        # eps = [0.03, 0.03, 0.015]
        eps = 0.035
        cup_pose = self.cup.get_functional_point(0, "pose").p
        coaster_pose = self.coaster.get_functional_point(0, "pose").p
        return (
            # np.all(np.abs(cup_pose - coaster_pose) < eps)
            np.sum(pow(cup_pose[:2] - coaster_pose[:2], 2)) < eps**2 and abs(cup_pose[2] - coaster_pose[2]) < 0.015
            and self.is_left_gripper_open() and self.is_right_gripper_open())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.cup,
                "B": self.coaster,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_cup",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "cup_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_cup_on_coaster",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "cup_on_coaster",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        tag = int(np.random.randint(0, 2))
        cup_thetas = [
            rotate_theta_side(self, side=-1),
            rotate_theta_side(self, side=1),
        ]
        coaster_thetas = [
            rotate_theta_center(self),
            rotate_theta_center(self),
        ]
        self.cup = create_actor(
            self,
            pose=rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=cup_thetas[tag],

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            ),
            modelname="021_cup",
            convex=True,
            model_id=0,
        )
        cup_pose = self.cup.get_pose().p

        while True:
            coaster_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=coaster_thetas[tag],

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            if np.sum(np.square(cup_pose[:2] - coaster_pose.p[:2])) < 0.01:
                continue
            break
        self.coaster = create_actor(
            self,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            model_id=0,
            is_static=True,
        )
        self.coaster.set_mass(0.1)
        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area(self.coaster, padding=0.05)
        self.delay(2)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        cup_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        cup_pose = self.cup.get_pose().p
        arm_tag = ArmTag("right" if cup_pose[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(cup_key or "A"))
        # self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag,
                pre_grasp_dis=0.1,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                gripper_pos=-0.1,
                grasp_dis=-0.02,
            )
        )
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["A"])

        coaster_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.coaster.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(coaster_key or "B"))
        self.move(
            self.place_actor(
                self.cup,
                arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.05,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag, z=0.05, move_axis="arm"))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {"{A}": "empty cup", "{B}": "coaster"}
        return self.info
