from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import glob
import numpy as np
import os


class place_object_scale_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def check_success(self):
        object_pose = self.object.get_pose().p
        scale_pose = self.scale.get_functional_point(0)
        distance_threshold = 0.035
        distance = np.linalg.norm(np.array(scale_pose[:2]) - np.array(object_pose[:2]))
        check_arm = (self.is_left_gripper_open if self.arm_tag == "left" else self.is_right_gripper_open)
        return (distance < distance_threshold and object_pose[2] > (scale_pose[2] - 0.01) and check_arm())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.scale,
                "B": self.object,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_object_on_scale",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_on_scale",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def _get_available_model_ids(self, modelname):
        asset_path = os.path.join("assets/objects", modelname)
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

        available_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
                available_ids.append(idx)
            except ValueError:
                continue
        return available_ids

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        object_list = ["047_mouse", "048_stapler", "050_bell"]
        self.selected_modelname = str(np.random.choice(object_list))
        available_model_ids = self._get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = int(np.random.choice(available_model_ids))

        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_obj_lim = rotate_theta_side(self, side=side)
        theta_scale_lim = rotate_theta_side(self, side=-side)

        while True:
            object_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=theta_obj_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
            )
            obj_cyl = world_to_robot(object_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(obj_cyl[1]) < 0.25:
                continue
            break
        self.arm_tag = ArmTag("right" if object_pose.p[0] > 0 else "left")

        while True:
            scale_pose = rand_pose_cyl(
                rlim=[0.4, 0.5],
                thetalim=theta_scale_lim,

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
            )
            if np.linalg.norm(scale_pose.p[:2] - object_pose.p[:2]) < 0.16:
                continue
            scale_cyl = world_to_robot(scale_pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(scale_cyl[1]) < 0.2:
                continue
            break

        self.object = create_actor(
            scene=self,
            pose=object_pose,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.1)
        # check_success() is evaluated from the first policy action, before
        # play_once() has a chance to select an arm during data collection.
        self.arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        self.scale_id = int(np.random.choice([0, 1, 5, 6], 1)[0])
        self.scale = create_actor(
            scene=self,
            pose=scale_pose,
            modelname="072_electronicscale",
            model_id=self.scale_id,
            convex=True,
            is_static=True,
        )
        self.scale.set_mass(0.1)

        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.scale, padding=0.05)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        self.enter_rotate_action_stage(1, focus_object_key=(object_key or "B"))
        self.move(self.grasp_actor(self.object, arm_tag=self.arm_tag,gripper_pos=0.1))
        self._set_carried_object_keys(["B"])
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.15))
        self.complete_rotate_subtask(1, carried_after=["B"])

        scale_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        place_pose = self.scale.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(scale_key or "A"))
        self.move(
            self.place_actor(
                self.object,
                arm_tag=self.arm_tag,
                target_pose=place_pose,
                constrain="free",
                pre_dis=0.05,
                dis=0.005,
            )
        )
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "electronic scale",
            "{B}": self._natural_model_label(self.selected_modelname),
            "{a}": str(self.arm_tag),
        }
        return self.info
