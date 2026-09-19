from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *
import numpy as np


class beat_block_hammer_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_R = 0.64

    def check_success(self):
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        return np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and self.check_actors_contact(
            self.hammer.get_name(), self.block.get_name())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.hammer,
                "B": self.block,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_hammer",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "hammer_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "hammer_block",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "block_hit",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def _sample_block_pose(self):
        for _ in range(100):
            pose = rand_pose_cyl(
                rlim=[0.45, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.76, 0.76],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.5],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if cyl[0] < 0.45 or abs(cyl[1]) < 0.12:
                continue
            if np.sum(np.square(pose.p[:2])) < 0.04:
                continue
            return pose
        return rand_pose_cyl(
            rlim=[0.5, 0.5],
            thetalim=rotate_theta_fixed(self, side=-1),

            zlim=[0.76, 0.76],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        hammer_pos = place_point_cyl(
            [0.44, 0.0, 0.8],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        # Keep hammer orientation in world frame (same as original task).
        # `place_pose_cyl` interprets quaternion in cylindrical local frame.
        hammer_pose = sapien.Pose(hammer_pos, [0, 0, 0.995, 0.105])
        self.hammer = create_actor(
            scene=self,
            pose=hammer_pose,
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )

        block_pose = self._sample_block_pose()
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.01)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])
        self._configure_rotate_subtask_plan()

    def play_once(self):
        hammer_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        block_pose = self.block.get_functional_point(0, "pose").p.tolist()
        block_cyl = world_to_robot(block_pose, self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if block_cyl[1] >= 0 else "right")

        self.enter_rotate_action_stage(1, focus_object_key=(hammer_key or "A"))
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["A"])

        block_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.64,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.block.get_functional_point(1, "pose")
        self.enter_rotate_action_stage(2, focus_object_key=(block_key or "B"))
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
                constrain="free",
            )
        )
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {"{A}": "hammer", "{a}": str(arm_tag)}
        return self.info
