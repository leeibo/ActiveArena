from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import transforms3d as t3d


class shake_bottle_horizontally_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_R = 0.60
    ROTATE_SCAN_SCENE_Z_BIAS = 0.90
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.90, -0.90)

    def check_success(self):
        bottle_pose = self.bottle.get_pose().p
        return bottle_pose[2] > 0.8 + self.table_z_bias

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.bottle,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "_shake_bottle_horizontally",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "bottle_shaken_horizontally",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, is_test=False, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(is_test=is_test, **kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        self.id_list = [i for i in range(20)]
        side = 1.0 if np.random.rand() < 0.5 else -1.0
        theta_lim = rotate_theta_side(self, side=side)
        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.3, 0.45],
                thetalim=theta_lim,

                zlim=[0.785, 0.785],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0, 0, 1, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 4],
            )
            bottle_cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(bottle_cyl[1]) < 0.35:
                continue
            break

        self.bottle_id = int(np.random.choice(self.id_list))
        self.bottle = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="001_bottle",
            convex=True,
            model_id=self.bottle_id,
        )
        self.bottle.set_mass(0.05)
        self.add_prohibit_area(self.bottle, padding=0.05)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        bottle_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.6,
            scan_z=0.9 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")
        self.enter_rotate_action_stage(1, focus_object_key=(bottle_key or "A"))
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0.2))
        self._set_carried_object_keys(["A"])

        target_quat = [0.707, 0, 0, 0.707]
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, quat=target_quat))

        y_rotation = t3d.euler.euler2quat(0, (np.pi / 2), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, target_quat)
        target_quat = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))

        quat1 = deepcopy(target_quat)
        quat2 = deepcopy(target_quat)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat1)
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat2)
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, quat=quat1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.03, quat=quat2))
        for _ in range(2):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, quat=quat1))
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.05, quat=quat2))

        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=target_quat))
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {
            "{A}": "bottle",
            "{a}": str(arm_tag),
        }
        return self.info
