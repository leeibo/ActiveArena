from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
import transforms3d as t3d
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC


class click_alarmclock_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    def _resolve_alarm_press_pose(self, arm_tag: ArmTag, pre_dis: float = 0.1):
        contact_point = self.alarm.get_contact_point(0, "list")
        if contact_point is None:
            return None

        top_down_key = "top_down_little_right" if str(arm_tag) == "left" else "top_down_little_left"
        quat_candidates = [
            GRASP_DIRECTION_DIC[top_down_key],
            GRASP_DIRECTION_DIC["top_down"],
            [0.5, -0.5, 0.5, 0.5],
        ]

        for quat in quat_candidates:
            pose = np.array(list(contact_point[:3]) + list(quat), dtype=np.float64)
            direction_mat = t3d.quaternions.quat2mat(np.array(quat, dtype=np.float64))
            pose[:3] += [pre_dis, 0, 0] @ np.linalg.inv(direction_mat)
            planned_pose = self.choose_best_pose(pose.tolist(), contact_point, arm_tag)
            if planned_pose is not None and len(planned_pose) == 7 and float(planned_pose[0]) != -1:
                return planned_pose
        fallback_pose = np.array(list(self.alarm.get_pose().p) + list(quat_candidates[0]), dtype=np.float64)
        fallback_pose[2] += 0.13
        return fallback_pose.tolist()

    def check_success(self):
        if self.stage_success_tag:
            return True
        if not self.check_arm_function():
            return False
        alarm_pose = self.alarm.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("046_alarm-clock")
        eps = [0.03, 0.03]
        for position in positions:
            if (np.all(np.abs(position[:2] - alarm_pose[:2]) < eps) and abs(position[2] - alarm_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.alarm,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "click_alarmclock_top_button",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "alarmclock_clicked",
                    "next_subtask_id": -1,
                }
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        while True:
            rand_pos = rand_pose_cyl(
                rlim=[0.44, 0.5],
                thetalim=rotate_theta_center(self),

                zlim=[0.741, 0.741],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            cyl = world_to_robot(rand_pos.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) < 0.2:
                continue
            break

        self.alarmclock_id = int(np.random.choice([1, 3], 1)[0])
        self.alarm = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="046_alarm-clock",
            convex=True,
            model_id=self.alarmclock_id,
            is_static=True,
        )
        self.add_prohibit_area(self.alarm, padding=0.05)
        self.check_arm_function = (
            self.is_left_gripper_close
            if self.alarm.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )
        self._configure_rotate_subtask_plan()

    def play_once(self):
        alarm_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        alarm_cyl = world_to_robot(self.alarm.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        arm_tag = ArmTag("left" if alarm_cyl[1] >= 0 else "right")

        self.enter_rotate_action_stage(1, focus_object_key=(alarm_key or "A"))
        grasp_pose = self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)
        if grasp_pose is None:
            press_pose = self._resolve_alarm_press_pose(arm_tag, pre_dis=0.1)
        else:
            press_pose = grasp_pose[:3] + [0.5, -0.5, 0.5, 0.5]
        if press_pose is None:
            self.plan_success = False
            self.info["info"] = {
                "{A}": "alarm clock",
                "{a}": str(arm_tag),
            }
            return self.info
        self.move(
            (
                ArmTag(arm_tag),
                [
                    Action(arm_tag, "move", press_pose),
                    Action(arm_tag, "close", target_gripper_pos=-0.1),
                ],
            )
        )

        self.move(self.move_by_displacement(arm_tag, z=-0.065))
        self.check_success()
        self.move(self.move_by_displacement(arm_tag, z=0.065))
        self.check_success()
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {
            "{A}": "alarm clock",
            "{a}": str(arm_tag),
        }
        return self.info
