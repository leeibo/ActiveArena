import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from copy import deepcopy

from ._base_task import Base_Task
from .utils import *


class search_object_place_pad(Base_Task):
    """Find a red block either on the table or in a cabinet, then place it on a pad."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    TASK_HOMESTATE = [
        [-0.11, -0.7, -0.8, 2.0, -0.9, 0.0, 0.0],
        [0.11, -0.7, 0.8, 2.0, 0.9, 0.0, 0.0],
    ]

    CABINET_MODEL_ID = 46653
    CABINET_R = 0.70
    CABINET_THETA_DEG_RANGE = (10.0, 24.0)
    CABINET_THETA_SIGN_CHOICES = (1.0,)
    CABINET_Z = 0.741
    CABINET_PRE_GRASP_DIS = 0.05
    CABINET_GRASP_DIS = 0.01
    CABINET_GRIPPER_POS = -0.02
    DRAWER_PULL_DIS = 0.20
    DRAWER_PULL_STEPS = 2
    DRAWER_OPEN_SUCCESS_DIS = 0.01
    CABINET_POST_PULL_BACKOFF_DIS = 0.03
    CABINET_ROTATE_LIM_ABS = (0.25, 1.0)
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.18
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"

    BLOCK_HALF_SIZE = 0.022
    BLOCK_COLOR = (0.90, 0.10, 0.10)
    BLOCK_MASS = 0.03
    TABLE_BLOCK_R = 0.44
    TABLE_BLOCK_THETA = -0.38
    DRAWER_BLOCK_OUTWARD_OFFSET = 0.07

    PAD_R = 0.36
    PAD_THETA = 0.0
    PAD_HALF_SIZE = (0.055, 0.045, 0.004)
    PAD_COLOR = (0.10, 0.35, 0.95)
    PAD_SUCCESS_XY = (0.065, 0.055)

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    OBJECT_PRE_GRASP_DIS = 0.11
    OBJECT_GRASP_DIS = 0.01
    OBJECT_APPROACH_CLEARANCE_Z = 0.12
    OBJECT_LIFT_Z = 0.10
    DRAWER_OBJECT_ESCAPE_DIS = 0.07
    PLACE_PRE_DIS = 0.08
    PLACE_DIS = 0.01

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        choices = kwargs.pop("cabinet_theta_sign_choices", self.CABINET_THETA_SIGN_CHOICES)
        if np.isscalar(choices):
            choices = [float(choices)]
        self.cabinet_theta_sign_choices = tuple(float(v) for v in choices if abs(float(v)) > 1e-9) or (1.0,)
        self.force_block_location = kwargs.pop("search_object_block_location", None)
        for cfg_key in ["left_embodiment_config", "right_embodiment_config"]:
            if cfg_key in kwargs and kwargs[cfg_key] is not None:
                cfg = deepcopy(kwargs[cfg_key])
                cfg["homestate"] = deepcopy(self.TASK_HOMESTATE)
                kwargs[cfg_key] = cfg
        super()._init_task_env_(**kwargs)

    @staticmethod
    def _quat_from_yaw(yaw):
        return t3d.euler.euler2quat(0.0, 0.0, float(yaw))

    def _pose_cyl(self, r, theta, z, q=(1, 0, 0, 0), quat_frame="cyl"):
        return rand_pose_cyl(
            [float(r), float(r)],
            [float(theta), float(theta)],
            [float(z), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=list(q),
            quat_frame=quat_frame,
        )

    def _cabinet_pose(self):
        theta_deg = float(np.random.uniform(*self.CABINET_THETA_DEG_RANGE))
        theta_deg *= float(np.random.choice(self.cabinet_theta_sign_choices))
        self.cabinet_theta_deg = theta_deg
        theta = np.deg2rad(theta_deg)
        x = float(self.robot_root_xy[0] + self.CABINET_R * np.cos(self.robot_yaw + theta))
        y = float(self.robot_root_xy[1] + self.CABINET_R * np.sin(self.robot_yaw + theta))
        yaw = float(np.arctan2(y - self.robot_root_xy[1], x - self.robot_root_xy[0]))
        return sapien.Pose([x, y, float(self.CABINET_Z)], self._quat_from_yaw(yaw))

    def _make_block(self, pose):
        block = create_box(
            scene=self,
            pose=pose,
            half_size=(self.BLOCK_HALF_SIZE,) * 3,
            color=self.BLOCK_COLOR,
            name="red_search_block",
        )
        block.set_mass(float(self.BLOCK_MASS))
        return block

    def _drawer_outward_dir(self):
        vec = np.array(self.robot_root_xy, dtype=np.float64) - np.array(self.cabinet.get_pose().p[:2], dtype=np.float64)
        return vec / max(float(np.linalg.norm(vec)), 1e-9)

    def _drawer_point(self):
        return np.array(self.cabinet.get_functional_point(0)[:3], dtype=np.float64)

    def _drawer_block_pose(self):
        drawer_pose = self.cabinet.get_functional_point(0, "pose")
        p = np.array(drawer_pose.p, dtype=np.float64)
        p[:2] += self._drawer_outward_dir() * float(self.DRAWER_BLOCK_OUTWARD_OFFSET)
        p[2] += float(self.BLOCK_HALF_SIZE + 0.002 - self.table_z_bias)
        return sapien.Pose(p, np.array(drawer_pose.q, dtype=np.float64))

    def _choose_block_on_table(self):
        if self.force_block_location in {"table", "drawer"}:
            return self.force_block_location == "table"
        return bool(np.random.rand() < 0.5)

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {"A": "lower", "B": "lower", "P": "lower"}
        self.object_labels = {"A": "red block", "B": "cabinet", "P": "blue pad"}
        self.cabinet_opened = False
        self.cabinet_arm_tag = None
        self.object_arm_tag = None
        self.initial_drawer_point = None
        self.block_was_on_table = self._choose_block_on_table()

        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=self._cabinet_pose(),
            modelname="036_cabinet",
            modelid=self.CABINET_MODEL_ID,
            fix_root_link=True,
        )
        self.pad = create_box(
            scene=self,
            pose=self._pose_cyl(self.PAD_R, self.PAD_THETA, 0.741, quat_frame="world"),
            half_size=self.PAD_HALF_SIZE,
            color=self.PAD_COLOR,
            name="search_object_pad",
            is_static=True,
        )
        block_pose = (
            self._pose_cyl(self.TABLE_BLOCK_R, self.TABLE_BLOCK_THETA, 0.74 + self.BLOCK_HALF_SIZE + 0.002)
            if self.block_was_on_table
            else self._drawer_block_pose()
        )
        self.object = self._make_block(block_pose)
        self.initial_object_z = float(self.object.get_pose().p[2])
        self.initial_drawer_point = self._drawer_point()

        self.add_prohibit_area(self.cabinet, padding=0.03)
        self.add_prohibit_area(self.pad, padding=0.08)
        self.add_prohibit_area(self.object, padding=0.05)
        self._configure_rotate_subtask_plan()

    def _configure_rotate_subtask_plan(self):
        subtasks = [
            dict(id=1, name="search_table_for_red_block", instruction_idx=1,
                 search_target_keys=["A"], action_target_keys=[], required_carried_keys=[],
                 carry_keys_after_done=[], allow_stage2_from_memory=False,
                 done_when="red_block_found_or_missing_on_table", next_subtask_id=2),
        ]
        if not self.block_was_on_table:
            subtasks += [
                dict(id=2, name="open_cabinet", instruction_idx=2,
                     search_target_keys=["B"], action_target_keys=["B"], required_carried_keys=[],
                     carry_keys_after_done=[], allow_stage2_from_memory=True,
                     done_when="cabinet_opened", next_subtask_id=3),
                dict(id=3, name="pick_red_block_from_cabinet", instruction_idx=3,
                     search_target_keys=["A"], action_target_keys=["A"], required_carried_keys=[],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=False,
                     done_when="red_block_grasped", next_subtask_id=4),
            ]
            place_id = 4
        else:
            subtasks += [
                dict(id=2, name="pick_red_block_from_table", instruction_idx=3,
                     search_target_keys=["A"], action_target_keys=["A"], required_carried_keys=[],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=True,
                     done_when="red_block_grasped", next_subtask_id=3),
            ]
            place_id = 3
        subtasks.append(
            dict(id=place_id, name="place_red_block_on_pad", instruction_idx=4,
                 search_target_keys=["P"], action_target_keys=["A", "P"], required_carried_keys=["A"],
                 carry_keys_after_done=[], allow_stage2_from_memory=True,
                 done_when="red_block_on_pad", next_subtask_id=-1)
        )
        self.place_subtask_id = place_id
        self.configure_rotate_subtask_plan(
            object_registry={"A": self.object, "B": self.cabinet, "P": self.pad},
            subtask_defs=subtasks,
            task_instruction="Find the red block; if it is not on the table, open the cabinet, then place it on the blue pad.",
        )

    def _project_rotate_registry_object(self, object_key, camera_pose=None, camera_spec=None):
        if str(object_key) == "A" and (not self.block_was_on_table) and (not self.cabinet_opened):
            return None
        return super()._project_rotate_registry_object(object_key, camera_pose=camera_pose, camera_spec=camera_spec)

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    @staticmethod
    def _mirror_lim(arm_tag, lim):
        lo, hi = sorted(abs(float(v)) for v in lim)
        return (lo, hi) if ArmTag(arm_tag) == "left" else (-hi, -lo)

    def _with_arm_lim(self, arm_tag, lim, fn):
        attr = "left_rotate_lim" if ArmTag(arm_tag) == "left" else "right_rotate_lim"
        old = list(getattr(self.robot, attr))
        try:
            setattr(self.robot, attr, list(lim))
            return fn()
        finally:
            setattr(self.robot, attr, old)

    def _open_cabinet(self, subtask_id):
        self.cabinet_arm_tag = ArmTag("left" if world_to_robot(self.cabinet.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)[1] >= 0 else "right")
        self.initial_drawer_point = self._drawer_point()
        self.enter_rotate_action_stage(subtask_id, focus_object_key="B")
        actions = self._with_arm_lim(
            self.cabinet_arm_tag,
            self._mirror_lim(self.cabinet_arm_tag, self.CABINET_ROTATE_LIM_ABS),
            lambda: self.grasp_actor(self.cabinet, arm_tag=self.cabinet_arm_tag,
                                     pre_grasp_dis=self.CABINET_PRE_GRASP_DIS,
                                     grasp_dis=self.CABINET_GRASP_DIS,
                                     gripper_pos=self.CABINET_GRIPPER_POS),
        )
        if not self.move(actions):
            return False
        pull_step = self._drawer_outward_dir() * (float(self.DRAWER_PULL_DIS) / float(max(int(self.DRAWER_PULL_STEPS), 1)))
        for _ in range(max(int(self.DRAWER_PULL_STEPS), 1)):
            if not self.move(self.move_by_displacement(self.cabinet_arm_tag, x=float(pull_step[0]), y=float(pull_step[1]))):
                return False
        self.cabinet_opened = bool(np.linalg.norm(self._drawer_point()[:2] - self.initial_drawer_point[:2]) > self.DRAWER_OPEN_SUCCESS_DIS)
        if not self.cabinet_opened:
            return False
        back = self._drawer_outward_dir() * float(self.CABINET_POST_PULL_BACKOFF_DIS)
        self.move(self.move_by_displacement(self.cabinet_arm_tag, x=float(back[0]), y=float(back[1])))
        self.move(self.open_gripper(self.cabinet_arm_tag))
        self.object_arm_tag = self.cabinet_arm_tag.opposite
        self.complete_rotate_subtask(subtask_id, carried_after=[])
        return bool(self.plan_success)

    def _pick_block(self, subtask_id):
        if self.object_arm_tag is None:
            theta = world_to_robot(self.object.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)[1]
            self.object_arm_tag = ArmTag("left" if theta >= 0 else "right")
        self.enter_rotate_action_stage(subtask_id, focus_object_key="A")
        if not self.move(self.grasp_actor(self.object, arm_tag=self.object_arm_tag,
                                          pre_grasp_dis=self.OBJECT_PRE_GRASP_DIS,
                                          grasp_dis=self.OBJECT_GRASP_DIS)):
            return False
        self._set_carried_object_keys(["A"])
        if not self.move(self.move_by_displacement(self.object_arm_tag, z=self.OBJECT_LIFT_Z)):
            return False
        self.complete_rotate_subtask(subtask_id, carried_after=["A"])
        return True

    def _place_on_pad(self):
        arm = self.object_arm_tag or ArmTag("left")
        self.enter_rotate_action_stage(self.place_subtask_id, focus_object_key="P")
        if not self.block_was_on_table:
            escape = self._drawer_outward_dir() * float(self.DRAWER_OBJECT_ESCAPE_DIS)
            if not self.move(self.move_by_displacement(arm, x=float(escape[0]), y=float(escape[1]))):
                return False
        pad = np.array(self.pad.get_pose().p, dtype=np.float64)
        ee_pose = np.array(self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose(), dtype=np.float64)
        block_pos = np.array(self.object.get_pose().p, dtype=np.float64)
        ee_to_block = block_pos - ee_pose[:3]
        target_center = pad.copy()
        target_center[2] = float(pad[2] + self.PAD_HALF_SIZE[2] + self.BLOCK_HALF_SIZE + 0.04)
        release_pose = ee_pose.copy()
        release_pose[:3] = target_center - ee_to_block
        if not self.move(self.move_to_pose(arm, release_pose)):
            return False
        if not self.move(self.open_gripper(arm)):
            return False
        self._set_carried_object_keys([])
        self.delay(10)
        self.complete_rotate_subtask(self.place_subtask_id, carried_after=[])
        return True

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        if self.block_was_on_table:
            key = self.search_and_focus_rotate_subtask(1, scan_r=self.SCAN_R, scan_z=scan_z, joint_name_prefer=self.SCAN_JOINT_NAME)
            if key is None:
                self.plan_success = False
            else:
                self.complete_rotate_subtask(1, carried_after=[])
                if not self._pick_block(2):
                    self.plan_success = False
        else:
            self.begin_rotate_subtask(1)
            self._set_rotate_subtask_state(
                subtask_idx=1,
                stage=1,
                focus_object_key=None,
                search_target_keys=["A"],
                action_target_keys=[],
                info_complete=0,
                camera_mode=1,
                camera_target_theta=float(self.TABLE_BLOCK_THETA),
            )
            self.delay(4, save_freq=1)
            self.complete_rotate_subtask(1, carried_after=[])
            self._reset_head_to_home_pose(save_freq=None)
            if not self._open_cabinet(2):
                self.plan_success = False
            else:
                obj_key = self.search_and_focus_rotate_subtask(3, scan_r=self.SCAN_R, scan_z=scan_z, joint_name_prefer=self.SCAN_JOINT_NAME)
                if obj_key is None or not self._pick_block(3):
                    self.plan_success = False
        if self.plan_success and not self._place_on_pad():
            self.plan_success = False
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        return {
            "{A}": "red block",
            "{B}": "cabinet",
            "{C}": "blue pad",
            "{a}": str(self.object_arm_tag or "left/right"),
            "{b}": str(self.cabinet_arm_tag or "left"),
            "{s}": "table" if self.block_was_on_table else "drawer",
        }

    def check_success(self):
        block = np.array(self.object.get_pose().p, dtype=np.float64)
        pad = np.array(self.pad.get_pose().p, dtype=np.float64)
        xy_ok = np.all(np.abs(block[:2] - pad[:2]) <= np.array(self.PAD_SUCCESS_XY, dtype=np.float64))
        z_ok = abs(float(block[2]) - float(pad[2] + self.PAD_HALF_SIZE[2] + self.BLOCK_HALF_SIZE)) < 0.025
        return bool(xy_ok and z_ok and self.is_left_gripper_open() and self.is_right_gripper_open())
