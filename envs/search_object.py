from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *


class search_object(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    CABINET_MODEL_ID = 46653
    OBJECT_LABEL = "small block"
    OBJECT_HALF_SIZE = 0.018
    OBJECT_COLOR = (0.10, 0.80, 0.20)
    OBJECT_COLOR_CANDIDATES = (
        (0.90, 0.20, 0.20),
        (0.15, 0.72, 0.25),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
        (0.88, 0.45, 0.16),
    )
    OBJECT_MASS = 0.03
    OBJECT_VARIANTS = (
        {
            "kind": "asset",
            "modelname": "073_rubikscube",
            "label": "rubik's cube",
            "base_q": (0.7071068, 0.7071068, 0.0, 0.0),
            "outward_offset": 0.02,
            "surface_z_offset": 0.0,
            "mass": OBJECT_MASS,
        },
    )

    CABINET_CYL_R = 0.7
    CABINET_CYL_THETA_DEG_RANGE = (8.0, 24.0)
    CABINET_THETA_SIGN_CHOICES = (1.0)
    CABINET_CYL_Z = 0.741
    CABINET_CYL_SPIN_DEG = 0.0

    TASK_HOMESTATE = [
        [-0.11, -0.7, -0.8, 2, -0.9, 0, 0],
        [0.11, -0.7, 0.8, 2, 0.9, 0, 0],
    ]

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    RIGHT_ARM_ROTATE_LIM = (-1.0, 0.0)
    CABINET_ROTATE_LIM_ABS_NEW = (0.25, 1.0)
    OBJECT_ROTATE_LIM_ABS_NEW = (0.0, 1.0)
    CABINET_POST_PULL_BACKOFF_DIS = 0.03
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.18
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"

    OBJECT_Z_BIAS = OBJECT_HALF_SIZE + 0.002
    OBJECT_OUTER_EDGE_OFFSET = 0.01
    # Keep cabinet-opening behavior local to this task instead of inheriting open_cabinet.
    DRAWER_OPEN_SUCCESS_DIS = 0.08
    DRAWER_PULL_TOTAL_DIS = 0.20
    DRAWER_PULL_STEPS = 2
    CABINET_PRE_GRASP_DIS = 0.05
    CABINET_GRASP_DIS = 0.01
    CABINET_GRIPPER_POS = -0.02
    OBJECT_PRE_GRASP_DIS_OLD = 0.09
    OBJECT_PRE_GRASP_DIS_NEW = 0.12
    OBJECT_GRASP_DIS = 0.01
    OBJECT_APPROACH_CLEARANCE_Z = 0.12
    OBJECT_APPROACH_PRE_GRASP_MARGIN_Z = 0.08
    OBJECT_LIFT_Z = 0.08
    SUCCESS_LIFT_Z = 0.03

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        # Legacy compatibility only. Final behavior is selected by cabinet side at runtime.
        kwargs.pop("right_arm_rotate_lim", self.RIGHT_ARM_ROTATE_LIM)
        cabinet_theta_sign_choices = kwargs.pop(
            "cabinet_theta_sign_choices",
            self.CABINET_THETA_SIGN_CHOICES,
        )
        if np.isscalar(cabinet_theta_sign_choices):
            cabinet_theta_sign_choices = [float(cabinet_theta_sign_choices)]
        else:
            cabinet_theta_sign_choices = [float(v) for v in cabinet_theta_sign_choices]
        cabinet_theta_sign_choices = [v for v in cabinet_theta_sign_choices if abs(v) > 1e-9]
        self.cabinet_theta_sign_choices = tuple(cabinet_theta_sign_choices or self.CABINET_THETA_SIGN_CHOICES)

        for cfg_key in ["left_embodiment_config", "right_embodiment_config"]:
            if cfg_key in kwargs and kwargs[cfg_key] is not None:
                cfg = deepcopy(kwargs[cfg_key])
                cfg["homestate"] = deepcopy(self.TASK_HOMESTATE)
                kwargs[cfg_key] = cfg

        super()._init_task_env_(**kwargs)

    @staticmethod
    def _quat_from_yaw(yaw_rad):
        return t3d.euler.euler2quat(0.0, 0.0, float(yaw_rad))

    def _world_xy_from_cyl(self, r, theta_deg):
        theta_rad = float(np.deg2rad(theta_deg))
        phi_world = float(self.robot_yaw + theta_rad)
        x = float(self.robot_root_xy[0] + float(r) * np.cos(phi_world))
        y = float(self.robot_root_xy[1] + float(r) * np.sin(phi_world))
        return x, y

    def _cabinet_pose_from_cyl(self, r, theta_deg, z, spin_deg):
        x, y = self._world_xy_from_cyl(r=r, theta_deg=theta_deg)
        radial_out_yaw = float(np.arctan2(y - self.robot_root_xy[1], x - self.robot_root_xy[0]))
        yaw = float(radial_out_yaw + np.deg2rad(spin_deg))
        return sapien.Pose([x, y, float(z)], self._quat_from_yaw(yaw))

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.cabinet,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "search_hidden_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_not_found",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "open_seen_cabinet",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "cabinet_opened",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_object_from_cabinet",
                    "instruction_idx": 3,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_grasped_and_lifted",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Search for the object; if it is not visible, open the cabinet and pick the object up.",
        )

    @staticmethod
    def _get_available_model_ids(modelname):
        model_dir = Path("assets/objects") / str(modelname)
        available_ids = []
        for json_path in model_dir.glob("model_data*.json"):
            suffix = json_path.stem.replace("model_data", "")
            if suffix.isdigit():
                available_ids.append(int(suffix))
        return sorted(available_ids)

    def _sample_block_color(self):
        color_idx = int(np.random.randint(len(self.OBJECT_COLOR_CANDIDATES)))
        return tuple(float(channel) for channel in self.OBJECT_COLOR_CANDIDATES[color_idx])

    def _compose_object_quat(self, drawer_pose, base_q=None):
        drawer_q = np.array(drawer_pose.q, dtype=np.float64)
        if base_q is None:
            return drawer_q
        return np.array(t3d.quaternions.qmult(drawer_q, np.array(base_q, dtype=np.float64)), dtype=np.float64)

    def _build_drawer_object_pose(self, drawer_pose, drawer_outward_dir, outward_offset, surface_z_offset, quat):
        return sapien.Pose(
            np.array(drawer_pose.p, dtype=np.float64)
            + np.array(
                [
                    float(drawer_outward_dir[0]) * float(outward_offset),
                    float(drawer_outward_dir[1]) * float(outward_offset),
                    float(surface_z_offset - self.table_z_bias),
                ],
                dtype=np.float64,
            ),
            np.array(quat, dtype=np.float64),
        )

    def _create_search_target_object(self, drawer_pose, drawer_outward_dir):
        variant = dict(self.OBJECT_VARIANTS[int(np.random.randint(len(self.OBJECT_VARIANTS)))])
        if variant["kind"] == "block":
            block_pose = self._build_drawer_object_pose(
                drawer_pose=drawer_pose,
                drawer_outward_dir=drawer_outward_dir,
                outward_offset=float(variant.get("outward_offset", self.OBJECT_OUTER_EDGE_OFFSET)),
                surface_z_offset=float(variant.get("surface_z_offset", self.OBJECT_Z_BIAS)),
                quat=self._compose_object_quat(drawer_pose),
            )
            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE, self.OBJECT_HALF_SIZE),
                color=self._sample_block_color(),
                name="search_object_block",
            )
            block.set_mass(float(variant.get("mass", self.OBJECT_MASS)))
            self.selected_modelname = None
            self.selected_model_id = None
            return block, str(variant.get("label", self.OBJECT_LABEL))

        modelname = str(variant["modelname"])
        available_model_ids = self._get_available_model_ids(modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {modelname}")
        model_id = int(np.random.choice(available_model_ids))
        object_pose = self._build_drawer_object_pose(
            drawer_pose=drawer_pose,
            drawer_outward_dir=drawer_outward_dir,
            outward_offset=float(variant.get("outward_offset", self.OBJECT_OUTER_EDGE_OFFSET)),
            surface_z_offset=float(variant.get("surface_z_offset", 0.0)),
            quat=self._compose_object_quat(drawer_pose, base_q=variant.get("base_q", None)),
        )
        obj = create_actor(
            scene=self,
            pose=object_pose,
            modelname=modelname,
            convex=True,
            model_id=model_id,
        )
        obj.set_mass(float(variant.get("mass", self.OBJECT_MASS)))
        self.selected_modelname = modelname
        self.selected_model_id = model_id
        return obj, str(variant.get("label", modelname))

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.cabinet_opened = False
        self.object_arm_tag = None
        self.cabinet_arm_tag = None
        self.initial_object_z = None
        self.object_label = str(self.OBJECT_LABEL)
        self.selected_modelname = None
        self.selected_model_id = None

        cabinet_theta_abs_deg = float(np.random.uniform(*self.CABINET_CYL_THETA_DEG_RANGE))
        theta_sign_choices = getattr(self, "cabinet_theta_sign_choices", self.CABINET_THETA_SIGN_CHOICES)
        self.cabinet_theta_deg = float(cabinet_theta_abs_deg * float(np.random.choice(theta_sign_choices)))
        cabinet_pose = self._cabinet_pose_from_cyl(
            r=self.CABINET_CYL_R,
            theta_deg=self.cabinet_theta_deg,
            z=self.CABINET_CYL_Z,
            spin_deg=self.CABINET_CYL_SPIN_DEG,
        )
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            pose=cabinet_pose,
            modelname="036_cabinet",
            modelid=self.CABINET_MODEL_ID,
            fix_root_link=True,
        )

        drawer_pose = self.cabinet.get_functional_point(0, "pose")
        drawer_outward_dir = self._get_drawer_outward_dir_xy()
        self.object, self.object_label = self._create_search_target_object(
            drawer_pose=drawer_pose,
            drawer_outward_dir=drawer_outward_dir,
        )
        self.initial_drawer_world_point = self._get_drawer_world_point()

        self.add_prohibit_area(self.cabinet, padding=0.03)
        self.add_prohibit_area(self.object, padding=0.03)
        self._configure_rotate_subtask_plan()

    def _project_rotate_registry_object(self, object_key, camera_pose=None, camera_spec=None):
        if str(object_key) == "A" and not bool(getattr(self, "cabinet_opened", False)):
            return None
        return super()._project_rotate_registry_object(
            object_key,
            camera_pose=camera_pose,
            camera_spec=camera_spec,
        )

    def _get_cabinet_arm_tag(self):
        cabinet_cyl = world_to_robot(self.cabinet.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if float(cabinet_cyl[1]) >= 0.0 else "right")

    def _get_drawer_outward_dir_xy(self):
        cabinet_xy = np.array(self.cabinet.get_pose().p[:2], dtype=np.float64)
        robot_xy = np.array(self.robot_root_xy, dtype=np.float64)
        direction = robot_xy - cabinet_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, -1.0], dtype=np.float64)
        return direction / norm

    def _get_drawer_world_point(self):
        return np.array(self.cabinet.get_functional_point(0)[:3], dtype=np.float64)

    def _get_drawer_pull_step_xy(self):
        direction = self._get_drawer_outward_dir_xy()
        step_dis = float(self.DRAWER_PULL_TOTAL_DIS) / float(max(int(self.DRAWER_PULL_STEPS), 1))
        return (direction * step_dis).tolist()

    def _retreat_cabinet_arm_after_open(self):
        if self.cabinet_arm_tag is None:
            return True

        backoff_dis = float(getattr(self, "CABINET_POST_PULL_BACKOFF_DIS", 0.0))
        if backoff_dis > 1e-9:
            backoff_xy = np.array(self._get_drawer_outward_dir_xy(), dtype=np.float64) * backoff_dis
            if not self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(backoff_xy[0]),
                    y=float(backoff_xy[1]),
                    move_axis="world",
                )
            ):
                return False

        # Release only after clearing the handle so the fingers do not push the drawer back.
        if not self.move(self.open_gripper(self.cabinet_arm_tag)):
            return False

        lateral_xy = self._get_upper_place_lateral_escape_xy(self.cabinet_arm_tag)
        if lateral_xy is None:
            return True
        if abs(float(lateral_xy[0])) <= 1e-9 and abs(float(lateral_xy[1])) <= 1e-9:
            return True
        return bool(
            self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(lateral_xy[0]),
                    y=float(lateral_xy[1]),
                    move_axis="world",
                )
            )
        )

    @staticmethod
    def _build_mirrored_rotate_lim(arm_tag, positive_rotate_lim):
        min_abs, max_abs = sorted(abs(float(v)) for v in positive_rotate_lim)
        if ArmTag(arm_tag) == "left":
            return (min_abs, max_abs)
        return (-max_abs, -min_abs)

    def _use_old_core(self):
        cabinet_arm_tag = self.cabinet_arm_tag
        if cabinet_arm_tag is None:
            cabinet_arm_tag = self._get_cabinet_arm_tag()
        return ArmTag(cabinet_arm_tag) == "left"

    def _get_cabinet_arm_rotate_lim(self, arm_tag):
        return self._build_mirrored_rotate_lim(arm_tag, self.CABINET_ROTATE_LIM_ABS_NEW)

    def _get_object_arm_rotate_lim(self, arm_tag):
        return self._build_mirrored_rotate_lim(arm_tag, self.OBJECT_ROTATE_LIM_ABS_NEW)

    def _run_with_arm_rotate_lim(self, arm_tag, rotate_lim, action_fn):
        rotate_attr = "left_rotate_lim" if ArmTag(arm_tag) == "left" else "right_rotate_lim"
        original_rotate_lim = list(getattr(self.robot, rotate_attr))
        try:
            setattr(self.robot, rotate_attr, list(rotate_lim))
            return action_fn()
        finally:
            setattr(self.robot, rotate_attr, original_rotate_lim)

    def _grasp_cabinet_with_new_core(self):
        cabinet_actions = self._run_with_arm_rotate_lim(
            self.cabinet_arm_tag,
            self._get_cabinet_arm_rotate_lim(self.cabinet_arm_tag),
            lambda: self.grasp_actor(
                self.cabinet,
                arm_tag=self.cabinet_arm_tag,
                pre_grasp_dis=self.CABINET_PRE_GRASP_DIS,
                grasp_dis=self.CABINET_GRASP_DIS,
                gripper_pos=self.CABINET_GRIPPER_POS,
            ),
        )
        self.move(cabinet_actions)
        return bool(self.plan_success)

    def _grasp_cabinet_for_current_side(self):
        # Cabinet contact posture stays symmetric across arms.
        return self._grasp_cabinet_with_new_core()

    def _choose_object_grasp_pose_old_core(self):
        return self.choose_grasp_pose(
            self.object,
            arm_tag=self.object_arm_tag,
            pre_dis=self.OBJECT_PRE_GRASP_DIS_OLD,
            target_dis=self.OBJECT_GRASP_DIS,
        )

    def _choose_object_grasp_pose_new_core(self):
        return self._run_with_arm_rotate_lim(
            self.object_arm_tag,
            self._get_object_arm_rotate_lim(self.object_arm_tag),
            lambda: self.choose_grasp_pose(
                self.object,
                arm_tag=self.object_arm_tag,
                pre_dis=self.OBJECT_PRE_GRASP_DIS_NEW,
                target_dis=self.OBJECT_GRASP_DIS,
            ),
        )

    def _choose_object_grasp_pose(self):
        if self._use_old_core():
            return self._choose_object_grasp_pose_old_core()
        return self._choose_object_grasp_pose_new_core()

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]
        if arm_tag == "left":
            plan_multi_pose = self.robot.left_plan_multi_path
        elif arm_tag == "right":
            plan_multi_pose = self.robot.right_plan_multi_path
        else:
            return None

        target_lst = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        traj_lst = plan_multi_pose(target_lst)
        best_pose = None
        best_step = None
        for i, pose in enumerate(target_lst):
            if traj_lst["status"][i] != "Success":
                continue
            step_count = len(traj_lst["position"][i])
            if best_step is None or step_count < best_step:
                best_pose = pose
                best_step = step_count
        return best_pose

    def _is_cabinet_drawer_opened(self):
        if self.initial_drawer_world_point is None:
            return False
        current_drawer_world_point = self._get_drawer_world_point()
        open_dis = float(
            np.linalg.norm(
                current_drawer_world_point[:2] - np.array(self.initial_drawer_world_point[:2], dtype=np.float64)
            )
        )
        return open_dis > float(self.DRAWER_OPEN_SUCCESS_DIS)

    def _get_arm_ee_pose(self, arm_tag):
        if arm_tag == "left":
            return np.array(self.robot.get_left_ee_pose(), dtype=np.float64)
        if arm_tag == "right":
            return np.array(self.robot.get_right_ee_pose(), dtype=np.float64)
        raise ValueError(f'arm_tag must be either "left" or "right", not {arm_tag}')

    def _build_object_grasp_transition_waypoints(self, arm_tag, pre_grasp_pose):
        current_pose = self._get_arm_ee_pose(arm_tag)
        pre_grasp_pose = np.array(pre_grasp_pose, dtype=np.float64)
        cabinet_ref_z = max(
            float(self.cabinet.get_pose().p[2]),
            float(self.object.get_pose().p[2]),
            float(self._get_drawer_world_point()[2]),
        )
        safe_z = max(
            float(current_pose[2]),
            float(pre_grasp_pose[2] + self.OBJECT_APPROACH_PRE_GRASP_MARGIN_Z),
            float(cabinet_ref_z + self.OBJECT_APPROACH_CLEARANCE_Z),
        )

        lift_pose = np.array(current_pose, dtype=np.float64)
        lift_pose[2] = safe_z

        front_pose = np.array(pre_grasp_pose, dtype=np.float64)
        front_pose[2] = safe_z
        return lift_pose.tolist(), front_pose.tolist()

    def _open_cabinet_drawer(self, cabinet_key):
        self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        self.object_arm_tag = self.cabinet_arm_tag.opposite
        self.initial_drawer_world_point = self._get_drawer_world_point()
        self.enter_rotate_action_stage(2, focus_object_key=(cabinet_key or "B"))
        if not self._grasp_cabinet_for_current_side():
            return False

        step_xy = self._get_drawer_pull_step_xy()
        pull_step_limit = max(int(self.DRAWER_PULL_STEPS), 1)
        if not bool(getattr(self, "need_plan", True)):
            # During replay the cabinet arm is only used for drawer pulling after grasping,
            # so the remaining cached plans on that arm are the exact pull budget.
            pull_step_limit = min(
                pull_step_limit,
                int(self._get_remaining_joint_path_count(self.cabinet_arm_tag)),
            )

        executed_pull_steps = 0
        for _ in range(pull_step_limit):
            self.move(
                self.move_by_displacement(
                    arm_tag=self.cabinet_arm_tag,
                    x=float(step_xy[0]),
                    y=float(step_xy[1]),
                )
            )
            executed_pull_steps += 1
            if not self.plan_success:
                break

        drawer_opened = self._is_cabinet_drawer_opened()
        replay_pull_budget_consumed = bool(
            (not bool(getattr(self, "need_plan", True)))
            and pull_step_limit > 0
            and executed_pull_steps >= pull_step_limit
        )
        self.cabinet_opened = bool(self.plan_success and (drawer_opened or replay_pull_budget_consumed))
        if not self.cabinet_opened:
            return False
        return self._retreat_cabinet_arm_after_open()

    def _grasp_and_lift_object(self, object_key):
        if self.object_arm_tag is None:
            self.object_arm_tag = self._get_cabinet_arm_tag().opposite
        self.enter_rotate_action_stage(3, focus_object_key=(object_key or "A"))
        self.initial_object_z = float(self.object.get_pose().p[2])
        pre_grasp_pose, grasp_pose = self._choose_object_grasp_pose()
        if pre_grasp_pose is None or grasp_pose is None:
            self.plan_success = False
            return False

        lift_pose, front_pose = self._build_object_grasp_transition_waypoints(
            self.object_arm_tag,
            pre_grasp_pose,
        )
        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=lift_pose))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm_tag=self.object_arm_tag, target_pose=front_pose))
        if not self.plan_success:
            return False

        self.move(
            (
                self.object_arm_tag,
                [
                    Action(self.object_arm_tag, "move", target_pose=pre_grasp_pose),
                    Action(
                        self.object_arm_tag,
                        "move",
                        target_pose=grasp_pose,
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(self.object_arm_tag, "close", target_gripper_pos=0.0),
                ],
            )
        )
        if not self.plan_success:
            return False
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=self.object_arm_tag, z=self.OBJECT_LIFT_Z))
        if not self.plan_success:
            return False
        self.delay(2)
        return True

    def _build_info(self):
        if self.cabinet_arm_tag is None:
            self.cabinet_arm_tag = self._get_cabinet_arm_tag()
        if self.object_arm_tag is None:
            self.object_arm_tag = self.cabinet_arm_tag.opposite
        return {
            "{A}": str(getattr(self, "object_label", self.OBJECT_LABEL)),
            "{B}": "cabinet",
            "{a}": str(self.object_arm_tag),
            "{b}": str(self.cabinet_arm_tag),
        }

    def play_once(self):
        scan_z = float(self.SCAN_Z_BIAS + self.table_z_bias)
        self._reset_head_to_home_pose(save_freq=None)

        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is not None:
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(1, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        cabinet_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if cabinet_key is None or not self._open_cabinet_drawer(cabinet_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(2, carried_after=[])

        self._reset_head_to_home_pose(save_freq=None)
        object_key = self.search_and_focus_rotate_subtask(
            3,
            scan_r=self.SCAN_R,
            scan_z=scan_z,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is None or not self._grasp_and_lift_object(object_key):
            self.plan_success = False
            self.info["info"] = self._build_info()
            return self.info
        self.complete_rotate_subtask(3, carried_after=["A"])

        self.info["info"] = self._build_info()
        return self.info

    def check_success(self):
        if self.initial_object_z is None or self.object_arm_tag is None:
            return False
        object_z = float(self.object.get_pose().p[2])
        gripper_close = (
            self.is_left_gripper_close()
            if self.object_arm_tag == "left"
            else self.is_right_gripper_close()
        )
        return bool(object_z > self.initial_object_z + self.SUCCESS_LIFT_Z and gripper_close)
