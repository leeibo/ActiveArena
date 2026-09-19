import sapien.core as sapien
import hashlib
import numpy as np
import pdb
from .planner import MplibPlanner
import numpy as np
import toppra as ta
import math
import yaml
import os
import tempfile
import xml.etree.ElementTree as ET
import transforms3d as t3d
from copy import deepcopy
from collections import OrderedDict
import sapien.core as sapien
import envs._GLOBAL_CONFIGS as CONFIGS
from envs.utils import transforms
from .planner import CuroboPlanner, create_rgb_axis_marker
import torch.multiprocessing as mp

class Robot:

    def __init__(self, scene, need_topp=False, **kwargs):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self._init_robot_(scene, need_topp, **kwargs)

    @staticmethod
    def _parse_joint_drive_overrides(raw_cfg, cfg_name):
        """
        Parse per-joint drive overrides from config.
        Expected format:
            cfg_name:
              joint_name_a: <number>
              joint_name_b: <number>
        """
        if raw_cfg is None:
            return {}
        if not isinstance(raw_cfg, dict):
            print(f"[Robot.drive] ignore '{cfg_name}': expected mapping, got {type(raw_cfg).__name__}")
            return {}

        overrides = {}
        for joint_name, val in raw_cfg.items():
            if isinstance(val, (int, float, np.floating)):
                overrides[str(joint_name)] = float(val)
            else:
                print(
                    f"[Robot.drive] ignore '{cfg_name}.{joint_name}': "
                    f"expected number, got {type(val).__name__}"
                )
        return overrides

    def _get_joint_drive_property(self, arm_tag, joint_name):
        if arm_tag == "left":
            stiffness = self.left_joint_stiffness_per_joint.get(joint_name, self.left_joint_stiffness)
            damping = self.left_joint_damping_per_joint.get(joint_name, self.left_joint_damping)
            return stiffness, damping
        stiffness = self.right_joint_stiffness_per_joint.get(joint_name, self.right_joint_stiffness)
        damping = self.right_joint_damping_per_joint.get(joint_name, self.right_joint_damping)
        return stiffness, damping

    def _init_robot_(self, scene, need_topp=False, **kwargs):
        # self.dual_arm = dual_arm_tag
        # self.plan_success = True

        self.scene = scene
        self.viewer = kwargs.get("viewer", None)
        self.verbose_robot_init_log = bool(kwargs.get("verbose_robot_init_log", False))
        self.verbose_planner_log = bool(kwargs.get("verbose_planner_log", False))
        self._target_markers = {}
        self._left_live_frame_marker = None
        self._right_live_frame_marker = None
        self._left_base_frame_marker = None
        self._reference_frame_marker = None
        self.left_gripper_links = []
        self._left_gripper_pair = None
        self._left_live_prev_rot = None
        self.left_base_link = None
        self.head_camera = None
        self.communication_flag = False
        self.left_planner = None
        self.right_planner = None
        self.left_mplib_planner = None
        self.right_mplib_planner = None
        self.left_conn = None
        self.right_conn = None
        self.left_proc = None
        self.right_proc = None

        self.left_js = None
        self.right_js = None
        self.head_joints = []
        self.head_joints_name = []
        self.head_homestate = [0.0, 0.0]
        self.head_entity = None
        self.torso_joints = []
        self.torso_joints_name = []
        self.torso_homestate = [0.0]
        self.torso_entity = None

        left_embodiment_args = kwargs["left_embodiment_config"]
        right_embodiment_args = kwargs["right_embodiment_config"]
        left_robot_file = kwargs["left_robot_file"]
        right_robot_file = kwargs["right_robot_file"]

        self.need_topp = bool(need_topp)

        self.left_urdf_path = os.path.abspath(
            os.path.join(left_robot_file, left_embodiment_args["urdf_path"])
        )
        self.left_srdf_path = left_embodiment_args.get("srdf_path", None)
        self.left_curobo_yml_path = os.path.join(left_robot_file, "curobo.yml")
        if self.left_srdf_path is not None:
            self.left_srdf_path = os.path.join(left_robot_file, self.left_srdf_path)
        self.left_joint_stiffness = left_embodiment_args.get("joint_stiffness", 1000)
        self.left_joint_damping = left_embodiment_args.get("joint_damping", 200)
        self.left_disable_gravity = bool(left_embodiment_args.get("disable_gravity", False))
        self.left_joint_stiffness_per_joint = self._parse_joint_drive_overrides(
            left_embodiment_args.get("joint_stiffness_per_joint"), "joint_stiffness_per_joint"
        )
        self.left_joint_damping_per_joint = self._parse_joint_drive_overrides(
            left_embodiment_args.get("joint_damping_per_joint"), "joint_damping_per_joint"
        )
        self.left_gripper_stiffness = left_embodiment_args.get("gripper_stiffness", 1000)
        self.left_gripper_damping = left_embodiment_args.get("gripper_damping", 200)
        self.left_gripper_force_limit = left_embodiment_args.get("gripper_force_limit", None)
        self.left_gripper_drive_velocity_scale = float(
            left_embodiment_args.get("gripper_drive_velocity_scale", 0.2)
        )
        self.left_planner_type = left_embodiment_args.get("planner", "mplib_RRT")
        self.left_move_group = left_embodiment_args["move_group"][0]
        self.left_ee_name = left_embodiment_args["ee_joints"][0]
        self.left_arm_joints_name = left_embodiment_args["arm_joints_name"][0]
        self.left_gripper_name = left_embodiment_args["gripper_name"][0]
        self.left_gripper_bias = left_embodiment_args["gripper_bias"]
        self.left_gripper_scale = left_embodiment_args["gripper_scale"]
        self.left_gripper_homestate = self._parse_gripper_homestate(left_embodiment_args, arm_idx=0)
        self.left_homestate = left_embodiment_args.get("homestate", [[0] * len(self.left_arm_joints_name)])[0]
        self.left_fix_gripper_name = left_embodiment_args.get("fix_gripper_name", [])
        self.left_delta_matrix = np.array(left_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.left_global_trans_matrix = np.array(
            left_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = left_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.left_entity_origion_pose = deepcopy(_entity_origion_pose)

        self.right_urdf_path = os.path.abspath(
            os.path.join(right_robot_file, right_embodiment_args["urdf_path"])
        )
        self.right_srdf_path = right_embodiment_args.get("srdf_path", None)
        if self.right_srdf_path is not None:
            self.right_srdf_path = os.path.join(right_robot_file, self.right_srdf_path)
        self.right_curobo_yml_path = os.path.join(right_robot_file, "curobo.yml")
        self.right_joint_stiffness = right_embodiment_args.get("joint_stiffness", 1000)
        self.right_joint_damping = right_embodiment_args.get("joint_damping", 200)
        self.right_disable_gravity = bool(right_embodiment_args.get("disable_gravity", False))
        self.right_joint_stiffness_per_joint = self._parse_joint_drive_overrides(
            right_embodiment_args.get("joint_stiffness_per_joint"), "joint_stiffness_per_joint"
        )
        self.right_joint_damping_per_joint = self._parse_joint_drive_overrides(
            right_embodiment_args.get("joint_damping_per_joint"), "joint_damping_per_joint"
        )
        self.right_gripper_stiffness = right_embodiment_args.get("gripper_stiffness", 1000)
        self.right_gripper_damping = right_embodiment_args.get("gripper_damping", 200)
        self.right_gripper_force_limit = right_embodiment_args.get("gripper_force_limit", None)
        self.right_gripper_drive_velocity_scale = float(
            right_embodiment_args.get("gripper_drive_velocity_scale", 0.2)
        )
        self.right_planner_type = right_embodiment_args.get("planner", "mplib_RRT")
        self.right_move_group = right_embodiment_args["move_group"][1]
        self.right_ee_name = right_embodiment_args["ee_joints"][1]
        self.right_arm_joints_name = right_embodiment_args["arm_joints_name"][1]
        self.right_gripper_name = right_embodiment_args["gripper_name"][1]
        self.right_gripper_bias = right_embodiment_args["gripper_bias"]
        self.right_gripper_scale = right_embodiment_args["gripper_scale"]
        self.right_gripper_homestate = self._parse_gripper_homestate(right_embodiment_args, arm_idx=1)
        self.right_homestate = right_embodiment_args.get("homestate", [[1] * len(self.right_arm_joints_name)])[1]
        self.right_fix_gripper_name = right_embodiment_args.get("fix_gripper_name", [])
        self.right_delta_matrix = np.array(right_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.right_inv_delta_matrix = np.linalg.inv(self.right_delta_matrix)
        self.right_global_trans_matrix = np.array(
            right_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = right_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])
        _entity_origion_pose = _entity_origion_pose[0 if len(_entity_origion_pose) == 1 else 1]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.right_entity_origion_pose = deepcopy(_entity_origion_pose)
        self.is_dual_arm = kwargs["dual_arm_embodied"]

        left_sapien_urdf = self._get_absolute_resource_urdf_for_sapien(self.left_urdf_path)
        right_sapien_urdf = self._get_absolute_resource_urdf_for_sapien(self.right_urdf_path)

        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

        self.left_perfect_direction = left_embodiment_args.get("grasp_perfect_direction",
                                                               ["front_right", "front_left"])[0]
        self.right_perfect_direction = right_embodiment_args.get("grasp_perfect_direction",
                                                                 ["front_right", "front_left"])[1]
        self.head_joints_name = left_embodiment_args.get(
            "head_joints_name",
            ["astribot_head_joint_1", "astribot_head_joint_2"],
        )
        if isinstance(self.head_joints_name, str):
            self.head_joints_name = [self.head_joints_name]
        self.head_homestate = self._parse_head_homestate(left_embodiment_args, len(self.head_joints_name))
        self.head_motion_max_vel = float(left_embodiment_args.get("head_motion_max_vel", 1.2))
        self.head_motion_acc = float(left_embodiment_args.get("head_motion_acc", 2.5))
        self.head_collision_filter_mode = str(left_embodiment_args.get("head_collision_filter_mode", "keep")).lower()
        self.torso_joints_name = left_embodiment_args.get("torso_joints_name", [])
        if isinstance(self.torso_joints_name, str):
            self.torso_joints_name = [self.torso_joints_name]
        self.torso_homestate = self._parse_torso_homestate(left_embodiment_args, len(self.torso_joints_name))
        self.torso_motion_max_vel = float(left_embodiment_args.get("torso_motion_max_vel", self.head_motion_max_vel))
        self.torso_motion_acc = float(left_embodiment_args.get("torso_motion_acc", self.head_motion_acc))
        # Torso facing can stop within a yaw range instead of strict center alignment.
        self.torso_face_deadband_rad = max(float(left_embodiment_args.get("torso_face_deadband_rad", 0.1)), 0.0)
        self.torso_face_world_deadband_rad = max(
            float(left_embodiment_args.get("torso_face_world_deadband_rad", self.torso_face_deadband_rad)), 0.0
        )
        self.torso_face_object_deadband_rad = max(
            float(left_embodiment_args.get("torso_face_object_deadband_rad", self.torso_face_deadband_rad)), 0.0
        )
        self.torso_face_hysteresis_rad = max(float(left_embodiment_args.get("torso_face_hysteresis_rad", 0.02)), 0.0)
        # Scan range shrink x: scan [theta_min + x, theta_max - x].
        self.scan_theta_inward_margin_rad = max(
            float(
                left_embodiment_args.get(
                    "scan_theta_inward_margin_rad",
                    left_embodiment_args.get("scan_theta_margin_rad", 0.0),
                )
            ),
            0.0,
        )

        if self.is_dual_arm:
            loader: sapien.URDFLoader = scene.create_urdf_loader()
            loader.fix_root_link = True
            self._entity = loader.load(left_sapien_urdf)
            self.left_entity = self._entity
            self.right_entity = self._entity
        else:
            arms_dis = kwargs["embodiment_dis"]
            self.left_entity_origion_pose.p += [-arms_dis / 2, 0, 0]
            self.right_entity_origion_pose.p += [arms_dis / 2, 0, 0]
            left_loader: sapien.URDFLoader = scene.create_urdf_loader()
            left_loader.fix_root_link = True
            right_loader: sapien.URDFLoader = scene.create_urdf_loader()
            right_loader.fix_root_link = True
            self.left_entity = left_loader.load(left_sapien_urdf)
            self.right_entity = right_loader.load(right_sapien_urdf)

        self.left_entity.set_root_pose(self.left_entity_origion_pose)
        self.right_entity.set_root_pose(self.right_entity_origion_pose)
        self._apply_robot_gravity_flags()
        if self.verbose_robot_init_log:
            print(
                f"[Robot.drive] left stiffness/damping=({self.left_joint_stiffness}, {self.left_joint_damping}), "
                f"right stiffness/damping=({self.right_joint_stiffness}, {self.right_joint_damping})"
            )
            print(
                f"[Robot.gripper_drive] left stiffness/damping/force_limit="
                f"({self.left_gripper_stiffness}, {self.left_gripper_damping}, {self.left_gripper_force_limit}), "
                f"right stiffness/damping/force_limit="
                f"({self.right_gripper_stiffness}, {self.right_gripper_damping}, {self.right_gripper_force_limit})"
            )
            print(
                f"[Robot.drive] left per-joint overrides: "
                f"stiffness={len(self.left_joint_stiffness_per_joint)}, "
                f"damping={len(self.left_joint_damping_per_joint)}; "
                f"right per-joint overrides: "
                f"stiffness={len(self.right_joint_stiffness_per_joint)}, "
                f"damping={len(self.right_joint_damping_per_joint)}"
            )
        self._print_left_right_symmetry_debug()

    @staticmethod
    def _get_absolute_resource_urdf_for_sapien(urdf_path: str) -> str:
        """Return a cached URDF whose relative resource paths are absolute.

        SAPIEN can intermittently resolve a relative mesh path against the
        process working directory after many scene reloads.  Keeping the
        generated file in /tmp and making every resource path absolute avoids
        that process-global resolver state while leaving the source asset
        untouched.
        """
        abs_urdf = os.path.abspath(urdf_path)
        stat = os.stat(abs_urdf)
        cache_key = hashlib.sha256(
            f"{abs_urdf}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
        ).hexdigest()[:20]
        cache_dir = os.path.join(tempfile.gettempdir(), "robotwin_sapien_urdf")
        cache_path = os.path.join(cache_dir, f"{cache_key}.urdf")

        if os.path.isfile(cache_path):
            try:
                ET.parse(cache_path)
                return cache_path
            except ET.ParseError:
                pass

        tree = ET.parse(abs_urdf)
        urdf_dir = os.path.dirname(abs_urdf)
        for element in tree.getroot().iter():
            for attribute in ("filename", "url"):
                resource_path = element.get(attribute)
                if not resource_path or os.path.isabs(resource_path):
                    continue
                if "://" in resource_path or resource_path.startswith("package://"):
                    continue
                absolute_resource = os.path.abspath(os.path.join(urdf_dir, resource_path))
                if not os.path.exists(absolute_resource):
                    raise FileNotFoundError(
                        f"URDF resource does not exist: {resource_path} "
                        f"(resolved from {abs_urdf})"
                    )
                element.set(attribute, absolute_resource)

        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", dir=cache_dir, prefix=f"{cache_key}.", suffix=".tmp", delete=False
        ) as handle:
            tmp_path = handle.name
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, cache_path)
        return cache_path

    def _set_entity_disable_gravity(self, entity, disable_gravity):
        if entity is None:
            return
        for link in entity.get_links():
            # SAPIEN 3.x articulation link exposes set_disable_gravity.
            if hasattr(link, "set_disable_gravity"):
                link.set_disable_gravity(bool(disable_gravity))

    def _apply_robot_gravity_flags(self):
        if self.is_dual_arm:
            # Dual-arm mode shares one articulation; either side requesting
            # disable-gravity means the whole robot articulation is disabled.
            disable = self.left_disable_gravity or self.right_disable_gravity
            self._set_entity_disable_gravity(self.left_entity, disable)
            if self.verbose_robot_init_log:
                print(f"[Robot.gravity] dual_arm disable_gravity={disable}")
            return

        self._set_entity_disable_gravity(self.left_entity, self.left_disable_gravity)
        self._set_entity_disable_gravity(self.right_entity, self.right_disable_gravity)
        if self.verbose_robot_init_log:
            print(
                f"[Robot.gravity] left disable_gravity={self.left_disable_gravity}, "
                f"right disable_gravity={self.right_disable_gravity}"
            )

    @staticmethod
    def _parse_gripper_homestate(embodiment_args, arm_idx=0):
        """
        Parse optional gripper_homestate from config.
        Supports scalar or list/tuple; default is 1.0 (open in RobotWin command space).
        """
        v = embodiment_args.get("gripper_homestate", 1.0)
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0:
                return 1.0
            if len(v) == 1:
                return float(v[0])
            idx = max(0, min(int(arm_idx), len(v) - 1))
            return float(v[idx])
        return 1.0

    @staticmethod
    def _parse_head_homestate(embodiment_args, joint_num):
        v = embodiment_args.get("head_homestate", [0.0] * max(int(joint_num), 0))
        if isinstance(v, (int, float, np.floating)):
            return [float(v)] * max(int(joint_num), 0)
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=np.float64).reshape(-1).tolist()
            if len(arr) >= joint_num:
                return arr[:joint_num]
            return arr + [0.0] * (joint_num - len(arr))
        return [0.0] * max(int(joint_num), 0)

    @staticmethod
    def _parse_torso_homestate(embodiment_args, joint_num):
        v = embodiment_args.get("torso_homestate", [0.0] * max(int(joint_num), 0))
        if isinstance(v, (int, float, np.floating)):
            return [float(v)] * max(int(joint_num), 0)
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.array(v, dtype=np.float64).reshape(-1).tolist()
            if len(arr) >= joint_num:
                return arr[:joint_num]
            return arr + [0.0] * (joint_num - len(arr))
        return [0.0] * max(int(joint_num), 0)

    def reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)

        if self.communication_flag:
            if hasattr(self, "left_conn") and self.left_conn:
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if hasattr(self, "right_conn") and self.right_conn:
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            if (
                not isinstance(getattr(self, "left_planner", None), CuroboPlanner)
                or not isinstance(getattr(self, "right_planner", None), CuroboPlanner)
            ):
                self.set_planner(scene=scene)

        self.init_joints()

    def get_grasp_perfect_direction(self, arm_tag):
        if arm_tag == "left":
            return self.left_perfect_direction
        elif arm_tag == "right":
            return self.right_perfect_direction

    def create_target_pose_list(self, origin_pose, center_pose, arm_tag=None):
        res_lst = []
        rotate_lim = (self.left_rotate_lim if arm_tag == "left" else self.right_rotate_lim)
        rotate_step = (rotate_lim[1] - rotate_lim[0]) / CONFIGS.ROTATE_NUM
        for i in range(CONFIGS.ROTATE_NUM):
            now_pose = transforms.rotate_along_axis(
                origin_pose,
                center_pose,
                [0, 1, 0],
                rotate_step * i + rotate_lim[0],
                axis_type="target",
                towards=[0, -1, 0],
            )
            res_lst.append(now_pose)
        return res_lst

    def get_constraint_pose(self, ori_vec: list, arm_tag=None):
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        return ori_vec[:3] + (ori_vec[-3:] @ np.linalg.inv(inv_delta_matrix)).tolist()

    @staticmethod
    def _find_arm_base_link(entity, move_group_name: str, arm_tag: str):
        candidates = []
        if isinstance(move_group_name, str):
            if "_link_" in move_group_name:
                prefix = move_group_name.split("_link_")[0]
                candidates.append(f"{prefix}_base_link")
            if move_group_name.endswith("_link"):
                candidates.append(f"{move_group_name[:-5]}_base_link")

        if arm_tag == "left":
            candidates.extend(["left_base_link", "left_arm_base_link"])
        elif arm_tag == "right":
            candidates.extend(["right_base_link", "right_arm_base_link"])
        candidates.extend(["base_link", "robot_base_link"])

        for name in candidates:
            link = entity.find_link_by_name(name)
            if link is not None:
                return link

        for link in entity.get_links():
            lname = link.get_name().lower()
            if arm_tag in lname and "base" in lname and "link" in lname:
                return link
        return None

    def init_joints(self):
        if self.left_entity is None or self.right_entity is None:
            raise ValueError("Robote entity is None")

        self.left_active_joints = self.left_entity.get_active_joints()
        self.right_active_joints = self.right_entity.get_active_joints()

        self.left_ee = self.left_entity.find_joint_by_name(self.left_ee_name)
        self.right_ee = self.right_entity.find_joint_by_name(self.right_ee_name)
        self.left_ee_link = self.left_entity.find_link_by_name(self.left_move_group)
        self.right_ee_link = self.right_entity.find_link_by_name(self.right_move_group)
        if self.left_ee_link is None:
            print(f"[Robot] left ee link '{self.left_move_group}' not found, fallback to left_ee child link")
            self.left_ee_link = self.left_ee.child_link if self.left_ee is not None else None
        if self.right_ee_link is None:
            print(f"[Robot] right ee link '{self.right_move_group}' not found, fallback to right_ee child link")
            self.right_ee_link = self.right_ee.child_link if self.right_ee is not None else None
        self.left_base_link = self._find_arm_base_link(self.left_entity, self.left_move_group, "left")
        if self.left_base_link is None:
            print("[Robot] left base link not found; left base marker will be disabled")

        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0

        self.left_arm_joints = [self.left_entity.find_joint_by_name(i) for i in self.left_arm_joints_name]
        self.right_arm_joints = [self.right_entity.find_joint_by_name(i) for i in self.right_arm_joints_name]
        missing_left_arm = [name for name, joint in zip(self.left_arm_joints_name, self.left_arm_joints) if joint is None]
        missing_right_arm = [name for name, joint in zip(self.right_arm_joints_name, self.right_arm_joints) if joint is None]
        if missing_left_arm or missing_right_arm:
            raise ValueError(
                f"Arm joints missing in URDF. left={missing_left_arm}, right={missing_right_arm}. "
                "Please check arm_joints_name in config.yml and URDF joint names."
            )

        self.head_entity = self.left_entity if self.left_entity is not None else self.right_entity
        if self.head_entity is not None and len(self.head_joints_name) > 0:
            self.head_joints = [self.head_entity.find_joint_by_name(i) for i in self.head_joints_name]
            missing_head = [name for name, joint in zip(self.head_joints_name, self.head_joints) if joint is None]
            if missing_head:
                print(f"[Robot] head joints missing in URDF: {missing_head}")
            self.head_joints = [j for j in self.head_joints if j is not None]
        else:
            self.head_joints = []
        if len(self.head_joints) == 0 and len(self.head_joints_name) > 0:
            print("[Robot] head control disabled (no valid head joints found)")

        self.torso_entity = self.left_entity if self.left_entity is not None else self.right_entity
        if self.torso_entity is not None and len(self.torso_joints_name) > 0:
            self.torso_joints = [self.torso_entity.find_joint_by_name(i) for i in self.torso_joints_name]
            missing_torso = [name for name, joint in zip(self.torso_joints_name, self.torso_joints) if joint is None]
            if self.verbose_robot_init_log and missing_torso:
                print(f"[Robot] torso joints missing in URDF: {missing_torso}")
            self.torso_joints = [j for j in self.torso_joints if j is not None]
        else:
            self.torso_joints = []
        if self.verbose_robot_init_log and len(self.torso_joints) == 0 and len(self.torso_joints_name) > 0:
            print("[Robot] torso control disabled (no valid torso joints found)")

        def get_gripper_joints(find, gripper_name: str, arm_tag: str):
            gripper = []
            missing = []

            base_name = gripper_name["base"]
            base_joint = find(base_name)
            if base_joint is None:
                missing.append(base_name)
            else:
                gripper.append((base_joint, 1.0, 0.0))

            for g in gripper_name["mimic"]:
                mimic_joint = find(g[0])
                if mimic_joint is None:
                    missing.append(g[0])
                    continue
                gripper.append((mimic_joint, g[1], g[2]))

            if missing:
                print(f"[Robot] {arm_tag} gripper joints missing in URDF: {missing}")
            if len(gripper) == 0:
                print(f"[Robot] {arm_tag} gripper disabled (no valid joints found)")
            return gripper

        self.left_gripper = get_gripper_joints(self.left_entity.find_joint_by_name, self.left_gripper_name, "left")
        self.right_gripper = get_gripper_joints(self.right_entity.find_joint_by_name, self.right_gripper_name, "right")
        self.left_gripper_links = [joint_info[0].child_link for joint_info in self.left_gripper if joint_info[0] is not None]
        self.gripper_name = deepcopy(self.left_fix_gripper_name) + deepcopy(self.right_fix_gripper_name)

        for g in self.left_gripper:
            if g[0] is None:
                continue
            self.gripper_name.append(g[0].child_link.get_name())
        for g in self.right_gripper:
            if g[0] is None:
                continue
            self.gripper_name.append(g[0].child_link.get_name())

        # camera link id
        self.left_camera = self.left_entity.find_link_by_name("left_camera")
        if self.left_camera is None:
            self.left_camera = self.left_entity.find_link_by_name("camera")
            if self.left_camera is None:
                print("No left camera link")
                self.left_camera = self.left_entity.get_links()[0]

        self.right_camera = self.right_entity.find_link_by_name("right_camera")
        if self.right_camera is None:
            self.right_camera = self.right_entity.find_link_by_name("camera")
            if self.right_camera is None:
                print("No right camera link")
                self.right_camera = self.right_entity.get_links()[0]

        self.head_camera = self.left_entity.find_link_by_name("camera_head")
        if self.head_camera is None:
            self.head_camera = self.left_entity.find_link_by_name("head_camera")
        if self.head_camera is None:
            print("No head-mounted camera link (camera_head/head_camera)")

        left_gripper_joint_set = {j[0] for j in self.left_gripper if j[0] is not None}
        right_gripper_joint_set = {j[0] for j in self.right_gripper if j[0] is not None}

        for i, joint in enumerate(self.left_active_joints):
            if joint in left_gripper_joint_set:
                continue
            stiffness, damping = self._get_joint_drive_property("left", joint.get_name())
            joint.set_drive_property(
                stiffness=stiffness,
                damping=damping,
            )
        for i, joint in enumerate(self.right_active_joints):
            if joint in right_gripper_joint_set:
                continue
            stiffness, damping = self._get_joint_drive_property("right", joint.get_name())
            joint.set_drive_property(
                stiffness=stiffness,
                damping=damping,
            )

        for joint in self.left_gripper:
            if joint[0] is None:
                continue
            self._set_joint_drive_property(
                joint[0],
                stiffness=self.left_gripper_stiffness,
                damping=self.left_gripper_damping,
                force_limit=self.left_gripper_force_limit,
            )
        for joint in self.right_gripper:
            if joint[0] is None:
                continue
            self._set_joint_drive_property(
                joint[0],
                stiffness=self.right_gripper_stiffness,
                damping=self.right_gripper_damping,
                force_limit=self.right_gripper_force_limit,
            )

        self._configure_head_collision_filter()

    @staticmethod
    def _iter_link_subtree(root_link):
        if root_link is None:
            return []
        ordered = []
        stack = [root_link]
        visited = set()
        while stack:
            link = stack.pop()
            if link is None:
                continue
            link_name = None
            try:
                link_name = link.get_name()
            except Exception:
                link_name = str(id(link))
            if link_name in visited:
                continue
            visited.add(link_name)
            ordered.append(link)
            children = []
            if hasattr(link, "get_children"):
                try:
                    children = list(link.get_children())
                except Exception:
                    children = []
            elif hasattr(link, "children"):
                try:
                    children = list(link.children)
                except Exception:
                    children = []
            stack.extend(children)
        return ordered

    @staticmethod
    def _get_link_collision_shapes(link):
        if link is None:
            return []
        if hasattr(link, "get_collision_shapes"):
            try:
                return list(link.get_collision_shapes())
            except Exception:
                pass
        try:
            return list(link.collision_shapes)
        except Exception:
            return []

    def _collect_head_collision_links(self):
        links = []
        for joint in self.head_joints:
            try:
                links.extend(self._iter_link_subtree(joint.get_child_link()))
            except Exception:
                continue
        if self.head_camera is not None:
            links.extend(self._iter_link_subtree(self.head_camera))

        unique = OrderedDict()
        for link in links:
            try:
                unique[link.get_name()] = link
            except Exception:
                continue
        return list(unique.values())

    def _collect_arm_collision_links(self):
        roots = []
        for joint in self.left_arm_joints + self.right_arm_joints:
            try:
                roots.append(joint.get_child_link())
            except Exception:
                continue
        for joint_info in self.left_gripper + self.right_gripper:
            joint = joint_info[0]
            if joint is None:
                continue
            try:
                roots.append(joint.get_child_link())
            except Exception:
                continue

        unique = OrderedDict()
        for root in roots:
            for link in self._iter_link_subtree(root):
                try:
                    unique[link.get_name()] = link
                except Exception:
                    continue
        return list(unique.values())

    @staticmethod
    def _set_collision_groups_or(link, bitmask):
        if int(bitmask) == 0:
            return
        for shape in Robot._get_link_collision_shapes(link):
            try:
                groups = list(shape.get_collision_groups())
                groups[2] = int(groups[2]) | int(bitmask)
                shape.set_collision_groups(groups)
            except Exception:
                continue

    @staticmethod
    def _disable_link_collisions(link):
        for shape in Robot._get_link_collision_shapes(link):
            try:
                groups = list(shape.get_collision_groups())
                groups[0] = 0
                groups[1] = 0
                shape.set_collision_groups(groups)
            except Exception:
                continue

    def _configure_head_collision_filter(self):
        mode = str(getattr(self, "head_collision_filter_mode", "keep")).lower()
        if mode in {"", "keep", "none", "off", "false"}:
            return

        head_links = self._collect_head_collision_links()
        if len(head_links) == 0:
            return

        if mode in {"disable_head", "disable_all", "disable_head_collisions"}:
            for link in head_links:
                self._disable_link_collisions(link)
            return

        if mode not in {"disable_head_vs_arms", "ignore_arm_links"}:
            print(f"[Robot.collision] unknown head_collision_filter_mode='{mode}', skip")
            return

        head_name_set = {link.get_name() for link in head_links}
        arm_links = [link for link in self._collect_arm_collision_links() if link.get_name() not in head_name_set]
        arm_links = [link for link in arm_links if len(self._get_link_collision_shapes(link)) > 0]
        if len(arm_links) == 0:
            return

        if len(arm_links) > 30:
            print(
                f"[Robot.collision] too many arm collision links ({len(arm_links)}), "
                "fallback to disabling head collisions entirely"
            )
            for link in head_links:
                self._disable_link_collisions(link)
            return

        head_bitmask = 0
        for idx, link in enumerate(sorted(arm_links, key=lambda item: item.get_name())):
            bit = 1 << idx
            head_bitmask |= bit
            self._set_collision_groups_or(link, bit)

        for link in head_links:
            self._set_collision_groups_or(link, head_bitmask)

    @staticmethod
    def _set_joint_drive_property(joint, stiffness, damping, force_limit=None):
        # Some SAPIEN builds support `force_limit` in set_drive_property while others
        # only accept stiffness/damping. Fall back gracefully for compatibility.
        if force_limit is not None:
            try:
                joint.set_drive_property(
                    stiffness=stiffness,
                    damping=damping,
                    force_limit=float(force_limit),
                )
                return
            except TypeError:
                try:
                    joint.set_drive_property(stiffness, damping, float(force_limit))
                    return
                except Exception:
                    pass
            except Exception:
                pass
        joint.set_drive_property(
            stiffness=stiffness,
            damping=damping,
        )

    @staticmethod
    def _clip_joint_target_to_limits(joint, target):
        try:
            limits = joint.get_limits()
            if limits is None or len(limits) == 0:
                return float(target)
            low, high = limits[0]
            return float(np.clip(target, low, high))
        except Exception:
            return float(target)

    @staticmethod
    def _is_wrap_equivalent_joint(joint):
        """
        Whether a revolute-like joint can be shifted by +/-2pi equivalently.
        We only treat joints as wrap-equivalent when limits are absent/non-finite
        or span is clearly larger than 2*pi.
        """
        try:
            limits = joint.get_limits()
            if limits is None or len(limits) == 0:
                return True
            low, high = limits[0]
            if not np.isfinite(low) or not np.isfinite(high):
                return True
            return float(high - low) > (2.0 * np.pi + 1e-3)
        except Exception:
            return False

    def _unwrap_arm_joint_path_shortest(self, path_pos, arm_tag):
        path = np.array(path_pos, dtype=np.float64, copy=True)
        if path.ndim != 2 or path.shape[0] <= 1:
            return path

        arm_joints = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        dof = min(path.shape[1], len(arm_joints))

        for j in range(dof):
            joint = arm_joints[j]
            if not self._is_wrap_equivalent_joint(joint):
                continue
            for i in range(1, path.shape[0]):
                delta = path[i, j] - path[i - 1, j]
                delta_short = (delta + np.pi) % (2.0 * np.pi) - np.pi
                path[i, j] = path[i - 1, j] + delta_short
        return path

    def _path_dt(self):
        dt = 1.0 / 250.0
        try:
            scene_dt = float(self.scene.get_timestep())
            if scene_dt > 0:
                dt = scene_dt
        except Exception:
            pass
        return dt

    def _recompute_path_velocity(self, path_pos):
        path_pos = np.asarray(path_pos, dtype=np.float64)
        vel = np.zeros_like(path_pos, dtype=np.float64)
        if path_pos.ndim != 2 or path_pos.shape[0] <= 1:
            return vel
        dt = self._path_dt()
        vel[:-1] = (path_pos[1:] - path_pos[:-1]) / dt
        vel[-1] = 0.0
        return vel

    def _postprocess_arm_plan_result(self, plan_res, arm_tag):
        if not isinstance(plan_res, dict):
            return plan_res
        if "position" not in plan_res:
            return plan_res

        status = plan_res.get("status", None)
        if isinstance(status, str) and status != "Success":
            return plan_res

        raw_pos = np.asarray(plan_res["position"], dtype=np.float64)
        if raw_pos.ndim != 2 or raw_pos.shape[0] == 0:
            return plan_res

        new_pos = self._unwrap_arm_joint_path_shortest(raw_pos, arm_tag=arm_tag)
        changed = float(np.max(np.abs(new_pos - raw_pos))) if new_pos.shape == raw_pos.shape else 0.0
        if self.verbose_planner_log and changed > 1e-3:
            print(f"[Robot.{arm_tag}_plan_path] unwrap continuous-joint spins, max_delta={changed:.4f} rad")
        plan_res["position"] = new_pos
        plan_res["velocity"] = self._recompute_path_velocity(new_pos)
        return plan_res

    def move_to_homestate(self):
        for i, joint in enumerate(self.left_arm_joints):
            joint.set_drive_target(self.left_homestate[i])

        for i, joint in enumerate(self.right_arm_joints):
            joint.set_drive_target(self.right_homestate[i])

        for i, joint in enumerate(self.head_joints):
            target = self.head_homestate[i] if i < len(self.head_homestate) else 0.0
            joint.set_drive_target(self._clip_joint_target_to_limits(joint, target))
            joint.set_drive_velocity_target(0.0)

        for i, joint in enumerate(self.torso_joints):
            target = self.torso_homestate[i] if i < len(self.torso_homestate) else 0.0
            joint.set_drive_target(self._clip_joint_target_to_limits(joint, target))
            joint.set_drive_velocity_target(0.0)

        self._sync_arm_qpos_to_drive_target()
        self._sync_head_qpos_to_drive_target()
        self._sync_torso_qpos_to_drive_target()

        # Initialize grippers together with arm homestate.
        left_cmd = float(np.clip(self.left_gripper_homestate, 0.0, 1.0))
        right_cmd = float(np.clip(self.right_gripper_homestate, 0.0, 1.0))
        self.left_gripper_val = left_cmd
        self.right_gripper_val = right_cmd

        left_active_target = self._gripper_cmd_to_active_rad(left_cmd, self.left_gripper_scale)
        right_active_target = self._gripper_cmd_to_active_rad(right_cmd, self.right_gripper_scale)
        self._set_gripper_mimic_targets(
            self.left_gripper,
            left_active_target,
            velocity_scale=self.left_gripper_drive_velocity_scale,
        )
        self._set_gripper_mimic_targets(
            self.right_gripper,
            right_active_target,
            velocity_scale=self.right_gripper_drive_velocity_scale,
        )
        self._sync_gripper_qpos_to_drive_target()
        self._zero_entity_qvel()

    def _sync_arm_qpos_to_drive_target(self):
        # Force arm qpos to configured homestate at initialization.
        # This avoids the initial frame drifting away from expected joint angles.
        entity_joint_groups = [
            (self.left_entity, self.left_arm_joints),
            (self.right_entity, self.right_arm_joints),
        ]
        for entity, joints in entity_joint_groups:
            if entity is None:
                continue
            qpos = entity.get_qpos().copy()
            active_joints = entity.get_active_joints()
            changed = False
            for joint in joints:
                if joint is None or joint not in active_joints:
                    continue
                idx = active_joints.index(joint)
                qpos[idx] = float(joint.get_drive_target()[0])
                changed = True
            if changed:
                entity.set_qpos(qpos)

    def _sync_head_qpos_to_drive_target(self):
        if self.head_entity is None or len(self.head_joints) == 0:
            return
        qpos = self.head_entity.get_qpos().copy()
        active_joints = self.head_entity.get_active_joints()
        changed = False
        for joint in self.head_joints:
            if joint is None or joint not in active_joints:
                continue
            idx = active_joints.index(joint)
            qpos[idx] = float(joint.get_drive_target()[0])
            changed = True
        if changed:
            self.head_entity.set_qpos(qpos)

    def _sync_torso_qpos_to_drive_target(self):
        if self.torso_entity is None or len(self.torso_joints) == 0:
            return
        qpos = self.torso_entity.get_qpos().copy()
        active_joints = self.torso_entity.get_active_joints()
        changed = False
        for joint in self.torso_joints:
            if joint is None or joint not in active_joints:
                continue
            idx = active_joints.index(joint)
            qpos[idx] = float(joint.get_drive_target()[0])
            changed = True
        if changed:
            self.torso_entity.set_qpos(qpos)

    def _sync_gripper_qpos_to_drive_target(self):
        # Align gripper qpos to just-set drive targets at initialization time,
        # avoiding random-looking equivalent angles for continuous joints.
        entity_joint_groups = [
            (self.left_entity, self.left_gripper),
            (self.right_entity, self.right_gripper),
        ]
        for entity, joints in entity_joint_groups:
            if entity is None:
                continue
            qpos = entity.get_qpos().copy()
            active_joints = entity.get_active_joints()
            changed = False
            for joint_info in joints:
                joint = joint_info[0]
                if joint is None or joint not in active_joints:
                    continue
                idx = active_joints.index(joint)
                qpos[idx] = float(joint.get_drive_target()[0])
                changed = True
            if changed:
                entity.set_qpos(qpos)

    def _zero_entity_qvel(self):
        # Remove leftover velocity so the initial state is deterministic.
        entities = [self.left_entity]
        if self.right_entity is not self.left_entity:
            entities.append(self.right_entity)
        for entity in entities:
            if entity is None:
                continue
            qvel = entity.get_qvel()
            if qvel is None:
                continue
            entity.set_qvel(np.zeros_like(qvel))

    def set_origin_endpose(self):
        self.left_original_pose = self.get_left_ee_pose()
        self.right_original_pose = self.get_right_ee_pose()

    def _print_left_right_symmetry_debug(self):
        if not self.verbose_robot_init_log:
            return

        def _fmt_vec(v):
            return list(np.round(np.array(v, dtype=np.float64), 6))

        print("[Robot.symmetry_check] ===== left/right config =====")
        print(f"[Robot.symmetry_check] left robot root:  p={_fmt_vec(self.left_entity_origion_pose.p)}, q={_fmt_vec(self.left_entity_origion_pose.q)}")
        print(f"[Robot.symmetry_check] right robot root: p={_fmt_vec(self.right_entity_origion_pose.p)}, q={_fmt_vec(self.right_entity_origion_pose.q)}")
        print(f"[Robot.symmetry_check] left gripper_bias={self.left_gripper_bias}, right gripper_bias={self.right_gripper_bias}, diff={self.left_gripper_bias - self.right_gripper_bias:.6f}")
        print(f"[Robot.symmetry_check] left delta_matrix:\n{np.round(self.left_delta_matrix, 6)}")
        print(f"[Robot.symmetry_check] right delta_matrix:\n{np.round(self.right_delta_matrix, 6)}")
        print(f"[Robot.symmetry_check] left global_trans_matrix:\n{np.round(self.left_global_trans_matrix, 6)}")
        print(f"[Robot.symmetry_check] right global_trans_matrix:\n{np.round(self.right_global_trans_matrix, 6)}")
        print("[Robot.symmetry_check] =================================")

    def print_info(self):
        if not self.verbose_robot_init_log:
            return
        print(
            "active joints: ",
            [joint.get_name() for joint in self.left_active_joints + self.right_active_joints],
        )
        print(
            "all links: ",
            [link.get_name() for link in self.left_entity.get_links() + self.right_entity.get_links()],
        )
        print("left arm joints: ", [joint.get_name() for joint in self.left_arm_joints])
        print("right arm joints: ", [joint.get_name() for joint in self.right_arm_joints])
        print("head joints: ", [joint.get_name() for joint in self.head_joints])
        print("torso joints: ", [joint.get_name() for joint in self.torso_joints])
        print("left gripper: ", [joint[0].get_name() for joint in self.left_gripper if joint[0] is not None])
        print("right gripper: ", [joint[0].get_name() for joint in self.right_gripper if joint[0] is not None])
        print("left ee: ", self.left_ee.get_name())
        print("right ee: ", self.right_ee.get_name())

    def set_planner(self, scene=None):
        abs_left_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.left_curobo_yml_path)
        abs_right_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.right_curobo_yml_path)

        self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path)

        if self.is_dual_arm:
            abs_left_curobo_yml_path = abs_left_curobo_yml_path.replace("curobo.yml", "curobo_left.yml")
            abs_right_curobo_yml_path = abs_right_curobo_yml_path.replace("curobo.yml", "curobo_right.yml")

        if not self.communication_flag:
            self.left_planner = CuroboPlanner(self.left_entity_origion_pose,
                                              self.left_arm_joints_name,
                                              [joint.get_name() for joint in self.left_entity.get_active_joints()],
                                              yml_path=abs_left_curobo_yml_path,
                                              verbose=self.verbose_planner_log)
            self.right_planner = CuroboPlanner(self.right_entity_origion_pose,
                                               self.right_arm_joints_name,
                                               [joint.get_name() for joint in self.right_entity.get_active_joints()],
                                               yml_path=abs_right_curobo_yml_path,
                                               verbose=self.verbose_planner_log)
        else:
            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                "origin_pose": self.left_entity_origion_pose,
                "joints_name": self.left_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                "yml_path": abs_left_curobo_yml_path,
                "verbose": self.verbose_planner_log,
            }

            right_args = {
                "origin_pose": self.right_entity_origion_pose,
                "joints_name": self.right_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                "yml_path": abs_right_curobo_yml_path,
                "verbose": self.verbose_planner_log,
            }

            self.left_proc = mp.Process(target=planner_process_worker, args=(left_child_conn, left_args))
            self.right_proc = mp.Process(target=planner_process_worker, args=(right_child_conn, right_args))

            self.left_proc.daemon = True
            self.right_proc.daemon = True

            self.left_proc.start()
            self.right_proc.start()

        if self.need_topp:
            left_topp_urdf = self._get_ascii_safe_urdf_for_mplib(self.left_urdf_path)
            right_topp_urdf = self._get_ascii_safe_urdf_for_mplib(self.right_urdf_path)
            # Use vanilla MPlib for TOPP to avoid Sapien constrained-planning
            # limitations on continuous revolute joints.
            self.left_mplib_planner = MplibPlanner(
                left_topp_urdf,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                None,
            )
            self.right_mplib_planner = MplibPlanner(
                right_topp_urdf,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                None,
            )

    def _get_ascii_safe_urdf_for_mplib(self, urdf_path: str) -> str:
        """
        MPlib may open URDF with ASCII default encoding when generating SRDF.
        For URDFs with non-ASCII comments, provide an ASCII-safe copy.
        """
        abs_urdf = os.path.abspath(urdf_path)
        try:
            with open(abs_urdf, "r", encoding="ascii") as f:
                f.read()
            return abs_urdf
        except UnicodeDecodeError:
            ascii_abs = abs_urdf + ".mplib_ascii.urdf"
            try:
                if os.path.exists(ascii_abs) and os.path.getsize(ascii_abs) > 0:
                    try:
                        ET.parse(ascii_abs)
                        print(f"[Robot.TOPP] use ASCII URDF for MPlib: {ascii_abs}")
                        return ascii_abs
                    except ET.ParseError:
                        pass

                with open(abs_urdf, "r", encoding="utf-8") as f:
                    txt = f.read()
                ascii_txt = txt.encode("ascii", errors="ignore").decode("ascii")
                ascii_dir = os.path.dirname(ascii_abs)
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="ascii",
                    dir=ascii_dir,
                    prefix=os.path.basename(ascii_abs) + ".",
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    tmp_abs = f.name
                    f.write(ascii_txt)
                os.replace(tmp_abs, ascii_abs)
                print(f"[Robot.TOPP] use ASCII URDF for MPlib: {ascii_abs}")
                return ascii_abs
            except Exception as e:
                if "tmp_abs" in locals() and os.path.exists(tmp_abs):
                    os.remove(tmp_abs)
                print(f"[Robot.TOPP] failed to create ASCII URDF fallback, use original. err={e}")
                return abs_urdf

    def update_world_pcd(self, world_pcd):
        try:
            self.left_planner.update_point_cloud(world_pcd, resolution=0.02)
            self.right_planner.update_point_cloud(world_pcd, resolution=0.02)
        except:
            print("Update world pointcloud wrong!")

    def update_world_cuboids(self, cuboids, curr_qpos=None):
        if curr_qpos is None:
            try:
                curr_qpos = self.left_entity.get_qpos().tolist()
            except Exception:
                curr_qpos = None
        if self.communication_flag:
            self.left_conn.send({"cmd": "update_world_cuboids", "cuboids": cuboids, "qpos": curr_qpos})
            self.right_conn.send({"cmd": "update_world_cuboids", "cuboids": cuboids, "qpos": curr_qpos})
            left_res = self.left_conn.recv()
            right_res = self.right_conn.recv()
            return left_res, right_res
        left_res = self.left_planner.set_world_extra_cuboids(cuboids, curr_joint_pos=curr_qpos)
        right_res = self.right_planner.set_world_extra_cuboids(cuboids, curr_joint_pos=curr_qpos)
        return left_res, right_res

    def _trans_from_gripper_to_endlink(self, target_pose, arm_tag=None):
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        target_pose_arr = np.array(target_pose)
        gripper_pose_pos, gripper_pose_quat = deepcopy(target_pose_arr[0:3]), deepcopy(target_pose_arr[-4:])
        if self.verbose_planner_log:
            print(f"gripper_pose_pos: {gripper_pose_pos}, gripper_pose_quat: {gripper_pose_quat}")
        gripper_pose_mat = t3d.quaternions.quat2mat(gripper_pose_quat)
        gripper_pose_pos += gripper_pose_mat @ np.array([0.12 - gripper_bias, 0, 0]).T
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = t3d.quaternions.mat2quat(gripper_pose_mat)
        return sapien.Pose(gripper_pose_pos, gripper_pose_quat)

    def left_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.left_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_grippers(now_val, target_val)

    def right_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.right_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_grippers(now_val, target_val)

    def left_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="left")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
            })
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="left",
            )

    def right_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="right")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
            })
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="right",
            )

    def _visualize_target(self, target_pose, name="target_marker"):
        if self.scene is None:
            return
        if isinstance(target_pose, list):
            target_pose = sapien.Pose(np.array(target_pose[0:3]), np.array(target_pose[3:7]))
        if name not in self._target_markers:
            self._target_markers[name] = create_rgb_axis_marker(self.scene, name=name)
        self._target_markers[name].set_pose(target_pose)
        if self.viewer is not None:
            # Render marker update without stepping simulation to avoid control-side effects.
            self.scene.update_render()
            self.viewer.render()

    @staticmethod
    def _safe_normalize(vec, eps=1e-9):
        n = np.linalg.norm(vec)
        if n < eps:
            return None
        return vec / n

    def _compute_gripper_opening_direction(self):
        links = [l for l in self.left_gripper_links if l is not None]
        if len(links) < 2:
            return None
        points = [np.array(l.get_pose().p, dtype=np.float64) for l in links]
        if self._left_gripper_pair is None:
            best_pair = None
            best_d = -1.0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    d = np.linalg.norm(points[i] - points[j])
                    if d > best_d:
                        best_d = d
                        best_pair = (i, j)
            if best_pair is None:
                return None
            self._left_gripper_pair = best_pair
        i, j = self._left_gripper_pair
        if i >= len(points) or j >= len(points):
            return None
        return self._safe_normalize(points[i] - points[j])

    def _compute_link_live_frame_pose(self, ee_link):
        if ee_link is None:
            return None
        ee_pose = ee_link.get_pose()
        r_link7 = t3d.quaternions.quat2mat(np.array(ee_pose.q, dtype=np.float64))

        # Axis remap requested by user:
        #   x_marker = -y_link7
        #   y_marker = -x_link7
        #   z_marker = x_marker x y_marker (right-handed)
        x_m = -r_link7[:, 1]
        y_m = -r_link7[:, 0]
        z_m = np.cross(x_m, y_m)
        r_marker = np.column_stack([x_m, y_m, z_m])

        q_marker = t3d.quaternions.mat2quat(r_marker)
        return sapien.Pose(np.array(ee_pose.p, dtype=np.float64), q_marker)

    def _compute_left_live_frame_pose(self):
        return self._compute_link_live_frame_pose(self.left_ee_link)

    def _compute_right_live_frame_pose(self):
        return self._compute_link_live_frame_pose(self.right_ee_link)

    def _compute_left_base_frame_pose(self):
        if self.left_base_link is None:
            return None
        base_pose = self.left_base_link.get_pose()
        r_base = t3d.quaternions.quat2mat(np.array(base_pose.q, dtype=np.float64))

        # User definition for left-base marker:
        #   x_marker = -y_base
        #   y_marker = -x_base
        #   z_marker = x_marker x y_marker (right-handed)
        x_m = -r_base[:, 1]
        y_m = -r_base[:, 0]
        z_m = np.cross(x_m, y_m)
        r_marker = np.column_stack([x_m, y_m, z_m])

        q_marker = t3d.quaternions.mat2quat(r_marker)
        return sapien.Pose(np.array(base_pose.p, dtype=np.float64), q_marker)

    @staticmethod
    def _get_reference_world_rotation():
        # reference frame definition:
        #   x_ref = +y_world
        #   y_ref = -x_world
        #   z_ref = x_ref x y_ref (right-handed)
        x_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_ref = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        z_ref = np.cross(x_ref, y_ref)
        return np.column_stack([x_ref, y_ref, z_ref])

    def _compute_reference_frame_pose(self):
        r_world_ref = self._get_reference_world_rotation()
        q_world_ref = t3d.quaternions.mat2quat(r_world_ref)
        return sapien.Pose(np.array([0.0, 0.0, 1.0], dtype=np.float64), q_world_ref)

    def get_left_live_frame_calibration_data(self):
        live_pose = self._compute_left_live_frame_pose()
        if live_pose is None:
            return None

        ref_pose = self._compute_reference_frame_pose()
        r_world_live = t3d.quaternions.quat2mat(np.array(live_pose.q, dtype=np.float64))
        r_world_ref = self._get_reference_world_rotation()
        r_ref_live = r_world_ref.T @ r_world_live

        return {
            "live_world_p": np.array(live_pose.p, dtype=np.float64),
            "live_world_q": np.array(live_pose.q, dtype=np.float64),
            "reference_world_p": np.array(ref_pose.p, dtype=np.float64),
            "reference_world_q": np.array(ref_pose.q, dtype=np.float64),
            "R_world_live": r_world_live,
            "R_world_ref": r_world_ref,
            "R_ref_live": r_ref_live,
        }

    def update_left_live_frame_marker(self):
        # Only visualize in UI mode to avoid affecting dataset rendering.
        if self.viewer is None or self.scene is None:
            return
        pose = self._compute_left_live_frame_pose()
        if pose is None:
            return
        if self._left_live_frame_marker is None:
            self._left_live_frame_marker = create_rgb_axis_marker(
                self.scene, axis_len=0.12, axis_radius=0.0035, name="left_live_frame_marker"
            )
        self._left_live_frame_marker.set_pose(pose)

    def update_right_live_frame_marker(self):
        # Only visualize in UI mode to avoid affecting dataset rendering.
        if self.viewer is None or self.scene is None:
            return
        pose = self._compute_right_live_frame_pose()
        if pose is None:
            return
        if self._right_live_frame_marker is None:
            self._right_live_frame_marker = create_rgb_axis_marker(
                self.scene, axis_len=0.12, axis_radius=0.0035, name="right_live_frame_marker"
            )
        self._right_live_frame_marker.set_pose(pose)

    def update_left_base_frame_marker(self):
        # Only visualize in UI mode to avoid affecting dataset rendering.
        if self.viewer is None or self.scene is None:
            return
        pose = self._compute_left_base_frame_pose()
        if pose is None:
            return
        if self._left_base_frame_marker is None:
            self._left_base_frame_marker = create_rgb_axis_marker(
                self.scene, axis_len=0.14, axis_radius=0.0038, name="left_base_frame_marker"
            )
        self._left_base_frame_marker.set_pose(pose)

    def update_reference_frame_marker(self):
        # Only visualize in UI mode to avoid affecting dataset rendering.
        if self.viewer is None or self.scene is None:
            return
        pose = self._compute_reference_frame_pose()
        if self._reference_frame_marker is None:
            self._reference_frame_marker = create_rgb_axis_marker(
                self.scene, axis_len=0.12, axis_radius=0.0035, name="reference_frame_marker"
            )
        self._reference_frame_marker.set_pose(pose)

    def left_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        if self.verbose_planner_log:
            print(f"[Robot.left_plan_path] gripper target: {target_pose}")
        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="left")
        if self.verbose_planner_log:
            print(
                f"[Robot.left_plan_path] endlink target: "
                f"p={list(np.round(np.array(trans_target_pose.p), 5))}, "
                f"q={list(np.round(np.array(trans_target_pose.q), 5))}"
            )
        self._visualize_target(trans_target_pose, name="left_target_marker")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
            })
            plan_res = self.left_conn.recv()
        else:
            plan_res = self.left_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="left",
            )
        if isinstance(plan_res, dict):
            plan_res = self._postprocess_arm_plan_result(plan_res, arm_tag="left")
            plan_res["debug_target_tcp_pose"] = deepcopy(target_pose)
            plan_res["debug_target_endlink_pose"] = deepcopy(
                trans_target_pose.p.tolist() + trans_target_pose.q.tolist()
            )
        return plan_res

    def right_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        if self.verbose_planner_log:
            print(f"[Robot.right_plan_path] gripper target: {target_pose}")
        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="right")
        if self.verbose_planner_log:
            print(
                f"[Robot.right_plan_path] endlink target: "
                f"p={list(np.round(np.array(trans_target_pose.p), 5))}, "
                f"q={list(np.round(np.array(trans_target_pose.q), 5))}"
            )
        self._visualize_target(trans_target_pose, name="right_target_marker")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
            })
            plan_res = self.right_conn.recv()
        else:
            plan_res = self.right_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="right",
            )

        # Attach debug targets so caller can compare with executed TCP.
        if isinstance(plan_res, dict):
            plan_res = self._postprocess_arm_plan_result(plan_res, arm_tag="right")
            plan_res["debug_target_tcp_pose"] = deepcopy(target_pose)
            plan_res["debug_target_endlink_pose"] = deepcopy(
                trans_target_pose.p.tolist() + trans_target_pose.q.tolist()
            )
        return plan_res

    # The data of gripper has been normalized
    def get_left_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.left_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.right_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_arm_real_jointState(self) -> list:
        jointState_list = []
        left_joints_qpos = self.left_entity.get_qpos()
        left_active_joints = self.left_entity.get_active_joints()
        for joint in self.left_arm_joints:
            jointState_list.append(left_joints_qpos[left_active_joints.index(joint)])
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_real_jointState(self) -> list:
        jointState_list = []
        right_joints_qpos = self.right_entity.get_qpos()
        right_active_joints = self.right_entity.get_active_joints()
        for joint in self.right_arm_joints:
            jointState_list.append(right_joints_qpos[right_active_joints.index(joint)])
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_head_jointState(self) -> list:
        if len(self.head_joints) == 0:
            return []
        return [float(joint.get_drive_target()[0]) for joint in self.head_joints]

    def get_head_real_jointState(self) -> list:
        if self.head_entity is None or len(self.head_joints) == 0:
            return []
        joint_state_list = []
        qpos = self.head_entity.get_qpos()
        active_joints = self.head_entity.get_active_joints()
        for joint in self.head_joints:
            if joint in active_joints:
                joint_state_list.append(float(qpos[active_joints.index(joint)]))
        return joint_state_list

    def get_torso_jointState(self) -> list:
        if len(self.torso_joints) == 0:
            return []
        return [float(joint.get_drive_target()[0]) for joint in self.torso_joints]

    def get_torso_real_jointState(self) -> list:
        if self.torso_entity is None or len(self.torso_joints) == 0:
            return []
        joint_state_list = []
        qpos = self.torso_entity.get_qpos()
        active_joints = self.torso_entity.get_active_joints()
        for joint in self.torso_joints:
            if joint in active_joints:
                joint_state_list.append(float(qpos[active_joints.index(joint)]))
        return joint_state_list

    def get_left_gripper_val(self):
        if len(self.left_gripper) == 0 or self.left_gripper[0][0] is None:
            print("No gripper")
            return 0
        return self.left_gripper_val

    def get_right_gripper_val(self):
        if len(self.right_gripper) == 0 or self.right_gripper[0][0] is None:
            print("No gripper")
            return 0
        return self.right_gripper_val

    def is_left_gripper_open(self):
        return self.left_gripper_val > 0.8

    def is_right_gripper_open(self):
        return self.right_gripper_val > 0.8

    def is_left_gripper_open_half(self):
        return self.left_gripper_val > 0.45

    def is_right_gripper_open_half(self):
        return self.right_gripper_val > 0.45

    def is_left_gripper_close(self):
        return self.left_gripper_val < 0.2

    def is_right_gripper_close(self):
        return self.right_gripper_val < 0.2

    # get move group joint pose
    def get_left_ee_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=False)

    def get_right_ee_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=False)

    # get gripper centor pose
    def get_left_tcp_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=True)

    def get_right_tcp_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=True)

    def get_left_orig_endpose(self):
        pose = self.left_ee.global_pose
        global_trans_matrix = self.left_global_trans_matrix
        pose.p = pose.p - self.left_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def get_right_orig_endpose(self):
        pose = self.right_ee.global_pose
        global_trans_matrix = self.right_global_trans_matrix
        pose.p = pose.p - self.right_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def get_left_endlink_pose(self):
        if self.left_ee_link is None:
            return None
        pose = self.left_ee_link.get_pose()
        return pose.p.tolist() + pose.q.tolist()

    def get_right_endlink_pose(self):
        if self.right_ee_link is None:
            return None
        pose = self.right_ee_link.get_pose()
        return pose.p.tolist() + pose.q.tolist()

    def get_arm_joint_tracking_error(self, arm_tag, target_joint_pos):
        if target_joint_pos is None:
            return None
        if arm_tag == "left":
            entity = self.left_entity
            arm_joints = self.left_arm_joints
        else:
            entity = self.right_entity
            arm_joints = self.right_arm_joints
        qpos = entity.get_qpos()
        active_joints = entity.get_active_joints()
        real = np.array([qpos[active_joints.index(j)] for j in arm_joints], dtype=np.float64)
        drive = np.array([float(j.get_drive_target()[0]) for j in arm_joints], dtype=np.float64)
        target = np.array(target_joint_pos, dtype=np.float64)
        diff = real - target
        drive_diff = drive - target
        return {
            "real": real.tolist(),
            "drive": drive.tolist(),
            "target": target.tolist(),
            "diff": diff.tolist(),
            "drive_diff": drive_diff.tolist(),
            "max_abs": float(np.max(np.abs(diff))),
            "l2": float(np.linalg.norm(diff)),
            "drive_max_abs": float(np.max(np.abs(drive_diff))),
            "drive_l2": float(np.linalg.norm(drive_diff)),
        }

    def _trans_endpose(self, arm_tag=None, is_endpose=False):
        if arm_tag is None:
            print("No arm tag")
            return
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        global_trans_matrix = (self.left_global_trans_matrix if arm_tag == "left" else self.right_global_trans_matrix)
        delta_matrix = (self.left_delta_matrix if arm_tag == "left" else self.right_delta_matrix)
        ee_pose = (self.left_ee.global_pose if arm_tag == "left" else self.right_ee.global_pose)
        # ee_pose = (self.left_ee.g if arm_tag == "left" else self.right_ee_link.global_pose)
        endpose_arr = np.eye(4)
        endpose_arr[:3, :3] = (t3d.quaternions.quat2mat(ee_pose.q) @ global_trans_matrix @ delta_matrix)
        dis = gripper_bias
        if is_endpose == False:
            dis -= 0.12
        endpose_arr[:3, 3] = ee_pose.p + endpose_arr[:3, :3] @ np.array([dis, 0, 0]).T
        res = (endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist())
        return res

    def _entity_qf(self, entity):
        # return
        qf = entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        entity.set_qf(qf)

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)

        joint_lst = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        for j in range(len(joint_lst)):
            joint = joint_lst[j]
            joint.set_drive_target(target_position[j])
            joint.set_drive_velocity_target(target_velocity[j])

    def set_head_joints(self, target_position, target_velocity=None):
        if len(self.head_joints) == 0:
            return False

        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        target_position = np.array(target_position, dtype=np.float64).reshape(-1)
        if target_velocity is None:
            target_velocity = np.zeros_like(target_position)
        target_velocity = np.array(target_velocity, dtype=np.float64).reshape(-1)

        for j, joint in enumerate(self.head_joints):
            if j >= target_position.shape[0]:
                break
            target = self._clip_joint_target_to_limits(joint, target_position[j])
            vel = float(target_velocity[j]) if j < target_velocity.shape[0] else 0.0
            joint.set_drive_target(target)
            joint.set_drive_velocity_target(vel)
        return True

    def set_head_joints_delta(self, delta_position):
        if len(self.head_joints) == 0:
            return False
        delta_position = np.array(delta_position, dtype=np.float64).reshape(-1)
        if delta_position.shape[0] < len(self.head_joints):
            pad = np.zeros(len(self.head_joints) - delta_position.shape[0], dtype=np.float64)
            delta_position = np.concatenate([delta_position, pad], axis=0)
        now = np.array(self.get_head_jointState(), dtype=np.float64)
        target = now + delta_position[: len(self.head_joints)]
        return self.set_head_joints(target, np.zeros_like(target))

    def set_torso_joints(self, target_position, target_velocity=None):
        if len(self.torso_joints) == 0:
            return False

        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        target_position = np.array(target_position, dtype=np.float64).reshape(-1)
        if target_velocity is None:
            target_velocity = np.zeros_like(target_position)
        target_velocity = np.array(target_velocity, dtype=np.float64).reshape(-1)

        for j, joint in enumerate(self.torso_joints):
            if j >= target_position.shape[0]:
                break
            target = self._clip_joint_target_to_limits(joint, target_position[j])
            vel = float(target_velocity[j]) if j < target_velocity.shape[0] else 0.0
            joint.set_drive_target(target)
            joint.set_drive_velocity_target(vel)
        return True

    def set_torso_joints_delta(self, delta_position):
        if len(self.torso_joints) == 0:
            return False
        delta_position = np.array(delta_position, dtype=np.float64).reshape(-1)
        if delta_position.shape[0] < len(self.torso_joints):
            pad = np.zeros(len(self.torso_joints) - delta_position.shape[0], dtype=np.float64)
            delta_position = np.concatenate([delta_position, pad], axis=0)
        now = np.array(self.get_torso_jointState(), dtype=np.float64)
        target = now + delta_position[: len(self.torso_joints)]
        return self.set_torso_joints(target, np.zeros_like(target))

    def get_normal_real_gripper_val(self):
        if len(self.left_gripper) == 0 or self.left_gripper[0][0] is None:
            normal_left_gripper_val = self.left_gripper_val
        else:
            normal_left_gripper_val = (
                (self.left_gripper[0][0].get_drive_target()[0] - self.left_gripper_scale[0]) / (
                    self.left_gripper_scale[1] - self.left_gripper_scale[0]
                )
            )

        if len(self.right_gripper) == 0 or self.right_gripper[0][0] is None:
            normal_right_gripper_val = self.right_gripper_val
        else:
            normal_right_gripper_val = (
                (self.right_gripper[0][0].get_drive_target()[0] - self.right_gripper_scale[0]) / (
                    self.right_gripper_scale[1] - self.right_gripper_scale[0]
                )
            )
        normal_left_gripper_val = np.clip(normal_left_gripper_val, 0, 1)
        normal_right_gripper_val = np.clip(normal_right_gripper_val, 0, 1)
        return [normal_left_gripper_val, normal_right_gripper_val]

    @staticmethod
    def _gripper_cmd_to_active_rad(cmd_val, gripper_scale):
        # RobotWin command is normalized in [0, 1].
        # Mapping to active-joint rad is controlled by gripper_scale.
        return gripper_scale[0] + cmd_val * (gripper_scale[1] - gripper_scale[0])

    @staticmethod
    def _set_gripper_mimic_targets(joints, active_target_rad, velocity_scale=0.2, velocity_limit=4.0):
        # Keep strict mimic relation:
        # follower_rad = active_target_rad * mimic_coeff + mimic_offset
        velocity_scale = float(max(velocity_scale, 0.0))
        velocity_limit = float(max(velocity_limit, 1e-6))
        for idx, joint_info in enumerate(joints):
            real_joint: sapien.physx.PhysxArticulationJoint = joint_info[0]
            if real_joint is None:
                continue
            if idx == 0:
                # First joint is active joint by construction.
                drive_target = float(active_target_rad)
            else:
                mimic_coeff = float(joint_info[1])
                mimic_offset = float(joint_info[2])
                drive_target = float(active_target_rad) * mimic_coeff + mimic_offset
            drive_velocity_target = float(
                np.clip(drive_target - real_joint.drive_target, -velocity_limit, velocity_limit) * velocity_scale
            )
            real_joint.set_drive_target(drive_target)
            real_joint.set_drive_velocity_target(drive_velocity_target)

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):  # gripper_val in [0,1]
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        gripper_val = np.clip(gripper_val, 0, 1)

        if arm_tag == "left":
            joints = self.left_gripper
            self.left_gripper_val = gripper_val
            gripper_scale = self.left_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[0]
        else:
            joints = self.right_gripper
            self.right_gripper_val = gripper_val
            gripper_scale = self.right_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[1]

        if not joints:
            print("No gripper")
            return

        if (gripper_val - real_gripper_val > gripper_eps
                and gripper_eps > 0) or (gripper_val - real_gripper_val < gripper_eps and gripper_eps < 0):
            gripper_val = real_gripper_val + gripper_eps  # TODO

        active_target_rad = self._gripper_cmd_to_active_rad(gripper_val, gripper_scale)
        velocity_scale = (
            self.left_gripper_drive_velocity_scale if arm_tag == "left" else self.right_gripper_drive_velocity_scale
        )
        self._set_gripper_mimic_targets(joints, active_target_rad, velocity_scale=velocity_scale)


def planner_process_worker(conn, args):
    import os
    from .planner import CuroboPlanner  # 或者绝对路径导入

    planner = CuroboPlanner(
        args["origin_pose"],
        args["joints_name"],
        args["all_joints"],
        yml_path=args["yml_path"],
        verbose=bool(args.get("verbose", False)),
    )

    while True:
        try:
            msg = conn.recv()
            if msg["cmd"] == "plan_path":
                result = planner.plan_path(
                    msg["qpos"],
                    msg["target_pose"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                )
                conn.send(result)

            elif msg["cmd"] == "plan_batch":
                result = planner.plan_batch(
                    msg["qpos"],
                    msg["target_pose_list"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                )
                conn.send(result)

            elif msg["cmd"] == "plan_grippers":
                result = planner.plan_grippers(
                    msg["now_val"],
                    msg["target_val"],
                )
                conn.send(result)

            elif msg["cmd"] == "update_point_cloud":
                planner.update_point_cloud(msg["pcd"], resolution=msg.get("resolution", 0.02))
                conn.send("ok")

            elif msg["cmd"] == "update_world_cuboids":
                n = planner.set_world_extra_cuboids(msg.get("cuboids", []), curr_joint_pos=msg.get("qpos", None))
                conn.send({"status": "ok", "num_cuboids": int(n)})

            elif msg["cmd"] == "reset":
                planner.motion_gen.reset(reset_seed=True)
                conn.send("ok")

            elif msg["cmd"] == "exit":
                conn.close()
                break

            else:
                conn.send({"error": f"Unknown command {msg['cmd']}"})

        except EOFError:
            break
        except Exception as e:
            conn.send({"error": str(e)})
