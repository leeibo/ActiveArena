from ._base_task import Base_Task
from ._fan_double_task_utils import *


class blocks_ranking_size_fan_double(Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan"
    ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY = "centered"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True
    FIXED_LAYER_HEAD_JOINT2_ONLY = True
    LAYER_SPECS = {
        "lower": {
            "r_min": 0.35,
            "r_max": 0.35,
            "theta_min_deg": -70.0,
            "theta_max_deg": 70.0,
        },
        "upper": {
            "inner_margin": 0.05,
            "outer_margin": 0.07,
            "max_cyl_r": 0.68,
            "theta_shrink": 0.96,
        },
    }
    BLOCK_DEFS = (
        {"key": "A", "label": "large block", "size_range": (0.030, 0.033), "color": (0.90, 0.15, 0.10)},
        {"key": "B", "label": "medium block", "size_range": (0.024, 0.027), "color": (0.10, 0.75, 0.20)},
        {"key": "C", "label": "small block", "size_range": (0.018, 0.021), "color": (0.15, 0.25, 0.90)},
    )
    BLOCK_SPAWN_MIN_DIST_SQ = 0.014

    TARGET_LAYER = "upper"
    TARGET_LOWER_LAYER_PROB = 0.3
    LOWER_TARGET_ROW_R = 0.43
    TARGET_ROW_SPEC = {
        "r": 0.68,
        "theta_deg": 34.0,
        "gap_theta_deg": 13.0,
        "z_offset": 0.0,
    }
    RANDOMIZE_TARGET_ROW_THETA = True
    TARGET_ROW_THETA_MARGIN_DEG = 6.0
    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    HEAD_RESET_SAVE_FREQ = None

    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_LIFT_Z = 0.12
    PLACE_RETREAT_Z = 0.08
    LOWER_PLACE_WITH_PLACE_ACTOR = True
    LOWER_PLACE_FUNCTIONAL_POINT_ID = 0
    LOWER_PLACE_PRE_DIS = 0.05
    LOWER_PLACE_DIS = 0.0
    LOWER_PLACE_CONSTRAIN = "free"
    LOWER_PLACE_PRE_DIS_AXIS = "fp"
    LOWER_PLACE_IS_OPEN = True
    LOWER_PLACE_RETREAT_Z = 0.0
    LOWER_PLACE_RETREAT_MOVE_AXIS = "arm"
    RETURN_TO_HOMESTATE_AFTER_PLACE = False

    DIRECT_RELEASE_TCP_BACKOFF = 0.12
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.06
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_RETREAT_Z = 0.06
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03, -0.06, 0.06)
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0, -6.0, 6.0)
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)

    UPPER_TO_LOWER_USE_HOVER_DROP = True
    UPPER_TO_LOWER_HOVER_Z_OFFSETS = (0.06, 0.08, 0.10)
    UPPER_TO_LOWER_DROP_YAW_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)
    UPPER_TO_LOWER_RELEASE_DELAY_STEPS = 15
    UPPER_TO_LOWER_RELEASE_RETREAT_Z = 0.08

    UPPER_PICK_ENTRY_Z_OFFSET = 0.10
    UPPER_PICK_PRE_GRASP_DIS = 0.10
    UPPER_PICK_GRASP_Z_BIAS = 0.02
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = -0.01

    SUCCESS_XY_TOL = 0.09
    SUCCESS_Z_TOL = 0.08

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        self.fixed_layer_head_joint2_only = bool(
            kwargs.get(
                "fixed_layer_head_joint2_only",
                getattr(self, "FIXED_LAYER_HEAD_JOINT2_ONLY", False),
            )
        )
        super()._init_task_env_(**kwargs)

    def _get_subtask_search_target_keys(self, subtask_idx):
        return get_subtask_search_target_keys(self, subtask_idx)

    def _get_subtask_upper_search_target_keys(self, subtask_idx):
        return get_subtask_upper_search_target_keys(self, subtask_idx)

    def _should_search_lower_before_upper_for_subtask(self, subtask_idx):
        return should_search_lower_before_upper_for_subtask(self, subtask_idx)

    def _has_unfinished_lower_search_phase(self):
        return has_unfinished_lower_search_phase(self)

    def _clear_rotate_target_search_history(self, object_key):
        clear_rotate_target_search_history(self, object_key)

    def _prepare_subtask_rotate_search(self, subtask_idx):
        prepare_subtask_rotate_search(self, subtask_idx)

    def _after_rotate_visibility_refresh(self, visibility_map):
        return None

    def _sample_target_layer(self):
        lower_prob = float(getattr(self, "TARGET_LOWER_LAYER_PROB", 0.0))
        return "lower" if float(np.random.random()) < lower_prob else normalize_layer(self.TARGET_LAYER)

    def _get_target_layer(self):
        return normalize_layer(getattr(self, "target_layer", self.TARGET_LAYER))

    def _get_target_row_spec(self):
        spec = dict(self.TARGET_ROW_SPEC)
        if self._get_target_layer() == "lower":
            spec["r"] = float(getattr(self, "LOWER_TARGET_ROW_R", spec["r"]))
        return spec

    def _sample_block_layers(self):
        return {"A": self._get_target_layer(), "B": "lower", "C": "lower"}

    def _sample_target_row_theta_deg(self):
        spec = self._get_target_row_spec()
        fallback_theta = float(spec["theta_deg"])
        if not bool(getattr(self, "RANDOMIZE_TARGET_ROW_THETA", False)):
            return fallback_theta

        layer_spec = get_layer_spec(self, self._get_target_layer())
        theta_min = float(np.rad2deg(layer_spec["thetalim"][0]))
        theta_max = float(np.rad2deg(layer_spec["thetalim"][1]))
        margin = float(getattr(self, "TARGET_ROW_THETA_MARGIN_DEG", 0.0))
        total_gap = float(spec.get("gap_theta_deg", 0.0)) * 2.0
        low = theta_min + total_gap + margin
        high = theta_max - margin
        if high < low:
            return min(max(fallback_theta, theta_min), theta_max)
        return float(np.random.uniform(low, high))

    def _target_point(self, target_idx, z_offset=0.0):
        spec = self._get_target_row_spec()
        base_theta_deg = float(getattr(self, "target_row_theta_deg", spec["theta_deg"]))
        theta_deg = base_theta_deg - float(spec["gap_theta_deg"]) * float(target_idx)
        return place_point_cyl(
            [
                float(spec["r"]),
                float(np.deg2rad(theta_deg)),
                get_layer_top_z(self, self._get_target_layer()) + float(spec.get("z_offset", 0.0)) + float(z_offset),
            ],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="array",
        )

    def _target_pose(self, target_idx):
        return pose_list_from_point(self._target_point(target_idx), quat=[0, 1, 0, 0])

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.blocks["A"],
                "B": self.blocks["B"],
                "C": self.blocks["C"],
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_medium_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "medium_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_medium_block_right_of_large",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "medium_block_placed",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "pick_small_block",
                    "instruction_idx": 3,
                    "search_target_keys": ["C"],
                    "action_target_keys": ["C"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["C"],
                    "allow_stage2_from_memory": True,
                    "done_when": "small_block_grasped",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "place_small_block_right_of_medium",
                    "instruction_idx": 4,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B", "C"],
                    "required_carried_keys": ["C"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "small_block_placed",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Arrange the large block, medium block, and small block from left to right by size.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = get_robot_root_xy_yaw(self)
        self.target_layer = self._sample_target_layer()
        sampled_layers = self._sample_block_layers()
        self.blocks = {}
        self.block_layers = {}
        self.block_sizes = {}
        self.target_row_theta_deg = self._sample_target_row_theta_deg()
        self.target_poses = {
            "A": self._target_pose(0),
            "B": self._target_pose(1),
            "C": self._target_pose(2),
        }
        target_xy_lst = [np.array(self.target_poses[key][:2], dtype=np.float64) for key in ("A", "B", "C")]

        existing_pose_lst = []
        for idx, block_def in enumerate(self.BLOCK_DEFS):
            key = block_def["key"]
            layer_name = normalize_layer(sampled_layers[key])
            block_size = float(np.random.uniform(*block_def["size_range"]))
            if key == "A":
                pose_point = self._target_point(0, z_offset=block_size)
                block_pose = sapien.Pose(pose_point.tolist(), [1, 0, 0, 0])
            else:
                block_pose = sample_pose_on_layer(
                    self,
                    layer_name,
                    z_offset=block_size,
                    existing_pose_lst=existing_pose_lst,
                    avoid_xy_lst=target_xy_lst,
                    min_dist_sq=self.BLOCK_SPAWN_MIN_DIST_SQ,
                )
            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_size, block_size, block_size),
                color=block_def["color"],
                name=f"{key}_block",
                is_static=(key == "A" and layer_name == self._get_target_layer()),
            )
            block.set_mass(0.03)
            self.blocks[key] = block
            self.block_layers[key] = layer_name
            self.block_sizes[key] = block_size
            existing_pose_lst.append(block_pose)
            self.add_prohibit_area(block, padding=0.06)

        self.block1 = self.blocks["A"]
        self.block2 = self.blocks["B"]
        self.block3 = self.blocks["C"]
        self.object_layers = dict(self.block_layers)
        self._configure_rotate_subtask_plan()

    def _pick(self, subtask_idx, key, arm_tag=None):
        return pick_object(
            self,
            subtask_idx,
            key,
            self.blocks[key],
            self.block_layers[key],
            arm_tag=arm_tag,
            lower_grasp_kwargs={
                "pre_grasp_dis": self.PICK_PRE_GRASP_DIS,
                "grasp_dis": self.PICK_GRASP_DIS,
                # "contact_point_id":0
            },
        )

    def _place(self, subtask_idx, key, arm_tag, focus_key):
        placed = place_object(
            self,
            subtask_idx,
            key,
            self.blocks[key],
            arm_tag,
            self.target_poses[key],
            self._get_target_layer(),
            place_kwargs={
                "functional_point_id": self.LOWER_PLACE_FUNCTIONAL_POINT_ID,
                "pre_dis": self.LOWER_PLACE_PRE_DIS,
                "dis": self.LOWER_PLACE_DIS,
                "constrain": self.LOWER_PLACE_CONSTRAIN,
                "pre_dis_axis": self.LOWER_PLACE_PRE_DIS_AXIS,
                "is_open": bool(self.LOWER_PLACE_IS_OPEN),
            },
            focus_object_key=focus_key,
            retreat_after_release=(key != "C"),
            return_after_upper_release=(key != "C"),
        )
        if placed:
            self.block_layers[str(key)] = self._get_target_layer()
            self.object_layers[str(key)] = self._get_target_layer()
        return placed

    def play_once(self):
        locked_arm_tag = None
        prev_subtask_idx = None
        for pick_idx, place_idx, key, focus_key in [(1, 2, "B", "A"), (3, 4, "C", "B")]:
            self._prepare_subtask_rotate_search(pick_idx)
            found_key = self.search_and_focus_rotate_subtask(
                pick_idx,
                scan_r=self.SCAN_R,
                scan_z=float(self.SCAN_Z_BIAS + self.table_z_bias),
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if found_key is None:
                self.plan_success = False
                break
            if locked_arm_tag is None:
                locked_arm_tag = get_object_arm_tag(self, self.blocks[key])
            arm_tag = ArmTag(locked_arm_tag)
            self._pick(pick_idx, key, arm_tag=arm_tag)
            if not self.plan_success:
                break
            prev_subtask_idx = pick_idx

            self._prepare_subtask_rotate_search(place_idx)
            found_focus = self.search_and_focus_rotate_subtask(
                place_idx,
                scan_r=self.SCAN_R,
                scan_z=float(self.SCAN_Z_BIAS + self.table_z_bias),
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if found_focus is None:
                self.plan_success = False
                break
            if not self._place(place_idx, key, arm_tag, found_focus or focus_key):
                break
            prev_subtask_idx = place_idx

        info_arm_tag = locked_arm_tag or get_object_arm_tag(self, self.blocks["B"])

        self.info["info"] = {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": str(info_arm_tag),
            "{b}": str(info_arm_tag),
            "{c}": str(info_arm_tag),
        }
        return self.info

    def check_success(self):
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        xy_ok = True
        z_ok = True
        for key in ("A", "B", "C"):
            pose = np.array(self.blocks[key].get_pose().p, dtype=np.float64).reshape(3)
            target = np.array(self.target_poses[key][:3], dtype=np.float64).reshape(3)
            xy_ok = xy_ok and bool(np.linalg.norm(pose[:2] - target[:2]) < self.SUCCESS_XY_TOL)
            z_ok = z_ok and bool(abs(pose[2] - target[2]) < self.SUCCESS_Z_TOL)

        c1 = world_to_robot(self.blocks["A"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        c2 = world_to_robot(self.blocks["B"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        c3 = world_to_robot(self.blocks["C"].get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        ordered = c1[1] > c2[1] > c3[1]
        same_arc = abs(c1[0] - c2[0]) < 0.16 and abs(c2[0] - c3[0]) < 0.16
        return bool(gripper_open and xy_ok and z_ok and ordered and same_arc)
