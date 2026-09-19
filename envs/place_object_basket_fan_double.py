from ._base_task import Base_Task
from ._fan_double_task_utils import *


class place_object_basket_fan_double(Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan"
    ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY = "centered"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True
    FIXED_LAYER_HEAD_JOINT2_ONLY = True
    BASKET_THETA_JITTER_DEG = 0.0
    UPPER_THETA_MARGIN_DEG = 6.0
    LAYER_SPECS = {
        "lower": {
            "r_min": 0.40,
            "r_max": 0.43,
            "theta_min_deg": -55.0,
            "theta_max_deg": 55.0,
        },
        "upper": {
            "inner_margin": 0.05,
            "outer_margin": 0.07,
            "max_cyl_r": 0.68,
            "theta_shrink": 0.96,
        },
    }

    OBJECT_LAYER = "lower"
    BASKET_LAYER = "upper"
    BASKET_LOWER_LAYER_PROB = 0.3
    OBJECT_CANDIDATES = {
        # "057_toycar": [0, 1, 2, 3, 4, 5],
        "071_can": [0, 1, 2, 3, 5, 6],
    }
    OBJECT_R_RANGE_BY_MODEL = {
        "057_toycar": [0.40, 0.43],
        "071_can": [0.40, 0.43],
    }
    OBJECT_ROTATE_RAND_BY_MODEL = {
        "057_toycar": True,
        "071_can": False,
    }
    OBJECT_PRE_GRASP_DIS_BY_MODEL = {
        "057_toycar": 0.10,
        "071_can": 0.08,
    }
    OBJECT_GRIPPER_POS_BY_MODEL = {
        "057_toycar": 0.2,
        "071_can": 0.2,
    }
    BASKET_MODEL_NAME = "076_breadbasket"
    BASKET_MODEL_IDS = [0]
    OBJECT_R_RANGE = [0.40, 0.43]
    # OBJECT_QPOS = [0.707225, 0.706849, -0.0100455, -0.00982061]
    OBJECT_QPOS = [0, 0, 0, 1]
    OBJECT_ROTATE_RAND = True
    OBJECT_ROTATE_LIM = [0.0, np.pi / 6, 0.0]
    OBJECT_PRE_GRASP_DIS = 0.30
    OBJECT_GRIPPER_POS = 0.0
    PICK_SUCCESS_Z_DELTA = 0.03
    OBJECT_POSE_SPECS = {
        "lower": {
            "r": 0.52,
            "theta_deg": -42.0,
            "z_offset": 0.0,
            "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061],
        },
        "upper": {
            "r": 0.69,
            "theta_deg": -12.0,
            "z_offset": 0.0,
            "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061],
        },
    }
    BASKET_POSE_SPECS = {
        "lower": {"r": 0.42, "theta_deg": -18.0, "z_offset": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
        "upper": {"r": 0.68, "theta_deg": 5.0, "z_offset": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
    }
    LIFT_BASKET_AFTER_PLACE = False

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.90
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    HEAD_RESET_SAVE_FREQ = None

    PICK_LIFT_Z = 0.12
    POST_GRASP_EXTRA_LIFT_Z = 0.00
    PLACE_RETREAT_Z = 0.0
    LOWER_PLACE_WITH_PLACE_ACTOR = True
    RETURN_TO_HOMESTATE_AFTER_PLACE = False

    DIRECT_RELEASE_TCP_BACKOFF = 0.12
    DIRECT_RELEASE_ENTRY_R_MARGIN_FROM_UPPER_INNER = 0.08
    DIRECT_RELEASE_TCP_Z_OFFSET = 0.09
    DIRECT_RELEASE_ENTRY_TCP_Z_OFFSET = 0.12
    DIRECT_RELEASE_APPROACH_TCP_Z_OFFSET = 0.10
    DIRECT_RELEASE_RETREAT_Z = 0.0
    DIRECT_RELEASE_R_OFFSETS = (0.0, -0.03, 0.03)
    DIRECT_RELEASE_THETA_OFFSETS_DEG = (0.0, -3.0, 3.0)
    DIRECT_RELEASE_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0)
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.0
    UPPER_PLACE_BODY_JOINT_NAME = "astribot_torso_joint_2"
    LOWER_PLACE_RETREAT_Z = 0.0
    UPPER_TO_LOWER_RELEASE_RETREAT_Z = 0.0

    UPPER_PICK_ENTRY_Z_OFFSET = 0.09
    UPPER_PICK_PRE_GRASP_DIS = 0.11
    UPPER_PICK_GRASP_Z_BIAS = 0.0
    UPPER_PICK_YAW_OFFSETS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)
    UPPER_PICK_GRIPPER_POS = 0.0

    SUCCESS_DIST = 0.18
    SUCCESS_Z_MIN_DELTA = 0.015

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        self.fixed_layer_head_joint2_only = bool(
            kwargs.get(
                "fixed_layer_head_joint2_only",
                getattr(self, "FIXED_LAYER_HEAD_JOINT2_ONLY", True),
            )
        )
        super()._init_task_env_(**kwargs)

    def _get_basket_pose_spec(self):
        basket_spec = dict(self.BASKET_POSE_SPECS[self.basket_layer])
        if self.basket_layer == "upper":
            basket_spec["theta_deg"] = self._sample_theta_deg_on_layer("upper")
        else:
            basket_spec["theta_deg"] = float(basket_spec.get("theta_deg", 0.0)) + float(
                np.random.uniform(-self.BASKET_THETA_JITTER_DEG, self.BASKET_THETA_JITTER_DEG)
            )
        return basket_spec

    def _sample_basket_layer(self):
        lower_prob = float(getattr(self, "BASKET_LOWER_LAYER_PROB", 0.0))
        return "lower" if float(np.random.random()) < lower_prob else normalize_layer(self.BASKET_LAYER)

    def _sample_theta_deg_on_layer(self, layer_name):
        layer_spec = get_layer_spec(self, layer_name)
        theta_min = float(np.rad2deg(layer_spec["thetalim"][0]))
        theta_max = float(np.rad2deg(layer_spec["thetalim"][1]))
        margin = float(getattr(self, "UPPER_THETA_MARGIN_DEG", 0.0)) if normalize_layer(layer_name) == "upper" else 0.0
        low = theta_min + margin
        high = theta_max - margin
        if high < low:
            return 0.5 * (theta_min + theta_max)
        return float(np.random.uniform(low, high))

    def _get_object_r_range(self):
        layer_spec = get_layer_spec(self, self.object_layer)
        return list(self.OBJECT_R_RANGE_BY_MODEL.get(self.object_name, layer_spec["rlim"]))

    def _get_object_theta_range(self):
        layer_spec = get_layer_spec(self, self.object_layer)
        return list(layer_spec["thetalim"])

    def _get_object_rotate_rand(self):
        return bool(self.OBJECT_ROTATE_RAND_BY_MODEL.get(self.object_name, self.OBJECT_ROTATE_RAND))

    def _get_object_pre_grasp_dis(self):
        return float(self.OBJECT_PRE_GRASP_DIS_BY_MODEL.get(self.object_name, self.OBJECT_PRE_GRASP_DIS))

    def _get_object_gripper_pos(self):
        return float(self.OBJECT_GRIPPER_POS_BY_MODEL.get(self.object_name, self.OBJECT_GRIPPER_POS))

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.object,
                "B": self.basket,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_object",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_object_into_basket",
                    "instruction_idx": 2,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": False,
                    "done_when": "object_in_basket",
                    "next_subtask_id": -1,
                },
            ],
            task_instruction="Put the object into the bread basket.",
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = get_robot_root_xy_yaw(self)
        self.object_layer = normalize_layer(self.OBJECT_LAYER)
        self.basket_layer = normalize_layer(self._sample_basket_layer())
        self.object_name = str(np.random.choice(list(self.OBJECT_CANDIDATES.keys())))
        self.object_id = int(np.random.choice(self.OBJECT_CANDIDATES[self.object_name]))
        self.basket_name = str(self.BASKET_MODEL_NAME)
        self.basket_id = int(np.random.choice(self.BASKET_MODEL_IDS))
        self.arm_tag = ArmTag({0: "left", 1: "right"}[int(np.random.randint(0, 2))])

        if self.object_layer == "lower":
            object_z = (
                get_layer_top_z(self, self.object_layer)
                + float(self.OBJECT_POSE_SPECS[self.object_layer].get("z_offset", 0.0))
            )
            object_pose = rand_pose_cyl(
                rlim=self._get_object_r_range(),
                thetalim=self._get_object_theta_range(),
                zlim=[object_z, object_z],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
            )
        else:
            object_spec = dict(self.OBJECT_POSE_SPECS[self.object_layer])
            if self.object_layer == "upper":
                object_spec["theta_deg"] = self._sample_theta_deg_on_layer("upper")
            object_pose = pose_from_cyl(
                self,
                self.object_layer,
                object_spec,
                default_qpos=self.OBJECT_QPOS,
                ret="pose",
            )
        self.object = create_actor(
            self,
            pose=object_pose,
            modelname=self.object_name,
            model_id=self.object_id,
            convex=True,
        )
        self.object.set_mass(0.05)

        basket_pose = pose_from_cyl(
            self,
            self.basket_layer,
            self._get_basket_pose_spec(),
            default_qpos=[0.5, 0.5, 0.5, 0.5],
            ret="pose",
        )
        self.basket = create_actor(
            self,
            pose=basket_pose,
            modelname=self.basket_name,
            model_id=self.basket_id,
            convex=True,
            is_static=not bool(self.LIFT_BASKET_AFTER_PLACE),
        )
        self.basket.set_mass(0.5)

        self.object_start_height = float(self.object.get_pose().p[2])
        self.basket_start_height = float(self.basket.get_pose().p[2])
        self.add_prohibit_area(self.object, padding=0.08)
        self.add_prohibit_area(self.basket, padding=0.08)
        self.object_layers = {"A": self.object_layer, "B": self.basket_layer}
        self._configure_rotate_subtask_plan()

    def _basket_target_pose(self, arm_tag):
        candidates = []
        for idx in (0, 1):
            pose = self.basket.get_functional_point(idx)
            if pose is not None:
                candidates.append(np.array(pose, dtype=np.float64).reshape(-1))
        if len(candidates) == 0:
            candidates = [np.array(self.basket.get_pose().p.tolist() + [1, 0, 0, 0], dtype=np.float64)]

        obj_xy = np.array(self.object.get_pose().p[:2], dtype=np.float64)
        target = min(candidates, key=lambda item: float(np.linalg.norm(item[:2] - obj_xy))).copy()
        target[3:] = [-1, 0, 0, 0] if ArmTag(arm_tag) == "left" else [0.05, 0, 0, 0.99]
        return target.tolist()

    def _lift_basket_if_requested(self, arm_tag):
        if not bool(self.LIFT_BASKET_AFTER_PLACE) or not self.plan_success:
            return
        lift_arm = ArmTag(arm_tag).opposite
        self.move(
            self.back_to_origin(arm_tag),
            self.grasp_actor(self.basket, arm_tag=lift_arm, pre_grasp_dis=0.08),
        )
        self.move(self.move_by_displacement(arm_tag=lift_arm, z=0.05))

    def _object_lifted_after_pick(self):
        object_z = float(self.object.get_pose().p[2])
        return bool(object_z - float(self.object_start_height) > float(self.PICK_SUCCESS_Z_DELTA))

    def _clear_rotate_target_search_history(self, object_key):
        key = str(object_key)
        state = self.discovered_objects.get(key, None)
        if state is not None:
            state.update(
                {
                    "discovered": False,
                    "visible_now": False,
                    "first_seen_frame": None,
                    "last_seen_frame": None,
                    "last_seen_subtask": 0,
                    "last_seen_stage": 0,
                    "last_uv_norm": None,
                    "last_world_point": None,
                }
            )
        if key in self.visible_objects:
            self.visible_objects[key] = False

    def _prepare_basket_rotate_search(self):
        return None

    def _should_enforce_rotate_stage1_search_order(self, subtask_idx, subtask_def=None):
        return bool(int(subtask_idx) == 2 and self.object_layer == "lower" and self.basket_layer == "upper")

    def _should_skip_rotate_head_home_reset(self, subtask_idx, prev_subtask_idx=None):
        if prev_subtask_idx is None or int(prev_subtask_idx) != 1:
            return False
        return bool(
            int(subtask_idx) == 2
            and self.object_layer == "lower"
            and self.basket_layer == "upper"
            and self._has_pending_lower_rotate_search_states()
        )

    def play_once(self):
        prev_subtask_idx = None
        object_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=float(self.SCAN_Z_BIAS + self.table_z_bias),
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )
        if object_key is None:
            self.plan_success = False
            arm_tag = self.arm_tag
        else:
            # print("pick_object")
            arm_tag = pick_object(
                self,
                1,
                "A",
                self.object,
                self.object_layer,
                arm_tag=self.arm_tag,
                lower_grasp_kwargs={
                    "pre_grasp_dis": self._get_object_pre_grasp_dis(),
                    "gripper_pos": self._get_object_gripper_pos(),
                },
            )
            # print("after pick_object")
            if self.plan_success and not self._object_lifted_after_pick():
                self.plan_success = False
            if self.plan_success:
                prev_subtask_idx = 1

        if self.plan_success:
            self._prepare_basket_rotate_search()
            basket_key = self.search_and_focus_rotate_subtask(
                2,
                scan_r=self.SCAN_R,
                scan_z=float(self.SCAN_Z_BIAS + self.table_z_bias),
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if basket_key is None:
                self.plan_success = False
            else:
                place_ok = place_object(
                    self,
                    2,
                    "A",
                    self.object,
                    arm_tag,
                    self._basket_target_pose(arm_tag),
                    self.basket_layer,
                    place_kwargs={
                        "dis": 0.02,
                        "is_open": True,
                        "constrain": "free",
                    },
                    focus_object_key=basket_key,
                    retreat_after_release=False,
                    return_after_upper_release=False,
                )
                if place_ok:
                    prev_subtask_idx = 2
                self._lift_basket_if_requested(arm_tag)

        self.info["info"] = {
            "{A}": self._natural_model_label(self.object_name),
            "{B}": self._natural_model_label(self.basket_name, fallback="bread basket"),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        obj_p = np.array(self.object.get_pose().p, dtype=np.float64).reshape(3)
        basket_p = np.array(self.basket.get_pose().p, dtype=np.float64).reshape(3)
        return bool(np.linalg.norm(obj_p - basket_p) < self.SUCCESS_DIST)
