from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien.core as sapien
from ._GLOBAL_CONFIGS import left_check_pose
from ._info_task_helpers import sample_info_color_specs


class check_block_color(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"

    BLOCK_RLIM = (0.44, 0.50)
    BLOCK_THETA_RATIO = 1.0
    BLOCK_MIN_ABS_THETA = 0.10
    BLOCK_MAX_SAMPLE_TRIES = 160
    BLOCK_HALF_SIZE = (0.024, 0.024, 0.024)
    BLOCK_COLOR = (0.72, 0.72, 0.72)
    BACK_PAD_AXIS = "y"
    BACK_PAD_HALF_SIZE = (0.012, 0.0015, 0.012)
    BACK_PAD_PROTRUSION = 0.00005
    BACK_PAD_SIGN = 1.0

    # Match the pad radial range used by move_stapler_pad_rotate_view.
    PAD_RLIM = (0.32, 0.42)
    PAD_HALF_SIZE = (0.04, 0.04, 0.0005)
    PAD_THETA_RATIO = 0.55
    TARGET_PAD_THETA_DEADBAND = 0.05

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    PLACE_RELEASE_CLEARANCE = 0.035
    INSPECT_HOLD_STEPS = 8

    PAD_COUNT = 3
    PAD_SPECS = ()

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        self.check_block_color_override = kwargs.pop(
            "check_block_color_override",
            kwargs.pop("inspect_underside_color_override", None),
        )
        super()._init_task_env_(**kwargs)

    def _sample_pad_specs(self):
        required = None
        if self.check_block_color_override is not None:
            required = str(self.check_block_color_override).strip().lower()
        color_specs = sample_info_color_specs(int(self.PAD_COUNT), required_label=required)
        return tuple((label, label.upper(), color) for label, color in color_specs)

    def _pad_layout(self):
        theta_half = rotate_theta_half(self)
        theta_abs = float(theta_half * self.PAD_THETA_RATIO)
        pad_thetas = np.linspace(-theta_abs, theta_abs, len(self.PAD_SPECS))
        pad_rs = np.full(len(self.PAD_SPECS), float(np.mean(self.PAD_RLIM)))
        return list(zip(pad_rs.tolist(), pad_thetas.tolist()))

    def _pose_from_cyl(self, r, theta, z, qpos=None, rotate_rand=False, rotate_lim=None):
        if qpos is None:
            qpos = [1, 0, 0, 0]
        if rotate_lim is None:
            rotate_lim = [0, 0, 0]
        return rand_pose_cyl(
            rlim=[float(r), float(r)],
            thetalim=[float(theta), float(theta)],
            zlim=[float(z), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=qpos,
            rotate_rand=rotate_rand,
            rotate_lim=rotate_lim,
        )

    def _target_block_pose(self):
        block_z = 0.741 + float(self.BLOCK_HALF_SIZE[2])
        target_pad_cyl = world_to_robot(self.target_pad.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        target_theta = float(target_pad_cyl[1])
        theta_half = float(rotate_theta_half(self)) * float(self.BLOCK_THETA_RATIO)
        for _ in range(int(self.BLOCK_MAX_SAMPLE_TRIES)):
            theta = float(np.random.uniform(-theta_half, theta_half))
            if abs(theta) <= float(self.BLOCK_MIN_ABS_THETA):
                continue
            if abs(theta - target_theta) <= 0.20:
                continue
            return self._pose_from_cyl(
                r=float(np.random.uniform(*self.BLOCK_RLIM)),
                theta=theta,
                z=block_z,
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, 0.75],
            )
        theta = -0.35 if abs(target_theta + 0.35) > abs(target_theta - 0.35) else 0.35
        return self._pose_from_cyl(
            r=float(np.mean(self.BLOCK_RLIM)),
            theta=theta,
            z=block_z,
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0.0, 0.0, 0.75],
        )

    def _create_back_pad_block(self, pose, pad_color):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name("target_block")
        entity.set_pose(pose)

        half_size = np.array(self.BLOCK_HALF_SIZE, dtype=np.float32)
        rigid_component = sapien.physx.PhysxRigidDynamicComponent()
        rigid_component.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half_size,
                material=scene.default_physical_material,
            )
        )

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(
            sapien.render.RenderShapeBox(
                half_size,
                sapien.render.RenderMaterial(base_color=[*self.BLOCK_COLOR, 1.0]),
            )
        )

        pad_half_size = np.array(self.BACK_PAD_HALF_SIZE, dtype=np.float32)
        pad_axis = str(getattr(self, "BACK_PAD_AXIS", "y")).lower()
        if pad_axis not in {"x", "y", "z"}:
            pad_axis = "y"
        axis_idx = {"x": 0, "y": 1, "z": 2}[pad_axis]
        pad_sign = 1.0 if float(self.BACK_PAD_SIGN) >= 0.0 else -1.0
        pad_offset = np.zeros(3, dtype=np.float64)
        pad_offset[axis_idx] = pad_sign * (
            float(self.BLOCK_HALF_SIZE[axis_idx])
            + float(pad_half_size[axis_idx])
            - float(self.BACK_PAD_PROTRUSION)
        )
        back_pad = sapien.render.RenderShapeBox(
            pad_half_size,
            sapien.render.RenderMaterial(base_color=[*tuple(float(v) for v in pad_color[:3]), 1.0]),
        )
        back_pad.set_local_pose(
            sapien.Pose(
                pad_offset.tolist(),
                [1.0, 0.0, 0.0, 0.0],
            )
        )
        render_component.attach(back_pad)

        entity.add_component(rigid_component)
        entity.add_component(render_component)
        scene.add_entity(entity)

        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": list(self.BLOCK_HALF_SIZE),
            "scale": list(self.BLOCK_HALF_SIZE),
            "target_pose": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            "contact_points_pose": [
                [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point of the colored pad on the back side of the gray block."],
        }
        return Actor(entity, data)

    def _get_target_arm_tag(self):
        pad_cyl = world_to_robot(self.target_pad.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)
        return ArmTag("left" if pad_cyl[1] > float(self.TARGET_PAD_THETA_DEADBAND) else "right")

    def _move_block_over_target_pad(self, arm_tag):
        ee_pose = np.array(
            self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(),
            dtype=np.float64,
        )
        block_pos = np.array(self.block.get_pose().p, dtype=np.float64)
        ee_to_block = block_pos - ee_pose[:3]

        target_center = np.array(self.target_pad.get_pose().p, dtype=np.float64)
        target_center[2] += (
            float(self.PAD_HALF_SIZE[2])
            + float(self.BLOCK_HALF_SIZE[2])
            + float(self.PLACE_RELEASE_CLEARANCE)
        )

        release_pose = ee_pose.copy()
        release_pose[:3] = target_center - ee_to_block
        return self.move_to_pose(arm_tag, release_pose)

    def _configure_rotate_subtask_plan(self):
        registry = {"A": self.block}
        for color_name, _, _ in self.PAD_SPECS:
            registry[self.pad_keys[color_name]] = self.sort_pads[color_name]

        target_pad_key = self.pad_keys[self.target_color_name]
        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_target_block",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": True,
                    "done_when": "target_block_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "inspect_target_block_backside",
                    "instruction_idx": 2,
                    "search_target_keys": [],
                    "action_target_keys": ["A"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "target_block_backside_seen",
                    "next_subtask_id": 3,
                },
                {
                    "id": 3,
                    "name": "restore_target_block_pose",
                    "instruction_idx": 3,
                    "search_target_keys": [],
                    "action_target_keys": ["A"],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": ["A"],
                    "allow_stage2_from_memory": False,
                    "done_when": "target_block_pose_restored",
                    "next_subtask_id": 4,
                },
                {
                    "id": 4,
                    "name": "place_target_block_on_matching_pad",
                    "instruction_idx": 4,
                    "search_target_keys": [target_pad_key],
                    "action_target_keys": ["A", target_pad_key],
                    "required_carried_keys": ["A"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "target_block_on_matching_pad",
                    "next_subtask_id": -1,
                },
            ],
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.PAD_SPECS = self._sample_pad_specs()

        self.sort_pads = {}
        self.pad_keys = {}
        self.pad_centers = {}
        for (color_name, pad_key, color_value), (pad_r, pad_theta) in zip(self.PAD_SPECS, self._pad_layout()):
            pad_pose = self._pose_from_cyl(
                r=pad_r,
                theta=pad_theta,
                z=0.741,
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            pad = create_box(
                scene=self,
                pose=pad_pose,
                half_size=self.PAD_HALF_SIZE,
                color=color_value,
                name=f"{color_name}_pad",
                is_static=True,
            )
            self.sort_pads[color_name] = pad
            self.pad_keys[color_name] = pad_key
            self.pad_centers[color_name] = pad.get_pose().p.tolist()

        if self.check_block_color_override is None:
            target_idx = int(np.random.choice(len(self.PAD_SPECS)))
        else:
            target_label = str(self.check_block_color_override).strip().lower()
            target_idx = [idx for idx, spec in enumerate(self.PAD_SPECS) if spec[0] == target_label][0]
        self.target_color_name, _, self.target_color_value = self.PAD_SPECS[target_idx]
        self.target_pad = self.sort_pads[self.target_color_name]
        # self.arm_tag = self._get_target_arm_tag()
        self.arm_tag = "left"

        block_pose = self._target_block_pose()
        self.block = self._create_back_pad_block(block_pose, self.target_color_value)

        self.add_prohibit_area(self.block, padding=0.07)
        for pad in self.sort_pads.values():
            self.add_prohibit_area(pad, padding=0.10)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        block_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=self.SCAN_R,
            scan_z=self.SCAN_Z_BIAS + self.table_z_bias,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )

        arm_tag = self.arm_tag
        self.enter_rotate_action_stage(1, focus_object_key=(block_key or "A"))
        self.move(self.grasp_actor(self.block, arm_tag=arm_tag, pre_grasp_dis=0.09, contact_point_id=0))
        self._set_carried_object_keys(["A"])
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self.inspect_return_joint_state = np.array(self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64).tolist()
        self.complete_rotate_subtask(1, carried_after=["A"])

        self.begin_rotate_subtask(2)
        self.enter_rotate_action_stage(2, focus_object_key="A")
        self.move((arm_tag, [Action(arm_tag, "move_joint", target_pose=left_check_pose)]))
        self.delay(int(self.INSPECT_HOLD_STEPS), save_freq=1)
        self.complete_rotate_subtask(2, carried_after=["A"])

        self.begin_rotate_subtask(3)
        self.enter_rotate_action_stage(3, focus_object_key="A")
        self.move((arm_tag, [Action(arm_tag, "move_joint", target_joint_pos=self.inspect_return_joint_state)]))
        self.complete_rotate_subtask(3, carried_after=["A"])

        target_pad_key = self.pad_keys[self.target_color_name]
        pad_key = self.search_and_focus_rotate_subtask(
            4,
            scan_r=self.SCAN_R,
            scan_z=self.SCAN_Z_BIAS + self.table_z_bias,
            joint_name_prefer=self.SCAN_JOINT_NAME,
        )

        target_pose = self.target_pad.get_functional_point(1, "pose")
        self.enter_rotate_action_stage(4, focus_object_key=(pad_key or target_pad_key))
        self.move(self._move_block_over_target_pad(arm_tag))
        self.move(self.open_gripper(arm_tag))
        self._set_carried_object_keys([])
        self.complete_rotate_subtask(4, carried_after=[])

        self.info["info"] = {
            "{A}": f"gray target block with a {self.target_color_name} pad on its back side",
            "{B}": f"{self.target_color_name} pad",
            "{C}": self.target_color_name,
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        block_pos = self.block.get_pose().p
        target_pos = self.target_pad.get_pose().p
        return bool(
            np.all(np.abs(block_pos[:2] - target_pos[:2]) < np.array([0.055, 0.055]))
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
